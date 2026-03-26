import gc
import queue
from typing import Generator

import numpy as np
import torch
from loguru import logger

from fish_speech.inference_engine.reference_loader import ReferenceLoader
from fish_speech.inference_engine.utils import InferenceResult, wav_chunk_header
from fish_speech.inference_engine.vq_manager import VQManager
from fish_speech.models.dac.modded_dac import DAC
from fish_speech.models.text2semantic.inference import (
    GenerateRequest,
    GenerateResponse,
    WrappedGenerateResponse,
)
from fish_speech.utils import autocast_exclude_mps, set_seed
from fish_speech.utils.schema import ServeTTSRequest


class TTSInferenceEngine(ReferenceLoader, VQManager):
    def __init__(
        self,
        llama_queue: queue.Queue,
        decoder_model: DAC,
        precision: torch.dtype,
        compile: bool,
    ) -> None:

        super().__init__()

        self.llama_queue = llama_queue
        self.decoder_model = decoder_model
        self.precision = precision
        self.compile = compile

    @torch.inference_mode()
    def inference(self, req: ServeTTSRequest) -> Generator[InferenceResult, None, None]:
        """
        Main inference function:
        - Loads the reference audio and text.
        - Calls the LLAMA model for inference.
        - Decodes the VQ tokens to audio.
        """

        ref_id: str | None = req.reference_id
        prompt_tokens, prompt_texts = [], []
        # Load the reference audio and text based on id or hash
        if ref_id is not None:
            prompt_tokens, prompt_texts = self.load_by_id(ref_id, req.use_memory_cache)

        elif req.references:
            prompt_tokens, prompt_texts = self.load_by_hash(
                req.references, req.use_memory_cache
            )

        # Set the random seed if provided
        if req.seed is not None:
            set_seed(req.seed)
            logger.warning(f"set seed: {req.seed}")

        # Get the symbolic tokens from the LLAMA model
        response_queue = self.send_Llama_request(req, prompt_tokens, prompt_texts)

        # Get the sample rate from the decoder model
        if hasattr(self.decoder_model, "spec_transform"):
            sample_rate = self.decoder_model.spec_transform.sample_rate
        else:
            sample_rate = self.decoder_model.sample_rate

        # If streaming, send the header
        if req.streaming:
            yield InferenceResult(
                code="header",
                audio=(
                    sample_rate,
                    np.array(wav_chunk_header(sample_rate=sample_rate)),
                ),
                error=None,
            )

        segments = []
        deferred_codes = []
        decoder_device = self.decoder_model.device

        # Check if VBAR (Dynamic VRAM) is managing the LLaMA model.
        _vbar_active = getattr(self, "_vbar_active", False)
        use_cpu_offload = (
            req.low_vram_mode or (len(req.text) > 500 and not req.streaming)
        ) and not _vbar_active

        if _vbar_active:
            logger.info(
                "VBAR active — decoder stays on GPU, LLaMA weights managed by allocator"
            )

        if use_cpu_offload and decoder_device.type == "cuda":
            self.decoder_model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                vram_after_decoder_offload = torch.cuda.memory_allocated() / 1024**3
                logger.info(
                    f"Low-VRAM mode: decoder offloaded to CPU, VRAM: {vram_after_decoder_offload:.2f} GB (LLaMA only)"
                )

        while True:
            wrapped_result = None
            while wrapped_result is None:
                try:
                    wrapped_result = response_queue.get(timeout=0.1)
                except Exception:
                    pass
                try:
                    import comfy.model_management as _mm

                    _mm.throw_exception_if_processing_interrupted()
                except ImportError:
                    pass

            if wrapped_result.status == "error":
                yield InferenceResult(
                    code="error",
                    audio=None,
                    error=(
                        wrapped_result.response
                        if isinstance(wrapped_result.response, Exception)
                        else Exception("Unknown error")
                    ),
                )
                break

            if not isinstance(wrapped_result.response, GenerateResponse):
                raise TypeError(
                    "Expected GenerateResponse, got {type(wrapped_result.response).__name__}"
                )

            result: GenerateResponse = wrapped_result.response
            if result.action != "next":
                if use_cpu_offload:
                    deferred_codes.append(result.codes.cpu())
                    yield InferenceResult(
                        code="segment",
                        audio=None,
                        error=None,
                    )
                else:
                    segment = self.get_audio_segment(result)
                    if req.streaming:
                        yield InferenceResult(
                            code="segment",
                            audio=(sample_rate, segment),
                            error=None,
                        )
                    segments.append(segment)
            else:
                break

        if deferred_codes:
            offload_resp = queue.Queue()
            self.llama_queue.put(
                GenerateRequest(
                    request={"__offload__": "cpu"},
                    response_queue=offload_resp,
                )
            )
            try:
                offload_resp.get(timeout=30)
                if torch.cuda.is_available():
                    vram_after_llama_offload = torch.cuda.memory_allocated() / 1024**3
                    logger.info(
                        f"Long-form: LLaMA offloaded to CPU, VRAM: {vram_after_llama_offload:.2f} GB (both models offloaded)"
                    )
            except queue.Empty:
                logger.warning(
                    "Long-form: LLaMA offload timed out, proceeding with caution."
                )

            self.decoder_model.to(decoder_device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                vram_before_decode = torch.cuda.memory_allocated() / 1024**3
                logger.info(
                    f"Long-form: decoder on GPU, VRAM: {vram_before_decode:.2f} GB, decoding {len(deferred_codes)} segments"
                )
            for codes in deferred_codes:
                segment = self._decode_codes(codes)
                segments.append(segment)
            logger.info("Long-form: all segments decoded")

        # Release CUDA cached blocks so other ComfyUI nodes can use the VRAM.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Edge case: no audio generated
        if len(segments) == 0:
            yield InferenceResult(
                code="error",
                audio=None,
                error=RuntimeError("No audio generated, please check the input text."),
            )
        else:
            # Streaming or not, return the final audio
            audio = np.concatenate(segments, axis=0)
            yield InferenceResult(
                code="final",
                audio=(sample_rate, audio),
                error=None,
            )

        return None

    def send_Llama_request(
        self, req: ServeTTSRequest, prompt_tokens: list, prompt_texts: list
    ) -> queue.Queue:
        """
        Send a request to the LLAMA model to generate the symbolic tokens.
        """

        # Prepare the request
        request = dict(
            device=self.decoder_model.device,
            max_new_tokens=req.max_new_tokens,
            text=req.text,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            temperature=req.temperature,
            compile=self.compile,
            chunk_length=req.chunk_length,
            prompt_tokens=prompt_tokens,
            prompt_text=prompt_texts,
            max_context_batches=req.max_context_batches,
        )

        # Create a queue to get the response
        response_queue = queue.Queue()

        # Send the request to the LLAMA model
        self.llama_queue.put(
            GenerateRequest(
                request=request,
                response_queue=response_queue,
            )
        )

        return response_queue

    def get_audio_segment(self, result: GenerateResponse) -> np.ndarray:
        with autocast_exclude_mps(
            device_type=self.decoder_model.device.type, dtype=self.precision
        ):
            segment = self.decode_vq_tokens(codes=result.codes)

        return segment.float().cpu().numpy()

    def _decode_codes(self, codes: torch.Tensor) -> np.ndarray:
        codes = codes.to(self.decoder_model.device)
        with autocast_exclude_mps(
            device_type=self.decoder_model.device.type, dtype=self.precision
        ):
            segment = self.decode_vq_tokens(codes=codes)

        return segment.float().cpu().numpy()
