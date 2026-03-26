"""VBAR-based dynamic VRAM offloading for the LLaMA model.

When ComfyUI's Dynamic VRAM (aimdo) is available, model weights are placed
in a Virtual Base Address Register (VBAR) and faulted in on demand during
each layer's forward pass.  When VRAM is insufficient the allocator evicts
lower-priority weights automatically — no manual .to("cpu") needed.

When aimdo is NOT available (e.g. AMD, old PyTorch, missing package) this
module is a no-op and the manual offloading path in inference.py is used
instead.
"""

import logging
from typing import Optional

logger = logging.getLogger("FishAudioS2")

_VBAR_AVAILABLE = False

try:
    import comfy_aimdo
    from comfy_aimdo.model_vbar import (
        ModelVBAR,
        vbar_fault,
        vbar_unpin,
        vbar_signature_compare,
    )

    _VBAR_AVAILABLE = True
except ImportError:
    comfy_aimdo = None
    ModelVBAR = None
    vbar_fault = None
    vbar_unpin = None
    vbar_signature_compare = None


def is_vbar_available() -> bool:
    return _VBAR_AVAILABLE


class VBARWeightManager:
    """Manages model weights inside a ComfyUI VBAR for demand-based offloading.

    Usage
    -----
    1. Create a VBARWeightManager after model loading.
    2. Call ``prepare_model()`` to copy every weight into VBAR-backed storage.
    3. Before each layer forward, call ``fault_layer_weights()``.
    4. After each layer forward, call ``unpin_layer_weights()``.

    The manager stores a CPU copy of every weight so it can re-populate a
    GPU tensor when the allocator evicts a weight (the "offloaded" path).
    """

    def __init__(self, device_index: int = 0):
        if not _VBAR_AVAILABLE:
            raise RuntimeError(
                "VBARWeightManager requires comfy_aimdo. "
                "Install ComfyUI's Dynamic VRAM or use manual offloading."
            )

        import torch

        gpu_total = torch.cuda.get_device_properties(device_index).total_memory
        self.vbar = ModelVBAR(gpu_total * 5, device=device_index)
        self.device_index = device_index
        self._cpu_weights: dict[str, "torch.Tensor"] = {}
        self._vbar_ptrs: dict[str, tuple] = {}
        self._signatures: dict[str, object] = {}
        self._layer_groups: dict[str, list[str]] = {}
        self._active_layer: Optional[str] = None
        logger.info(
            f"VBAR created for device {device_index}, virtual size {gpu_total * 5 / 1024**3:.1f} GB"
        )

    # ------------------------------------------------------------------
    # Preparation
    # ------------------------------------------------------------------

    def prepare_model(self, model) -> None:
        """Scan *model* and copy every leaf weight into the VBAR.

        Weights are grouped by their parent layer name so we can
        fault / unpin an entire layer at once during forward passes.
        """
        import torch

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self._add_weight(name, param.data, torch.float32)

        for name, buf in model.named_buffers():
            if buf is None or name.endswith(("kv_cache.k_cache", "kv_cache.v_cache")):
                continue
            if buf.dtype in (torch.bool, torch.int, torch.long, torch.int8):
                continue
            self._add_weight(name, buf.data, buf.dtype)

        self._group_by_layer()
        logger.info(
            f"VBAR: {len(self._cpu_weights)} weights prepared, "
            f"{len(self._layer_groups)} layer groups"
        )

    def _add_weight(self, name: str, tensor: "torch.Tensor", dtype) -> None:
        import torch

        cpu_copy = tensor.detach().cpu().to(dtype).contiguous()
        nbytes = cpu_copy.numel() * cpu_copy.element_size()
        vbar_ptr = self.vbar.alloc(nbytes)
        self._cpu_weights[name] = cpu_copy
        self._vbar_ptrs[name] = vbar_ptr

    def _group_by_layer(self) -> None:
        for name in self._vbar_ptrs:
            parts = name.rsplit(".", 1)
            layer_name = parts[0] if len(parts) > 1 else name
            self._layer_groups.setdefault(layer_name, []).append(name)

    # ------------------------------------------------------------------
    # Layer-level fault / unpin
    # ------------------------------------------------------------------

    def fault_layer_weights(self, layer_name: str) -> dict[str, "torch.Tensor"]:
        """Fault in all weights for *layer_name*.

        Returns a dict mapping weight name -> (vbar_ptr, gpu_tensor | None).
        If a weight was successfully faulted the value is a VBAR-backed tensor;
        if it was offloaded the value is ``None`` (caller must use CPU fallback).
        """
        import torch

        weights_for_layer = self._layer_groups.get(layer_name, [])
        result = {}
        for wname in weights_for_layer:
            ptr = self._vbar_ptrs[wname]
            sig = vbar_fault(ptr)
            if sig is not None:
                if not vbar_signature_compare(sig, self._signatures.get(wname)):
                    gpu_tensor = comfy_aimdo.torch.aimdo_to_tensor(
                        ptr, torch.device(f"cuda:{self.device_index}")
                    )
                    cpu = self._cpu_weights[wname]
                    gpu_tensor = gpu_tensor.view(cpu.dtype).view(cpu.shape)
                    gpu_tensor.copy_(cpu)
                    self._signatures[wname] = sig
                result[wname] = ptr
            else:
                result[wname] = None
        return result

    def unpin_layer_weights(self, layer_name: str) -> None:
        weights_for_layer = self._layer_groups.get(layer_name, [])
        for wname in weights_for_layer:
            ptr = self._vbar_ptrs[wname]
            if ptr is not None:
                vbar_unpin(ptr)

    def get_cpu_weight(self, name: str) -> "torch.Tensor":
        return self._cpu_weights[name]

    def prioritize(self) -> None:
        self.vbar.prioritize()

    def analyze(self) -> None:
        comfy_aimdo.control.analyze()
