"""VBAR-based dynamic VRAM offloading for the LLaMA model.

Three memory management paths:

1. **VBAR explicit** (aimdo VBAR API available): Model moved to CPU.
   Weights allocated in VBAR virtual space. Per-layer fault/unpin during
   forward passes gives fine-grained control over what's in VRAM.

2. **Aimdo auto-allocator** (aimdo installed, no VBAR API): Model stays
   on GPU normally. No manual .to("cpu") calls — aimdo's custom CUDA
   allocator automatically evicts weights under VRAM pressure.

3. **Manual fallback** (no aimdo): Original .to("cpu") ping-pong
   between LLaMA and decoder.

BNB (bitsandbytes) models always use path 3 — VBAR is incompatible
with quantized layers.
"""

import logging
from typing import Optional

logger = logging.getLogger("FishAudioS2")

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

_VBAR_AVAILABLE = False
_AIMDO_AVAILABLE = False

try:
    import comfy_aimdo

    _AIMDO_AVAILABLE = True
except ImportError:
    comfy_aimdo = None

if _AIMDO_AVAILABLE:
    try:
        from comfy_aimdo.model_vbar import (
            ModelVBAR,
            vbar_fault,
            vbar_unpin,
            vbar_signature_compare,
        )

        _VBAR_AVAILABLE = True
    except ImportError:
        ModelVBAR = None
        vbar_fault = None
        vbar_unpin = None
        vbar_signature_compare = None


def is_vbar_available() -> bool:
    return _VBAR_AVAILABLE


def is_aimdo_available() -> bool:
    return _AIMDO_AVAILABLE


# ---------------------------------------------------------------------------
# VBAR Weight Manager
# ---------------------------------------------------------------------------

_SKIP_BUFFER_SUFFIXES = (
    "kv_cache.k_cache",
    "kv_cache.v_cache",
    "causal_mask",
)


class VBARWeightManager:
    """Manages model weights inside a ComfyUI VBAR.

    Flow:
    1. ``prepare_model(model)`` — moves model to CPU, allocates VBAR
       virtual space for every weight, keeps CPU copies as source of truth.
    2. ``swap_in_layer(layer, layer_name)`` — faults in VBAR weights for
       one transformer layer, temporarily replaces ``param.data`` /
       ``buf.data`` with VBAR-backed GPU tensors (or CPU→GPU fallback).
    3. ``swap_out_layer(layer, layer_name)`` — unpins VBAR weights,
       restores param/buf data to CPU tensors.
    """

    def __init__(self, device_index: int = 0):
        if not _VBAR_AVAILABLE:
            raise RuntimeError("VBARWeightManager requires comfy_aimdo VBAR API")

        import torch

        gpu_total = torch.cuda.get_device_properties(device_index).total_memory
        self.vbar = ModelVBAR(gpu_total * 5, device=device_index)
        self.device_index = device_index
        self._cpu_weights: dict[str, "torch.Tensor"] = {}
        self._vbar_ptrs: dict[str, tuple] = {}
        self._signatures: dict[str, object] = {}
        self._layer_groups: dict[str, list[str]] = {}
        logger.info(
            f"VBAR created for device {device_index}, "
            f"virtual size {gpu_total * 5 / 1024**3:.1f} GB"
        )

    # ------------------------------------------------------------------
    # Preparation
    # ------------------------------------------------------------------

    def prepare_model(self, model) -> None:
        """Move model to CPU, allocate VBAR space for all weights.

        After this call the model's parameters reside on CPU. During
        forward passes ``swap_in_layer`` temporarily moves them to GPU
        via VBAR fault.
        """
        import torch

        # Collect all weights into VBAR
        for name, param in model.named_parameters():
            self._add_weight(name, param.data)

        for name, buf in model.named_buffers():
            if buf is None:
                continue
            if any(name.endswith(s) for s in _SKIP_BUFFER_SUFFIXES):
                continue
            if buf.dtype in (
                torch.bool,
                torch.int,
                torch.long,
                torch.int8,
                torch.uint8,
            ):
                continue
            self._add_weight(name, buf.data)

        self._group_by_layer()

        # Move the entire model to CPU — VRAM is now managed by VBAR
        model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Keep small read-only buffers on GPU permanently.
        # These are indexed by CUDA tensors during forward and are too small
        # to benefit from VBAR management (~few MB total).
        _GPU_BUFFERS = ("causal_mask", "freqs_cis", "fast_freqs_cis")
        device = torch.device(f"cuda:{self.device_index}")
        for name, buf in model.named_buffers():
            if name.endswith(_GPU_BUFFERS):
                buf.data = buf.data.to(device, non_blocking=True)

        logger.info(
            f"VBAR: {len(self._cpu_weights)} weights in {len(self._layer_groups)} "
            f"layer groups, model moved to CPU"
        )

    def _add_weight(self, name: str, tensor: "torch.Tensor") -> None:
        cpu_copy = tensor.detach().cpu().contiguous()
        nbytes = cpu_copy.numel() * cpu_copy.element_size()
        vbar_ptr = self.vbar.alloc(nbytes)
        self._cpu_weights[name] = cpu_copy
        self._vbar_ptrs[name] = vbar_ptr

    def _group_by_layer(self) -> None:
        # Group weights by their parent module name (e.g. "layers.3.attention.wqkv")
        for name in self._vbar_ptrs:
            parts = name.rsplit(".", 1)
            layer_name = parts[0] if len(parts) > 1 else name
            self._layer_groups.setdefault(layer_name, []).append(name)

    # ------------------------------------------------------------------
    # Per-layer swap in / out
    # ------------------------------------------------------------------

    def swap_in_layer(self, layer: "torch.nn.Module", layer_name: str) -> None:
        """Fault in VBAR weights for *layer* and replace param/buf data."""
        import torch

        device = torch.device(f"cuda:{self.device_index}")
        prefix = layer_name + "."

        for wname, cpu in self._cpu_weights.items():
            if not wname.startswith(prefix):
                continue

            short = wname[len(layer_name) + 1 :]
            ptr = self._vbar_ptrs[wname]

            target = self._resolve_attr(layer, short)
            if target is None:
                continue

            sig = vbar_fault(ptr)
            if sig is not None:
                gpu_tensor = comfy_aimdo.torch.aimdo_to_tensor(ptr, device)
                gpu_tensor = gpu_tensor.view(cpu.dtype).view(cpu.shape)
                if not vbar_signature_compare(sig, self._signatures.get(wname)):
                    gpu_tensor.copy_(cpu)
                    self._signatures[wname] = sig
                self._set_data(layer, short, gpu_tensor)
            else:
                gpu_copy = cpu.to(device, non_blocking=True)
                self._set_data(layer, short, gpu_copy)

    def swap_out_layer(self, layer: "torch.nn.Module", layer_name: str) -> None:
        """Unpin VBAR weights and restore CPU tensors."""
        import torch

        prefix = layer_name + "."

        for wname, ptr in self._vbar_ptrs.items():
            if not wname.startswith(prefix):
                continue
            if ptr is not None:
                vbar_unpin(ptr)

            short = wname[len(layer_name) + 1 :]
            cpu = self._cpu_weights[wname]
            self._set_data(layer, short, cpu)

    # ------------------------------------------------------------------
    # Non-layer weights (embeddings, norm, output) — swapped once
    # ------------------------------------------------------------------

    def swap_in_module(self, module: "torch.nn.Module", prefix: str) -> None:
        """Swap in all VBAR weights for a top-level module (embeddings, etc.)."""
        self.swap_in_layer(module, prefix)

    def swap_out_module(self, module: "torch.nn.Module", prefix: str) -> None:
        """Swap out all VBAR weights for a top-level module."""
        self.swap_out_layer(module, prefix)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_attr(module, attr_path: str):
        """Resolve dotted attribute path on a module."""
        parts = attr_path.split(".")
        obj = module
        for p in parts[:-1]:
            obj = getattr(obj, p, None)
            if obj is None:
                return None
        return getattr(obj, parts[-1], None) if parts else None

    @staticmethod
    def _set_data(module, attr_path: str, tensor) -> None:
        """Set .data on a parameter or buffer via dotted path."""
        import torch

        parts = attr_path.split(".")
        parent = module
        for p in parts[:-1]:
            parent = getattr(parent, p)
        leaf = getattr(parent, parts[-1])
        if isinstance(leaf, torch.nn.Parameter):
            leaf.data = tensor
        else:
            # Buffer or regular tensor attribute
            parent.__dict__[parts[-1]] = tensor

    def prioritize(self) -> None:
        self.vbar.prioritize()
