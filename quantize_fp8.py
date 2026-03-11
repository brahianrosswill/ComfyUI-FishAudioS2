"""
Fish Audio S2 Pro — FP8 Weight Quantization Script
====================================================
Quantizes the s2-pro LLaMA backbone to FP8 (float8_e4m3fn) using torchao.
Handles sharded safetensors automatically (merges both shards into one file).
Creates a NEW folder — never overwrites the original model.

Output folder layout (ready to upload to HuggingFace as-is):
    s2-pro-fp8/
        model.safetensors          <- merged, quantized (FP8 weights + fp32 scales + bf16 rest)
        config.json                <- copied from s2-pro
        tokenizer.json             <- copied from s2-pro
        tokenizer_config.json      <- copied from s2-pro
        special_tokens_map.json    <- copied from s2-pro
        chat_template.jinja        <- copied from s2-pro
        codec.pth                  <- copied from s2-pro (unchanged, bf16)
        quantization_info.json     <- metadata about this quantization

Key format in model.safetensors:
    "<layer_name>"        -> float8_e4m3fn  (quantized weight)
    "<layer_name>.scale"  -> float32        (per-row dequant scale)
    "<other tensors>"     -> bfloat16       (embeddings, norms, etc.)

Requirements:
    - torchao (already installed)
    - safetensors (already installed)
    - CUDA GPU with compute capability 8.9+ recommended (RTX 4090 / 5090)

Usage (run from the ComfyUI-FishAudioS2 custom node folder):

    "E:\\...\\python_embeded\\python.exe" quantize_fp8.py ^
        --input  "E:\\...\\ComfyUI\\models\\fishaudioS2\\s2-pro" ^
        --output "E:\\...\\ComfyUI\\models\\fishaudioS2\\s2-pro-fp8"
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Quantize s2-pro to FP8 (float8_e4m3fn)")
    p.add_argument("--input",  "-i", required=True,
                   help="Path to s2-pro folder (contains config.json + sharded safetensors)")
    p.add_argument("--output", "-o", required=True,
                   help="Output folder for FP8 model (must NOT be the same as input)")
    p.add_argument("--device", default="cuda",
                   help="Device for quantization (default: cuda)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Setup sys.path so fish_speech imports work from embedded python
# ---------------------------------------------------------------------------

def setup_paths():
    script_dir = Path(__file__).resolve().parent
    for sub in ("fish_speech_src", "auto_gptq_src"):
        p = script_dir / sub
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
            print(f"[FP8] sys.path += {p}")


# ---------------------------------------------------------------------------
# GPU check
# ---------------------------------------------------------------------------

def check_gpu(device: str):
    import torch
    if not torch.cuda.is_available() or device == "cpu":
        print("[FP8] WARNING: No CUDA — quantization on CPU will be very slow.")
        return
    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    print(f"[FP8] GPU : {name}")
    print(f"[FP8] CC  : {cap[0]}.{cap[1]}", end="  ")
    if cap[0] > 8 or (cap[0] == 8 and cap[1] >= 9):
        print("(native FP8 matmuls supported ✓)")
    else:
        print("(FP8 matmuls NOT natively supported — needs Ada/Blackwell i.e. RTX 4090/5090)")


# ---------------------------------------------------------------------------
# Quantize and save
# ---------------------------------------------------------------------------

def quantize_and_save(input_path: Path, output_path: Path, device: str):
    import torch
    from torchao.quantization import quantize_, Float8WeightOnlyConfig
    from safetensors.torch import save_file
    from torchao.quantization import Float8Tensor

    # ------------------------------------------------------------------ #
    # 1. Load model — from_pretrained handles sharded safetensors natively
    # ------------------------------------------------------------------ #
    print(f"\n[FP8] Loading model from : {input_path}")
    print(f"[FP8] (merging shards automatically...)")
    t0 = time.time()

    from fish_speech.models.text2semantic.llama import DualARTransformer
    model = DualARTransformer.from_pretrained(str(input_path), load_weights=True)
    model = model.to(device=device, dtype=torch.bfloat16).eval()

    t_load = time.time() - t0
    total_params  = sum(p.numel() for p in model.parameters())
    linear_params = sum(
        p.numel()
        for m in model.modules()
        if isinstance(m, torch.nn.Linear)
        for p in m.parameters()
    )
    other_params = total_params - linear_params

    print(f"[FP8] Loaded in {t_load:.1f}s")
    print(f"[FP8] Total params    : {total_params  / 1e9:.3f} B")
    print(f"[FP8] Linear -> fp8   : {linear_params / 1e9:.3f} B  ({linear_params*100//total_params}%)")
    print(f"[FP8] Other  -> bf16  : {other_params  / 1e9:.3f} B  ({other_params*100//total_params}%)")

    # ------------------------------------------------------------------ #
    # 2. Quantize all nn.Linear weights to float8_e4m3fn
    # ------------------------------------------------------------------ #
    print(f"\n[FP8] Quantizing linear layers ...")
    t1 = time.time()
    quantize_(model, Float8WeightOnlyConfig())
    print(f"[FP8] Done in {time.time() - t1:.1f}s")

    # ------------------------------------------------------------------ #
    # 3. Build a plain state-dict with no torchao subclass tensors
    #    Float8Tensor exposes:
    #      .qdata  -> float8_e4m3fn  weight
    #      .scale  -> float32        per-row scale  (shape [out, 1])
    # ------------------------------------------------------------------ #
    print(f"\n[FP8] Extracting tensors ...")
    save_sd = {}
    n_fp8 = n_bf16 = 0

    for name, param in model.named_parameters():
        if isinstance(param, Float8Tensor):
            save_sd[name]              = param.qdata.cpu().contiguous()
            save_sd[f"{name}.scale"]   = param.scale.cpu().contiguous()
            n_fp8 += 1
        else:
            save_sd[name] = param.detach().cpu().to(torch.bfloat16).contiguous()
            n_bf16 += 1

    # Buffers (freqs_cis, causal_mask, etc.) — keep as-is
    for name, buf in model.named_buffers():
        key = f"_buf.{name}"
        if key not in save_sd:
            save_sd[key] = buf.detach().cpu().contiguous()

    print(f"[FP8] fp8 weight tensors : {n_fp8}  (+ {n_fp8} scale tensors)")
    print(f"[FP8] bf16 tensors       : {n_bf16}")
    print(f"[FP8] total tensors      : {len(save_sd)}")

    # ------------------------------------------------------------------ #
    # 4. Create output folder and save model.safetensors
    #    The loader checks for:
    #      1. model.safetensors.index.json  (sharded)
    #      2. model.safetensors             (single — what we produce)
    #      3. model.pth                     (legacy)
    # ------------------------------------------------------------------ #
    output_path.mkdir(parents=True, exist_ok=True)

    out_file = output_path / "model.safetensors"
    print(f"\n[FP8] Saving -> {out_file}")
    t2 = time.time()
    save_file(save_sd, str(out_file))
    size_gb = out_file.stat().st_size / 1e9
    orig_gb = total_params * 2 / 1e9   # bf16 reference size
    print(f"[FP8] Saved in {time.time() - t2:.1f}s")
    print(f"[FP8] File size : {size_gb:.2f} GB  (original bf16 was ~{orig_gb:.2f} GB)")

    # ------------------------------------------------------------------ #
    # 5. Copy all supporting files from s2-pro unchanged
    # ------------------------------------------------------------------ #
    print(f"\n[FP8] Copying supporting files ...")
    support_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "README.md",
        "LICENSE.md",
    ]
    for fname in support_files:
        src = input_path / fname
        if src.is_file():
            shutil.copy2(src, output_path / fname)
            print(f"[FP8]   {fname}")

    # Codec — must be present for inference, keep bf16
    codec_names = [
        "codec.pth",
        "firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
        "decoder.pth",
        "vocoder.pth",
    ]
    for fname in codec_names:
        src = input_path / fname
        if src.is_file():
            shutil.copy2(src, output_path / fname)
            print(f"[FP8]   {fname}  (codec, kept as bf16)")
            break
    else:
        print("[FP8]   WARNING: codec file not found — copy codec.pth manually!")

    # ------------------------------------------------------------------ #
    # 6. Write quantization_info.json
    # ------------------------------------------------------------------ #
    import torchao
    info = {
        "quantization_method": "torchao_Float8WeightOnly",
        "weight_dtype": "float8_e4m3fn",
        "scale_dtype": "float32",
        "scale_granularity": "per_row",
        "activation_dtype": "bfloat16",
        "torchao_version": torchao.__version__,
        "torch_version": torch.__version__,
        "source_model": "fishaudio/s2-pro",
        "total_params_B": round(total_params / 1e9, 3),
        "fp8_linear_params_B": round(linear_params / 1e9, 3),
        "bf16_other_params_B": round(other_params / 1e9, 3),
        "output_size_GB": round(size_gb, 2),
        "key_format": {
            "<layer_name>": "float8_e4m3fn quantized weight",
            "<layer_name>.scale": "float32 per-row dequantization scale",
            "_buf.<name>": "bf16/fp32 buffer (freqs_cis, causal_mask, etc.)",
            "other": "bfloat16 (embeddings, norms, non-linear layers)"
        },
        "inference_requirements": {
            "torchao": ">= 0.8.0",
            "compute_capability": ">= 8.9 (RTX 4090 / 5090) for native FP8 matmuls"
        },
        "notes": (
            "All nn.Linear weights are float8_e4m3fn. "
            "Activations are bfloat16 (weight-only quantization). "
            "codec.pth is unchanged bfloat16."
        ),
    }
    with open(output_path / "quantization_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"[FP8]   quantization_info.json")

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    total_time = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"[FP8] Complete!  Total time: {total_time:.0f}s")
    print(f"[FP8] Output folder: {output_path}")
    print(f"[FP8] Contents:")
    for f in sorted(output_path.iterdir()):
        mb = f.stat().st_size / 1e6
        print(f"[FP8]   {f.name:<45} {mb:>8.1f} MB")
    print(f"\n[FP8] Next steps:")
    print(f"  1. Upload to HuggingFace:")
    print(f"       huggingface-cli upload YOUR_USERNAME/s2-pro-fp8 \"{output_path}\"")
    print(f"  2. Add the HF repo to HF_MODELS in nodes/loader.py")
    print(f"  3. Add FP8 loading support to loader.py (reads .scale tensors)")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    setup_paths()

    input_path  = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    # Safety checks
    if not input_path.is_dir():
        print(f"[FP8] ERROR: Input not found: {input_path}")
        sys.exit(1)
    if not (input_path / "config.json").is_file():
        print(f"[FP8] ERROR: No config.json in {input_path} — not a valid s2-pro folder")
        sys.exit(1)
    if output_path == input_path:
        print("[FP8] ERROR: --output must be different from --input (never overwrite the original!)")
        sys.exit(1)
    if output_path.exists() and any(output_path.iterdir()):
        print(f"[FP8] ERROR: Output folder already exists and is not empty: {output_path}")
        print(f"[FP8]        Delete it first or choose a different --output path.")
        sys.exit(1)

    check_gpu(args.device)
    print(f"\n[FP8] Input  : {input_path}")
    print(f"[FP8] Output : {output_path}")
    print(f"[FP8] Device : {args.device}\n")

    quantize_and_save(input_path, output_path, args.device)
