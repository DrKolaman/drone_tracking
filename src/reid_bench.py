"""Benchmark DINOv2 vs DINOv3 backbones as frozen ReID embedders.

Measures, on the *actual* GPU, what matters for re-identification:
  * peak VRAM (does it even fit in 4 GB?)
  * throughput in crops/sec (a tracker hands us many person-crops per frame)
  * latency per batch
  * embedding dimension (size of the per-identity descriptor)

ReID usage is inference-only: crop -> backbone -> CLS/pooled vector ->
L2-normalize -> cosine similarity. No training, no grad. So we run everything
under inference_mode() in fp16.

DINOv3 weights are gated on Hugging Face. If a model can't be fetched
(no token / license not accepted / offline) it is reported as SKIPPED and the
rest still run. Pass HF_TOKEN env var to unlock the gated DINOv3 repos.

Usage:
    python3 src/reid_bench.py                 # default model set, res 224 + 384
    python3 src/reid_bench.py --batch 1 8 32  # sweep batch sizes
    python3 src/reid_bench.py --res 224       # single resolution
"""
from __future__ import annotations

import argparse
import time
import traceback

import torch
from transformers import AutoModel

# (label, hf_repo, gated?) — small/base only; ViT-L is borderline on 4 GB.
MODELS = [
    ("DINOv2  ViT-S/14", "facebook/dinov2-small", False),
    ("DINOv2  ViT-B/14", "facebook/dinov2-base", False),
    ("DINOv3  ViT-S/16", "facebook/dinov3-vits16-pretrain-lvd1689m", True),
    ("DINOv3  ViT-B/16", "facebook/dinov3-vitb16-pretrain-lvd1689m", True),
    ("DINOv3  ConvNeXt-T", "facebook/dinov3-convnext-tiny-pretrain-lvd1689m", True),
]


def snap(res: int, patch: int) -> int:
    """Round a requested side length to a whole number of patches."""
    return max(patch, round(res / patch) * patch)


def embed(model, pixel_values):
    """Pooled descriptor, shape (B, D). Works for ViT (CLS) and ConvNeXt."""
    out = model(pixel_values=pixel_values)
    if getattr(out, "pooler_output", None) is not None:
        return out.pooler_output
    h = out.last_hidden_state
    if h.dim() == 4:          # ConvNeXt: (B, C, H, W) -> global avg pool
        return h.mean(dim=(2, 3))
    return h[:, 0]            # ViT: CLS token


def bench_one(label, repo, batch_sizes, resolutions, device, dtype):
    model = AutoModel.from_pretrained(repo, dtype=dtype).to(device).eval()
    patch = getattr(model.config, "patch_size", 16)
    load_mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0

    rows = []
    for res in resolutions:
        side = snap(res, patch)
        for bs in batch_sizes:
            if device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            x = torch.randn(bs, 3, side, side, device=device, dtype=dtype)
            with torch.inference_mode():
                for _ in range(3):                 # warmup
                    dim = embed(model, x).shape[-1]
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                iters = 10
                for _ in range(iters):
                    embed(model, x)
                if device == "cuda":
                    torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) / iters
            peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
            rows.append((side, bs, dim, dt * 1e3, bs / dt, peak))
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return load_mem, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--res", type=int, nargs="+", default=[224, 384])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    if device == "cuda":
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Device: {name}  ({total:.1f} GB)   dtype={dtype}\n")
    else:
        print(f"Device: CPU   dtype={dtype}  (no CUDA — numbers are CPU-bound)\n")

    hdr = f"{'res':>5} {'batch':>6} {'dim':>5} {'ms/batch':>9} {'crops/s':>8} {'peak GB':>8}"
    for label, repo, gated in MODELS:
        print(f"### {label}   [{repo}]")
        try:
            load_mem, rows = bench_one(label, repo, args.batch, args.res, device, dtype)
            print(f"    weights resident: {load_mem:.2f} GB")
            print("    " + hdr)
            for side, bs, dim, ms, cps, peak in rows:
                flag = "  OOM-risk" if peak > 3.6 else ""
                print(f"    {side:>5} {bs:>6} {dim:>5} {ms:>9.1f} {cps:>8.0f} {peak:>8.2f}{flag}")
        except Exception as e:
            tag = "gated — set HF_TOKEN & accept license" if gated else "load failed"
            print(f"    SKIPPED ({tag}): {type(e).__name__}: {str(e)[:120]}")
            if not gated:
                traceback.print_exc()
        print()


if __name__ == "__main__":
    main()
