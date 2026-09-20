"""Cost profiling: latency (synchronised, after warm-up), peak memory."""
from __future__ import annotations

import resource
import time

import torch


@torch.no_grad()
def latency_ms(model, in_dim: int, device: str, batch: int, iters: int = 200, warmup: int = 50) -> float:
    model.eval()
    x = torch.randn(batch, in_dim, device=device)
    for _ in range(warmup):
        model(x)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        model(x)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def peak_memory_mb(device: str) -> dict:
    out = {"peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    if device.startswith("cuda"):
        out["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 2**20
    return out
