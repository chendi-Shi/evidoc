"""Exercise real CUDA forward/backward kernels before starting training.

Writes reports/training_gpu_probe.json. Does not install or change drivers.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    report = {"checked_at": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
              "success": False, "cuda_available": False}
    try:
        import torch
        report.update(torch_version=torch.__version__, cuda_build=torch.version.cuda,
                      cudnn_version=torch.backends.cudnn.version(), cuda_available=torch.cuda.is_available())
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is false")
        device = torch.device("cuda:0")
        report.update(gpu_name=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
                      total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                      bf16_supported=torch.cuda.is_bf16_supported(), architectures=torch.cuda.get_arch_list())
        torch.manual_seed(42)
        torch.cuda.reset_peak_memory_stats()
        dtype = torch.bfloat16 if report["bf16_supported"] else torch.float16
        layer = torch.nn.Linear(1024, 1024, device=device, dtype=dtype)
        optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-3)
        inputs = torch.randn(16, 128, 1024, device=device, dtype=dtype)
        losses = []
        started = time.perf_counter()
        for _ in range(8):
            optimizer.zero_grad(set_to_none=True)
            loss = layer(inputs).float().square().mean()
            loss.backward()
            if not all(torch.isfinite(parameter.grad).all().item() for parameter in layer.parameters()):
                raise RuntimeError("CUDA backward produced non-finite gradients")
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        q = torch.randn(1, 4, 128, 64, device=device, dtype=dtype, requires_grad=True)
        attention = torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
        attention.float().square().mean().backward()
        torch.cuda.synchronize()
        if not all(math.isfinite(loss) for loss in losses) or not torch.isfinite(attention).all().item():
            raise RuntimeError("CUDA kernels produced non-finite results")
        report.update(success=True, dtype=str(dtype), elapsed_seconds=time.perf_counter() - started,
                      losses=losses, sdpa_finite=bool(torch.isfinite(attention).all().item()),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except Exception as error:
        report.update(error=str(error), traceback=traceback.format_exc())
    destination = ROOT / "reports" / "training_gpu_probe.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
