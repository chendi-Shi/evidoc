"""Read-only diagnostics for local models, GPU driver, runtime and endpoints."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from download_models import verify
from run_models import healthy, read_status


def diagnose(check_hashes: bool = False) -> dict:
    manifest = json.loads((ROOT / "models" / "manifest.json").read_text(encoding="utf-8"))
    report = {"python": sys.version.split()[0], "platform": platform.platform(), "gpu": None,
              "models": {}, "runtime": {}, "servers": read_status()}
    nvidia = shutil.which("nvidia-smi")
    if nvidia:
        result = subprocess.run([nvidia, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=15,
                                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        report["gpu"] = result.stdout.strip() or result.stderr.strip()
    for role, model in manifest["models"].items():
        path = ROOT / "models" / model["filename"]
        item = {"path": str(path), "exists": path.exists(), "expected_size": model["size"],
                "actual_size": path.stat().st_size if path.exists() else 0}
        item["complete_size"] = item["actual_size"] == item["expected_size"]
        if check_hashes:
            item["sha256_verified"] = verify(path, model["sha256"], model["size"])
        report["models"][role] = item
    for backend in ("cpu", "vulkan"):
        report["runtime"][backend] = [str(p) for p in (ROOT / "runtime" / f"llama-{backend}").rglob("llama-server.exe")]
    report["health"] = {role: healthy(role) for role in ("generation", "embedding")}
    probe = ROOT / "reports" / "training_gpu_probe.json"
    if probe.exists():
        report["training_gpu_probe"] = json.loads(probe.read_text(encoding="utf-8"))
    else:
        report["training_gpu_probe"] = {"state": "not_run", "command": "python scripts/probe_training_gpu.py"}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-models", action="store_true", help="Read all model bytes and verify SHA-256")
    args = parser.parse_args()
    print(json.dumps(diagnose(args.verify_models), ensure_ascii=False, indent=2))
