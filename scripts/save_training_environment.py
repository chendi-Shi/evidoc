"""Record exact local package versions and GPU probe evidence for reproducibility."""
from datetime import datetime, timezone
from importlib.metadata import distributions
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
report_dir = ROOT / "reports"
report_dir.mkdir(parents=True, exist_ok=True)
packages = sorted(({"name": package.metadata["Name"], "version": package.version}
                   for package in distributions()), key=lambda package: package["name"].lower())
check = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
report = {"recorded_at": datetime.now(timezone.utc).isoformat(), "python": sys.version,
          "executable": sys.executable, "platform": platform.platform(), "packages": packages,
          "pip_check": {"exit_code": check.returncode, "stdout": check.stdout.strip(), "stderr": check.stderr.strip()}}
for key, path in (("gpu_probe", report_dir / "training_gpu_probe.json"),
                  ("torch_wheel", ROOT / "runtime" / "torch-wheel.json")):
    if path.exists():
        report[key] = json.loads(path.read_text(encoding="utf-8"))
(report_dir / "training_environment.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
(report_dir / "training_requirements_lock.txt").write_text(
    "# Windows Python 3.12; install torch from the verified cu118 wheel first.\n" +
    "\n".join(f"{package['name']}=={package['version']}" for package in packages) + "\n", encoding="utf-8")
print(json.dumps({"packages": len(packages), "pip_check": report["pip_check"], "report": str(report_dir / "training_environment.json")}, indent=2))
