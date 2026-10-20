"""Atomic lightweight progress for the local portfolio UI."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
from datetime import datetime, timezone
from uuid import uuid4

STATUS_PATH = Path(__file__).resolve().parents[1] / "reports" / "training_status.json"


def write_json_atomic(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}-{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    for attempt in range(10):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt == 9:
                temporary.unlink(missing_ok=True)
                raise
            time.sleep(.05)


def publish_status(**updates):
    status = {"state": "not_started", "model": None, "quantization": "none", "steps": 0, "loss": None,
              "elapsed_hours": 0, "budget_hours": 72, "peak_vram_gb": 0, "message": "尚未启动训练",
              "history": [], "artifacts": []}
    if STATUS_PATH.exists():
        try:
            status.update(json.loads(STATUS_PATH.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass
    status.update(updates)
    status["updated_at"] = datetime.now(timezone.utc).isoformat()
    status["history"] = status.get("history", [])[-1000:]
    write_json_atomic(STATUS_PATH, status)
    return status
