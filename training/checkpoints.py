"""Validate complete local Trainer checkpoints without importing torch or CUDA."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import struct
import warnings
import zipfile

from .status import write_json_atomic

MARKER = "checkpoint_complete.json"
REQUIRED_FILES = (
    "adapter_model.safetensors", "adapter_config.json", "optimizer.pt",
    "scheduler.pt", "trainer_state.json", "monitor_state.json", "rng_state.pth",
)


def _json(path: Path) -> dict:
    def invalid_constant(value):
        raise ValueError(f"Non-finite JSON number: {value}")
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid_constant)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _step(path: Path) -> int:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    if not match or not path.is_dir():
        raise ValueError(f"Not a checkpoint directory: {path}")
    return int(match.group(1))


def _hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _validate_safetensors(path: Path):
    size = path.stat().st_size
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError("Truncated safetensors length")
        length = struct.unpack("<Q", raw)[0]
        if not 2 <= length <= min(100_000_000, size - 8):
            raise ValueError("Invalid safetensors header length")
        header = json.loads(stream.read(length))
    if not isinstance(header, dict):
        raise ValueError("Invalid safetensors header")
    byte_width = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
                  "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
                  "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8}
    regions = []
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(tensor, dict):
            raise ValueError("Invalid safetensors tensor description")
        shape, offsets = tensor.get("shape"), tensor.get("data_offsets")
        if (not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape)
                or not isinstance(offsets, list) or len(offsets) != 2
                or any(type(n) is not int or n < 0 for n in offsets)):
            raise ValueError("Invalid safetensors shape or offsets")
        width = byte_width.get(tensor.get("dtype"))
        if width is None or offsets[1] - offsets[0] != math.prod(shape) * width:
            raise ValueError("Safetensors tensor size mismatch")
        regions.append(tuple(offsets))
    if not regions:
        raise ValueError("Safetensors has no adapter tensors")
    end = 0
    for start, next_end in sorted(regions):
        if start != end:
            raise ValueError("Safetensors has overlapping or missing tensor data")
        end = next_end
    if end != size - 8 - length:
        raise ValueError("Safetensors data is truncated or has trailing bytes")


def _validate_torch_zip(path: Path):
    # Inspect the archive only: never unpickle or allocate model tensors.
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        pickles = [name for name in names if name.endswith("/data.pkl")]
        if len(pickles) != 1 or pickles[0][:-len("data.pkl")] + "version" not in names:
            raise ValueError(f"{path.name} is not a PyTorch ZIP checkpoint")
        corrupt = archive.testzip()
        if corrupt is not None:
            raise ValueError(f"{path.name} has an invalid CRC: {corrupt}")


def _files(path: Path) -> dict:
    step = _step(path)
    for name in REQUIRED_FILES:
        item = path / name
        if not item.is_file() or item.stat().st_size == 0:
            raise ValueError(f"Missing or empty checkpoint file: {name}")
    trainer = _json(path / "trainer_state.json")
    monitor = _json(path / "monitor_state.json")
    _json(path / "adapter_config.json")
    if type(trainer.get("global_step")) is not int or trainer["global_step"] != step:
        raise ValueError("Trainer global_step differs from checkpoint directory")
    if type(monitor.get("step")) is not int or monitor["step"] != step:
        raise ValueError("Monitor step differs from checkpoint directory")
    elapsed = monitor.get("training_seconds")
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("Invalid cumulative checkpoint training_seconds")
    _validate_safetensors(path / "adapter_model.safetensors")
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        _validate_torch_zip(path / name)
    return {"path": str(path.resolve()), "global_step": step, "monitor": monitor,
            "trainer": trainer, "files": {name: {"size": (path / name).stat().st_size,
                "sha256": _hash(path / name)} for name in REQUIRED_FILES}}


def validate_checkpoint(path: Path | str) -> dict:
    """Raise ValueError for partial/corrupt checkpoints; return inspected metadata."""
    path = Path(path)
    try:
        result = _files(path)
        if (path / MARKER).exists():
            marker = _json(path / MARKER)
            if (marker.get("format") != 1 or marker.get("global_step") != result["global_step"]
                    or marker.get("files") != result["files"]):
                raise ValueError("Checkpoint commit manifest does not match file sizes/hashes")
            result["format"] = "committed"
        else:
            manifest = path.parent / "run_manifest.json"
            if manifest.exists():
                run = _json(manifest)
                threshold = run.get("checkpoint_commit_required_from_step", 1)
                if run.get("checkpoint_format", 0) >= 1 and result["global_step"] >= threshold:
                    raise ValueError("Checkpoint commit manifest is missing")
            result["format"] = "legacy"
        return result
    except (OSError, ValueError, TypeError, KeyError, zipfile.BadZipFile, RuntimeError) as exc:
        raise ValueError(f"Invalid checkpoint {path}: {exc}") from exc


def latest_checkpoint(run_dir: Path | str) -> Path:
    """Choose the newest complete checkpoint, preserving incomplete files for audit."""
    run_dir = Path(run_dir)
    candidates = [p for p in run_dir.glob("checkpoint-*")
                  if p.is_dir() and re.fullmatch(r"checkpoint-\d+", p.name)]
    for path in sorted(candidates, key=_step, reverse=True):
        try:
            validate_checkpoint(path)
            return path
        except ValueError as exc:
            warnings.warn(str(exc), RuntimeWarning, stacklevel=2)
    raise ValueError(f"No complete resumable checkpoint in {run_dir}")


def commit_checkpoint(path: Path | str) -> dict:
    """Call only after Trainer and monitor have finished writing every file."""
    path = Path(path)
    result = _files(path)
    marker = {"format": 1, "global_step": result["global_step"], "files": result["files"]}
    write_json_atomic(path / MARKER, marker)
    return marker
