"""CPU-only helpers for exclusive workers and auditable checkpoint recovery."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
from uuid import uuid4


class ActiveRunError(RuntimeError):
    """A rejected duplicate must not overwrite the live run's status."""


def reject_live_worker(output: Path):
    """Also protect workers launched before per-run locks were introduced."""
    import psutil
    output = output.resolve()
    for process in psutil.process_iter(["pid", "cmdline"]):
        if process.pid == os.getpid():
            continue
        command = process.info.get("cmdline") or []
        if "-m" not in command or "--output" not in command:
            continue
        try:
            if command[command.index("-m") + 1] != "training.train":
                continue
            target = Path(command[command.index("--output") + 1])
            if not target.is_absolute():
                target = Path(process.cwd()) / target
            if target.resolve() == output:
                raise ActiveRunError(f"GPU worker {process.pid} is still using {output}")
        except (psutil.NoSuchProcess, psutil.AccessDenied, IndexError):
            continue


def acquire_run_lock(path: Path):
    """Keep the returned file open until the owning process finishes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    handle.seek(0)
    handle.write(b"1")
    handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise ActiveRunError(f"Another process owns this run: {path}") from exc
    return handle


def consumed_training_seconds(previous: dict, *monitors: dict) -> float:
    """Keep spent time even after a crash immediately following a resume."""
    candidates = [previous.get("training_seconds", 0), previous.get("previous_training_seconds", 0)]
    candidates.extend(item.get("training_seconds", 0) for item in monitors)
    values = [float(value) for value in candidates]
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Invalid cumulative training time in recovery metadata")
    return max(values)


def consumed_experiment_seconds(summary: dict, output: Path) -> float:
    """Account for a worker that outlived its supervisor before recovery."""
    elapsed = float(summary.get("elapsed_seconds", 0))
    summed_training = 0.0
    for record in summary.get("runs", []):
        run = output / record["name"]
        metadata = []
        for name in ("run_manifest.json", "monitor_state.json"):
            if (run / name).is_file():
                metadata.append(json.loads((run / name).read_text(encoding="utf-8")))
        spent = consumed_training_seconds({}, *metadata)
        summed_training += spent
        if "experiment_seconds_at_start" in record:
            elapsed = max(elapsed, float(record["experiment_seconds_at_start"]) +
                          max(0, spent - float(record.get("training_seconds_at_start", 0))))
    return consumed_training_seconds({"training_seconds": max(elapsed, summed_training)})


def archive_replayed_metrics(output: Path, checkpoint_step: int) -> list[str]:
    """Preserve original logs, then remove abandoned steps from the main curves.

    A hard interruption may leave metrics beyond the latest committed model.
    Replaying those steps must not produce two different losses at the same step.
    Partial trailing records are kept in the byte-exact archive for diagnosis.
    """
    archives = []
    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
    for name in ("metrics.jsonl", "metrics.csv"):
        path = output / name
        if not path.exists():
            continue
        original = path.read_bytes()
        text = original.decode("utf-8", errors="replace")
        if name.endswith(".jsonl"):
            retained = []
            for line in text.splitlines():
                try:
                    row = json.loads(line)
                    if int(row["step"]) <= checkpoint_step:
                        retained.append(line)
                except (ValueError, KeyError, TypeError):
                    continue
            cleaned = "\n".join(retained) + ("\n" if retained else "")
        else:
            reader = csv.DictReader(io.StringIO(text))
            stream = io.StringIO(newline="")
            if reader.fieldnames:
                writer = csv.DictWriter(stream, fieldnames=reader.fieldnames)
                writer.writeheader()
                for row in reader:
                    try:
                        if int(row["step"]) <= checkpoint_step and None not in row:
                            writer.writerow(row)
                    except (ValueError, KeyError, TypeError):
                        continue
            cleaned = stream.getvalue()
        if cleaned.encode("utf-8") == original:
            continue
        archive = output / "recovery-history" / f"{suffix}-{name}"
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(original)
        temporary = path.with_name(f".{path.name}-{uuid4().hex}.tmp")
        temporary.write_bytes(cleaned.encode("utf-8"))
        os.replace(temporary, path)
        archives.append(str(archive.resolve()))
    return archives
