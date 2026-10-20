import csv
import json
from pathlib import Path

import pytest

from training.recovery import (acquire_run_lock, archive_replayed_metrics, consumed_training_seconds,
                               consumed_experiment_seconds, reject_live_worker)


def test_resume_crash_does_not_reset_consumed_budget():
    assert consumed_training_seconds({"previous_training_seconds": 140}, {"training_seconds": 100}) == 140
    assert consumed_training_seconds({"training_seconds": 130}, {"training_seconds": 170}) == 170
    with pytest.raises(ValueError):
        consumed_training_seconds({"training_seconds": float("nan")})


def test_orphan_worker_time_is_included_without_double_counting(tmp_path):
    for name, seconds in (("first", 60), ("second", 500)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "monitor_state.json").write_text(json.dumps({"training_seconds": seconds}))
    summary = {"elapsed_seconds": 100, "runs": [{"name": "first"}, {"name": "second"}]}
    assert consumed_experiment_seconds(summary, tmp_path) == 560
    # Known start includes model loading and time from earlier runs.
    summary["runs"][1].update(experiment_seconds_at_start=120, training_seconds_at_start=10)
    assert consumed_experiment_seconds(summary, tmp_path) == 610
    summary["elapsed_seconds"] = 650
    assert consumed_experiment_seconds(summary, tmp_path) == 650


def test_replay_archives_original_bytes_and_keeps_checkpoint_history(tmp_path):
    log = tmp_path / "metrics.jsonl"
    original = b'{"step": 1, "loss": 1.0}\n{"step": 2, "loss": 0.5}\n{"step":'
    log.write_bytes(original)
    table = tmp_path / "metrics.csv"
    table.write_bytes(b"step,loss\r\n1,1.0\r\n2,0.5\r\n3,\r\n")
    archives = archive_replayed_metrics(tmp_path, 1)
    assert len(archives) == 2
    assert Path(archives[0]).read_bytes() == original
    assert [json.loads(row)["step"] for row in log.read_text().splitlines()] == [1]
    with table.open(newline="") as stream:
        assert list(csv.DictReader(stream)) == [{"step": "1", "loss": "1.0"}]
    assert archive_replayed_metrics(tmp_path, 1) == []


def test_worker_lock_prevents_duplicate_and_releases_on_close(tmp_path):
    path = tmp_path / ".training.lock"
    first = acquire_run_lock(path)
    try:
        with pytest.raises(RuntimeError, match="owns this run"):
            acquire_run_lock(path)
    finally:
        first.close()
    acquire_run_lock(path).close()


def test_old_worker_without_lock_is_detected_by_exact_output(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import psutil
    process = SimpleNamespace(pid=999999, info={"cmdline": ["python", "-m", "training.train", "--output", "run"]},
                              cwd=lambda: str(tmp_path))
    monkeypatch.setattr(psutil, "process_iter", lambda attrs: [process])
    reject_live_worker(tmp_path / "other-run")
    with pytest.raises(RuntimeError, match="still using"):
        reject_live_worker(tmp_path / "run")
