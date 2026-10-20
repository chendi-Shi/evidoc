"""Exercise checkpoint stop/resume on real data without changing train.py.

Uses a fresh, fixed runs/resume-smoke directory. This is a two-step functional
check, not a model-quality experiment. Existing evidence is never overwritten.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs" / "resume-smoke"
REPORT = ROOT / "reports" / "resume_smoke.json"
COMMAND = [sys.executable, "-u", "-m", "training.train",
           "--model", "models/Qwen3.5-2B", "--output", "runs/resume-smoke",
           "--epochs", "1", "--max-steps", "2", "--accumulation", "2",
           "--max-length", "512", "--logging-steps", "1", "--save-steps", "1",
           "--eval-steps", "1", "--eval-samples", "2"]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save(report: dict) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(REPORT)


def run_stage(name: str, command: list[str], request_stop: bool = False) -> dict:
    log = ROOT / "reports" / f"resume-smoke-{name}.log"
    started = time.monotonic()
    stop_written = False
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            while process.poll() is None:
                # This file is created by train.py after its empty-output guard.
                if request_stop and not stop_written and (OUTPUT / "tokenization_report.json").is_file():
                    (OUTPUT / "STOP_REQUESTED").write_text("Authorized stop/resume functional test.\n", encoding="utf-8")
                    stop_written = True
                    print("Stop request written after training created its output.", flush=True)
                if time.monotonic() - started > 600:
                    raise TimeoutError(f"{name} exceeded the 600-second functional-test deadline")
                time.sleep(.1)
        except BaseException:
            if process.poll() is None:
                process.terminate()  # Only the child created by this controller.
                process.wait(timeout=15)
            raise
    result = {"command": command, "returncode": process.returncode, "log": str(log),
              "seconds": time.monotonic() - started, "stop_request_written": stop_written}
    print(json.dumps({"stage": name, **result}), flush=True)
    if process.returncode:
        raise RuntimeError(f"{name} failed with exit {process.returncode}; see {log}")
    if request_stop and not stop_written:
        raise RuntimeError("Training finished before the controller could request the step-one stop")
    return result


def inspect_checkpoint(step: int) -> dict:
    path = OUTPUT / f"checkpoint-{step}"
    required = ("adapter_model.safetensors", "adapter_config.json", "optimizer.pt", "scheduler.pt",
                "trainer_state.json", "monitor_state.json", "rng_state.pth")
    sizes = {name: (path / name).stat().st_size for name in required}
    if not all(sizes.values()):
        raise RuntimeError(f"Checkpoint {step} has empty files")
    trainer = read_json(path / "trainer_state.json")
    monitor = read_json(path / "monitor_state.json")
    if trainer["global_step"] != step or monitor["step"] != step:
        raise RuntimeError(f"Checkpoint {step} has incorrect trainer/monitor step")
    # Torch is imported only in a short, CPU-only checkpoint-inspection child.
    code = """import json,sys,torch
p=torch.load(sys.argv[1],map_location='cpu',weights_only=True)
steps=sorted({int(s['step']) for s in p['state'].values() if 'step' in s})
print(json.dumps({'optimizer_steps':steps,'optimizer_state_entries':len(p['state'])}))
"""
    inspected = subprocess.run([sys.executable, "-c", code, str(path / "optimizer.pt")],
                               cwd=ROOT, capture_output=True, text=True, timeout=30, check=True,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    optimizer = json.loads(inspected.stdout)
    if optimizer["optimizer_steps"] != [step] or not optimizer["optimizer_state_entries"]:
        raise RuntimeError(f"Checkpoint {step} has unexpected AdamW state: {optimizer}")
    return {"path": str(path), "files_bytes": sizes, "global_step": trainer["global_step"],
            "monitor_step": monitor["step"], "monitor_history": monitor["history"],
            **optimizer}


def main() -> int:
    if OUTPUT.exists() or REPORT.exists():
        raise RuntimeError("Resume-smoke output/report already exists; preserve it and use a deliberately fresh test.")
    report = {"started_at": datetime.now(timezone.utc).isoformat(), "success": False,
              "purpose": "Real-data two-step checkpoint mechanics test; no model-quality claim",
              "stages": []}
    save(report)
    try:
        report["stages"].append(run_stage("initial", COMMAND, request_stop=True))
        report["checkpoint_1"] = inspect_checkpoint(1)
        before = read_json(OUTPUT / "run_manifest.json")
        if before["global_step"] != 1 or before["stop_reason"] != "interrupt" or before["training_complete"]:
            raise RuntimeError(f"Initial training did not stop at step one: {before}")
        report["initial_run_manifest"] = before
        save(report)
        # Delete only the stop marker created above, after checking the stop.
        (OUTPUT / "STOP_REQUESTED").unlink()
        resumed = COMMAND + ["--resume", "runs/resume-smoke/checkpoint-1"]
        report["stages"].append(run_stage("resumed", resumed))
        report["checkpoint_2"] = inspect_checkpoint(2)
        after = read_json(OUTPUT / "run_manifest.json")
        trainer = read_json(OUTPUT / "trainer_state.json")
        metrics = [json.loads(line) for line in (OUTPUT / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        loss_steps = [entry["step"] for entry in metrics if "loss" in entry]
        if after["global_step"] != 2 or trainer["global_step"] != 2 or loss_steps != [1, 2]:
            raise RuntimeError(f"Resume did not progress exactly from step one to two: {loss_steps}")
        if after.get("resumed_from") != str((OUTPUT / "checkpoint-1").resolve()):
            raise RuntimeError("Run manifest did not record the exact resumed checkpoint")
        if not after["training_complete"] or after["previous_training_seconds"] <= 0:
            raise RuntimeError("Resume completion or cumulative-time accounting failed")
        adapter = OUTPUT / "adapter" / "adapter_model.safetensors"
        if not adapter.is_file() or adapter.stat().st_size <= 0:
            raise RuntimeError("Final adapter was not saved")
        report.update(success=True, resumed_run_manifest=after, optimizer_log_steps=loss_steps,
                      final_adapter=str(adapter), final_adapter_bytes=adapter.stat().st_size)
    except Exception as error:
        report.update(error=repr(error), traceback=traceback.format_exc())
        print(report["traceback"], flush=True)
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        save(report)
    print(json.dumps({"success": report["success"], "report": str(REPORT)}), flush=True)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
