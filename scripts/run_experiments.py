"""Run meaningful GPU LoRA ablations under a cumulative wall-clock ceiling.

Never keeps the GPU busy just to fill the budget. All runs share a fixed data
split, seed and learning rate; only adapter rank/target coverage changes.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from training.status import write_json_atomic, publish_status
from training.checkpoints import latest_checkpoint
from training.recovery import acquire_run_lock, consumed_training_seconds, consumed_experiment_seconds

ROOT = Path(__file__).resolve().parents[1]
RUNS = [
    {"name": "r8-hybrid", "rank": 8, "targets": "auto"},
    {"name": "r16-hybrid", "rank": 16, "targets": "auto"},
    {"name": "r8-full-attention", "rank": 8, "targets": "q_proj,v_proj"},
]


def atomic_json(path, data):
    write_json_atomic(path, data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(ROOT / "models/Qwen3.5-2B"))
    parser.add_argument("--data", default="data/training/cmrc2018")
    parser.add_argument("--pilot", type=Path, required=True, help="Passed pilot_report.json from the same model/sequence configuration")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/experiment-72h")
    parser.add_argument("--hours", type=float, default=72)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--accumulation", type=int, default=16)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--seeds", default="42,43,44", help="Independent model/loader seeds; dataset split stays fixed")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-after", action="store_true", help="Evaluate completed adapters after the full training suite finishes")
    parser.add_argument("--plan-only", action="store_true", help="Validate the pilot and print the plan without starting training")
    args = parser.parse_args()
    if not 0 < args.hours <= 96:
        raise ValueError("Budget must be greater than zero and no more than 96 hours")
    seeds = list(dict.fromkeys(int(value.strip()) for value in args.seeds.split(",")))
    if not seeds or len(seeds) > 3:
        raise ValueError("Choose one to three independent seeds")
    configurations = [{**preset, "seed": seed, "name": f"{preset['name']}-s{seed}"} for seed in seeds for preset in RUNS]
    pilot = json.loads(args.pilot.read_text(encoding="utf-8"))
    manifest = json.loads((args.pilot.parent / "run_manifest.json").read_text(encoding="utf-8"))
    for key, expected in (("model", args.model), ("data", args.data), ("max_length", args.max_length), ("accumulation", args.accumulation)):
        actual = manifest["arguments"][key]
        same = (Path(actual).resolve() == Path(expected).resolve() if key in {"model", "data"} else actual == expected)
        if not same:
            raise ValueError(f"Pilot configuration differs: {key}. Repeat pilot using planned configuration.")
    if manifest.get("stop_reason") != "epochs_or_step_cap" or manifest.get("global_step", 0) < 2:
        raise ValueError("Pilot did not complete enough GPU optimizer steps")
    probe = json.loads((ROOT / "reports/training_gpu_probe.json").read_text(encoding="utf-8"))
    if not probe.get("success"):
        raise RuntimeError("GPU CUDA probe must pass. No CPU training fallback.")
    if args.plan_only:
        print(json.dumps({"gpu": probe["gpu_name"], "runs": configurations, "budget_hours": args.hours,
            "estimated_training_hours_before_eval_and_saving": pilot.get("estimated_epoch_plan_hours", 0) * len(configurations),
            "evaluate_after": args.evaluate_after}, ensure_ascii=False, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / "experiment.json"
    if summary_path.exists() and not args.resume:
        raise ValueError("Experiment exists; use --resume or choose a fresh output")
    summary = (json.loads(summary_path.read_text(encoding="utf-8")) if args.resume else {
        "created_at": datetime.now(timezone.utc).isoformat(), "state": "running", "budget_hours": args.hours,
        "model": args.model, "data": args.data, "gpu": probe["gpu_name"], "device": "cuda:0", "gpu_verified": True,
        "pilot": str(args.pilot.resolve()), "pilot_measurement": pilot, "runs": [], "elapsed_seconds": 0,
        "configurations": configurations,
        "estimated_training_hours_before_eval_and_saving": pilot.get("estimated_epoch_plan_hours", 0) * len(configurations),
        "note": "Time is an upper bound, not a promise. Stop early on convergence; no idle filler computation.",
    })
    # Process-owned lock prevents two supervisors from training this experiment.
    lock_handle = acquire_run_lock(args.output / ".supervisor.lock")
    if args.resume and summary.get("child_pid"):
        import psutil
        try:
            child = psutil.Process(summary["child_pid"])
            if abs(child.create_time() - summary.get("child_created_at", 0)) < 1 and child.is_running():
                raise RuntimeError("The previous GPU worker is still alive; request a safe stop before resuming.")
        except psutil.NoSuchProcess:
            pass
    previous_seconds = consumed_experiment_seconds(summary, args.output)
    summary["elapsed_seconds"] = previous_seconds
    started = time.monotonic()
    stop_requested = False
    def request_stop(*_):
        nonlocal stop_requested
        stop_requested = True
        print("Stop requested; waiting for the GPU job to save its checkpoint.", flush=True)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    if os.name == "nt":
        # Keeps the system awake only while this process lives; display may sleep.
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
    if args.resume:
        for key, expected in (("model", args.model), ("data", args.data), ("configurations", configurations)):
            if summary.get(key) != expected:
                raise ValueError(f"Experiment resume configuration differs: {key}")
        if summary["budget_hours"] != args.hours:
            raise ValueError("Resume must preserve the original cumulative budget_hours")
        saved_args = summary.get("arguments", {})
        for key in ("max_length", "accumulation", "epochs"):
            # Older experiment summaries stored these in each worker command.
            values = [saved_args[key]] if key in saved_args else [
                run["command"][run["command"].index("--" + key.replace("_", "-")) + 1]
                for run in summary["runs"] if "--" + key.replace("_", "-") in run.get("command", [])]
            if any(float(value) != float(getattr(args, key)) for value in values):
                raise ValueError(f"Experiment resume configuration differs: {key}")
        (args.output / "STOP_REQUESTED").unlink(missing_ok=True)
        summary["state"] = "running"
    summary["arguments"] = {key: str(value) if isinstance(value, Path) else value
                            for key, value in vars(args).items()}
    summary["command"] = [sys.executable, "-m", "scripts.run_experiments", *sys.argv[1:]]
    summary["evaluate_after"] = args.evaluate_after
    for configuration in configurations:
        name = configuration["name"]
        if any(run["name"] == name and run["state"] == "completed" for run in summary["runs"]):
            continue
        run_dir = args.output / name
        # A worker may have finished saving before the supervisor was killed.
        final_path = run_dir / "run_manifest.json"
        if args.resume and final_path.exists():
            final_info = json.loads(final_path.read_text(encoding="utf-8"))
            if final_info.get("training_complete") and (run_dir / "adapter/adapter_model.safetensors").is_file():
                for record in summary["runs"]:
                    if record["name"] == name:
                        record.update(state="completed", stop_reason=final_info.get("stop_reason"),
                                      recovered_completion=True)
                atomic_json(summary_path, summary)
                continue
        remaining = args.hours * 3600 - previous_seconds - (time.monotonic() - started)
        if remaining < 300:
            summary["state"] = "budget_reached"
            break
        spent = 0.0
        command = [sys.executable, "-m", "training.train", "--model", args.model, "--data", args.data,
            "--output", str(run_dir), "--rank", str(configuration["rank"]), "--targets", configuration["targets"],
            "--max-length", str(args.max_length), "--accumulation", str(args.accumulation),
            "--epochs", str(args.epochs), "--hours", str(min(remaining / 3600, 96)),
            "--quantization", "none", "--dtype", "bf16", "--seed", str(configuration["seed"]), "--logging-steps", "1"]
        if run_dir.exists() and any(run_dir.iterdir()):
            if not args.resume:
                raise ValueError(f"Existing run output {run_dir}; use --resume")
            checkpoint = latest_checkpoint(run_dir)
            command += ["--resume", str(checkpoint)]
            # --resume explicitly requests continuation after the old stop request.
            (run_dir / "STOP_REQUESTED").unlink(missing_ok=True)
            previous_run = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            monitors = [json.loads((checkpoint / "monitor_state.json").read_text(encoding="utf-8"))]
            if (run_dir / "monitor_state.json").exists():
                monitors.append(json.loads((run_dir / "monitor_state.json").read_text(encoding="utf-8")))
            spent = consumed_training_seconds(previous_run, *monitors)
            command[command.index("--hours") + 1] = str(min(96, (remaining + spent) / 3600))
        record = {"name": name, "configuration": configuration, "command": command, "state": "running",
                  "experiment_seconds_at_start": previous_seconds + time.monotonic() - started,
                  "training_seconds_at_start": spent}
        summary["runs"] = [r for r in summary["runs"] if r["name"] != name] + [record]
        atomic_json(summary_path, summary)
        print(f"Starting GPU experiment {name}", flush=True)
        process = subprocess.Popen(command, cwd=ROOT, env=env)
        import psutil
        summary["child_created_at"] = psutil.Process(process.pid).create_time()
        budget_stop = False
        while process.poll() is None:
            summary["elapsed_seconds"] = previous_seconds + time.monotonic() - started
            summary["child_pid"] = process.pid
            budget_stop = summary["elapsed_seconds"] >= args.hours * 3600
            if (args.output / "STOP_REQUESTED").exists():
                stop_requested = True
            if budget_stop or stop_requested:
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "STOP_REQUESTED").write_text("wall_clock_budget" if budget_stop else "user_stop", encoding="utf-8")
                summary["state"] = "saving_before_stop"
            atomic_json(summary_path, summary)
            time.sleep(15)
        record["exit_code"] = process.returncode
        final_manifest = run_dir / "run_manifest.json"
        final_info = json.loads(final_manifest.read_text(encoding="utf-8")) if final_manifest.exists() else {}
        record["state"] = ("completed" if process.returncode == 0 and final_info.get("training_complete")
                           else "interrupted" if process.returncode == 0 else "failed")
        record["stop_reason"] = final_info.get("stop_reason", "unknown")
        summary["elapsed_seconds"] = previous_seconds + time.monotonic() - started
        atomic_json(summary_path, summary)
        if process.returncode:
            summary["state"] = "failed"
            atomic_json(summary_path, summary)
            raise SystemExit(process.returncode)
        if record["state"] != "completed":
            summary["state"] = "budget_reached" if budget_stop or final_info.get("stop_reason") == "wall_clock_budget" else "interrupted"
            break
    else:
        summary["state"] = "completed"
    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(summary_path, summary)
    if args.evaluate_after and summary["state"] == "completed":
        publish_status(state="preparing", phase="evaluation", message="训练实验已完成，正在 GPU 上进行同题基础模型/适配器评测。")
        summary["evaluation_state"] = "running"
        atomic_json(summary_path, summary)
        evaluation = subprocess.Popen([sys.executable, "-m", "scripts.evaluate_experiments", str(args.output)], cwd=ROOT, env=env)
        import psutil
        summary["child_pid"] = evaluation.pid
        summary["child_created_at"] = psutil.Process(evaluation.pid).create_time()
        atomic_json(summary_path, summary)
        while evaluation.poll() is None:
            time.sleep(15)
        summary["evaluation_state"] = "completed" if evaluation.returncode == 0 else "failed"
        atomic_json(summary_path, summary)
        publish_status(state="completed" if evaluation.returncode == 0 else "failed", phase="evaluation",
            message="GPU 训练及保留集评测完成。" if evaluation.returncode == 0 else "适配器训练已完成，后续评测失败，请查看实验日志。",
            artifacts=[str((args.output / "final_comparison.json").resolve()), str(summary_path.resolve())])
        if evaluation.returncode:
            raise SystemExit(evaluation.returncode)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
