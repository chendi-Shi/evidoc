"""Inspect, safely pause, or resume the existing experiment without using CUDA.

Resume restores the recorded experiment settings; it never resets its time budget.
Process identity uses executable, module and experiment path, never a PID alone.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

import psutil

from training.status import write_json_atomic

ROOT = Path(__file__).resolve().parents[1]
MODULES = {"scripts.run_experiments", "training.train", "scripts.evaluate_experiments", "training.evaluate"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def resolve_path(value, base):
    path = Path(value)
    return (path if path.is_absolute() else Path(base) / path).resolve()


def option(command, name):
    for index, token in enumerate(command):
        if token == name and index + 1 < len(command):
            return command[index + 1]
        if token.startswith(name + "="):
            return token.split("=", 1)[1]
    return None


def invocation(command, cwd, root):
    """Return the exact allowed Python module and the arguments after it."""
    if "-m" in command:
        index = command.index("-m") + 1
        if index < len(command) and command[index] in MODULES:
            return command[index], command[index + 1:]
        return None, []
    for index, token in enumerate(command[1:], 1):
        if token.startswith("-"):
            continue
        candidate = resolve_path(token, cwd)
        for module in MODULES:
            if candidate == root / (module.replace(".", "/") + ".py"):
                return module, command[index + 1:]
        # Only the first non-option can be a script; do not scan payload text.
        break
    return None, []


def find_live_processes(output, root=ROOT):
    """Find this experiment's supervisor/worker/evaluation, including orphans."""
    root, output = Path(root).resolve(), Path(output).resolve()
    python = root / ".tools/python/python.exe"
    matches = []
    for process in psutil.process_iter():
        try:
            if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
                continue
            command = process.cmdline()
            if not command or Path(process.exe()).resolve() != python:
                continue
            cwd = Path(process.cwd()).resolve()
            module, arguments = invocation(command, cwd, root)
            if module is None:
                continue
            target = (arguments[0] if module == "scripts.evaluate_experiments" and arguments
                      else option(arguments, "--output"))
            if target is None:
                continue
            target_path = resolve_path(target, cwd)
            if module in {"scripts.run_experiments", "scripts.evaluate_experiments"}:
                same_experiment = target_path == output
            else:
                same_experiment = target_path != output and output in target_path.parents
            if same_experiment:
                matches.append({"pid": process.pid, "created_at": process.create_time(),
                                "module": module, "command": command, "cwd": str(cwd)})
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
        except psutil.AccessDenied:
            # An inaccessible unrelated OS process is not evidence of a worker.
            continue
    return matches


def command_settings(command):
    values = {}
    for key in ("model", "data", "pilot", "output", "hours", "max_length", "accumulation", "epochs", "seeds"):
        value = option(command, "--" + key.replace("_", "-"))
        if value is not None:
            values[key] = value
    # This helper is called with a proven supervisor command only.
    values["evaluate_after"] = "--evaluate-after" in command
    return values


def reconstruct_command(summary, output, root=ROOT, active=None, live=None):
    """Recover both modern settings and legacy runs without guessing a budget."""
    root, output = Path(root).resolve(), Path(output).resolve()
    values = {}
    pilot_value = summary.get("pilot") or summary.get("arguments", {}).get("pilot")
    if not pilot_value:
        raise ValueError("Experiment has no recorded pilot path; cannot reconstruct identical settings.")
    pilot_path = resolve_path(pilot_value, root)
    manifest = pilot_path.parent / "run_manifest.json"
    if manifest.exists():
        pilot = read_json(manifest).get("arguments", {})
        for key in ("model", "data", "max_length", "accumulation", "epochs"):
            if key in pilot:
                values[key] = pilot[key]
    # The first worker command preserves actual full-run settings, unlike a pilot.
    for run in summary.get("runs", []):
        command = run.get("command", [])
        for key in ("model", "data", "max_length", "accumulation", "epochs"):
            value = option(command, "--" + key.replace("_", "-"))
            if value is not None:
                values[key] = value
        if command:
            break
    for key in ("model", "data", "evaluate_after"):
        if key in summary:
            values[key] = summary[key]
    if "budget_hours" in summary:
        values["hours"] = summary["budget_hours"]
    configs = summary.get("configurations", [])
    if configs:
        values["seeds"] = ",".join(str(seed) for seed in dict.fromkeys(item["seed"] for item in configs))
    values["pilot"] = str(pilot_path)
    records = []
    if active and active.get("output") and resolve_path(active["output"], root) == output:
        records.append(active)
    records.append(summary)
    records.extend(item for item in (live or []) if item.get("module") == "scripts.run_experiments")
    for record in records:
        command = record.get("command") or record.get("supervisor_command")
        if command:
            module, _ = invocation(command, Path(record.get("cwd", root)), root)
            recorded_output = option(command, "--output")
            if module == "scripts.run_experiments" and recorded_output and resolve_path(recorded_output, record.get("cwd", root)) == output:
                values.update(command_settings(command))
        values.update({key: value for key, value in record.get("arguments", {}).items()
                       if key in {"model", "data", "hours", "max_length", "accumulation", "epochs", "seeds", "pilot", "evaluate_after"}})
    required = ("model", "data", "pilot", "hours", "max_length", "accumulation", "epochs", "seeds", "evaluate_after")
    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError("Recorded settings are incomplete: " + ", ".join(missing) + "; preserve evidence and restore the original supervisor command.")
    if not isinstance(values["evaluate_after"], bool):
        raise ValueError("Recorded evaluate_after must be a boolean.")
    if not 0 < float(values["hours"]) <= 96:
        raise ValueError("Recorded training budget is invalid.")
    if "budget_hours" in summary and float(values["hours"]) != float(summary["budget_hours"]):
        raise ValueError("Recorded supervisor settings conflict with the experiment's original time budget.")
    if isinstance(values["seeds"], list):
        values["seeds"] = ",".join(str(seed) for seed in values["seeds"])
    command = [str(root / ".tools/python/python.exe"), "-u", "-m", "scripts.run_experiments"]
    for key in ("model", "data", "pilot", "hours", "max_length", "accumulation", "epochs", "seeds"):
        command.extend(["--" + key.replace("_", "-"), str(values[key])])
    command.extend(["--output", str(output), "--resume"])
    if values["evaluate_after"]:
        command.append("--evaluate-after")
    return command


def checkpoint_inventory(summary, output):
    from training.checkpoints import latest_checkpoint
    completed = {run["name"] for run in summary.get("runs", []) if run.get("state") == "completed"}
    names = list(dict.fromkeys(item["name"] for item in summary.get("configurations", summary.get("runs", []))))
    entries = []
    for name in names:
        if name in completed:
            continue
        run_dir = (Path(output) / name).resolve()
        if Path(output).resolve() not in run_dir.parents:
            raise ValueError("Run name escapes the experiment directory.")
        entry = {"name": name, "checkpoint": None}
        if not run_dir.exists():
            entry["state"] = "not_started"
            entries.append(entry)
            continue
        try:
            checkpoint = latest_checkpoint(run_dir)
            entry["checkpoint"] = str(checkpoint) if checkpoint else None
        except (ValueError, OSError) as error:
            entry["checkpoint_error"] = str(error)
        entries.append(entry)
    return entries


def launch(command, output, root=ROOT):
    """Detach with fresh logs and retain enough metadata to repeat this command."""
    root = Path(root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid4().hex[:6]
    runtime = root / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    stdout_path, stderr_path = [runtime / f"experiment-{stamp}.{stream}.log" for stream in ("stdout", "stderr")]
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}
    settings = {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        process = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                   stdout=stdout, stderr=stderr, **settings)
    record = {"pid": process.pid, "output": str(output), "stdout": str(stdout_path), "stderr": str(stderr_path),
              "started_at": datetime.now(timezone.utc).isoformat(), "command": command, "cwd": str(root)}
    (runtime / "experiment.pid").write_text(str(process.pid), encoding="ascii")
    write_json_atomic(root / "reports/active_experiment.json", record)
    return record


def stop_targets(live, output, root):
    """Also reach orphan workers, whose supervisor can no longer relay a stop."""
    output, root = Path(output).resolve(), Path(root).resolve()
    targets = [output / "STOP_REQUESTED"]
    for item in live:
        if item.get("module") != "training.train":
            continue
        command = item.get("command", [])
        cwd = Path(item.get("cwd", root)).resolve()
        if not command or resolve_path(command[0], cwd) != root / ".tools/python/python.exe":
            continue
        module, arguments = invocation(command, cwd, root)
        run_output = option(arguments, "--output")
        if module != "training.train" or not run_output:
            continue
        run_dir = resolve_path(run_output, cwd)
        # Resolve before checking ancestry, including symlinks and ../ segments.
        if run_dir != output and output in run_dir.parents:
            targets.append(run_dir / "STOP_REQUESTED")
    return list(dict.fromkeys(targets))


def manage(action, output, dry_run=False, root=ROOT):
    root = Path(root).resolve()
    output = resolve_path(output, root)
    summary = read_json(output / "experiment.json")
    live = find_live_processes(output, root)
    result = {"action": action, "dry_run": dry_run, "output": str(output),
              "experiment_state": summary.get("state"), "budget_hours": summary.get("budget_hours"),
              "elapsed_seconds": summary.get("elapsed_seconds"), "processes": live}
    if action == "stop":
        result["state"] = "would_request_stop" if dry_run else "stop_requested"
        result["stop_file"] = str(output / "STOP_REQUESTED")
        targets = stop_targets(live, output, root)
        result["stop_files"] = [str(path) for path in targets]
        if not dry_run:
            for path in targets:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("user_stop", encoding="utf-8")
        result["message"] = ("A safe stop is requested from the supervisor and identified training workers, including orphans. Workers save after the current optimizer step; no process is killed."
                             if live else "No matching live process was found. The stop request is retained for this experiment; no process is killed.")
        return result
    result["checkpoints"] = checkpoint_inventory(summary, output)
    active_path = root / "reports/active_experiment.json"
    active = read_json(active_path) if active_path.exists() else None
    try:
        command = reconstruct_command(summary, output, root, active, live)
        result["command"] = command
        result["display_command"] = subprocess.list2cmdline(command)
    except ValueError as error:
        result["reconstruction_error"] = str(error)
        if action == "resume" and not live:
            raise
    if live:
        result["state"] = "already_running"
        result["message"] = "This experiment already has a live supervisor, worker, or evaluator; no duplicate process was started."
    elif action == "status":
        result["state"] = "not_running"
    elif dry_run:
        result["state"] = "would_resume"
    else:
        if not Path(command[0]).is_file():
            raise FileNotFoundError("Project Python is missing: " + command[0])
        # Recheck immediately before launch; the runner's process lock closes races.
        live = find_live_processes(output, root)
        if live:
            result.update(state="already_running", processes=live)
        else:
            result.update(state="resume_launched", launch=launch(command, output, root))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", required=True, choices=("status", "stop", "resume"))
    parser.add_argument("--output", default="runs/experiment-96h")
    parser.add_argument("--dry-run", action="store_true", help="Read status and show the command without writing files or starting/stopping anything")
    args = parser.parse_args(argv)
    try:
        result = manage(args.action, args.output, args.dry_run)
    except (OSError, ValueError, KeyError) as error:
        print(json.dumps({"state": "error", "error": str(error)}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
