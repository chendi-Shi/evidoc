"""Own and supervise EviDoc's two local llama.cpp processes.

  python scripts/run_models.py start --backend auto --background
  python scripts/run_models.py status
  python scripts/run_models.py stop

No driver changes are made. Auto tries Vulkan, then a separately installed CPU
runtime. Stop signals this supervisor only; unrelated processes are never killed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime"
STATUS = RUNTIME / "status.json"
STOP = RUNTIME / "stop.request"
LOCK = RUNTIME / "supervisor.lock"
PORTS = {"generation": 8081, "embedding": 8082}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_status() -> dict:
    try:
        return json.loads(STATUS.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"state": "stopped", "backend": None}


def save_status(state: dict) -> None:
    state["updated_at"] = now()
    temp = STATUS.with_suffix(".tmp")
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, STATUS)


def healthy(role: str) -> bool:
    try:
        with urlopen(f"http://127.0.0.1:{PORTS[role]}/health", timeout=2) as response:
            return response.status == 200 and json.load(response).get("status") == "ok"
    except (URLError, TimeoutError, OSError, ValueError):
        return False


def endpoint_occupied(role: str) -> bool:
    import socket
    with socket.socket() as connection:
        connection.settimeout(0.3)
        return connection.connect_ex(("127.0.0.1", PORTS[role])) == 0


def executable(backend: str) -> Path:
    found = list((RUNTIME / f"llama-{backend}").rglob("llama-server.exe"))
    if not found:
        raise FileNotFoundError(f"Install {backend} runtime: python scripts/download_models.py --runtime-only --runtime {backend}")
    return found[0]


def launch(role: str, backend: str, manifest: dict, state: dict) -> tuple[subprocess.Popen, object]:
    model = manifest["models"][role]
    path = ROOT / "models" / model["filename"]
    if not path.is_file() or path.stat().st_size != model["size"]:
        raise FileNotFoundError(f"Missing/incomplete {path.name}; run scripts/download_models.py first")
    command = [str(executable(backend)), "-m", str(path), "--host", "127.0.0.1",
               "--port", str(PORTS[role]), "--alias", f"evidoc-{'chat' if role == 'generation' else role}",
               "-c", "4096", "-np", "1", "-ngl", "99" if backend == "vulkan" else "0",
               "-t", str(max(1, min(8, (os.cpu_count() or 4) // 2))), "--no-webui"]
    if role == "generation":
        command += ["-b", "512", "-ub", "256", "-n", "768", "--jinja",
                    "--chat-template-kwargs", '{"enable_thinking":false}']
    else:
        command += ["--embeddings", "--pooling", "last", "-b", "4096", "-ub", "4096"]
    log_path = RUNTIME / f"{role}-{backend}.log"
    log = log_path.open("wb")
    process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    state[role] = {"state": "loading", "backend": backend, "pid": process.pid,
                   "model": model["repo"], "filename": model["filename"], "url": f"http://127.0.0.1:{PORTS[role]}/v1",
                   "alias": f"evidoc-{'chat' if role == 'generation' else role}", "context_size": 4096,
                   "log": str(log_path), "command": command}
    save_status(state)
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        if STOP.exists():
            terminate(process)
            log.close()
            raise InterruptedError("Stop requested")
        if process.poll() is not None:
            log.close()
            detail = log_path.read_text(encoding="utf-8", errors="replace")[-2500:]
            raise RuntimeError(f"{role}/{backend} exited {process.returncode}: {detail}")
        if healthy(role):
            state[role]["state"] = "ready"
            save_status(state)
            return process, log
        time.sleep(1)
    terminate(process)
    log.close()
    raise TimeoutError(f"{role}/{backend} did not become healthy within 300 seconds; see {log_path}")


def terminate(process: subprocess.Popen) -> None:
    """Use retained child process handles, never a discovered or user-supplied PID."""
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def supervise(backend: str) -> int:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    try:
        lock_fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f"Supervisor lock exists: {LOCK}. Check status; remove only after verifying the recorded process exited.", file=sys.stderr)
        return 1
    os.write(lock_fd, str(os.getpid()).encode())
    os.close(lock_fd)
    STOP.unlink(missing_ok=True)
    state = {"state": "starting", "backend": backend, "supervisor_pid": os.getpid(), "started_at": now(), "attempt_errors": []}
    children: list[tuple[subprocess.Popen, object]] = []
    try:
        for role in PORTS:
            if endpoint_occupied(role):
                raise RuntimeError(f"Port {PORTS[role]} is already occupied; EviDoc will not stop an unrelated process")
        save_status(state)
        manifest = json.loads((ROOT / "models" / "manifest.json").read_text(encoding="utf-8"))
        # The user requires GPU execution. CPU is available only by explicit opt-in.
        options = ["vulkan"] if backend == "auto" else [backend]
        for candidate in options:
            try:
                for role in PORTS:
                    children.append(launch(role, candidate, manifest, state))
                state.update(state="ready", backend=candidate)
                save_status(state)
                print(f"Models ready using {candidate}: 8081 generation, 8082 embeddings", flush=True)
                break
            except (OSError, RuntimeError, TimeoutError) as error:
                for process, log in children:
                    terminate(process)
                    log.close()
                children.clear()
                state["attempt_errors"].append({"backend": candidate, "error": str(error)})
                for role in PORTS:
                    if role in state:
                        state[role]["state"] = "stopped"
                save_status(state)
                if STOP.exists() or candidate == options[-1]:
                    raise
                print(f"{candidate} unavailable; trying CPU. {error}", flush=True)
        while not STOP.exists():
            if any(process.poll() is not None for process, _ in children):
                raise RuntimeError("A model server stopped unexpectedly; inspect runtime logs")
            time.sleep(1)
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as error:
        state.update(state="error", error=str(error))
        print(str(error), file=sys.stderr, flush=True)
        return 1
    finally:
        for process, log in children:
            terminate(process)
            log.close()
        if state.get("state") != "error":
            state["state"] = "stopped"
        for role in PORTS:
            if role in state:
                state[role]["state"] = "stopped"
        save_status(state)
        LOCK.unlink(missing_ok=True)
        STOP.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "status", "stop"], nargs="?", default="start")
    parser.add_argument("--backend", choices=["auto", "cpu", "vulkan"], default="auto")
    parser.add_argument("--background", action="store_true")
    args = parser.parse_args()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    if args.action == "status":
        state = read_status()
        state["health"] = {role: healthy(role) for role in PORTS}
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return 0 if all(state["health"].values()) else 1
    if args.action == "stop":
        if not LOCK.exists():
            print("No EviDoc supervisor is running")
            return 0
        STOP.write_text(now(), encoding="utf-8")
        for _ in range(35):
            if not LOCK.exists():
                print("EviDoc model processes stopped")
                return 0
            time.sleep(1)
        print("Stop requested; supervisor has not acknowledged. No unrelated process was terminated.")
        return 1
    if args.background:
        if LOCK.exists():
            print("Supervisor lock already exists; use status")
            return 1
        flags = (subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else 0
        with (RUNTIME / "supervisor.log").open("ab") as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "start", "--backend", args.backend],
                                       cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=flags, start_new_session=os.name != "nt")
        print(f"Supervisor started (PID {process.pid}); inspect scripts/run_models.py status and runtime/supervisor.log")
        return 0
    return supervise(args.backend)


if __name__ == "__main__":
    raise SystemExit(main())
