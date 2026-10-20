"""Download pinned GGUF files with resumable transfers and SHA-256 verification.

Uses only Python's standard library. Existing valid files are retained. Interrupted
transfers remain as .part files and resume on the next invocation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "models" / "manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, expected_hash: str, expected_size: int | None = None) -> bool:
    return (path.is_file() and (expected_size is None or path.stat().st_size == expected_size)
            and sha256(path) == expected_hash)


def download_parallel(url: str, target: Path, expected_hash: str, expected_size: int, connections: int) -> Path:
    """Resume independently verified HTTP ranges, then verify the complete object."""
    if verify(target, expected_hash, expected_size):
        print(f"Verified existing {target.name}", flush=True)
        return target
    if target.exists():
        raise ValueError(f"Existing file failed verification: {target}")
    partial = target.with_suffix(target.suffix + ".part")
    directory = target.with_suffix(target.suffix + ".parts")
    directory.mkdir(parents=True, exist_ok=True)
    chunk_size = 16 * 1024 * 1024
    ranges = [(start, min(start + chunk_size, expected_size)) for start in range(0, expected_size, chunk_size)]
    # Import any previous sequential transfer without throwing downloaded bytes away.
    if partial.exists():
        with partial.open("rb") as previous:
            for index, (start, end) in enumerate(ranges):
                available = min(end - start, max(0, partial.stat().st_size - start))
                chunk = directory / f"{index:05d}.part"
                if available and (not chunk.exists() or chunk.stat().st_size < available):
                    previous.seek(start)
                    with chunk.open("wb") as output:
                        output.write(previous.read(available))
    def transfer(index: int, start: int, end: int) -> int:
        chunk = directory / f"{index:05d}.part"
        length = end - start
        for attempt in range(10):
            have = chunk.stat().st_size if chunk.exists() else 0
            if have == length:
                return length
            if have > length:
                raise ValueError(f"Oversize range {chunk}")
            headers = {"User-Agent": "EviDoc/0.1", "Accept-Encoding": "identity", "Range": f"bytes={start + have}-{end - 1}"}
            try:
                with urlopen(Request(url, headers=headers), timeout=35) as response:
                    crange = response.headers.get("Content-Range", "")
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", crange)
                    if response.status != 206 or not match or tuple(map(int, match.groups())) != (start + have, end - 1, expected_size):
                        raise ValueError(f"Server did not honor requested byte range: {crange}")
                    with chunk.open("ab") as output:
                        while block := response.read(min(1024 * 1024, length - have)):
                            output.write(block)
                            have += len(block)
                        output.flush()
                    if have != length:
                        raise OSError("Incomplete byte range")
                return length
            except (HTTPError, URLError, TimeoutError, OSError) as error:
                if attempt == 9:
                    raise
                print(f"{target.name}: range {index} retry: {error}", flush=True)
                time.sleep(min(2 * (attempt + 1), 10))
        raise RuntimeError("Range retries exhausted")
    completed = 0
    last_print = 0.0
    with ThreadPoolExecutor(max_workers=connections) as pool:
        pending = [pool.submit(transfer, index, start, end) for index, (start, end) in enumerate(ranges)]
        for future in as_completed(pending):
            completed += future.result()
            if time.monotonic() - last_print > 10 or completed == expected_size:
                print(f"{target.name}: {completed:,} / {expected_size:,} bytes complete", flush=True)
                last_print = time.monotonic()
    with partial.open("wb") as output:
        for index in range(len(ranges)):
            with (directory / f"{index:05d}.part").open("rb") as chunk:
                shutil.copyfileobj(chunk, output, length=8 * 1024 * 1024)
        output.flush()
        os.fsync(output.fileno())
    if not verify(partial, expected_hash, expected_size):
        raise ValueError(f"SHA-256/size verification failed: {partial}")
    partial.replace(target)
    # Remove only the explicit chunk files created by this download operation.
    for index in range(len(ranges)):
        (directory / f"{index:05d}.part").unlink()
    directory.rmdir()
    print(f"Verified SHA-256: {target.name}", flush=True)
    return target


def download(url: str, target: Path, expected_hash: str, expected_size: int | None = None) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if verify(target, expected_hash, expected_size):
        print(f"Verified existing {target.name}", flush=True)
        return target
    if target.exists():
        raise ValueError(f"Existing file failed verification; preserve or remove it manually: {target}")
    partial = target.with_suffix(target.suffix + ".part")
    if verify(partial, expected_hash, expected_size):
        partial.replace(target)
        print(f"Verified completed partial: {target.name}", flush=True)
        return target
    for attempt in range(1, 9):
        offset = partial.stat().st_size if partial.exists() else 0
        if expected_size and offset == expected_size:
            break
        if expected_size and offset > expected_size:
            raise ValueError(f"Partial file is too large: {partial}")
        headers = {"User-Agent": "EviDoc/0.1", "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with urlopen(Request(url, headers=headers), timeout=60) as response:
                if offset and response.status == 206:
                    crange = response.headers.get("Content-Range", "")
                    if not crange.startswith(f"bytes {offset}-"):
                        raise ValueError(f"Unexpected Content-Range: {crange}")
                    mode = "ab"
                elif response.status == 200:
                    offset, mode = 0, "wb"
                else:
                    raise ValueError(f"Unexpected HTTP status: {response.status}")
                total = expected_size or (offset + int(response.headers.get("Content-Length", "0")))
                last_print = time.monotonic()
                with partial.open(mode) as output:
                    while block := response.read(1024 * 1024):
                        output.write(block)
                        offset += len(block)
                        if time.monotonic() - last_print >= 10:
                            print(f"{target.name}: {offset:,} / {total:,} bytes", flush=True)
                            last_print = time.monotonic()
                    output.flush()
                    os.fsync(output.fileno())
            break
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            if attempt == 8:
                raise
            print(f"Retry {attempt}/8: {error}", file=sys.stderr, flush=True)
            time.sleep(min(attempt * 2, 10))
    if not verify(partial, expected_hash, expected_size):
        raise ValueError(f"SHA-256/size verification failed; partial file retained: {partial}")
    partial.replace(target)
    print(f"Verified SHA-256: {target.name}", flush=True)
    return target


def verify_hub_metadata(model: dict) -> None:
    """Ensure the current Hub object still matches our pinned release manifest."""
    url = f"https://huggingface.co/api/models/{model['repo']}?blobs=true"
    with urlopen(Request(url, headers={"User-Agent": "EviDoc/0.1"}), timeout=30) as response:
        metadata = json.load(response)
    item = next(x for x in metadata["siblings"] if x["rfilename"] == model["filename"])
    lfs = item.get("lfs", {})
    if lfs.get("sha256") != model["sha256"] or lfs.get("size") != model["size"]:
        raise ValueError(f"Hub metadata changed for {model['filename']}; review manifest before downloading")


def install_runtime(backend: str, manifest: dict) -> Path:
    release = manifest["runtime"][backend]
    archive = ROOT / "runtime" / f"llama-{backend}.zip"
    download(release["url"], archive, release["sha256"])
    destination = ROOT / "runtime" / f"llama-{backend}"
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for entry in bundle.infolist():
            candidate = (destination / entry.filename).resolve()
            if not candidate.is_relative_to(destination.resolve()):
                raise ValueError("Invalid archive path")
        bundle.extractall(destination)
    candidates = list(destination.rglob("llama-server.exe"))
    if not candidates:
        raise ValueError("Downloaded archive has no llama-server.exe")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["all", "generation", "embedding"], default="all")
    parser.add_argument("--runtime", choices=["none", "cpu", "vulkan", "all"], default="none")
    parser.add_argument("--runtime-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--connections", type=int, choices=range(1, 17), default=4)
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not args.runtime_only:
        for role, model in manifest["models"].items():
            if args.model not in ("all", role):
                continue
            target = ROOT / "models" / model["filename"]
            if args.verify_only:
                if not verify(target, model["sha256"], model["size"]):
                    raise ValueError(f"Missing or corrupt model: {target}")
                print(f"Verified {target.name}")
                continue
            verify_hub_metadata(model)
            url = f"https://huggingface.co/{model['repo']}/resolve/main/{model['filename']}"
            if args.connections > 1:
                download_parallel(url, target, model["sha256"], model["size"], args.connections)
            else:
                download(url, target, model["sha256"], model["size"])
    for backend in ("vulkan", "cpu"):
        if args.runtime in (backend, "all"):
            print(install_runtime(backend, manifest))


if __name__ == "__main__":
    main()
