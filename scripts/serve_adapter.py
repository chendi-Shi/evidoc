"""Serve the validation-selected local Qwen3.5-2B adapter on CUDA:0.

python -m scripts.serve_adapter --inspect  # CPU-only provenance verification
python -m scripts.serve_adapter           # foreground, localhost:8083
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket

from inference.adapter import CudaAdapter, select_adapter, verify_artifacts
from inference.service import create_app


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/Qwen3.5-2B")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--experiment", type=Path, default=ROOT / "runs/experiment-96h")
    source.add_argument("--bundle", type=Path, help="Use an exported deployment directory without training checkpoints")
    parser.add_argument("--port", type=int, default=8083)
    parser.add_argument("--inspect", action="store_true", help="Verify chosen artifacts without loading CUDA")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be in 1..65535")
    if args.inspect:
        if args.bundle:
            from inference.bundle import verify_bundle

            identity = verify_bundle(args.bundle, args.model)
        else:
            identity = verify_artifacts(select_adapter(args.experiment), args.model)
        print(json.dumps(identity, ensure_ascii=False, indent=2))
        return
    # Reserve the socket before any model allocation to avoid a second server
    # consuming GPU memory when this port is already in use.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        server_socket.bind(("127.0.0.1", args.port))
        import uvicorn

        app = create_app(loader=lambda: CudaAdapter(args.model, args.experiment, bundle=args.bundle))
        config = uvicorn.Config(app, host="127.0.0.1", port=args.port, workers=1)
        uvicorn.Server(config).run(sockets=[server_socket])


if __name__ == "__main__":
    main()
