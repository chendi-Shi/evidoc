"""Export a validation-selected LoRA deployment ZIP without training data or optimizer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from inference.bundle import export_bundle


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/Qwen3.5-2B")
    parser.add_argument("--experiment", type=Path, default=ROOT / "runs/experiment-96h")
    parser.add_argument("--output", type=Path, default=ROOT / "dist/evidoc-adapter")
    parser.add_argument("--archive", type=Path, default=ROOT / "dist/evidoc-adapter.zip")
    args = parser.parse_args()
    print(json.dumps(export_bundle(args.experiment, args.model, args.output, args.archive), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
