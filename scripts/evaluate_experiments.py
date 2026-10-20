"""Evaluate completed adapters on one fixed held-out sample; no invented scores."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

from training.status import write_json_atomic

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--refusal-samples", type=int, default=50)
    args = parser.parse_args()
    experiment = json.loads((args.experiment / "experiment.json").read_text(encoding="utf-8"))
    if experiment["state"] in {"running", "saving_before_stop"}:
        raise RuntimeError("Wait for training to stop before occupying the same GPU for evaluation.")
    env = {**os.environ, "PYTHONUTF8": "1", "CUDA_VISIBLE_DEVICES": "0", "HF_HUB_OFFLINE": "1"}
    results = []
    for run in experiment["runs"]:
        if run["state"] != "completed":
            continue
        run_dir = args.experiment / run["name"]
        target = run_dir / "heldout-evaluation"
        comparison = target / "comparison.json"
        if not comparison.exists():
            info = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            command = [sys.executable, "-m", "training.evaluate", "--model", experiment["model"],
                "--data", experiment["data"], "--adapter", str(run_dir / "adapter"), "--output", str(target),
                "--max-length", str(info["arguments"]["max_length"]), "--samples", str(args.samples),
                "--refusal-samples", str(args.refusal_samples)]
            subprocess.run(command, cwd=ROOT, env=env, check=True)
        report = json.loads(comparison.read_text(encoding="utf-8"))
        if "adapter" not in report.get("results", {}):
            raise RuntimeError(f"Incomplete evaluation in {target}; preserve it and rerun to a new output directory.")
        if results and results[0]["example_ids"] != report["example_ids"]:
            raise RuntimeError("Evaluation sample IDs differ; comparisons would not be fair.")
        results.append({"name": run["name"], "seed": run["configuration"]["seed"],
                        "rank": run["configuration"]["rank"], "targets": run["configuration"]["targets"],
                        "example_ids": report["example_ids"], "results": report["results"],
                        "delta": report["delta_adapter_minus_base"]})
    grouped = {}
    for row in results:
        key = f"r{row['rank']}-{row['targets']}"
        grouped.setdefault(key, []).append(row)
    aggregate = {}
    for key, rows in grouped.items():
        aggregate[key] = {"seeds": [row["seed"] for row in rows]}
        for metric in ("normalized_em", "character_f1", "answerable_false_refusal_rate"):
            values = [row["results"]["adapter"][metric] for row in rows]
            aggregate[key][metric] = {"mean": statistics.mean(values),
                                      "sample_stdev": statistics.stdev(values) if len(values) > 1 else None}
    result = {"completed_comparisons": len(results), "aggregate": aggregate, "runs": results,
              "note": "Fixed held-out sample, custom metrics; not an official benchmark or guaranteed improvement."}
    write_json_atomic(args.experiment / "final_comparison.json", result)
    print(json.dumps({"completed_comparisons": len(results), "aggregate": aggregate}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
