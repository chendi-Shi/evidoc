"""Validate local experiment evidence and export a statistics-only release report.

No torch, model loading or network access is used. Dataset rows, predictions,
example IDs, local paths and process information never enter the export schema.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics

from training.data import REFUSAL, answer_scores, normalize_answer


METRICS = ("normalized_em", "character_f1", "missing_context_exact_refusal",
           "answerable_false_refusal_rate")
ARGUMENTS = ("quantization", "dtype", "max_length", "rank", "targets", "dropout",
             "accumulation", "learning_rate", "epochs", "save_steps", "eval_steps",
             "eval_samples", "early_stopping_patience", "seed")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def select(value, keys):
    return {key: value[key] for key in keys}


def file_hashes(value):
    """Only filenames and SHA-256 digests may cross the release boundary."""
    for name, digest in value.items():
        require(re.fullmatch(r"[A-Za-z0-9_.-]+", name), "Unsafe fingerprint filename")
        require(re.fullmatch(r"[0-9a-f]{64}", digest), "Invalid SHA-256 digest")
    return dict(value)


def validate_public_report(value):
    forbidden = {"prediction", "predictions", "answers", "question", "context",
                 "example_ids", "command", "command_line", "pid", "adapter", "output"}
    def walk(item):
        if isinstance(item, dict):
            require(not (set(item) & forbidden), "Private field in public export")
            for child in item.values():
                walk(child)
        elif isinstance(item, list):
            for child in item:
                walk(child)
        elif isinstance(item, str):
            require(not re.search(r"[A-Za-z]:[\\/]|\\\\|^/", item),
                    "Absolute local path in public export")
    walk(value)


def verify_predictions(path, expected, expected_ids):
    require(sha256(path) == expected["predictions_sha256"], "Prediction digest mismatch")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    answers = [row for row in rows if not row["missing_context"]]
    probes = [row for row in rows if row["missing_context"]]
    require([row["id"] for row in answers] == expected_ids, "Prediction sample order differs")
    require(len(answers) == expected["answered_count"], "Answer count mismatch")
    require(len(probes) == expected["refusal_probe_count"], "Refusal probe count mismatch")
    require([row["id"] for row in probes] == expected_ids[:len(probes)], "Probe IDs differ")
    scores = [answer_scores(row["prediction"], row["answers"]) for row in answers]
    recomputed = {
        "normalized_em": statistics.mean(row["normalized_em"] for row in scores),
        "character_f1": statistics.mean(row["character_f1"] for row in scores),
        "answerable_false_refusal_rate": statistics.mean(
            normalize_answer(row["prediction"]) == normalize_answer(REFUSAL) for row in answers),
        "missing_context_exact_refusal": statistics.mean(
            answer_scores(row["prediction"], [REFUSAL])["normalized_em"] for row in probes) if probes else None,
    }
    for metric, number in recomputed.items():
        require(number == expected[metric] if number is None else
                math.isclose(number, expected[metric], abs_tol=1e-12), "Prediction metric mismatch: " + metric)


def build_report(experiment: Path, project: Path):
    state = read_json(experiment / "experiment.json")
    final = read_json(experiment / "final_comparison.json")
    dataset_path = project / state["data"] / "manifest.json"
    dataset = read_json(dataset_path)
    snapshot_path = project / state["model"] / "snapshot_manifest.json"
    snapshot = read_json(snapshot_path)
    require(state["state"] == "completed" and state.get("evaluation_state") == "completed",
            "Training and final evaluation must both be complete")
    require(len(state["runs"]) == len(state["configurations"]) == final["completed_comparisons"],
            "Experiment or comparison count is incomplete")
    final_by_name = {row["name"]: row for row in final["runs"]}
    require(len(final_by_name) == len(state["runs"]), "Duplicate or missing final comparisons")
    exported, shared_ids, protocol = [], None, None
    for record in state["runs"]:
        name = record["name"]
        require(re.fullmatch(r"[A-Za-z0-9_-]+", name), "Unsafe run name")
        require(record["state"] == "completed" and record["exit_code"] == 0, "Incomplete run: " + name)
        run_dir = experiment / name
        manifest = read_json(run_dir / "run_manifest.json")
        comparison = read_json(run_dir / "heldout-evaluation/comparison.json")
        tokenization = read_json(run_dir / "tokenization_report.json")
        monitor = read_json(run_dir / "monitor_state.json")
        require(manifest["training_complete"], "Missing completed training manifest")
        for key in ("rank", "seed", "targets"):
            require(manifest["arguments"][key] == record["configuration"][key] == final_by_name[name][key],
                    "Run configuration mismatch: " + key)
        fingerprints = manifest["model_fingerprint"]["local_files_sha256"]
        require(fingerprints["snapshot_manifest.json"] == sha256(snapshot_path)
                and snapshot["sha256"] in fingerprints.values(), "Model fingerprint differs from pinned snapshot")
        require(math.isfinite(monitor["best_loss"]) and monitor["best_loss"] >= 0,
                "Invalid validation loss")
        require(manifest["dataset_manifest_sha256"] == comparison["dataset_manifest_sha256"] == sha256(dataset_path),
                "Dataset manifest differs from experiment")
        require(manifest["dataset_revision"] == comparison["dataset_revision"] == dataset["revision"],
                "Dataset revision mismatch")
        require(final_by_name[name]["results"] == comparison["results"], "Aggregate differs from run report")
        require(final_by_name[name]["example_ids"] == comparison["example_ids"], "Aggregate sample differs")
        if shared_ids is None:
            shared_ids = comparison["example_ids"]
        require(shared_ids == comparison["example_ids"], "Runs use different held-out samples")
        current_protocol = {**select(comparison, ("source_test_rows", "retained_test_rows", "evaluated_rows",
            "overlength_filtered", "max_length")),
            "generation": select(comparison["generation"], ("do_sample", "max_new_tokens", "enable_thinking")),
            "empty_context_probe_count": comparison["results"]["base"]["refusal_probe_count"]}
        if protocol is None:
            protocol = current_protocol
        require(protocol == current_protocol, "Runs use different evaluation settings")
        adapter_hashes = file_hashes(comparison["adapter_files_sha256"])
        for filename, digest in adapter_hashes.items():
            require(sha256(run_dir / "adapter" / filename) == digest, "Adapter digest mismatch")
        results = {}
        for variant in ("base", "adapter"):
            metrics = comparison["results"][variant]
            verify_predictions(run_dir / "heldout-evaluation" / f"{variant}_predictions.jsonl", metrics, shared_ids)
            results["base" if variant == "base" else "finetuned"] = select(metrics,
                (*METRICS, "answered_count", "refusal_probe_count", "elapsed_seconds", "predictions_sha256"))
        checkpoint = manifest["best_checkpoint"].replace("\\", "/").rsplit("/", 1)[-1]
        require(re.fullmatch(r"checkpoint-\d+", checkpoint), "Invalid best checkpoint")
        exported.append({
            "name": name,
            "configuration": select(manifest["arguments"], ARGUMENTS),
            "lora": select(manifest["lora"], ("rank", "alpha", "dropout", "targets",
                "trainable_parameters", "total_parameters", "adapted_module_counts")),
            "training": {**select(manifest, ("created_at_utc", "finished_at_utc", "global_step",
                "training_seconds", "peak_allocated_gib", "stop_reason")),
                "best_checkpoint_step": int(checkpoint.split("-")[1]),
                "best_validation_loss": monitor["best_loss"],
                "train_rows_retained": tokenization["train"]["kept_rows"],
                "validation_rows_retained": tokenization["validation"]["kept_rows"]},
            "environment": {**select(manifest["model"], ("class", "dtype", "quantization", "attention",
                "gpu", "torch", "cuda_runtime")),
                "packages": select(manifest["packages"], ("torch", "transformers", "peft", "accelerate", "safetensors"))},
            "model_files_sha256": file_hashes(manifest["model_fingerprint"]["local_files_sha256"]),
            "adapter_files_sha256": adapter_hashes,
            "results": results,
        })
    grouped = {}
    for row in exported:
        config = row["configuration"]
        key = f"r{config['rank']}-{config['targets']}"
        grouped.setdefault(key, []).append(row)
    aggregate = {}
    for key, rows in grouped.items():
        aggregate[key] = {"seeds": [row["configuration"]["seed"] for row in rows]}
        for metric in METRICS:
            values = [row["results"]["finetuned"][metric] for row in rows]
            aggregate[key][metric] = {"mean": statistics.mean(values),
                "sample_stdev": statistics.stdev(values) if len(values) > 1 else None}
            if metric in final["aggregate"][key]:
                require(aggregate[key][metric] == final["aggregate"][key][metric], "Aggregate statistics differ")
    report = {
        "schema_version": 1,
        "status": "training_and_final_evaluation_complete",
        "completed_runs": len(exported),
        "source_evidence_sha256": {"experiment.json": sha256(experiment / "experiment.json"),
            "final_comparison.json": sha256(experiment / "final_comparison.json")},
        "experiment": select(state, ("created_at", "finished_at", "budget_hours", "elapsed_seconds", "gpu")),
        "base_model": select(snapshot, ("repo", "revision", "sha256")),
        "dataset": {**select(dataset, ("dataset", "repository", "revision", "seed", "validation_fraction")),
            "manifest_sha256": sha256(dataset_path),
            "source_files_sha256": file_hashes({Path(k).name: v for k, v in dataset["raw_hashes_or_hf_fingerprints"].items()}),
            "splits": {key: select(value, ("sha256", "rows", "groups", "missing_context_rows"))
                for key, value in dataset["splits"].items()}},
        "evaluation": {**protocol,
            "ordered_sample_ids_sha256": hashlib.sha256(json.dumps(shared_ids, ensure_ascii=False,
                separators=(",", ":")).encode("utf-8")).hexdigest(),
            "metric_scope": "Custom normalized EM and character-multiset F1; not official CMRC scoring.",
            "refusal_scope": "Artificial empty-context policy probe, not general unanswerable QA accuracy.",
            "selection_note": "Comparative results on one fixed held-out sample; configuration choice from these scores is exploratory."},
        "aggregate": aggregate,
        "deployment_selection": {
            "rule": "Lowest recorded validation loss across completed runs; held-out scores are not used for selection.",
            "run": min(exported, key=lambda row: (row["training"]["best_validation_loss"], row["name"]))["name"],
        },
        "runs": exported,
    }
    validate_public_report(report)
    return report


def render_files(report):
    csv_stream = io.StringIO(newline="")
    writer = csv.writer(csv_stream, lineterminator="\n")
    writer.writerow(["run", "rank", "targets", "seed", "final_step", "best_checkpoint_step",
                     "training_seconds", "base_em", "finetuned_em", "base_character_f1", "finetuned_character_f1"])
    for row in report["runs"]:
        config, training, result = row["configuration"], row["training"], row["results"]
        writer.writerow([row["name"], config["rank"], config["targets"], config["seed"], training["global_step"],
            training["best_checkpoint_step"], training["training_seconds"], result["base"]["normalized_em"],
            result["finetuned"]["normalized_em"], result["base"]["character_f1"], result["finetuned"]["character_f1"]])
    return {"experiment_summary.json": json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            "per_run_metrics.csv": csv_stream.getvalue()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=Path("runs/experiment-96h"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/results"))
    parser.add_argument("--check", action="store_true", help="Verify existing exports match local evidence without writing")
    args = parser.parse_args()
    report = build_report(args.experiment, Path(__file__).resolve().parents[1])
    files = render_files(report)
    if not args.check:
        args.output.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        path = args.output / name
        if args.check:
            require(path.read_text(encoding="utf-8") == content, "Export is stale: " + name)
        else:
            path.write_text(content, encoding="utf-8", newline="\n")
    print(json.dumps({"verified_runs": report["completed_runs"], "files": list(files), "check_only": args.check}))


if __name__ == "__main__":
    main()
