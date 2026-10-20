"""Bounded, resumable single-GPU LoRA training; run a pilot before a long job."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import signal
import subprocess
import time

from .data import digest, encode_partition, file_digest, pad_features, pilot_estimate, read_jsonl
from .modeling import attach_lora, load_model, load_tokenizer, resolve_revision
from .status import publish_status, write_json_atomic
from .checkpoints import validate_checkpoint, commit_checkpoint
from .recovery import (ActiveRunError, acquire_run_lock, archive_replayed_metrics,
                       consumed_training_seconds, reject_live_worker)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--revision")
    parser.add_argument("--data", type=Path, default=Path("data/training/cmrc2018"))
    parser.add_argument("--output", type=Path, default=Path("runs/lora-qwen35-2b-r8-s42"))
    parser.add_argument("--quantization", choices=("none", "nf4"), default="none")
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16"), default="auto")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--targets", default="auto", help="auto covers hybrid attention projections; or comma-separated suffixes")
    parser.add_argument("--dropout", type=float, default=.05)
    parser.add_argument("--accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--hours", type=float, default=72, help="Wall-clock upper bound after model loading, never a minimum runtime")
    parser.add_argument("--max-steps", type=int, default=-1, help="Optional lower step cap; cannot exceed the epoch budget")
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--eval-samples", type=int, default=128)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--pilot-steps", type=int, default=20)
    parser.add_argument("--resume", type=Path, help="Explicit Trainer checkpoint directory, including optimizer/scheduler state")
    return parser.parse_args()


def verify_data(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for split, info in manifest["splits"].items():
        if digest((directory / info["file"]).read_bytes()) != info["sha256"]:
            raise ValueError(f"Dataset integrity mismatch in {split}; prepare a new version instead of editing it")
    return manifest


def package_versions() -> dict:
    result = {}
    for package in ("torch", "transformers", "peft", "accelerate", "datasets", "bitsandbytes", "safetensors"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def model_fingerprint(model_id: str, revision: str | None) -> dict:
    path = Path(model_id)
    if not path.is_dir():
        return {"repository": model_id, "revision": revision}
    names = {"snapshot_manifest.json", "config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"}
    files = [item for item in path.iterdir() if item.is_file() and (
        item.name in names or item.suffix == ".safetensors" or item.name.startswith("pytorch_model")
        or item.name.endswith(".index.json"))]
    return {"local_files_sha256": {item.name: file_digest(item) for item in sorted(files)}}


def validate_resume(previous: dict, current: dict):
    if previous.get("pilot"):
        raise ValueError("Pilot checkpoints cannot silently become a full experiment; start the planned run from the base model")
    immutable = ("model", "revision", "max_length", "rank", "targets", "dropout", "accumulation", "quantization",
                 "dtype", "learning_rate", "epochs", "max_steps", "seed", "eval_steps", "save_steps", "eval_samples",
                 "early_stopping_patience")
    for key in immutable:
        if previous["arguments"].get(key) != current["arguments"].get(key):
            raise ValueError(f"Resume configuration differs for {key}")
    for key in ("dataset_manifest_sha256", "model_fingerprint", "lora"):
        if previous.get(key) != current.get(key):
            raise ValueError(f"Resume identity differs for {key}")
    for key in ("dtype", "revision", "class"):
        if previous["model"].get(key) != current["model"].get(key):
            raise ValueError(f"Resume resolved model configuration differs for {key}")


def main():
    args = parse_args()
    if not 128 <= args.max_length <= 4096 or not 1 <= args.rank <= 128 or args.accumulation < 1:
        raise ValueError("Invalid sequence length, rank or gradient accumulation")
    if not 0 < args.epochs <= 3 or not 0 < args.hours <= 96:
        raise ValueError("Use 0 < epochs <= 3 and a wall-clock ceiling of at most 96 hours; run explicit ablations separately")
    if min(args.save_steps, args.eval_steps, args.logging_steps, args.pilot_steps, args.eval_samples, args.early_stopping_patience) < 1:
        raise ValueError("Step intervals and evaluation/pilot sizes must be positive")
    checkpoint_info = None
    if args.resume:
        if args.pilot or args.resume.resolve().parent != args.output.resolve():
            raise ValueError("Resume needs a non-pilot checkpoint inside the same output directory")
        checkpoint_info = validate_checkpoint(args.resume)
    if args.output.exists() and any(p.name != ".training.lock" for p in args.output.iterdir()) and not args.resume:
        raise ValueError("Output directory is nonempty. Choose a fresh run or explicitly resume a checkpoint.")
    if (args.output / "STOP_REQUESTED").exists():
        raise ValueError("STOP_REQUESTED is still present. Explicitly remove the stop request before resuming.")
    reject_live_worker(args.output)
    worker_lock = acquire_run_lock(args.output / ".training.lock")
    publish_status(state="preparing", model=args.model, quantization=args.quantization,
                   budget_hours=args.hours, steps=0, loss=None, elapsed_hours=0,
                   device=None, gpu_verified=False,
                   message="准备数据、分词器与模型；尚未开始优化器训练。", history=[], artifacts=[])

    import torch
    from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed

    manifest = verify_data(args.data)
    revision = resolve_revision(args.model, args.revision)
    fingerprint = model_fingerprint(args.model, revision)
    tokenizer = load_tokenizer(args.model, revision)
    train, train_report = encode_partition(tokenizer, read_jsonl(args.data / "train.jsonl"), args.max_length)
    validation_raw = read_jsonl(args.data / "validation.jsonl")
    validation, validation_report = encode_partition(tokenizer, validation_raw, args.max_length)
    if not train or not validation:
        raise ValueError("No training/validation rows remain after the documented length filter")
    validation = validation[:args.eval_samples]
    epoch_steps = math.ceil(math.ceil(len(train) / args.accumulation) * args.epochs)
    max_steps = min(args.pilot_steps, epoch_steps) if args.pilot else (
        min(args.max_steps, epoch_steps) if args.max_steps > 0 else -1)
    # Save and eval share boundaries, ensuring best-checkpoint selection works.
    checkpoint_interval = min(args.save_steps, args.eval_steps)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"max_length": args.max_length, "train": train_report, "validation": validation_report,
              "validation_rows_used_each_eval": len(validation),
              "validation_selection": "first deterministic hash-shuffled retained examples; no held-out test peeking"}
    (args.output / "tokenization_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"train_kept": len(train), "train_overlength_filtered": train_report["overlength_removed"],
                      "validation_kept": validation_report["kept_rows"], "epoch_step_budget": epoch_steps}, ensure_ascii=False))
    set_seed(args.seed)
    model, model_info = load_model(args.model, revision, args.quantization, args.dtype)
    model, lora_info = attach_lora(model, args.quantization, args.rank, args.targets, args.dropout)
    model.print_trainable_parameters()
    torch.cuda.reset_peak_memory_stats()
    run_info = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "dataset_manifest_sha256": digest((args.data / "manifest.json").read_bytes()),
                "dataset_revision": manifest["revision"], "model": model_info, "lora": lora_info,
                "model_fingerprint": fingerprint,
                "packages": package_versions(), "supervision": "assistant answer and end-of-turn tokens only; prompt labels -100",
                "pilot": args.pilot, "training_complete": False, "checkpoint_format": 1,
                "checkpoint_commit_required_from_step": (checkpoint_info["global_step"] + 1 if checkpoint_info else 1)}
    info_path = args.output / "run_manifest.json"
    if args.resume:
        previous_path = args.resume.parent / "run_manifest.json"
        if not previous_path.exists():
            raise ValueError("Resume checkpoint is missing its parent run_manifest.json")
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        validate_resume(previous, run_info)
        run_info["resumed_from"] = str(args.resume.resolve())
        monitors = [json.loads((args.resume / "monitor_state.json").read_text(encoding="utf-8"))]
        if (args.output / "monitor_state.json").exists():
            monitors.append(json.loads((args.output / "monitor_state.json").read_text(encoding="utf-8")))
        run_info["previous_training_seconds"] = consumed_training_seconds(previous, *monitors)
        # Older checkpoints remain valid after this run adopts commit manifests.
        run_info["checkpoint_commit_required_from_step"] = min(
            previous.get("checkpoint_commit_required_from_step", run_info["checkpoint_commit_required_from_step"]),
            run_info["checkpoint_commit_required_from_step"])
    previous_seconds = run_info.get("previous_training_seconds", 0)
    run_info["training_seconds"] = previous_seconds
    remaining_seconds = args.hours * 3600 - previous_seconds
    if remaining_seconds <= 0:
        raise ValueError("The cumulative wall-clock budget has already been consumed")
    write_json_atomic(info_path, run_info)

    class AnswerCollator:
        def __call__(self, features):
            return {key: torch.tensor(value, dtype=torch.long)
                    for key, value in pad_features(features, tokenizer.pad_token_id).items()}

    class RunMonitor(TrainerCallback):
        def __init__(self):
            self.started = self.step_started = 0.0
            self.step_times = []
            self.interrupted = False
            self.reason = "epochs_or_step_cap"
            self.best_loss = float("inf")
            self.bad_evals = 0
            self.history = []
            if args.resume and (args.resume / "monitor_state.json").exists():
                saved = json.loads((args.resume / "monitor_state.json").read_text(encoding="utf-8"))
                self.best_loss = saved.get("best_loss") if saved.get("best_loss") is not None else float("inf")
                self.bad_evals = saved.get("bad_evals", 0)
                self.history = saved.get("history", [])

        def persist(self, state, checkpoint=None):
            elapsed = previous_seconds + time.monotonic() - self.started
            run_info.update(training_seconds=elapsed, global_step=state.global_step,
                            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
            write_json_atomic(info_path, run_info)
            saved = {"training_seconds": elapsed, "best_loss": self.best_loss if math.isfinite(self.best_loss) else None,
                     "bad_evals": self.bad_evals, "history": self.history[-1000:], "step": state.global_step}
            write_json_atomic(args.output / "monitor_state.json", saved)
            if checkpoint:
                write_json_atomic(checkpoint / "monitor_state.json", saved)

        def request_stop(self, *_):
            self.interrupted = True
            print("Stop requested. Finishing the current optimizer step, then saving a resumable checkpoint.", flush=True)

        def on_train_begin(self, args_, state, control, **kwargs):
            self.started = time.monotonic()
            publish_status(state="running", steps=state.global_step, device="cuda:0", gpu_verified=True,
                           gpu=torch.cuda.get_device_name(0), message="GPU 微调运行中。")

        def on_step_begin(self, args_, state, control, **kwargs):
            self.step_started = time.monotonic()

        def on_step_end(self, args_, state, control, **kwargs):
            self.step_times.append(time.monotonic() - self.step_started)
            if (args.output / "STOP_REQUESTED").exists():
                self.interrupted = True
            if self.interrupted or time.monotonic() - self.started >= remaining_seconds:
                self.reason = "interrupt" if self.interrupted else "wall_clock_budget"
                control.should_training_stop = control.should_save = True
            return control

        def on_evaluate(self, args_, state, control, metrics=None, **kwargs):
            loss = (metrics or {}).get("eval_loss")
            if loss is not None and math.isfinite(loss):
                if loss < self.best_loss - 1e-4:
                    self.best_loss, self.bad_evals = loss, 0
                else:
                    self.bad_evals += 1
                if not args.pilot and self.bad_evals >= args.early_stopping_patience:
                    self.reason = "validation_early_stop"
                    control.should_training_stop = control.should_save = True
            return control

        def on_log(self, args_, state, control, logs=None, **kwargs):
            for metric in ("loss", "grad_norm", "eval_loss"):
                if metric in (logs or {}) and not math.isfinite(float(logs[metric])):
                    raise FloatingPointError(f"Non-finite {metric}={logs[metric]}; stopping instead of hiding an invalid run")
            entry = {"utc": datetime.now(timezone.utc).isoformat(), "step": state.global_step,
                     "elapsed_seconds": round(previous_seconds + time.monotonic() - self.started, 3), **(logs or {}),
                     "allocated_gib": torch.cuda.memory_allocated() / 2**30,
                     "reserved_gib": torch.cuda.memory_reserved() / 2**30,
                     "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}
            try:
                query = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,memory.used",
                                        "--format=csv,noheader,nounits", "--id=0"],
                                       capture_output=True, text=True, timeout=3, check=True)
                temperature, utilization, memory = query.stdout.strip().split(",")
                entry.update(gpu_temperature_c=float(temperature), gpu_utilization_percent=float(utilization),
                             gpu_process_and_system_memory_mib=float(memory))
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            with (args.output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fields = ("utc", "step", "elapsed_seconds", "loss", "eval_loss", "learning_rate", "grad_norm",
                      "allocated_gib", "reserved_gib", "peak_allocated_gib", "gpu_temperature_c", "gpu_utilization_percent")
            csv_path = args.output / "metrics.csv"
            new_file = not csv_path.exists()
            with csv_path.open("a", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
                if new_file:
                    writer.writeheader()
                writer.writerow(entry)
            if "loss" in entry:
                self.history.append({"step": state.global_step, "loss": entry["loss"]})
            publish_status(state="running", steps=state.global_step,
                           loss=entry.get("loss", self.history[-1]["loss"] if self.history else None),
                           elapsed_hours=(previous_seconds + time.monotonic() - self.started) / 3600,
                           peak_vram_gb=entry["peak_allocated_gib"], history=self.history,
                           message="GPU 微调运行中；按验证损失、轮数与总时间上限停止。")
            self.persist(state)

        def on_save(self, args_, state, control, **kwargs):
            checkpoint = args.output / f"checkpoint-{state.global_step}"
            self.persist(state, checkpoint)
            commit_checkpoint(checkpoint)

    monitor = RunMonitor()
    previous_handlers = {sig: signal.signal(sig, monitor.request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    training_args = TrainingArguments(
        output_dir=str(args.output), per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.accumulation, learning_rate=args.learning_rate,
        num_train_epochs=args.epochs, max_steps=max_steps, warmup_ratio=.03, lr_scheduler_type="cosine",
        weight_decay=.01, max_grad_norm=1.0, bf16=model_info["dtype"] == "bf16", fp16=model_info["dtype"] == "fp16",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps, logging_nan_inf_filter=False, save_strategy="steps", save_steps=checkpoint_interval,
        save_total_limit=3, save_only_model=False, eval_strategy="no" if args.pilot else "steps",
        eval_steps=checkpoint_interval, prediction_loss_only=True, eval_accumulation_steps=1,
        load_best_model_at_end=not args.pilot, metric_for_best_model="eval_loss", greater_is_better=False,
        optim="adamw_torch", report_to=[], dataloader_num_workers=0, dataloader_pin_memory=False,
        seed=args.seed, data_seed=args.seed, remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train,
                      eval_dataset=validation, data_collator=AnswerCollator(),
                      processing_class=tokenizer, callbacks=[monitor])
    try:
        if args.resume:
            run_info["recovery_log_archives"] = archive_replayed_metrics(args.output, checkpoint_info["global_step"])
            write_json_atomic(info_path, run_info)
        result = trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
        trainer.save_model(str(args.output / "adapter"))
        tokenizer.save_pretrained(args.output / "adapter")
        trainer.save_state()
        run_info.update(training_complete=not args.pilot and monitor.reason not in {"interrupt", "wall_clock_budget"}, stop_reason=monitor.reason,
                        global_step=trainer.state.global_step, metrics=result.metrics,
                        best_checkpoint=trainer.state.best_model_checkpoint)
        if args.pilot:
            estimate = pilot_estimate(monitor.step_times, len(train), args.accumulation, args.epochs, args.hours)
            estimate.update(peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                            adapter_is_pilot_only=True)
            (args.output / "pilot_report.json").write_text(json.dumps(estimate, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(estimate, ensure_ascii=False, indent=2))
    except torch.cuda.OutOfMemoryError:
        run_info.update(training_complete=False, stop_reason="out_of_memory")
        print("CUDA OOM. Try Qwen/Qwen3.5-0.8B or --max-length 256 in a fresh run, then repeat the pilot. No automatic sample truncation was performed.", flush=True)
        raise
    except Exception as exc:
        run_info.update(training_complete=False, stop_reason="error", error=repr(exc))
        raise
    finally:
        run_info["training_seconds"] = previous_seconds + (time.monotonic() - monitor.started if monitor.started else 0)
        run_info["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        run_info["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json_atomic(info_path, run_info)
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    publish_status(state="completed", steps=trainer.state.global_step,
                   elapsed_hours=run_info["training_seconds"] / 3600,
                   peak_vram_gb=run_info["peak_allocated_gib"], history=monitor.history,
                   message=("短跑测试结束；请根据实测速度选择正式训练计划。" if args.pilot else
                            f"本次训练已停止（{monitor.reason}），适配器和检查点已保存。"),
                   artifacts=[str((args.output / name).resolve()) for name in ("adapter", "run_manifest.json", "metrics.csv")])


if __name__ == "__main__":
    try:
        main()
    except ActiveRunError:
        # The existing GPU worker continues; do not mark its dashboard failed.
        raise
    except BaseException as exc:
        publish_status(state="failed", message=str(exc))
        raise
