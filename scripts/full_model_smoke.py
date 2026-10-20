"""Run two synthetic full-model GPU steps; never save the resulting adapter.

This verifies device compatibility and memory, not model quality or real-data
training. Run from the project with its independent Python installation.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_REPORT = ROOT / "reports" / "full_model_gpu_smoke.json"


def publish(report: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    def json_default(value):
        if isinstance(value, (set, frozenset)):
            return sorted(value)
        raise TypeError(f"Unsupported report value: {type(value).__name__}")

    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    temporary.replace(destination)


def card_memory() -> dict:
    """Sample entire-card usage, including display/other processes, in MiB."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--id=0", "--query-gpu=name,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        name, total, used, free = next(csv.reader(result.stdout.splitlines()))
        return {"name": name.strip(), "total_mib": int(total), "used_mib": int(used),
                "free_mib": int(free), "sampled_at": datetime.now(timezone.utc).isoformat()}
    except Exception as error:
        return {"error": str(error)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    if not 128 <= args.sequence_length <= 4096:
        parser.error("--sequence-length must be between 128 and 4096")
    destination = args.output.resolve()
    started = time.perf_counter()
    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "success": False,
        "purpose": "Synthetic token GPU compatibility and memory test; not a semantic training experiment.",
        "adapter_saved": False,
        "batch_size": 1,
        "sequence_length": args.sequence_length,
        "masked_prompt_tokens": args.sequence_length - 32,
        "supervised_tokens": 32,
        "requested_optimizer_steps": 2,
        "steps": [],
        "phase": "importing",
        "card_memory_before": card_memory(),
        "card_memory_sampling": "nvidia-smi entire-card snapshots before loading, after loading and after each completed optimizer step; sampled maximum, not continuous kernel peak",
    }
    publish(report, destination)
    torch = None
    try:
        import torch
        from training.modeling import attach_lora, load_model

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; CPU offload is forbidden for this test")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("This test requires CUDA BF16 support")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.cuda.reset_peak_memory_stats()
        report.update(
            torch_version=torch.__version__, cuda_runtime=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0),
            total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
            phase="loading_model",
        )
        publish(report, destination)
        load_started = time.perf_counter()
        model, model_info = load_model(str(ROOT / "models" / "Qwen3.5-2B"), None, "none", "bf16")
        model, lora_info = attach_lora(model, "none", 8, "auto", .05)
        model.train()
        torch.cuda.synchronize()
        parameters = list(model.parameters())
        all_cuda = all(parameter.device.type == "cuda" and parameter.device.index == 0 for parameter in parameters)
        if not all_cuda:
            raise RuntimeError("Some model parameters are not on CUDA:0")
        if not model.is_gradient_checkpointing:
            raise RuntimeError("Gradient checkpointing is not enabled")
        trainable = [parameter for parameter in parameters if parameter.requires_grad]
        if not trainable:
            raise RuntimeError("No trainable LoRA parameters")
        report.update(
            model=model_info, lora=lora_info,
            load_seconds=time.perf_counter() - load_started,
            all_parameters_cuda=all_cuda,
            gradient_checkpointing=bool(model.is_gradient_checkpointing),
            parameter_devices=dict(Counter(str(parameter.device) for parameter in parameters)),
            parameter_dtypes=dict(Counter(str(parameter.dtype) for parameter in parameters)),
            model_allocated_bytes=torch.cuda.memory_allocated(),
            card_memory_loaded=card_memory(),
            phase="running_steps",
        )
        publish(report, destination)
        print(json.dumps({"phase": "loaded", "load_seconds": report["load_seconds"],
                          "allocated_gib": report["model_allocated_bytes"] / 2**30,
                          "trainable_parameters": lora_info["trainable_parameters"]}), flush=True)

        optimizer = torch.optim.AdamW(trainable, lr=1e-4, foreach=False)
        inputs = torch.randint(1, model.config.vocab_size, (1, args.sequence_length), device="cuda:0")
        labels = inputs.clone()
        labels[:, :-32] = -100
        attention_mask = torch.ones_like(inputs)
        for step in range(1, 3):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            step_started = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(input_ids=inputs, attention_mask=attention_mask, labels=labels, use_cache=False)
                loss = output.loss
            torch.cuda.synchronize()
            forward_seconds = time.perf_counter() - step_started
            loss_value = loss.detach().float().item()
            if not math.isfinite(loss_value):
                raise RuntimeError(f"Step {step} produced non-finite loss: {loss_value}")
            backward_started = time.perf_counter()
            loss.backward()
            torch.cuda.synchronize()
            backward_seconds = time.perf_counter() - backward_started
            gradients = [parameter.grad for parameter in trainable if parameter.grad is not None]
            if len(gradients) != len(trainable):
                raise RuntimeError(f"Only {len(gradients)} of {len(trainable)} trainable tensors received gradients")
            if not torch.stack([torch.isfinite(gradient).all() for gradient in gradients]).all().item():
                raise RuntimeError(f"Step {step} produced non-finite LoRA gradients")
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0).item()
            optimizer_started = time.perf_counter()
            optimizer.step()
            torch.cuda.synchronize()
            optimizer_seconds = time.perf_counter() - optimizer_started
            item = {
                "step": step, "loss": loss_value, "loss_finite": True,
                "gradients_finite": True, "gradient_norm_before_clip": gradient_norm,
                "forward_seconds": forward_seconds, "backward_seconds": backward_seconds,
                "optimizer_seconds": optimizer_seconds,
                "total_seconds": time.perf_counter() - step_started,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "card_memory": card_memory(),
            }
            report["steps"].append(item)
            publish(report, destination)
            print(json.dumps(item), flush=True)
            del output, loss, gradients
        report.update(success=True, phase="complete")
    except Exception as error:
        report.update(phase="failed", error=repr(error), traceback=traceback.format_exc())
        print(report["traceback"], flush=True)
    finally:
        report.update(finished_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter() - started)
        if torch is not None and torch.cuda.is_available():
            report.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved())
        report["card_memory_final"] = card_memory()
        samples = [report["card_memory_before"], report.get("card_memory_loaded", {}), report["card_memory_final"]]
        samples += [step["card_memory"] for step in report["steps"]]
        valid_samples = [sample for sample in samples if "used_mib" in sample]
        if valid_samples:
            report["sampled_peak_card_used_mib"] = max(sample["used_mib"] for sample in valid_samples)
            report["sampled_min_card_free_mib"] = min(sample["free_mib"] for sample in valid_samples)
        publish(report, destination)
        print(json.dumps({"success": report["success"], "phase": report["phase"],
                          "report": str(destination), "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
