"""Bounded extractive inference using the training prompt and verified adapter.

Substring verification establishes a quotation's provenance. It does not prove
that the model selected the right entity, date, condition, or answer.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

from training.data import REFUSAL, file_digest, messages_for, normalize_answer


GROUNDING_SCOPE = "exact_source_substring_only; not semantic entailment or factuality verification"


@dataclass(frozen=True)
class Limits:
    max_input_tokens: int = 448
    max_new_tokens: int = 64
    max_sources: int = 3
    max_windows_per_source: int = 2
    max_windows: int = 6
    overlap_characters: int = 40

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass(frozen=True)
class Selection:
    run: Path
    checkpoint: Path
    validation_loss: float
    manifest: dict


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def select_adapter(experiment: Path) -> Selection:
    """Choose completed runs by validation loss; never read held-out test metrics."""
    experiment = experiment.resolve()
    candidates = []
    validation_protocols = set()
    for manifest_path in sorted(experiment.glob("*/run_manifest.json")):
        manifest = read_json(manifest_path)
        if not manifest.get("training_complete") or manifest.get("pilot"):
            continue
        monitor = read_json(manifest_path.parent / "monitor_state.json")
        loss = monitor.get("best_loss")
        if type(loss) not in (int, float) or not math.isfinite(loss) or loss < 0:
            raise ValueError(f"Invalid validation loss in {manifest_path.parent.name}")
        arguments = manifest.get("arguments", {})
        validation_protocols.add(json.dumps({
            "dataset": manifest.get("dataset_manifest_sha256"),
            "eval_samples": arguments.get("eval_samples"),
            "max_length": arguments.get("max_length"),
            "dtype": arguments.get("dtype"),
            "quantization": arguments.get("quantization"),
            "model_fingerprint": manifest.get("model_fingerprint"),
        }, sort_keys=True))
        # Saved paths can be from Windows or a different checkout; only the
        # checkpoint basename is portable, and it must stay inside this run.
        name = str(manifest.get("best_checkpoint", "")).replace("\\", "/").rsplit("/", 1)[-1]
        if not re.fullmatch(r"checkpoint-\d+", name):
            raise ValueError(f"Missing valid best checkpoint for {manifest_path.parent.name}")
        run = manifest_path.parent.resolve()
        checkpoint = (run / name).resolve()
        if run.parent != experiment or checkpoint.parent != run or not checkpoint.is_dir():
            raise ValueError(f"Missing or external checkpoint: {checkpoint}")
        candidates.append(Selection(run, checkpoint, float(loss), manifest))
    if not candidates:
        raise ValueError("No completed training runs with validation checkpoints found")
    if len(validation_protocols) != 1:
        raise ValueError("Runs have different validation/model protocols; cannot compare their losses")
    return min(candidates, key=lambda candidate: (candidate.validation_loss, candidate.run.name))


def verify_artifacts(selection: Selection, model_dir: Path) -> dict:
    """Verify selected checkpoint and exact base/tokenizer bytes without CUDA."""
    from training.checkpoints import validate_checkpoint

    model_dir = model_dir.resolve()
    manifest = selection.manifest
    arguments = manifest.get("arguments", {})
    if arguments.get("dtype") != "bf16" or arguments.get("quantization") != "none":
        raise ValueError("This serving profile requires the experiment's nonquantized BF16 model")
    if arguments.get("max_length") != 512:
        raise ValueError("This serving profile requires a 512-token training protocol")
    checkpoint = validate_checkpoint(selection.checkpoint)
    best = checkpoint["trainer"].get("best_metric")
    if type(best) not in (float, int) or not math.isclose(best, selection.validation_loss, rel_tol=1e-6):
        raise ValueError("Selected checkpoint does not match the recorded best validation loss")
    if checkpoint["trainer"].get("best_global_step") != checkpoint["global_step"]:
        raise ValueError("Selected checkpoint was not the best validation step")
    fingerprint = manifest.get("model_fingerprint", {}).get("local_files_sha256", {})
    if not fingerprint or not any(name.endswith(".safetensors") for name in fingerprint):
        raise ValueError("Training manifest lacks base model weight fingerprints")
    for name, expected in fingerprint.items():
        item = (model_dir / name).resolve()
        if item.parent != model_dir or not item.is_file() or file_digest(item) != expected:
            raise ValueError(f"Base model/tokenizer fingerprint mismatch: {name}")
    snapshot = read_json(model_dir / "snapshot_manifest.json")
    if snapshot.get("repo") != "Qwen/Qwen3.5-2B" or "snapshot_manifest.json" not in fingerprint:
        raise ValueError("Model snapshot identity is not the trained Qwen3.5-2B snapshot")
    return {
        "model": "Qwen3.5-2B + Evidoc LoRA",
        "model_id": "Qwen/Qwen3.5-2B",
        "model_revision": snapshot.get("revision"),
        "adapter_run": selection.run.name,
        "checkpoint": selection.checkpoint.name,
        "checkpoint_step": checkpoint["global_step"],
        "adapter_sha256": checkpoint["files"]["adapter_model.safetensors"]["sha256"],
        "adapter_config_sha256": checkpoint["files"]["adapter_config.json"]["sha256"],
        "model_files_sha256": fingerprint,
        "validation_loss": selection.validation_loss,
        "selection_rule": "lowest validation loss among completed runs; held-out test scores unused",
        "device": "cuda:0", "dtype": "bf16", "quantization": "none",
    }


def prompt_ids(tokenizer, question: str, context: str) -> list[int]:
    return tokenizer.apply_chat_template(messages_for({"question": question, "context": context}),
        tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)


def source_windows(tokenizer, question: str, text: str, limits: Limits) -> tuple[list[dict], bool]:
    """Exact contiguous windows, preferring sentence ends, with bounded overlap.

    Windows are cut without consulting model outputs or any reference answer.
    An oversized single sentence is split at a character boundary. Token length
    is checked for every final prompt, never silently truncated by the tokenizer.
    """
    if len(prompt_ids(tokenizer, question, "")) >= limits.max_input_tokens:
        raise ValueError("Question leaves no source-token budget; shorten the question")
    windows, start = [], 0
    while start < len(text) and len(windows) < limits.max_windows_per_source:
        remainder = text[start:]
        if len(prompt_ids(tokenizer, question, remainder)) <= limits.max_input_tokens:
            end = len(text)
        else:
            low, high = 0, len(remainder)
            while low < high:
                mid = (low + high + 1) // 2
                if len(prompt_ids(tokenizer, question, remainder[:mid])) <= limits.max_input_tokens:
                    low = mid
                else:
                    high = mid - 1
            if low == 0:
                raise ValueError("Question leaves no source-token budget; shorten the question")
            # Use the latest sentence boundary in the final half of this window.
            boundaries = [match.end() for match in re.finditer(r"[。！？!?；;\n]", remainder[:low])]
            cut = next((point for point in reversed(boundaries) if point >= max(1, low // 2)), low)
            end = start + cut
        context = text[start:end]
        ids = prompt_ids(tokenizer, question, context)
        # Some tokenizers have non-monotonic merges at a boundary; bound the
        # final actual prompt even when the binary search's approximation varied.
        while len(ids) > limits.max_input_tokens and end > start:
            end -= 1
            context = text[start:end]
            ids = prompt_ids(tokenizer, question, context)
        if end <= start:
            raise ValueError("No source text fits the declared token budget")
        windows.append({"text": context, "start": start, "end": end, "prompt": ids})
        if end == len(text):
            return windows, False
        overlap = min(limits.overlap_characters, max(0, (end - start) // 4))
        start = end - overlap
    return windows, bool(windows and windows[-1]["end"] < len(text))


def answer_sources(engine, question: str, sources: list[dict]) -> dict:
    """Return the first source-contained model answer in retrieval order."""
    limits = engine.limits
    attempts, warnings = [], []
    identity = engine.identity
    if len(sources) > limits.max_sources:
        warnings.append(f"Only the first {limits.max_sources} retrieved sources were examined")
    base = {"backend": "trained_adapter", "model": identity["model"],
            "model_info": identity, "grounding_scope": GROUNDING_SCOPE}
    for source in sources[:limits.max_sources]:
        windows, incomplete = source_windows(engine.tokenizer, question, source["text"], limits)
        if incomplete:
            warnings.append(f"Source [{source['citation']}] exceeds the window budget; its tail was not examined")
        for window in windows:
            if len(attempts) >= limits.max_windows:
                warnings.append("Request window budget reached; remaining passages were not examined")
                break
            prediction = engine.predict(window["prompt"])
            answer = prediction["text"].strip()
            status = "unsupported_output"
            if prediction.get("finish_reason") == "length":
                status = "generation_length_limit"
            elif normalize_answer(answer) == normalize_answer(REFUSAL):
                status = "model_refused"
            elif answer and answer in window["text"]:
                offset = window["start"] + window["text"].index(answer)
                attempts.append({"source": source["citation"], "window_start": window["start"],
                                 "window_end": window["end"], "status": "exact_substring",
                                 "input_tokens": len(window["prompt"])})
                return {**base, "answer": f"{answer} [{source['citation']}]", "answer_mode": "finetuned",
                        "citation_valid": True, "citations": [{"source": source["citation"], "quote": answer,
                        "start": offset, "end": offset + len(answer)}], "attempts": attempts,
                        "warnings": warnings}
            attempts.append({"source": source["citation"], "window_start": window["start"],
                             "window_end": window["end"], "status": status,
                             "input_tokens": len(window["prompt"])})
        if len(attempts) >= limits.max_windows:
            break
    return {**base, "answer": REFUSAL, "answer_mode": "refused", "citation_valid": False,
            "citations": [], "attempts": attempts, "warnings": warnings}


class CudaAdapter:
    """Lazily loaded frozen LoRA model. Only constructed by the serving entry point."""
    def __init__(self, model_dir: Path, experiment: Path | None = None, *, bundle: Path | None = None):
        self.limits = Limits()
        if bundle is not None:
            from .bundle import verify_bundle

            self.identity = verify_bundle(bundle, model_dir)
            adapter_path = bundle.resolve()
        else:
            if experiment is None:
                raise ValueError("Provide an experiment or a deployment bundle")
            selection = select_adapter(experiment)
            self.identity = verify_artifacts(selection, model_dir)
            adapter_path = selection.checkpoint
        # Delay all GPU-dependent imports until after artifact verification.
        import torch
        from peft import PeftModel
        from training.modeling import load_model, load_tokenizer

        self.tokenizer = load_tokenizer(str(model_dir.resolve()), None)
        model, info = load_model(str(model_dir.resolve()), None, "none", "bf16")
        self.model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False,
                                              use_safetensors=True, local_files_only=True)
        self.model.eval()
        self.model.config.use_cache = True
        self.model.requires_grad_(False)
        if any(parameter.device.type != "cuda" or parameter.device.index != 0 for parameter in self.model.parameters()):
            raise RuntimeError("Loaded adapter contains non-CUDA parameters")
        self.identity.update(gpu=info["gpu"], torch=info["torch"], cuda_runtime=info["cuda_runtime"])
        self.ready = True

    def predict(self, prompt: list[int]) -> dict:
        import torch

        if len(prompt) > self.limits.max_input_tokens:
            raise ValueError("Input exceeds the declared token budget")
        with torch.inference_mode():
            inputs = torch.tensor([prompt], dtype=torch.long, device="cuda:0")
            output = self.model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                do_sample=False, max_new_tokens=self.limits.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id, use_cache=True)
            generated = output[0, len(prompt):].tolist()
        eos = self.model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        exhausted = len(generated) >= self.limits.max_new_tokens and generated[-1] not in eos_ids
        return {"text": self.tokenizer.decode(generated, skip_special_tokens=True),
                "finish_reason": "length" if exhausted else "stop"}
