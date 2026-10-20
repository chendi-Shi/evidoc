"""Greedy base/adapter comparison on the SAME held-out examples and prompts.

Reports normalized exact match, character F1, and a controlled empty-context
refusal probe. These are custom experiment metrics, not official CMRC scores.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from .data import REFUSAL, answer_scores, digest, encode_example, file_digest, messages_for, normalize_answer, read_jsonl, write_jsonl
from .modeling import load_model, load_tokenizer, resolve_revision
from .status import write_json_atomic
from .train import verify_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--revision")
    parser.add_argument("--adapter", type=Path, help="Omit for a baseline; when provided evaluates base AND adapter")
    parser.add_argument("--data", type=Path, default=Path("data/training/cmrc2018"))
    parser.add_argument("--output", type=Path, default=Path("reports/training_evaluation"))
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--samples", type=int, default=200, help="0 means all retained test examples; fixed hash order")
    parser.add_argument("--refusal-samples", type=int, default=50)
    parser.add_argument("--quantization", choices=("none", "nf4"), default="none")
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16"), default="auto")
    args = parser.parse_args()
    if args.samples < 0 or args.refusal_samples < 0 or args.max_new_tokens < 1:
        raise ValueError("Invalid evaluation sizes")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Evaluation output is nonempty; choose another directory to preserve evidence")
    manifest = verify_data(args.data)
    revision = resolve_revision(args.model, args.revision)
    tokenizer = load_tokenizer(args.model, revision)
    source = read_jsonl(args.data / "test.jsonl")
    eligible = [row for row in source if not row["is_missing_context"] and encode_example(tokenizer, row, args.max_length) is not None]
    examples = eligible[:args.samples] if args.samples else eligible
    if not examples:
        raise ValueError("No test examples remain under the declared length policy")
    model, model_info = load_model(args.model, revision, args.quantization, args.dtype)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
    model.eval()
    model.config.use_cache = True
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "model": model_info,
              "adapter": str(args.adapter.resolve()) if args.adapter else None,
              "adapter_files_sha256": {path.name: file_digest(path) for path in args.adapter.iterdir()
                                         if path.is_file() and (path.suffix == ".safetensors" or path.name == "adapter_config.json")} if args.adapter else {},
              "dataset_revision": manifest["revision"], "dataset_manifest_sha256": digest((args.data / "manifest.json").read_bytes()),
              "source_test_rows": len(source), "retained_test_rows": len(eligible), "evaluated_rows": len(examples),
              "overlength_filtered": len(source) - len(eligible), "example_ids": [row["id"] for row in examples],
              "generation": {"do_sample": False, "max_new_tokens": args.max_new_tokens, "enable_thinking": False},
              "metric_scope": "Custom normalized EM and character-multiset F1; not official CMRC scoring. Exact empty-context refusal is an artificial policy probe, not real unanswerable QA accuracy.",
              "length_policy": "Full example <= max_length; no answer-centered cropping", "max_length": args.max_length,
              "results": {}}
    import torch

    for variant in (["base", "adapter"] if args.adapter else ["base"]):
        context = model.disable_adapter() if args.adapter and variant == "base" else nullcontext()
        predictions = []
        start = time.monotonic()
        with context, torch.inference_mode():
            for index, row in enumerate(examples):
                for missing in ([False, True] if index < args.refusal_samples else [False]):
                    prompt = tokenizer.apply_chat_template(messages_for(row, missing), tokenize=True,
                                                           return_dict=False, add_generation_prompt=True, enable_thinking=False)
                    input_ids = torch.tensor([prompt], dtype=torch.long, device="cuda:0")
                    output = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                                            do_sample=False, max_new_tokens=args.max_new_tokens,
                                            pad_token_id=tokenizer.pad_token_id, use_cache=True)
                    prediction = tokenizer.decode(output[0, len(prompt):], skip_special_tokens=True).strip()
                    predictions.append({"id": row["id"], "missing_context": missing, "prediction": prediction,
                                        "answers": [REFUSAL] if missing else row["answers"],
                                        **answer_scores(prediction, [REFUSAL] if missing else row["answers"])})
                if (index + 1) % 10 == 0:
                    print(f"{variant}: {index + 1}/{len(examples)}", flush=True)
        answers = [row for row in predictions if not row["missing_context"]]
        refusals = [row for row in predictions if row["missing_context"]]
        metrics = {"normalized_em": sum(row["normalized_em"] for row in answers) / len(answers),
                   "character_f1": sum(row["character_f1"] for row in answers) / len(answers),
                   "missing_context_exact_refusal": sum(row["normalized_em"] for row in refusals) / len(refusals) if refusals else None,
                   "answered_count": len(answers), "refusal_probe_count": len(refusals),
                   "elapsed_seconds": time.monotonic() - start,
                   "answerable_false_refusal_rate": sum(normalize_answer(row["prediction"]) == normalize_answer(REFUSAL) for row in answers) / len(answers),
                   "predictions_sha256": write_jsonl(args.output / f"{variant}_predictions.jsonl", predictions)}
        report["results"][variant] = metrics
        write_json_atomic(args.output / "comparison.json", report)
        print(json.dumps({variant: metrics}, ensure_ascii=False, indent=2))
    if args.adapter:
        report["delta_adapter_minus_base"] = {metric: report["results"]["adapter"][metric] - report["results"]["base"][metric]
                                                for metric in ("normalized_em", "character_f1", "answerable_false_refusal_rate")}
        write_json_atomic(args.output / "comparison.json", report)


if __name__ == "__main__":
    main()
