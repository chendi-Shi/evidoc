"""Data safety and metric tests run without torch, CUDA or network access."""
import json
from pathlib import Path
import tempfile
import unittest

from training.data import (
    REFUSAL, add_missing_context, answer_scores, context_key, encode_example,
    encode_partition, normalize_row, pad_features, pilot_estimate, read_jsonl,
    split_documents, write_jsonl,
)
from training.prepare_data import original_rows
from training.train import model_fingerprint, validate_resume, verify_data
from training.modeling import _json_safe


class TinyTokenizer:
    """Deterministic stand-in to test masking, not an actual model tokenizer."""
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking, return_dict=False):
        assert enable_thinking is False
        assert return_dict is False
        prompt = "system:" + messages[0]["content"] + "user:" + messages[1]["content"] + "assistant:"
        if len(messages) == 3:
            prompt += messages[2]["content"] + "\x02"
        return [ord(char) for char in prompt]


def row(index, split="train", title="", context=None, question=None):
    answer = f"答案{index}"
    return normalize_row({"id": f"{split}_{index}_QUERY_0", "title": title,
                          "context": context or f"独立文档{index}，规定为{answer}。",
                          "question": question or f"文档{index}的规定？",
                          "answers": {"text": [answer], "answer_start": [0]}}, split)


class TrainingDataTests(unittest.TestCase):
    def test_source_and_context_groups_never_cross_splits(self):
        rows = [row(index) for index in range(150)]
        # Multiple contexts of one titled article must remain together.
        rows += [row(1000, title="同一篇文章"), row(1001, title="同一篇文章")]
        partitions, report = split_documents(rows, seed=42)
        train_groups = {item["group_id"] for item in partitions["train"]}
        val_groups = {item["group_id"] for item in partitions["validation"]}
        self.assertTrue(train_groups and val_groups)
        self.assertFalse(train_groups & val_groups)
        located = [name for name, items in partitions.items() if any(item["source_document"] == "title:同一篇文章" for item in items)]
        self.assertEqual(len(located), 1)
        reverse, _ = split_documents(list(reversed(rows)), seed=42)
        self.assertEqual({key: {r["id"] for r in items} for key, items in partitions.items()},
                         {key: {r["id"] for r in items} for key, items in reverse.items()})

    def test_official_dev_is_held_out_and_train_overlap_removed(self):
        training = row(9)
        dev = row(9, split="validation", question="不同的问题")
        partitions, report = split_documents([training, dev])
        self.assertEqual(partitions["train"], [])
        self.assertEqual(partitions["validation"], [])
        self.assertEqual([item["id"] for item in partitions["test"]], [dev["id"]])
        self.assertEqual(report["cross_split_train_rows_removed"], 1)

    def test_normalized_context_and_invalid_supervision(self):
        self.assertEqual(context_key("Ａ B\n中文"), context_key("AB中文"))
        self.assertIsNone(normalize_row({"context": "资料", "question": "问题", "answers": {"text": ["编造"]}}, "train"))
        parsed = normalize_row({"context": "答案在这里", "question": "在哪里", "answers": [{"text": "这里"}]}, "train")
        self.assertEqual(parsed["answer"], "这里")

    def test_original_article_titles_are_preserved(self):
        payload = {"data": [{"title": "源文章", "paragraphs": [{"context": "文章中有答案", "qas": [
            {"id": "TRAIN_1_QUERY_0", "question": "有什么", "answers": [{"text": "答案"}]}]}]}]}
        raw = list(original_rows(payload, "train"))
        self.assertEqual(normalize_row(raw[0], "train")["source_document"], "title:源文章")

    def test_assistant_labels_and_padding_only(self):
        tokenizer = TinyTokenizer()
        encoded = encode_example(tokenizer, row(1), 512)
        unmasked = [label for label in encoded["labels"] if label != -100]
        self.assertEqual("".join(chr(value) for value in unmasked), "答案1\x02")
        self.assertTrue(all(label == -100 for label in encoded["labels"][:-len(unmasked)]))
        shorter = {key: value[:-1] for key, value in encoded.items()}
        batch = pad_features([encoded, shorter], pad_token_id=0)
        self.assertEqual(batch["labels"][1][-1], -100)
        self.assertEqual(batch["attention_mask"][1][-1], 0)

    def test_overlength_is_reported_without_truncating_answer(self):
        short = row(1)
        long = {**row(2), "context": "资料" * 1000 + "答案2"}
        encoded, report = encode_partition(TinyTokenizer(), [short, long], 512)
        self.assertEqual(len(encoded), 1)
        self.assertEqual(report["overlength_removed"], 1)
        self.assertEqual(report["removed_ids"], [long["id"]])
        self.assertIn("答案2", long["context"])

    def test_incompatible_chat_template_fails_closed(self):
        class BrokenTokenizer(TinyTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                result = super().apply_chat_template(messages, **kwargs)
                return result if len(messages) == 2 else [999] + result

        with self.assertRaisesRegex(ValueError, "boundary"):
            encode_example(BrokenTokenizer(), row(1), 512)

    def test_missing_context_augmentation_stays_in_existing_groups(self):
        rows = [row(index) for index in range(100)]
        split, _ = split_documents(rows)
        augmented = add_missing_context(split["train"], .25, 42)
        negatives = [item for item in augmented if item["is_missing_context"]]
        self.assertTrue(negatives)
        self.assertTrue(all(item["context"] == "" and item["answer"] == REFUSAL for item in negatives))
        self.assertTrue({item["group_id"] for item in negatives} <= {item["group_id"] for item in split["train"]})
        self.assertEqual(augmented, add_missing_context(split["train"], .25, 42))

    def test_jsonl_hash_detects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            checksum = write_jsonl(directory / "train.jsonl", [row(1)])
            (directory / "manifest.json").write_text(json.dumps({"splits": {
                "train": {"file": "train.jsonl", "sha256": checksum}}}), encoding="utf-8")
            self.assertEqual(read_jsonl(directory / "train.jsonl")[0]["answer"], "答案1")
            verify_data(directory)
            (directory / "train.jsonl").write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "integrity"):
                verify_data(directory)

    def test_metrics_and_pilot_estimate_are_not_invented_results(self):
        self.assertEqual(answer_scores("ＡＢ，中文。", ["AB中文"]), {"normalized_em": 1.0, "character_f1": 1.0})
        self.assertEqual(answer_scores("甲", ["甲乙"])["character_f1"], 2 / 3)
        self.assertEqual(answer_scores("丙", ["甲乙"])["normalized_em"], 0)
        result = pilot_estimate([20, 16, 10, 10, 10], 1000, 16, 3, 72)
        self.assertEqual(result["median_step_seconds"], 10)
        self.assertEqual(result["recommended_steps"], 189)
        self.assertLess(result["estimated_epoch_plan_hours"], 72)

    def test_resume_rejects_changed_hyperparameters_and_weight_bytes(self):
        previous = {"arguments": {"rank": 8, "epochs": 3}, "model": {"dtype": "bf16"},
                    "model_fingerprint": {"sha": "original"}, "dataset_manifest_sha256": "data", "lora": {"r": 8}}
        current = json.loads(json.dumps(previous))
        validate_resume(previous, current)
        current["arguments"]["epochs"] = 4
        with self.assertRaisesRegex(ValueError, "epochs"):
            validate_resume(previous, current)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "model.safetensors").write_bytes(b"original")
            before = model_fingerprint(directory, None)
            (path / "model.safetensors").write_bytes(b"modified")
            self.assertNotEqual(before, model_fingerprint(directory, None))

    def test_model_loading_info_sets_are_json_safe(self):
        value = _json_safe({"missing_keys": set(), "unexpected_keys": {"z", "a"}, "shape": (2, 3)})
        self.assertEqual(json.loads(json.dumps(value)), {"missing_keys": [], "unexpected_keys": ["a", "z"], "shape": [2, 3]})


if __name__ == "__main__":
    unittest.main()
