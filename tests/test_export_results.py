"""Release-boundary and evidence checks need no GPU or downloaded dataset."""
import json

import pytest

from scripts.export_results import file_hashes, sha256, validate_public_report, verify_predictions
from training.data import REFUSAL


@pytest.mark.parametrize("value", [
    {"question": "private"}, {"runs": [{"example_ids": ["test-id"]}]},
    {"location": "C:\\Users\\someone\\private.json"}, {"location": "/home/person/private"},
    {"location": "\\\\host\\share"},
])
def test_private_content_and_absolute_paths_are_rejected(value):
    with pytest.raises(ValueError):
        validate_public_report(value)


def test_fingerprint_keys_cannot_smuggle_paths():
    with pytest.raises(ValueError):
        file_hashes({"../private.json": "a" * 64})
    assert file_hashes({"adapter_config.json": "a" * 64}) == {"adapter_config.json": "a" * 64}


def test_predictions_are_recomputed_not_just_trusted(tmp_path):
    path = tmp_path / "predictions.jsonl"
    rows = [
        {"id": "fictional-1", "missing_context": False, "prediction": "答案", "answers": ["答案"]},
        {"id": "fictional-1", "missing_context": True, "prediction": REFUSAL, "answers": [REFUSAL]},
    ]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    expected = {"predictions_sha256": sha256(path), "answered_count": 1, "refusal_probe_count": 1,
                "normalized_em": 1.0, "character_f1": 1.0,
                "answerable_false_refusal_rate": 0.0, "missing_context_exact_refusal": 1.0}
    verify_predictions(path, expected, ["fictional-1"])
    with pytest.raises(ValueError, match="metric mismatch"):
        verify_predictions(path, {**expected, "character_f1": 0.9}, ["fictional-1"])
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_predictions(path, expected, ["fictional-1"])
