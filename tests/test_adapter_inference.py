"""Serving contract and provenance checks require no GPU or downloaded models."""
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from inference.adapter import Limits, answer_sources, select_adapter, source_windows, verify_artifacts
from inference.service import create_app
from training.data import REFUSAL, file_digest


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        assert kwargs["add_generation_prompt"] is True
        return list(range(20 + len(messages[-1]["content"])))


class FakeEngine:
    ready = True
    tokenizer = FakeTokenizer()
    limits = Limits(max_input_tokens=100)
    identity = {"model": "fixture adapter", "adapter_run": "fixture", "adapter_sha256": "abc"}

    def __init__(self, *predictions):
        self.predictions = iter(predictions)
        self.calls = 0

    def predict(self, prompt):
        self.calls += 1
        result = next(self.predictions)
        if isinstance(result, Exception):
            raise result
        return result if isinstance(result, dict) else {"text": result, "finish_reason": "stop"}


def test_exact_quote_and_identity_with_short_answer():
    engine = FakeEngine("600元")
    sources = [{"citation": 2, "text": "上海住宿费用上限为600元。"}]
    result = answer_sources(engine, "上海住宿费用上限是多少？", sources)
    assert result["answer"] == "600元 [2]"
    assert result["answer_mode"] == "finetuned"
    assert result["model_info"]["adapter_sha256"] == "abc"
    citation = result["citations"][0]
    assert sources[0]["text"][citation["start"]:citation["end"]] == citation["quote"] == "600元"
    assert "not semantic" in result["grounding_scope"]


def test_unsupported_output_refuses_instead_of_fabricating_quote():
    result = answer_sources(FakeEngine("900元"), "费用？", [{"citation": 1, "text": "费用600元。"}])
    assert result["answer_mode"] == "refused"
    assert not result["citation_valid"]
    assert result["attempts"][0]["status"] == "unsupported_output"


def test_refusal_and_length_limit_skip_to_next_source():
    engine = FakeEngine(REFUSAL, {"text": "900元", "finish_reason": "length"}, "600元")
    sources = [{"citation": n, "text": "该文档费用600元，外地费用900元。"} for n in (1, 2, 3)]
    result = answer_sources(engine, "费用？", sources)
    assert result["answer"] == "600元 [3]"
    assert [attempt["status"] for attempt in result["attempts"]] == ["model_refused", "generation_length_limit", "exact_substring"]


def test_first_supported_answer_stops_at_retrieval_order():
    engine = FakeEngine("600元")
    result = answer_sources(engine, "费用？", [
        {"citation": 1, "text": "费用600元。"}, {"citation": 2, "text": "费用900元。"}])
    assert result["answer"] == "600元 [1]" and engine.calls == 1


def test_windows_preserve_source_offsets_overlap_and_token_bounds():
    text = "第一句有一些内容。" * 30
    windows, incomplete = source_windows(FakeTokenizer(), "问题？", text, Limits(max_input_tokens=100))
    assert len(windows) == 2 and incomplete
    for window in windows:
        assert window["text"] == text[window["start"]:window["end"]]
        assert len(window["prompt"]) <= 100
        assert window["text"].endswith("。")
    assert 0 < windows[1]["start"] < windows[0]["end"]


def test_long_sentence_split_and_request_work_bounded():
    engine = FakeEngine(*([REFUSAL] * 6))
    sources = [{"citation": i, "text": "无标点长句" * 100} for i in range(1, 8)]
    result = answer_sources(engine, "问题？", sources)
    assert engine.calls == 6 and result["answer_mode"] == "refused"
    assert any("first 3" in warning for warning in result["warnings"])
    assert any("tail" in warning for warning in result["warnings"])


def test_empty_sources_do_not_generate():
    engine = FakeEngine()
    result = answer_sources(engine, "问题？", [])
    assert result["answer_mode"] == "refused" and engine.calls == 0


def test_service_health_validation_busy_and_error_releases_lock():
    engine = FakeEngine(RuntimeError("CUDA out of memory"), "600元")
    app = create_app(engine)
    body = {"question": "费用？", "sources": [{"citation": 1, "text": "费用600元。"}]}
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["status"] == "ok" and health["backend"] == "trained_adapter"
        assert health["concurrency"] == 1
        assert client.post("/answer", json={**body, "question": "  "}).status_code == 422
        assert client.post("/answer", json={**body, "sources": body["sources"] * 2}).status_code == 422
        assert client.post("/answer", json=body, headers={"Origin": "https://example.com"}).status_code == 403
        assert client.get("/health", headers={"Host": "untrusted.example"}).status_code == 403
        app.state.generation_lock.acquire()
        assert client.post("/answer", json=body).json()["detail"]["code"] == "model_busy"
        app.state.generation_lock.release()
        error = client.post("/answer", json=body)
        assert error.status_code == 503 and error.json()["detail"]["message"] == "CUDA out of memory"
        assert not app.state.generation_lock.locked()
        assert client.post("/answer", json=body).json()["answer_mode"] == "finetuned"


def test_service_question_too_long_for_model_does_not_generate():
    engine = FakeEngine()
    with TestClient(create_app(engine)) as client:
        result = client.post("/answer", json={"question": "长" * 100,
                                             "sources": [{"citation": 1, "text": "一些证据"}]})
        assert result.status_code == 422 and result.json()["detail"]["code"] == "input_token_budget"
        assert engine.calls == 0


def write_run(root: Path, name: str, loss: float, complete=True, dataset="same"):
    run = root / name
    (run / "checkpoint-50").mkdir(parents=True)
    manifest = {"training_complete": complete, "pilot": False, "dataset_manifest_sha256": dataset,
                "arguments": {"dtype": "bf16", "quantization": "none", "max_length": 512, "eval_samples": 128},
                "best_checkpoint": f"old\\absolute\\{name}\\checkpoint-50"}
    (run / "run_manifest.json").write_text(json.dumps(manifest))
    (run / "monitor_state.json").write_text(json.dumps({"best_loss": loss}))
    return run


def test_selection_uses_validation_not_test_and_skips_incomplete(tmp_path):
    write_run(tmp_path, "low-test-score", .10)
    write_run(tmp_path, "high-test-score", .20)
    write_run(tmp_path, "unfinished", .01, complete=False)
    (tmp_path / "final_comparison.json").write_text("invalid JSON: never consulted")
    chosen = select_adapter(tmp_path)
    assert chosen.run.name == "low-test-score" and chosen.validation_loss == .10
    assert chosen.checkpoint.parent == chosen.run


def test_selection_rejects_incomparable_validation_protocol(tmp_path):
    write_run(tmp_path, "first", .1)
    write_run(tmp_path, "second", .2, dataset="different")
    with pytest.raises(ValueError, match="different validation"):
        select_adapter(tmp_path)


def test_artifact_fingerprint_mismatch_rejected_without_loading_cuda(tmp_path, monkeypatch):
    run = write_run(tmp_path / "experiment", "run", .1)
    model = tmp_path / "model"
    model.mkdir()
    (model / "weights.safetensors").write_bytes(b"fixture")
    (model / "snapshot_manifest.json").write_text(json.dumps({"repo": "Qwen/Qwen3.5-2B", "revision": "fixture"}))
    manifest = json.loads((run / "run_manifest.json").read_text())
    manifest["model_fingerprint"] = {"local_files_sha256": {
        path.name: file_digest(path) for path in model.iterdir()}}
    (run / "run_manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr("training.checkpoints.validate_checkpoint", lambda _: {
        "trainer": {"best_metric": .1, "best_global_step": 50}, "global_step": 50,
        "files": {name: {"sha256": "fixture"} for name in ("adapter_model.safetensors", "adapter_config.json")}})
    chosen = select_adapter(tmp_path / "experiment")
    assert verify_artifacts(chosen, model)["adapter_run"] == "run"
    (model / "weights.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        verify_artifacts(chosen, model)
