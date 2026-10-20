"""Boundary tests for the UI -> local trained adapter service integration."""
import httpx
import pytest
from fastapi.testclient import TestClient
from app.api import create_app
from app.config import Settings
from app.llm import LocalModels, validate_adapter_citations


def test_short_adapter_span_requires_exact_answer_and_source():
    sources = [{"citation": 2, "text": "校准周期为17天，负责人暂未指定。"}]
    result = {"answer": "17天 [2]", "citations": [{"source": 2, "quote": "17天"}]}
    assert validate_adapter_citations(result, sources)
    assert not validate_adapter_citations({**result, "answer": "19天 [2]"}, sources)
    assert not validate_adapter_citations({**result, "citations": [{"source": True, "quote": "17天"}]}, sources)
    assert not validate_adapter_citations({**result, "citations": [{"source": 2, "quote": "19天"}]}, sources)


def test_unavailable_service_is_not_advertised_as_loaded(monkeypatch):
    def bad_health(*args, **kwargs):
        return httpx.Response(200, json={"status": "ok"}, request=httpx.Request("GET", args[0]))
    monkeypatch.setattr(httpx, "get", bad_health)
    assert not LocalModels(Settings()).generation_status()["ready"]


def test_adapter_client_rejects_unverified_span(monkeypatch):
    def post(self, url, **kwargs):
        return httpx.Response(200, json={"answer_mode": "finetuned", "answer": "900元 [1]",
            "citations": [{"source": 1, "quote": "900元"}]}, request=httpx.Request("POST", url))
    monkeypatch.setattr(httpx.Client, "post", post)
    with pytest.raises(ValueError, match="逐字"):
        LocalModels(Settings()).generate("上限？", [{"citation": 1, "text": "上限600元"}])


def test_api_keeps_adapter_identity_and_notices(tmp_path):
    app = create_app(Settings(db_path=tmp_path / "store.sqlite3"))
    app.state.models.ready = lambda _: False
    app.state.models.generation_status = lambda: {"ready": True, "model": "Qwen3.5-2B + LoRA", "backend": "trained_adapter"}
    def answer(question, sources):
        return {"answer_mode": "finetuned", "answer": "600 元/人/晚 [1]", "citation_valid": True,
                "model": "Qwen3.5-2B + LoRA", "model_info": {"adapter_run": "test-run"}, "warnings": ["出处检查不证明语义正确性。"]}
    app.state.models.generate = answer
    with TestClient(app) as client:
        client.post("/api/demo")
        result = client.post("/api/ask", json={"question": "上海住宿上限是多少？"}).json()
        assert result["answer_mode"] == "finetuned"
        assert result["model_info"]["adapter_run"] == "test-run"
        assert result["model"] == "Qwen3.5-2B + LoRA"
        assert any("出处检查" in warning for warning in result["warnings"])
        assert result["sources"] and result["timings"]["total_ms"] >= 0


def test_remote_model_urls_are_rejected():
    with pytest.raises(ValueError, match="本机"):
        Settings(adapter_url="https://example.com")


@pytest.mark.parametrize("error", [ValueError("引用无效"), httpx.ConnectError("unavailable")])
def test_api_adapter_error_preserves_evidence(tmp_path, error):
    app = create_app(Settings(db_path=tmp_path / "store.sqlite3"))
    app.state.models.ready = lambda _: False
    app.state.models.generation_status = lambda: {"ready": True}
    def fail(*args):
        raise error
    app.state.models.generate = fail
    with TestClient(app) as client:
        client.post("/api/demo")
        result = client.post("/api/ask", json={"question": "上海住宿上限是多少？", "mode": "bm25"}).json()
        assert result["answer_mode"] == "extractive"
        assert result["model"] == "evidence-only"
        assert result["warnings"] and result["sources"]


def test_question_limit_matches_adapter(tmp_path):
    with TestClient(create_app(Settings(db_path=tmp_path / "docs.sqlite3"))) as client:
        assert client.post("/api/ask", json={"question": "文" * 501}).status_code == 422
