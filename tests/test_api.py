from fastapi.testclient import TestClient
from app.api import create_app
from app.config import Settings
from app.llm import validate_citations


def test_local_lifecycle_and_evidence_fallback(tmp_path):
    app = create_app(Settings(db_path=tmp_path / "test.sqlite3"))
    app.state.models.ready = lambda _: False
    app.state.models.generation_status = lambda: {"ready": False}
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert client.post("/api/ask", json={"question": "  "}).status_code == 422
        assert client.post("/api/demo").status_code == 200
        assert len(client.get("/api/documents").json()["documents"]) == 4
        result = client.post("/api/ask", json={"question": "上海住宿费用上限是多少？"}).json()
        assert result["answer_mode"] == "extractive"
        assert result["sources"] and result["warnings"]
        assert "不是" in result["warnings"][-1]
        doc_id = result["sources"][0]["document_id"]
        assert client.delete("/api/documents/" + doc_id).status_code == 200
        assert client.delete("/api/documents/" + doc_id).status_code == 404
        assert client.post("/api/demo", headers={"Origin": "https://untrusted.example"}).status_code == 403


def test_quote_validation_is_not_just_citation_number():
    sources = [{"citation": 1, "text": "上海住宿上限为600元。"}]
    assert validate_citations({"answer": "上限600元。[1]", "citations": [{"source": 1, "quote": "住宿上限为600元"}]}, sources)
    assert not validate_citations({"answer": "上限900元。[1]", "citations": [{"source": 1, "quote": "住宿上限为900元"}]}, sources)
    assert not validate_citations({"answer": "上限600元。[2]", "citations": [{"source": 1, "quote": "住宿上限为600元"}]}, sources)
