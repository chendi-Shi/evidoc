"""Exercise document upload/retrieval/actual CUDA adapter/citation through the app.

Requires an already running scripts.serve_adapter service. Uses a temporary
document library; never modifies the user's documents or starts training.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx
from fastapi.testclient import TestClient

from app.api import create_app
from app.config import Settings
from app.llm import validate_adapter_citations
from training.data import normalize_answer

ROOT = Path(__file__).resolve().parents[1]


def verify(adapter_url: str) -> dict:
    health = httpx.get(adapter_url + "/health", timeout=5, trust_env=False)
    health.raise_for_status()
    identity = health.json()
    if identity.get("backend") != "trained_adapter" or not str(identity.get("device", "")).startswith("cuda"):
        raise ValueError("A real trained CUDA adapter service is required")
    cases = []
    with TemporaryDirectory(prefix="evidoc-acceptance-") as temporary:
        settings = Settings(db_path=Path(temporary) / "docs.sqlite3", adapter_url=adapter_url,
                            generation_backend="adapter")
        app = create_app(settings)
        with TestClient(app) as client:
            empty = client.post("/api/ask", json={"question": "上海住宿上限是多少？", "mode": "bm25"})
            empty.raise_for_status()
            cases.append({"case": "empty_library", "passed": empty.json()["answer_mode"] == "refused"})
            # An actual multipart upload, not a mocked retriever or a model-only call.
            for name in ("reimbursement.md", "leave.md", "project_operations.md"):
                path = ROOT / "data/demo" / name
                response = client.post("/api/documents", files={"files": (name, path.read_bytes(), "text/markdown")})
                response.raise_for_status()
            for question, expected in [
                ("上海的住宿上限是多少？", "600"),
                ("年假申请需要提前多少个工作日？", "3个工作日"),
                ("P1事故需要在多长时间内响应？", "10分钟"),
            ]:
                response = client.post("/api/ask", json={"question": question, "mode": "bm25", "top_k": 3})
                response.raise_for_status()
                answer = response.json()
                matching_model = answer.get("model_info", {}).get("adapter_sha256") == identity["adapter_sha256"]
                passed = (answer.get("answer_mode") == "finetuned" and matching_model
                          and normalize_answer(expected) in normalize_answer(answer.get("answer", ""))
                          and validate_adapter_citations(answer, answer["sources"]))
                cases.append({"case": "synthetic_document_qa", "question": question, "expected_contains": expected,
                              "answer": answer.get("answer"), "answer_mode": answer.get("answer_mode"),
                              "citation_valid": answer.get("citation_valid"), "model_identity_matches": matching_model,
                              "timings": answer.get("timings"), "passed": passed})
            # An unavailable adapter must not be labelled as a generated model answer.
            app.state.models.generation_status = lambda: {"ready": False}
            fallback = client.post("/api/ask", json={"question": "上海住宿上限是多少？", "mode": "bm25"}).json()
            cases.append({"case": "adapter_offline_fallback", "passed": fallback["answer_mode"] == "extractive"})
            document_ids = [item["id"] for item in client.get("/api/documents").json()["documents"]]
            deleted = all(client.delete("/api/documents/" + doc_id).status_code == 200 for doc_id in document_ids)
            cases.append({"case": "delete_documents", "passed": deleted and not client.get("/api/documents").json()["documents"]})
    return {"schema_version": 1, "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "Integration smoke test on synthetic project documents; not a general accuracy benchmark.",
            "model_info": {key: identity.get(key) for key in ("model", "model_id", "adapter_run", "adapter_sha256", "validation_loss", "device", "dtype")},
            "passed": all(case["passed"] for case in cases), "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-url", default="http://127.0.0.1:8083")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation/results/delivery_smoke.json")
    args = parser.parse_args()
    result = verify(args.adapter_url)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
