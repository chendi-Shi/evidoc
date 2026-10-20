"""Small synthetic retrieval smoke evaluation, kept separate from user documents."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
from time import perf_counter

from .config import ROOT
from .ingest import extract_document
from .retrieval import Store


def evaluate(models, embedding_model: str) -> dict:
    questions = json.loads((ROOT / "evaluation/questions.json").read_text(encoding="utf-8"))
    notes = ["12 道原创虚构制度题，仅作检索冒烟评测，不代表公共基准或真实业务准确率。",
             "Recall@5 按命中的相关文档比例计算；MRR@5 按首个相关片段的倒数排名计算。",
             "不可回答项仅统计检索器返回空结果的比例，不是生成模型的拒答准确率。"]
    results = []
    with tempfile.TemporaryDirectory(prefix="evidoc-eval-") as temp:
        store = Store(Path(temp) / "eval.sqlite3")
        for path in sorted((ROOT / "data/demo").glob("*.md")):
            content = path.read_bytes()
            store.add_document(path.name, extract_document(path.name, content), hashlib.sha256(content).hexdigest())
        has_dense = models.ready(models.settings.embedding_url)
        query_vectors = {}
        if has_dense:
            try:
                for chunk in store.all_chunks():
                    store.store_embeddings({chunk["id"]: models.embed([chunk["text"]])[0]}, embedding_model)
                for question in questions:
                    query_vectors[question["id"]] = models.embed([question["question"]], query=True)[0]
            except Exception as exc:
                notes.append(f"语义评测未完成：{type(exc).__name__}；未将词法结果冒充语义结果。")
                has_dense = False
        else:
            notes.append("语义模型未启动，暂仅评测 BM25。")
        for mode in ("bm25", "dense", "hybrid"):
            if mode != "bm25" and not has_dense:
                continue
            recalls, reciprocal, refusals, latencies, details = [], [], [], [], []
            for question in questions:
                start = perf_counter()
                result = store.search(question["question"], k=5, mode=mode,
                    query_vector=query_vectors.get(question["id"]) if mode != "bm25" else None,
                    embedding_model=embedding_model)
                latencies.append((perf_counter() - start) * 1000)
                names = [hit["name"] for hit in result["hits"]]
                relevant = set(question["relevant_documents"])
                if question["unanswerable"]:
                    refusals.append(float(not names))
                else:
                    recalls.append(len(set(names) & relevant) / len(relevant))
                    reciprocal.append(next((1 / rank for rank, name in enumerate(names, 1) if name in relevant), 0))
                details.append({"id": question["id"], "retrieved_documents": names, "relevant_documents": sorted(relevant)})
            results.append({"mode": mode, "mode_used": mode, "recall_at_5": sum(recalls) / len(recalls),
                "mrr_at_5": sum(reciprocal) / len(reciprocal), "unanswerable_accuracy": sum(refusals) / len(refusals),
                "latency_ms": sum(latencies) / len(latencies), "queries": len(questions), "details": details})
    notes.append("检索延迟不含预先计算的查询向量耗时；端到端耗时请查看问答页面。")
    report = {"available": True, "created_at": datetime.now(timezone.utc).isoformat(), "dataset": "Evidoc synthetic retrieval smoke v1",
              "sample_count": len(questions), "results": results, "notes": notes}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/retrieval_evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
