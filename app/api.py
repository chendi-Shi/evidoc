from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import threading
from time import perf_counter
from typing import Literal
from urllib.parse import urlparse

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .config import ROOT, Settings
from .evaluation import evaluate
from .ingest import extract_document
from .llm import LocalModels, extractive
from .retrieval import Store


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    mode: Literal["bm25", "dense", "hybrid"] = "hybrid"
    top_k: int = Field(default=5, ge=1, le=8)
    document_ids: list[str] = Field(default_factory=list, max_length=100)

    @field_validator("question")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("请输入问题。")
        return value.strip()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    store, models = Store(settings.db_path), LocalModels(settings)
    app = FastAPI(title="Evidoc — 本地训练与证据问答", version="0.1.0", docs_url="/api/docs")
    app.state.store, app.state.models = store, models
    generation_lock = threading.Lock()
    indexing_lock = threading.Lock()
    evaluation_lock = threading.Lock()
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"])

    @app.middleware("http")
    async def local_origin(request: Request, call_next):
        origin = request.headers.get("origin")
        if request.method not in {"GET", "HEAD", "OPTIONS"} and origin:
            if urlparse(origin).netloc != request.headers.get("host"):
                return JSONResponse({"detail": "仅接受本机同源操作。"}, status_code=403)
        return await call_next(request)

    def index_missing() -> dict:
        if not indexing_lock.acquire(blocking=False):
            return {"indexed": 0, "warnings": ["另一项索引任务正在进行。"]}
        try:
            missing = store.chunks_missing_embeddings(settings.embedding_model)
            if not missing:
                return {"indexed": 0, "warnings": []}
            if not models.ready(settings.embedding_url):
                return {"indexed": 0, "warnings": ["语义模型未启动，文档已建立 BM25 索引。启动模型后可重建语义索引。"]}
            count = 0
            for chunk in missing:
                try:
                    vector = models.embed([chunk["text"]])[0]
                    store.store_embeddings({chunk["id"]: vector}, settings.embedding_model)
                    count += 1
                except Exception as exc:
                    return {"indexed": count, "warnings": [f"语义索引中断（{type(exc).__name__}），已保存完成部分。"]}
            return {"indexed": count, "warnings": []}
        finally:
            indexing_lock.release()

    @app.get("/api/status")
    def status():
        generation = models.generation_status()
        model_ready, embedding_ready = generation["ready"], models.ready(settings.embedding_url)
        backend = "not_running"
        status_file = ROOT / "runtime/status.json"
        if status_file.exists():
            try:
                backend = json.loads(status_file.read_text(encoding="utf-8")).get("backend", "unknown")
            except (ValueError, OSError):
                pass
        hardware = {"gpu": "未检测到 NVIDIA 显卡", "vram_gb": None, "driver": "unknown"}
        try:
            flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
            result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3, creationflags=flags)
            gpu, memory, driver = result.stdout.strip().splitlines()[0].split(",")
            hardware = {"gpu": gpu.strip(), "vram_gb": round(float(memory) / 1024, 1), "driver": driver.strip()}
        except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
            pass
        return {"app_name": "Evidoc · 见微", "model_name": generation.get("model", settings.model_name), "model_ready": model_ready,
                "embedding_ready": embedding_ready, "backend": generation.get("backend", backend) if model_ready else "not_running",
                "model_info": generation,
                "hardware": hardware, "mode": ("finetuned" if settings.generation_backend == "adapter" else "generated") if model_ready else "extractive", **store.stats()}

    @app.get("/api/documents")
    def documents():
        return {"documents": store.list_documents()}

    @app.post("/api/documents")
    def upload(files: list[UploadFile] = File(...)):
        if len(files) > 10:
            raise HTTPException(400, "每批最多上传 10 个文件。")
        added, warnings = [], []
        for file in files:
            name = (file.filename or "document").replace("\\", "/").rsplit("/", 1)[-1]
            try:
                if store.stats()["documents"] >= settings.max_documents:
                    raise ValueError("演示版最多保存 100 份文档，请先清理不需要的文档。")
                content = file.file.read(settings.max_file_bytes + 1)
                if len(content) > settings.max_file_bytes:
                    raise ValueError("单个文件不得超过 20 MB。")
                added.append(store.add_document(name, extract_document(name, content), hashlib.sha256(content).hexdigest()))
            except ValueError as exc:
                warnings.append(f"{name}: {exc}")
            finally:
                file.file.close()
        warnings.extend(index_missing()["warnings"] if added else [])
        if not added:
            raise HTTPException(400, "；".join(warnings) or "没有有效文件。")
        return {"documents": added, "warnings": warnings}

    @app.post("/api/demo")
    def demo():
        added = []
        for path in sorted((ROOT / "data/demo").glob("*.md")):
            content = path.read_bytes()
            added.append(store.add_document(path.name, extract_document(path.name, content), hashlib.sha256(content).hexdigest()))
        return {"documents": added, "warnings": index_missing()["warnings"]}

    @app.delete("/api/documents/{document_id}")
    def delete(document_id: str):
        if not store.delete_document(document_id):
            raise HTTPException(404, "文档不存在。")
        return {"deleted": True}

    @app.post("/api/reindex")
    def reindex():
        return index_missing()

    def retrieve(payload: Question):
        started = perf_counter()
        vector, warnings = None, []
        if payload.mode != "bm25" and models.ready(settings.embedding_url):
            try:
                vector = models.embed([payload.question], query=True)[0]
            except Exception as exc:
                warnings.append(f"语义查询失败（{type(exc).__name__}），使用词法检索。")
        result = store.search(payload.question, k=payload.top_k, mode=payload.mode, query_vector=vector,
            embedding_model=settings.embedding_model, document_ids=payload.document_ids or None)
        sources, seen, remaining = [], set(), min(settings.context_chars, 3500)
        for hit in result["hits"]:
            parent = hit["parent_text"]
            key = (hit["document_id"], hit["page"], parent)
            if key in seen or remaining < 150:
                continue
            seen.add(key)
            text = parent[:remaining]
            remaining -= len(text)
            sources.append({"citation": len(sources) + 1, "document_id": hit["document_id"],
                "name": hit["name"], "page": hit["page"], "text": text, "score": hit["score"]})
        elapsed = round((perf_counter() - started) * 1000, 2)
        return {"sources": sources, "mode_used": result["mode_used"], "mode_requested": payload.mode,
                "warnings": warnings + result["warnings"], "timings": {"retrieval_ms": elapsed}}

    @app.post("/api/search")
    def search(payload: Question):
        return retrieve(payload)

    @app.post("/api/ask")
    def ask(payload: Question):
        if not generation_lock.acquire(blocking=False):
            raise HTTPException(409, "已有问答正在执行，请等待完成。")
        try:
            start = perf_counter()
            result = retrieve(payload)
            generation_start = perf_counter()
            answer = extractive(result["sources"])
            if result["sources"] and models.generation_status()["ready"]:
                try:
                    answer = models.generate(payload.question, result["sources"])
                except Exception as exc:
                    result["warnings"].append(str(exc) if isinstance(exc, ValueError) else f"模型服务未完成回答（{type(exc).__name__}），已保留原文证据。")
            elif result["sources"]:
                result["warnings"].append("生成模型未启动。当前展示原文摘录，不是模型回答。")
            # Merge service notices without dropping retrieval warnings.
            result["warnings"].extend(answer.pop("warnings", []))
            result.update(answer)
            result.setdefault("model", "evidence-only")
            result["timings"].update(generation_ms=round((perf_counter() - generation_start) * 1000, 2), total_ms=round((perf_counter() - start) * 1000, 2))
            result["trace"] = [
                {"step": "检索", "detail": f"{result['mode_used']} · {len(result['sources'])} 个父级证据片段", "ms": result["timings"]["retrieval_ms"]},
                {"step": "回答", "detail": answer["answer_mode"], "ms": result["timings"]["generation_ms"]},
                {"step": "引用检查", "detail": "仅校验来源编号和逐字摘录，不代表语义正确性已验证。", "ms": 0},
            ]
            return result
        finally:
            generation_lock.release()

    @app.get("/api/evaluation")
    def get_evaluation():
        path = ROOT / "reports/retrieval_evaluation.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"available": False}

    @app.post("/api/evaluate")
    def run_evaluation():
        if not evaluation_lock.acquire(blocking=False):
            raise HTTPException(409, "评测正在进行。")
        try:
            return evaluate(models, settings.embedding_model)
        finally:
            evaluation_lock.release()

    @app.get("/api/training")
    def training():
        status_path = ROOT / "reports/training_status.json"
        if status_path.exists():
            try:
                return json.loads(status_path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                pass
        if (ROOT / "evaluation/results/experiment_summary.json").exists():
            return {"state": "published", "model": "Qwen3.5-2B", "history": [],
                    "message": "仓库附带已完成实验的统计结果。本机没有训练运行记录；下方展示发布时保存的评测摘要。"}
        return {"state": "preparing", "model": "Qwen3.5-2B", "quantization": "BF16 LoRA（待 GPU 实测）",
            "steps": 0, "loss": None, "elapsed_hours": 0, "budget_hours": 72, "peak_vram_gb": None,
            "message": "正在准备训练环境。完成 GPU 短训练后才能估算时长；当前尚未开始长训练。", "history": [], "artifacts": []}

    @app.get("/api/experiments")
    def experiments():
        # Only the audited, portable summary is exposed; raw predictions and
        # machine-specific process metadata stay in local experiment storage.
        path = ROOT / "evaluation/results/experiment_summary.json"
        if not path.exists():
            return {"available": False}
        try:
            return {"available": True, **json.loads(path.read_text(encoding="utf-8"))}
        except (ValueError, OSError):
            raise HTTPException(503, "实验结果摘要暂时无法读取。")

    @app.get("/")
    def home():
        return FileResponse(ROOT / "static/index.html")

    app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
    return app


app = create_app()
