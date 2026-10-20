"""SQLite document store and inspectable Chinese/English hybrid retrieval.

The dense threshold is a conservative heuristic, not a calibrated confidence
probability. Reciprocal rank fusion combines rankings rather than raw scores.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sqlite3
from typing import Iterator
from uuid import uuid4

import numpy as np


CHILD_CHARS = 350
PARENT_CHARS = 1000
DENSE_MIN_SIMILARITY = 0.30
RRF_K = 60
MAX_DOCUMENT_CHARS = 2_000_000


def tokenize(text: str) -> list[str]:
    """English words plus Chinese bigrams; single characters support acronyms.

    Chinese characters receive a separate prefix to avoid collisions. Bigrams
    preserve useful local word evidence without downloading a segmenter.
    """
    tokens: list[str] = []
    for part in re.findall(r"[\u3400-\u9fff]+|[a-z0-9]+(?:[._-][a-z0-9]+)*", text.lower()):
        if "\u3400" <= part[0] <= "\u9fff":
            tokens.extend("zh:" + part[index:index + 2] for index in range(len(part) - 1))
            # Single-character indexing is useful only for a one-character run;
            # indexing every character produces too many spurious matches.
            if len(part) == 1:
                tokens.append("zh:" + part)
        else:
            tokens.append(part)
    return tokens


def _segments(text: str, limit: int, overlap: int = 0) -> list[str]:
    result: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + limit, len(text))
        if end < len(text):
            candidates = [text.rfind(mark, start + limit // 2, end) for mark in ("\n", "。", "！", "？", ". ", "; ")]
            boundary = max(candidates)
            if boundary >= 0:
                end = boundary + 1
        segment = text[start:end].strip()
        if segment:
            result.append(segment)
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return result


def _valid_vector(value: list[float]) -> list[float]:
    if not isinstance(value, (list, tuple)) or not value or len(value) > 16384:
        raise ValueError("Embedding 必须是非空数值数组，且维度不能超过 16384。")
    try:
        result = [float(component) for component in value]
    except (ValueError, TypeError) as exc:
        raise ValueError("Embedding 包含非数值内容。") from exc
    if not all(math.isfinite(component) for component in result) or not any(result):
        raise ValueError("Embedding 必须包含有限数值，且不能为零向量。")
    return result


class Store:
    """Each operation uses its own SQLite connection; writes are atomic."""

    def __init__(self, db_path: Path | str):
        self._memory_anchor: sqlite3.Connection | None = None
        self._uri = str(db_path) == ":memory:"
        if self._uri:
            self.db_path = f"file:evidoc-{uuid4().hex}?mode=memory&cache=shared"
            self._memory_anchor = sqlite3.connect(self.db_path, uri=True, check_same_thread=False)
        else:
            path = Path(db_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            self.db_path = str(path)
        with self._connect() as connection:
            if not self._uri:
                connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL,
                    content_hash TEXT NOT NULL UNIQUE, pages INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    page INTEGER NOT NULL, text TEXT NOT NULL, parent_text TEXT NOT NULL,
                    ordinal INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS chunks_document ON chunks(document_id);
                CREATE TABLE IF NOT EXISTS embeddings (
                    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                    model TEXT NOT NULL, vector TEXT NOT NULL,
                    PRIMARY KEY(chunk_id, model)
                );
            """)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path, uri=self._uri, timeout=20)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def add_document(self, name: str, pages: list[dict], content_hash: str) -> dict:
        if not name.strip() or not content_hash.strip():
            raise ValueError("文档名称和内容摘要不能为空。")
        if len(name) > 300 or len(content_hash) > 256:
            raise ValueError("文档名称或内容摘要过长。")
        if not pages or len(pages) > 500:
            raise ValueError("文档必须包含 1 至 500 个文字页面。")
        normalized = []
        for item in pages:
            text, page = item.get("text"), item.get("page")
            if not isinstance(text, str) or not isinstance(page, int) or isinstance(page, bool) or page < 1:
                raise ValueError("文档页面需要有效的文字及正整数页码。")
            if text.strip():
                normalized.append({"text": text.strip(), "page": page})
        if not normalized:
            raise ValueError("文档没有可检索的文字。")
        if sum(len(item["text"]) for item in normalized) > MAX_DOCUMENT_CHARS:
            raise ValueError("文档不能超过 200 万字符。")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute("SELECT id FROM documents WHERE content_hash=?", (content_hash,)).fetchone()
            if existing:
                result = self._document(connection, existing["id"])
                result["deduplicated"] = True
                return result
            doc_id = uuid4().hex
            connection.execute("INSERT INTO documents VALUES (?, ?, ?, ?, ?)", (
                doc_id, name.strip(), content_hash, len({page["page"] for page in normalized}),
                datetime.now(timezone.utc).isoformat(),
            ))
            ordinal = 0
            for page in normalized:
                for parent in _segments(page["text"], PARENT_CHARS):
                    for child in _segments(parent, CHILD_CHARS, overlap=50):
                        connection.execute("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
                            (uuid4().hex, doc_id, page["page"], child, parent, ordinal))
                        ordinal += 1
            result = self._document(connection, doc_id)
            result["deduplicated"] = False
            return result

    @staticmethod
    def _document(connection: sqlite3.Connection, doc_id: str) -> dict:
        row = connection.execute("""SELECT d.id, d.name, d.pages, d.created_at,
            COUNT(c.id) AS chunks FROM documents d LEFT JOIN chunks c ON c.document_id=d.id
            WHERE d.id=? GROUP BY d.id""", (doc_id,)).fetchone()
        return dict(row)

    def list_documents(self) -> list[dict]:
        with self._connect() as connection:
            return [dict(row) for row in connection.execute("""SELECT d.id, d.name, d.pages, d.created_at,
                COUNT(c.id) AS chunks FROM documents d LEFT JOIN chunks c ON c.document_id=d.id
                GROUP BY d.id ORDER BY d.created_at, d.id""")]

    def delete_document(self, doc_id: str) -> bool:
        with self._connect() as connection:
            return connection.execute("DELETE FROM documents WHERE id=?", (doc_id,)).rowcount > 0

    def all_chunks(self) -> list[dict]:
        with self._connect() as connection:
            return [dict(row) for row in connection.execute("""SELECT c.id, c.document_id,
                d.name, c.page, c.text, c.parent_text FROM chunks c
                JOIN documents d ON d.id=c.document_id ORDER BY d.created_at, d.id, c.ordinal""")]

    def store_embeddings(self, mapping: dict[str, list[float]], model: str) -> None:
        if not model.strip():
            raise ValueError("Embedding 模型名称不能为空。")
        vectors = {chunk_id: _valid_vector(vector) for chunk_id, vector in mapping.items()}
        dimensions = {len(vector) for vector in vectors.values()}
        if len(dimensions) > 1:
            raise ValueError("同一模型的 Embedding 维度必须一致。")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            previous = connection.execute("SELECT vector FROM embeddings WHERE model=? LIMIT 1", (model,)).fetchone()
            if previous and dimensions and len(json.loads(previous["vector"])) not in dimensions:
                raise ValueError("Embedding 维度与此模型已保存的向量不一致。")
            for chunk_id, vector in vectors.items():
                # A document might be deleted while embedding inference runs.
                if not connection.execute("SELECT 1 FROM chunks WHERE id=?", (chunk_id,)).fetchone():
                    continue
                connection.execute("""INSERT INTO embeddings VALUES (?, ?, ?)
                    ON CONFLICT(chunk_id, model) DO UPDATE SET vector=excluded.vector""",
                    (chunk_id, model, json.dumps(vector, separators=(",", ":"))))

    def chunks_missing_embeddings(self, model: str) -> list[dict]:
        with self._connect() as connection:
            missing = {row["id"] for row in connection.execute("""SELECT c.id FROM chunks c
                LEFT JOIN embeddings e ON e.chunk_id=c.id AND e.model=?
                WHERE e.chunk_id IS NULL""", (model,))}
        return [chunk for chunk in self.all_chunks() if chunk["id"] in missing]

    def stats(self) -> dict:
        with self._connect() as connection:
            return {
                "documents": connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
                "chunks": connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
            }

    def search(self, query: str, k: int = 5, mode: str = "hybrid",
               query_vector: list[float] | None = None, embedding_model: str | None = None,
               document_ids: list[str] | None = None) -> dict:
        if mode not in {"bm25", "dense", "hybrid"}:
            raise ValueError("检索模式必须为 bm25、dense 或 hybrid。")
        if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 50:
            raise ValueError("检索数量必须为 1 至 50。")
        if len(query) > 2000:
            raise ValueError("检索问题不能超过 2000 字符。")
        chunks = self.all_chunks()
        if document_ids is not None:
            allowed = set(document_ids)
            chunks = [chunk for chunk in chunks if chunk["document_id"] in allowed]
        if not query.strip() or not chunks:
            return {"hits": [], "mode_used": "bm25" if mode == "hybrid" else mode, "warnings": []}

        sparse = self._bm25(chunks, query)
        dense: dict[str, float] = {}
        warnings: list[str] = []
        dense_available = False
        if mode != "bm25":
            if query_vector is None or not embedding_model:
                warnings.append("语义向量不可用，已回退至 BM25 关键词检索。")
            else:
                try:
                    vector = np.asarray(_valid_vector(query_vector), dtype=np.float64)
                    with self._connect() as connection:
                        stored = {row["chunk_id"]: json.loads(row["vector"]) for row in connection.execute(
                            "SELECT chunk_id, vector FROM embeddings WHERE model=?", (embedding_model,))}
                    eligible = [(chunk["id"], stored[chunk["id"]]) for chunk in chunks
                                if chunk["id"] in stored and len(stored[chunk["id"]]) == len(vector)]
                    if not eligible:
                        warnings.append("当前文档没有匹配的语义向量，已回退至 BM25 关键词检索。")
                    else:
                        dense_available = True
                        if len(eligible) < len(chunks):
                            warnings.append("部分文本块缺少匹配的语义向量；混合检索仍使用这些文本块的 BM25 结果。")
                        vector = vector / np.linalg.norm(vector)
                        for chunk_id, values in eligible:
                            candidate = np.asarray(values, dtype=np.float64)
                            norm = np.linalg.norm(candidate)
                            similarity = float(np.dot(vector, candidate / norm)) if norm > 0 else 0.0
                            if math.isfinite(similarity) and similarity >= DENSE_MIN_SIMILARITY:
                                dense[chunk_id] = similarity
                        warnings.append("语义相似度阈值为经验值 0.30，未经校准，不能视作答案可信度。")
                except (ValueError, TypeError) as exc:
                    warnings.append(f"语义向量无效，已回退至 BM25 关键词检索：{exc}")

        mode_used = mode if mode == "bm25" or dense_available else "bm25"
        if mode_used == "dense":
            scores = dense.copy()
        elif mode_used == "hybrid":
            scores: dict[str, float] = {}
            for ranking in (sparse, dense):
                for rank, (chunk_id, _) in enumerate(sorted(ranking.items(), key=lambda item: item[1], reverse=True), 1):
                    scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
        else:
            scores = sparse.copy()
        lookup = {chunk["id"]: chunk for chunk in chunks}
        hits = [{**lookup[chunk_id], "score": round(score, 8),
                 "bm25_score": round(sparse.get(chunk_id, 0.0), 8),
                 "dense_score": round(dense.get(chunk_id, 0.0), 8)}
                for chunk_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]]
        return {"hits": hits, "mode_used": mode_used, "warnings": warnings}

    @staticmethod
    def _bm25(chunks: list[dict], query: str) -> dict[str, float]:
        query_terms = sorted(set(tokenize(query)))
        if not query_terms:
            return {}
        counts = [Counter(tokenize(chunk["text"])) for chunk in chunks]
        lengths = [sum(counter.values()) for counter in counts]
        average_length = sum(lengths) / max(len(lengths), 1) or 1.0
        frequencies = {term: sum(term in counter for counter in counts) for term in query_terms}
        result = {}
        for chunk, counter, length in zip(chunks, counts, lengths):
            score = 0.0
            for term in query_terms:
                frequency = counter.get(term, 0)
                if not frequency:
                    continue
                inverse_frequency = math.log(1.0 + (len(chunks) - frequencies[term] + 0.5) / (frequencies[term] + 0.5))
                score += inverse_frequency * frequency * 2.5 / (frequency + 1.5 * (0.25 + 0.75 * length / average_length))
            if score > 0:
                result[chunk["id"]] = score
        return result
