"""Loopback-only FastAPI contract for the trained adapter; one GPU request at a time."""
from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from threading import Lock

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .adapter import GROUNDING_SCOPE, answer_sources


class Source(BaseModel):
    model_config = ConfigDict(extra="ignore")
    citation: int = Field(ge=1, le=1000, strict=True)
    text: str = Field(min_length=1, max_length=12000)
    name: str | None = Field(default=None, max_length=500)
    page: int | None = Field(default=None, ge=1)


class AnswerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=500)
    sources: list[Source] = Field(max_length=12)

    @field_validator("question")
    @classmethod
    def clean_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question must not be blank")
        return value.strip()

    @model_validator(mode="after")
    def unique_citations(self):
        if len({source.citation for source in self.sources}) != len(self.sources):
            raise ValueError("Source citation IDs must be unique")
        return self


def create_app(engine=None, *, loader=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        if app.state.engine is None:
            if loader is None:
                raise RuntimeError("No trained adapter loader was configured")
            app.state.engine = loader()
        yield

    app = FastAPI(title="Evidoc trained adapter", version="1.0.0", lifespan=lifespan)
    app.state.engine = engine
    app.state.generation_lock = Lock()

    @app.middleware("http")
    async def local_origin(request: Request, call_next):
        # The web app calls this server-side. Deny browser cross-origin writes,
        # including DNS-rebinding hosts, rather than exposing an unauthenticated GPU API.
        host = request.headers.get("host", "").split(":", 1)[0]
        if host not in {"127.0.0.1", "localhost", "testserver"}:
            return JSONResponse({"detail": {"code": "invalid_host"}}, status_code=403)
        if request.method not in {"GET", "HEAD", "OPTIONS"} and request.headers.get("origin"):
            return JSONResponse({"detail": {"code": "browser_origin_not_allowed"}}, status_code=403)
        return await call_next(request)

    @app.get("/health")
    def health():
        current = app.state.engine
        if current is None or not current.ready:
            raise HTTPException(503, detail={"code": "model_not_ready"})
        return {"status": "ok", "ready": True, "backend": "trained_adapter", **current.identity,
                "limits": current.limits.as_dict(), "concurrency": 1,
                "busy": app.state.generation_lock.locked(), "grounding_scope": GROUNDING_SCOPE}

    @app.post("/answer")
    def answer(body: AnswerRequest):
        current = app.state.engine
        if current is None or not current.ready:
            raise HTTPException(503, detail={"code": "model_not_ready"})
        if not app.state.generation_lock.acquire(blocking=False):
            raise HTTPException(503, detail={"code": "model_busy", "message": "One GPU request is already running"},
                                headers={"Retry-After": "2"})
        try:
            return answer_sources(current, body.question, [source.model_dump() for source in body.sources])
        except ValueError as exc:
            raise HTTPException(422, detail={"code": "input_token_budget", "message": str(exc)}) from exc
        except RuntimeError as exc:
            logging.getLogger(__name__).exception("Adapter inference failed")
            message = "CUDA out of memory" if "out of memory" in str(exc).lower() else "Model inference failed; inspect the service log"
            raise HTTPException(503, detail={"code": "inference_failed", "message": message}) from exc
        finally:
            app.state.generation_lock.release()

    return app
