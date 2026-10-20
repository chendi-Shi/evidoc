from dataclasses import dataclass
import os
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]


def local_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Evidoc 模型地址必须是本机 HTTP 地址，文档不会发送至远端 API。")
    return value.rstrip("/")


@dataclass(frozen=True)
class Settings:
    db_path: Path = Path(os.environ.get("EVIDOC_DB", str(ROOT / "data" / "evidoc.sqlite3")))
    chat_url: str = local_url(os.environ.get("EVIDOC_CHAT_URL", "http://127.0.0.1:8081"))
    embedding_url: str = local_url(os.environ.get("EVIDOC_EMBED_URL", "http://127.0.0.1:8082"))
    adapter_url: str = local_url(os.environ.get("EVIDOC_ADAPTER_URL", "http://127.0.0.1:8083"))
    generation_backend: str = os.environ.get("EVIDOC_GENERATION_BACKEND", "adapter")
    model_name: str = "Qwen3.5-2B · 本地 LoRA"
    embedding_model: str = "Qwen3-Embedding-0.6B-Q8_0"
    max_file_bytes: int = 20 * 1024 * 1024
    max_documents: int = 100
    generation_timeout: int = 240
    context_chars: int = 5000

    def __post_init__(self):
        if self.generation_backend not in {"adapter", "gguf"}:
            raise ValueError("EVIDOC_GENERATION_BACKEND must be adapter or gguf")
        for url in (self.chat_url, self.embedding_url, self.adapter_url):
            local_url(url)
