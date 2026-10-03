from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="AW_", extra="ignore")

    data_dir: Path = Path("data")
    mode: Literal["vllm", "demo"] = "vllm"
    llm_base_url: str = "http://127.0.0.1:8000/v1"
    llm_api_key: str = "local-dev-key"
    llm_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    embedding_backend: Literal["bm25", "vllm"] = "bm25"
    embedding_base_url: str = "http://127.0.0.1:8001/v1"
    embedding_api_key: str = "local-dev-key"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    api_token: str = ""
    request_timeout: float = Field(120, ge=1, le=600)
    max_steps: int = Field(5, ge=1, le=12)
    max_tokens: int = Field(1024, ge=64, le=4096)
    context_chars: int = Field(12000, ge=4000, le=48000)
    history_turns: int = Field(8, ge=1, le=30)
    chunk_size: int = Field(700, ge=100, le=2000)
    chunk_overlap: int = Field(100, ge=0)
    top_k: int = Field(4, ge=1, le=10)
    max_upload_mb: int = Field(10, ge=1, le=50)
    max_document_chars: int = Field(500000, ge=1000)
    max_chunks: int = Field(10000, ge=10)

    @model_validator(mode="after")
    def validate_chunking(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self
