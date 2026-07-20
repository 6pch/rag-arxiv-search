"""Configuración central del pipeline, cargada desde variables de entorno.

Centralizar la config aquí (en vez de repartir os.getenv por todo el código)
es una decisión deliberada: facilita hacer sweeps de hiperparámetros para la
evaluación (ver src/evaluate.py --sweep) sin tocar el resto del código.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
    papers_dir: str = os.getenv("PAPERS_DIR", "./data/papers")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "76"))
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "4"))
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")


settings = Settings()
