"""Construcción e indexado del vector store (Chroma) a partir de los papers ingeridos."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.chunking import split_documents
from src.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_embedding_function() -> HuggingFaceEmbeddings:
    """Embeddings locales por defecto (gratis, sin límite de rate) en vez de
    depender de una API externa para la parte del pipeline que más se ejecuta.
    """
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def load_documents_from_manifest(papers_dir: str | None = None) -> list[Document]:
    """Reconstruye los Document a partir del manifest + PDFs guardados en la ingesta."""
    from pypdf import PdfReader

    papers_path = Path(papers_dir or settings.papers_dir)
    manifest_path = papers_path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No se encontró {manifest_path}. Ejecuta antes: python -m src.ingest"
        )

    metadata_list = json.loads(manifest_path.read_text())
    documents = []
    for meta in metadata_list:
        pdf_path = papers_path / f"{meta['arxiv_id']}.pdf"
        if not pdf_path.exists():
            logger.warning("PDF no encontrado, se omite: %s", pdf_path)
            continue
        reader = PdfReader(str(pdf_path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        documents.append(Document(page_content=text, metadata=meta))
    return documents


def build_vectorstore(
    documents: list[Document] | None = None,
    persist_dir: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> Chroma:
    documents = documents or load_documents_from_manifest()
    chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    logger.info("Indexando %d chunks en Chroma...", len(chunks))
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embedding_function(),
        persist_directory=persist_dir or settings.chroma_persist_dir,
    )
    logger.info("Índice construido en %s", persist_dir or settings.chroma_persist_dir)
    return vectorstore


def load_vectorstore(persist_dir: str | None = None) -> Chroma:
    return Chroma(
        persist_directory=persist_dir or settings.chroma_persist_dir,
        embedding_function=get_embedding_function(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Construir el vector store")
    parser.add_argument("action", choices=["build"])
    args = parser.parse_args()
    if args.action == "build":
        build_vectorstore()


if __name__ == "__main__":
    main()
