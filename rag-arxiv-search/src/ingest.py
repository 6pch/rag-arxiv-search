"""Ingesta de papers desde la API pública de arXiv.

Descarga metadatos + PDF de papers según una query de arXiv (p.ej. "cat:cs.LG"
para Machine Learning) y extrae el texto completo del PDF para su posterior
chunking e indexado.

Para adaptar este proyecto a otro dominio (legal, normativa técnica, etc.),
este es el único módulo que normalmente necesitas reescribir: el resto del
pipeline (chunking, vectorstore, rag_pipeline, evaluate) es agnóstico a la
fuente de datos, siempre que le entregues objetos `Document` de LangChain.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import arxiv
from langchain_core.documents import Document
from pypdf import PdfReader
from tqdm import tqdm

from src.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def fetch_papers(query: str, max_results: int = 50) -> list[arxiv.Result]:
    """Consulta la API de arXiv y devuelve los metadatos de los resultados."""
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    return list(client.results(search))


def download_and_extract(result: arxiv.Result, out_dir: Path) -> Document | None:
    """Descarga el PDF de un paper y extrae su texto completo como Document."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{result.get_short_id()}.pdf"

    try:
        if not pdf_path.exists():
            result.download_pdf(dirpath=str(out_dir), filename=pdf_path.name)

        reader = PdfReader(str(pdf_path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)

        if not text.strip():
            logger.warning("PDF sin texto extraíble, se omite: %s", result.title)
            return None

        return Document(
            page_content=text,
            metadata={
                "source": result.entry_id,
                "title": result.title,
                "authors": ", ".join(a.name for a in result.authors),
                "published": str(result.published.date()),
                "arxiv_id": result.get_short_id(),
            },
        )
    except Exception as exc:  # noqa: BLE001 - queremos loggear y continuar el batch
        logger.error("Fallo procesando %s: %s", result.entry_id, exc)
        return None


def ingest(query: str, max_results: int, out_dir: str | None = None) -> list[Document]:
    out_path = Path(out_dir or settings.papers_dir)
    logger.info("Buscando papers en arXiv: query='%s' max_results=%d", query, max_results)
    results = fetch_papers(query, max_results)

    documents: list[Document] = []
    for result in tqdm(results, desc="Descargando y extrayendo papers"):
        doc = download_and_extract(result, out_path)
        if doc is not None:
            documents.append(doc)

    manifest_path = out_path / "manifest.json"
    manifest_path.write_text(
        json.dumps([d.metadata for d in documents], indent=2, ensure_ascii=False)
    )
    logger.info("Ingesta completa: %d papers procesados (de %d encontrados)", len(documents), len(results))
    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingesta de papers desde arXiv")
    parser.add_argument("--query", default="cat:cs.LG", help="Query de arXiv, ej. 'cat:cs.LG'")
    parser.add_argument("--max-results", type=int, default=50)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    ingest(args.query, args.max_results, args.out_dir)


if __name__ == "__main__":
    main()
