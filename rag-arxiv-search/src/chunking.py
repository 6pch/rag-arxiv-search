"""Estrategias de división de texto (chunking).

Se expone una función parametrizable por tamaño/overlap para poder comparar
configuraciones en la evaluación (ver src/evaluate.py --sweep), en vez de
fijar un único tamaño "porque sí".
"""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings


def split_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Divide documentos largos en chunks con overlap, preservando metadata.

    Se usa RecursiveCharacterTextSplitter con separadores jerárquicos
    (párrafo > línea > frase > palabra) porque respeta mejor la estructura
    semántica de un paper que un split por longitud fija sin criterio.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # Añadimos el índice de chunk dentro del documento origen: útil para
    # depurar qué parte concreta del paper se recuperó.
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    return chunks
