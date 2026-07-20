from langchain_core.documents import Document

from src.chunking import split_documents


def make_doc(text: str) -> Document:
    return Document(page_content=text, metadata={"title": "Test Paper", "arxiv_id": "0000.0000"})


def test_split_respects_chunk_size_bounds():
    long_text = "Esta es una frase de prueba repetida para simular un paper largo. " * 200
    doc = make_doc(long_text)

    chunks = split_documents([doc], chunk_size=200, chunk_overlap=30)

    assert len(chunks) > 1
    # Con el splitter recursivo, algunos chunks pueden superar levemente el
    # tamaño objetivo si no hay separador natural cercano; comprobamos que
    # se mantienen dentro de un margen razonable.
    assert all(len(c.page_content) <= 260 for c in chunks)


def test_split_preserves_metadata():
    doc = make_doc("Texto corto de prueba.")
    chunks = split_documents([doc], chunk_size=50, chunk_overlap=5)

    assert all(c.metadata["title"] == "Test Paper" for c in chunks)
    assert all("chunk_id" in c.metadata for c in chunks)


def test_short_document_produces_single_chunk():
    doc = make_doc("Texto muy corto.")
    chunks = split_documents([doc], chunk_size=500, chunk_overlap=50)

    assert len(chunks) == 1
    assert chunks[0].page_content == "Texto muy corto."
