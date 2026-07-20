"""Tests del pipeline RAG.

Se mockean el LLM y el retriever para no depender de credenciales de API
ni de red en CI — el objetivo es verificar el ensamblado del pipeline
(formato del contexto, propagación de fuentes), no la calidad del LLM en sí
(eso es responsabilidad de la evaluación con RAGAS sobre el sistema real).
"""
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from src.rag_pipeline import RagAnswer, _format_context


def test_format_context_includes_titles_and_content():
    docs = [
        Document(page_content="Contenido A", metadata={"title": "Paper A"}),
        Document(page_content="Contenido B", metadata={"title": "Paper B"}),
    ]

    formatted = _format_context(docs)

    assert "Paper A" in formatted
    assert "Paper B" in formatted
    assert "Contenido A" in formatted
    assert "Contenido B" in formatted


def test_format_context_handles_missing_title():
    docs = [Document(page_content="Contenido sin título", metadata={})]
    formatted = _format_context(docs)
    assert "desconocido" in formatted


@patch("src.rag_pipeline.get_llm")
def test_answer_question_returns_answer_and_sources(mock_get_llm):
    mock_llm = MagicMock()
    mock_get_llm.return_value = mock_llm

    mock_vectorstore = MagicMock()
    mock_retriever = MagicMock()
    fake_docs = [Document(page_content="texto", metadata={"title": "Paper X"})]
    mock_retriever.invoke.return_value = fake_docs
    mock_vectorstore.as_retriever.return_value = mock_retriever

    from src.rag_pipeline import answer_question

    with patch("src.rag_pipeline.build_chain") as mock_build_chain:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Respuesta generada"
        mock_build_chain.return_value = (mock_chain, mock_retriever)

        result = answer_question(mock_vectorstore, "¿Qué es X?")

    assert isinstance(result, RagAnswer)
    assert result.answer == "Respuesta generada"
    assert result.sources == fake_docs
