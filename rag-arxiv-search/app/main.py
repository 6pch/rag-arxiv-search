"""Interfaz Streamlit del asistente de investigación RAG."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Permite ejecutar `streamlit run app/main.py` desde la raíz del repo
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.rag_pipeline import answer_question
from src.vectorstore import load_vectorstore

st.set_page_config(page_title="ArXiv Research Assistant", page_icon="🔬", layout="wide")

st.title("🔬 ArXiv Research Assistant")
st.caption(
    "Asistente RAG sobre papers científicos indexados localmente. "
    "Las respuestas se basan únicamente en los documentos ingeridos — no en el conocimiento general del LLM."
)


@st.cache_resource
def get_vectorstore():
    try:
        return load_vectorstore()
    except Exception:
        return None


vectorstore = get_vectorstore()

if vectorstore is None:
    st.warning(
        "No se encontró un índice construido. Ejecuta primero:\n\n"
        "```\npython -m src.ingest --query \"cat:cs.LG\" --max-results 50\n"
        "python -m src.vectorstore build\n```"
    )
    st.stop()

with st.sidebar:
    st.subheader("Configuración")
    top_k = st.slider("Documentos recuperados (top-k)", 1, 10, settings.retrieval_top_k)
    st.markdown("---")
    st.caption(f"Modelo: `{settings.llm_model}`")
    st.caption(f"Embeddings: `{settings.embedding_model}`")

question = st.text_input("Haz una pregunta sobre los papers indexados:")

if question:
    with st.spinner("Recuperando contexto y generando respuesta..."):
        result = answer_question(vectorstore, question, top_k=top_k)

    st.markdown("### Respuesta")
    st.write(result.answer)

    with st.expander(f"📄 Fuentes recuperadas ({len(result.sources)})"):
        for doc in result.sources:
            st.markdown(f"**{doc.metadata.get('title', 'Sin título')}**")
            st.caption(
                f"Autores: {doc.metadata.get('authors', 'N/A')} · "
                f"Publicado: {doc.metadata.get('published', 'N/A')}"
            )
            st.text(doc.page_content[:400] + "...")
            st.markdown("---")
