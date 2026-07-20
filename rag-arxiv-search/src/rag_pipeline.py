"""Cadena RAG: recuperación de contexto + generación de respuesta con citas.

Se construye con LCEL (LangChain Expression Language) en vez de las cadenas
legacy (RetrievalQA) para tener control explícito sobre qué se retorna
(respuesta + fuentes) y poder inspeccionar el contexto recuperado, algo
imprescindible para la evaluación con RAGAS.
"""
from __future__ import annotations

from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from src.config import settings

SYSTEM_PROMPT = """Eres un asistente de investigación que responde preguntas \
basándote EXCLUSIVAMENTE en el contexto de papers científicos proporcionado.

Reglas estrictas:
- Si la respuesta no está en el contexto, di explícitamente que no tienes \
información suficiente en los documentos indexados. No inventes datos.
- Cita el título del paper de origen entre corchetes al final de cada \
afirmación, ej: [Attention Is All You Need].
- Sé preciso y técnico; no simplifiques en exceso conceptos científicos.

Contexto recuperado:
{context}
"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


@dataclass
class RagAnswer:
    answer: str
    sources: list[Document]


def get_llm():
    """Instancia el LLM generador según el proveedor configurado.

    Se abstrae aquí para poder cambiar de OpenAI a Anthropic (o a un modelo
    local vía Ollama) cambiando solo una variable de entorno.
    """
    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=settings.llm_model, temperature=0)

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=settings.llm_model, temperature=0)


def _format_context(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(
        f"[{d.metadata.get('title', 'desconocido')}]\n{d.page_content}" for d in docs
    )


def build_chain(vectorstore: Chroma, top_k: int | None = None):
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k or settings.retrieval_top_k})
    llm = get_llm()

    chain = (
        {"context": retriever | _format_context, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def answer_question(vectorstore: Chroma, question: str, top_k: int | None = None) -> RagAnswer:
    chain, retriever = build_chain(vectorstore, top_k=top_k)
    sources = retriever.invoke(question)
    answer = chain.invoke(question)
    return RagAnswer(answer=answer, sources=sources)
