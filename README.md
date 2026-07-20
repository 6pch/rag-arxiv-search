# 🔬 ArXiv Research Assistant — RAG especializado con evaluación cuantitativa

Asistente de preguntas y respuestas sobre papers científicos (arXiv, dominio ML/IA por defecto,
configurable a cualquier otro dominio) construido con una arquitectura RAG completa: ingesta,
chunking, embeddings, vector store, generación aumentada y **evaluación cuantitativa con RAGAS**
(no solo "funciona bonito en la demo").

> 💡 El objetivo de este proyecto no es "otro chat con tus PDFs", sino demostrar criterio de
> ingeniería: decisiones de chunking justificadas, comparación de estrategias de recuperación,
> métricas de calidad medidas (no intuidas) y un pipeline reproducible.

---

## Arquitectura

```
┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌──────────────┐
│  Ingesta     │───▶│  Chunking     │───▶│  Embeddings +  │───▶│  Vector Store │
│  (arXiv API) │    │  (recursive/  │    │  indexado      │    │  (Chroma)     │
│  + PDF parse │    │   semantic)   │    │                │    │               │
└─────────────┘    └──────────────┘    └───────────────┘    └───────┬──────┘
                                                                      │
┌──────────────┐    ┌───────────────┐    ┌────────────────┐         │
│  Streamlit UI │◀───│  RAG Chain     │◀───│  Retriever      │◀────────┘
│               │    │  (LLM + prompt)│    │  (top-k + rerank)│
└──────────────┘    └───────┬───────┘    └────────────────┘
                              │
                     ┌────────▼────────┐
                     │  Evaluación RAGAS │
                     │  (faithfulness,   │
                     │  answer relevancy,│
                     │  context precision│
                     │  /recall)         │
                     └───────────────────┘
```

## Stack

| Componente          | Tecnología                              |
|---------------------|------------------------------------------|
| Orquestación RAG     | LangChain                                |
| Vector DB            | ChromaDB (local, sin infraestructura extra) |
| Embeddings           | `sentence-transformers` (local, gratis) o OpenAI embeddings |
| LLM generador        | OpenAI GPT-4o-mini (configurable a Anthropic/local con Ollama) |
| Evaluación            | RAGAS                                    |
| UI                    | Streamlit                                |
| Fuente de datos       | arXiv API (paquete `arxiv`)              |
| Tests                 | Pytest                                   |
| CI                     | GitHub Actions                          |

## Por qué estas decisiones (documentar esto en el README real es lo que impresiona)

- **Chunking recursivo con overlap del 15%**: los papers científicos tienen estructura densa
  (fórmulas, referencias cruzadas); un chunk demasiado pequeño rompe el contexto matemático,
  uno demasiado grande diluye la relevancia del embedding. Se comparan 3 tamaños en `eval/`.
- **Embeddings locales por defecto**: evita depender de una API de pago para la parte de
  indexado, que es la que más se ejecuta. El LLM generador sí puede ser de pago (se usa mucho menos).
- **RAGAS en vez de "funciona en mis 5 pruebas manuales"**: se mide con un dataset de evaluación
  de preguntas/respuestas de referencia (`eval/eval_dataset.json`), y se reportan métricas
  reproducibles en `eval/results/`.

## Instalación

```bash
git clone <tu-repo>
cd rag-research-assistant
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # añade tu OPENAI_API_KEY (o ANTHROPIC_API_KEY)
```

## Uso

```bash
# 1. Ingerir papers (ejemplo: últimos papers de ML en arXiv)
python -m src.ingest --query "cat:cs.LG" --max-results 100

# 2. Indexar (chunking + embeddings + Chroma)
python -m src.vectorstore build

# 3. Lanzar la UI
streamlit run app/main.py

# 4. Evaluar el pipeline con RAGAS
python -m src.evaluate --eval-set eval/eval_dataset.json
```

## Resultados de evaluación (ejemplo — reemplaza con tus números reales)

| Métrica              | Chunk 256 tokens | Chunk 512 tokens | Chunk 1024 tokens |
|-----------------------|:---:|:---:|:---:|
| Faithfulness           | 0.81 | **0.89** | 0.85 |
| Answer Relevancy        | 0.84 | **0.91** | 0.88 |
| Context Precision        | 0.77 | **0.86** | 0.80 |
| Context Recall            | 0.79 | 0.83 | **0.85** |

> Ejecuta `python -m src.evaluate --sweep` para regenerar esta tabla con tus propios datos.

## Roadmap / posibles extensiones (buenas para hablar en entrevista)

- [ ] Re-ranking con cross-encoder tras la recuperación inicial
- [ ] Hybrid search (BM25 + embeddings)
- [ ] Caché semántica de queries repetidas
- [ ] Soporte multi-dominio (legal, normativa técnica) cambiando solo `src/ingest.py`
- [ ] Despliegue con Docker Compose + API FastAPI además de la UI Streamlit

## Estructura del repo

```
rag-research-assistant/
├── src/
│   ├── ingest.py          # descarga y parseo de papers desde arXiv
│   ├── chunking.py        # estrategias de división de texto
│   ├── vectorstore.py     # construcción e indexado en Chroma
│   ├── rag_pipeline.py    # cadena de recuperación + generación
│   └── evaluate.py        # evaluación cuantitativa con RAGAS
├── app/
│   └── main.py            # interfaz Streamlit
├── eval/
│   └── eval_dataset.json  # preguntas/respuestas de referencia
├── tests/
│   ├── test_chunking.py
│   └── test_pipeline.py
├── .github/workflows/ci.yml
├── requirements.txt
└── Dockerfile
```

## Licencia

MIT
