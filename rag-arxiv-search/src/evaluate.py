"""Evaluación cuantitativa del pipeline RAG con RAGAS.

Esto es lo que distingue este proyecto de un tutorial de "chat con tus PDFs":
en vez de juzgar la calidad a ojo, se mide con métricas estándar de la
literatura de RAG:

- faithfulness: ¿la respuesta se sostiene en el contexto recuperado, o alucina?
- answer_relevancy: ¿la respuesta responde realmente a la pregunta hecha?
- context_precision: de lo recuperado, ¿qué proporción es relevante?
- context_recall: ¿se recuperó todo el contexto necesario para responder bien?

Uso:
    python -m src.evaluate --eval-set eval/eval_dataset.json
    python -m src.evaluate --sweep   # compara varios tamaños de chunk
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from src.config import settings
from src.rag_pipeline import answer_question
from src.vectorstore import build_vectorstore, load_documents_from_manifest, load_vectorstore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_eval_set(path: str) -> list[dict]:
    return json.loads(Path(path).read_text())


def run_evaluation(vectorstore, eval_set: list[dict]) -> pd.DataFrame:
    """Ejecuta el pipeline sobre cada pregunta del eval set y calcula métricas RAGAS."""
    records = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for item in eval_set:
        result = answer_question(vectorstore, item["question"])
        records["question"].append(item["question"])
        records["answer"].append(result.answer)
        records["contexts"].append([d.page_content for d in result.sources])
        records["ground_truth"].append(item["ground_truth"])

    dataset = Dataset.from_dict(records)
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    return scores.to_pandas()


def sweep_chunk_sizes(eval_set: list[dict], sizes: list[int]) -> pd.DataFrame:
    """Compara distintos tamaños de chunk para justificar la elección final (ver README)."""
    documents = load_documents_from_manifest()
    summary_rows = []

    for size in sizes:
        overlap = int(size * 0.15)
        logger.info("Evaluando chunk_size=%d overlap=%d", size, overlap)
        vectorstore = build_vectorstore(
            documents=documents,
            persist_dir=f"./data/chroma_eval_{size}",
            chunk_size=size,
            chunk_overlap=overlap,
        )
        df = run_evaluation(vectorstore, eval_set)
        summary_rows.append(
            {
                "chunk_size": size,
                "faithfulness": df["faithfulness"].mean(),
                "answer_relevancy": df["answer_relevancy"].mean(),
                "context_precision": df["context_precision"].mean(),
                "context_recall": df["context_recall"].mean(),
            }
        )

    return pd.DataFrame(summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluar el pipeline RAG con RAGAS")
    parser.add_argument("--eval-set", default="eval/eval_dataset.json")
    parser.add_argument("--sweep", action="store_true", help="Comparar varios chunk sizes")
    args = parser.parse_args()

    eval_set = load_eval_set(args.eval_set)
    out_dir = Path("eval/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.sweep:
        df = sweep_chunk_sizes(eval_set, sizes=[256, 512, 1024])
        df.to_csv(out_dir / "chunk_size_sweep.csv", index=False)
        logger.info("\n%s", df.to_string(index=False))
    else:
        vectorstore = load_vectorstore()
        df = run_evaluation(vectorstore, eval_set)
        df.to_csv(out_dir / "evaluation_results.csv", index=False)
        logger.info("\n%s", df.to_string(index=False))


if __name__ == "__main__":
    main()
