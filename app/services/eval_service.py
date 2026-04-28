import json
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness
from ragas.metrics import ResponseRelevancy
from ragas.metrics import SemanticSimilarity
from ragas.metrics.collections import ContextPrecision

from app.rag.pipeline import run_rag_pipeline


def load_eval_rows(file_path: str) -> list[dict]:
    path = Path(file_path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_eval_dataset(file_path: str) -> Dataset:
    rows = load_eval_rows(file_path)

    user_inputs = []
    responses = []
    references = []
    retrieved_contexts = []

    for row in rows:
        result = run_rag_pipeline(row["question"])

        contexts = [chunk["content"] for chunk in result["sources"]]

        user_inputs.append(row["question"])
        responses.append(result["answer"])
        references.append(row["reference"])
        retrieved_contexts.append(contexts)

    dataset = Dataset.from_dict(
        {
            "user_input": user_inputs,
            "response": responses,
            "reference": references,
            "retrieved_contexts": retrieved_contexts,
        }
    )
    return dataset


def run_rag_eval(file_path: str):
    dataset = build_eval_dataset(file_path)

    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            ContextPrecision(),
            ResponseRelevancy(),
            SemanticSimilarity(),
        ],
    )

    return result