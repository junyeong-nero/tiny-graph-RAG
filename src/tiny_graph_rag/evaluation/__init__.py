"""Evaluation framework for Tiny-Graph-RAG retrieval quality."""

from .dataset import EvalExample, load_dataset
from .metrics import (
    compute_mrr,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
)
from .runner import EvalResult, EvalSummary, EvaluationRunner, save_eval_output

__all__ = [
    "EvalExample",
    "EvalResult",
    "EvalSummary",
    "EvaluationRunner",
    "compute_mrr",
    "compute_ndcg_at_k",
    "compute_precision_at_k",
    "compute_recall_at_k",
    "load_dataset",
    "save_eval_output",
]
