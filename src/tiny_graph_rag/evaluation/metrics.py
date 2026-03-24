"""Retrieval evaluation metrics.

Implements Precision@k, Recall@k, MRR, and nDCG with binary relevance.
All functions handle empty ground truth safely by returning 0.0.
"""

import math


def compute_precision_at_k(
    retrieved: list[str],
    relevant: list[str],
    k: int,
) -> float:
    """Compute Precision@k.

    Precision@k = |retrieved[:k] intersection relevant| / k

    Args:
        retrieved: Ordered list of retrieved entity names.
        relevant: List of ground-truth relevant entity names.
        k: Cutoff rank.

    Returns:
        Precision@k score in [0.0, 1.0].
    """
    if k <= 0:
        return 0.0
    if not relevant:
        return 0.0

    relevant_set = _normalize_set(relevant)
    retrieved_at_k = [_normalize(name) for name in retrieved[:k]]
    seen: set[str] = set()
    hits = 0
    for name in retrieved_at_k:
        if name in relevant_set and name not in seen:
            hits += 1
            seen.add(name)
    return hits / k


def compute_recall_at_k(
    retrieved: list[str],
    relevant: list[str],
    k: int,
) -> float:
    """Compute Recall@k.

    Recall@k = |retrieved[:k] intersection relevant| / |relevant|

    Args:
        retrieved: Ordered list of retrieved entity names.
        relevant: List of ground-truth relevant entity names.
        k: Cutoff rank.

    Returns:
        Recall@k score in [0.0, 1.0].
    """
    if k <= 0:
        return 0.0
    if not relevant:
        return 0.0

    relevant_set = _normalize_set(relevant)
    retrieved_at_k = [_normalize(name) for name in retrieved[:k]]
    seen: set[str] = set()
    hits = 0
    for name in retrieved_at_k:
        if name in relevant_set and name not in seen:
            hits += 1
            seen.add(name)
    return hits / len(relevant_set)


def compute_mrr(
    retrieved: list[str],
    relevant: list[str],
) -> float:
    """Compute Mean Reciprocal Rank (MRR).

    MRR = 1 / rank_of_first_relevant_item

    Args:
        retrieved: Ordered list of retrieved entity names.
        relevant: List of ground-truth relevant entity names.

    Returns:
        MRR score in [0.0, 1.0]. Returns 0.0 if no relevant item found.
    """
    if not relevant or not retrieved:
        return 0.0

    relevant_set = _normalize_set(relevant)
    for rank, name in enumerate(retrieved, start=1):
        if _normalize(name) in relevant_set:
            return 1.0 / rank
    return 0.0


def compute_ndcg_at_k(
    retrieved: list[str],
    relevant: list[str],
    k: int,
) -> float:
    """Compute normalized Discounted Cumulative Gain at k (nDCG@k).

    Uses binary relevance: relevance = 1 if item is in relevant set, else 0.

    DCG@k  = sum_{i=1}^{k} rel_i / log2(i + 1)
    IDCG@k = sum_{i=1}^{min(k, |relevant|)} 1 / log2(i + 1)
    nDCG@k = DCG@k / IDCG@k

    Args:
        retrieved: Ordered list of retrieved entity names.
        relevant: List of ground-truth relevant entity names.
        k: Cutoff rank.

    Returns:
        nDCG@k score in [0.0, 1.0].
    """
    if k <= 0:
        return 0.0
    if not relevant:
        return 0.0

    relevant_set = _normalize_set(relevant)

    # Compute DCG@k
    dcg = 0.0
    seen: set[str] = set()
    for i, name in enumerate(retrieved[:k]):
        normalized = _normalize(name)
        if normalized in relevant_set and normalized not in seen:
            rel = 1.0
            seen.add(normalized)
        else:
            rel = 0.0
        dcg += rel / math.log2(i + 2)  # i+2 because i is 0-based, formula uses 1-based+1

    # Compute ideal DCG@k
    ideal_hits = min(k, len(relevant_set))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def _normalize(name: str) -> str:
    """Normalize entity name for comparison."""
    return name.lower().strip()


def _normalize_set(names: list[str]) -> set[str]:
    """Build a normalized set from a list of names."""
    return {_normalize(n) for n in names}
