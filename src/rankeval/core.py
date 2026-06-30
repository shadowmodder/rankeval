"""Ranking evaluation metrics. Pure Python — no dependencies.

All functions take ``rel``: a list of non-negative relevance scores in ranked
order.  ``rel[0]`` is the top-ranked item, ``rel[1]`` the second, and so on.
Relevance 0 means not relevant; any positive value means relevant (higher =
more relevant for NDCG graded-relevance scoring).
"""
from __future__ import annotations
import math


def dcg(rel, k=None):
    """Discounted Cumulative Gain at K.

    DCG@K = sum_{i=1}^{K} (2^rel_i - 1) / log2(i + 1)

    Position i is 1-indexed, so the first item has discount 1/log2(2) = 1.
    """
    n = len(rel) if k is None else min(k, len(rel))
    return sum((2.0 ** rel[i] - 1.0) / math.log2(i + 2) for i in range(n))


def ndcg(rel, k=None):
    """Normalized DCG at K.

    NDCG@K = DCG@K / IDCG@K, where IDCG is DCG for the ideal (sorted) ranking.
    Returns 0.0 when no relevant items exist.
    """
    ideal = sorted(rel, reverse=True)
    idcg = dcg(ideal, k)
    if idcg == 0.0:
        return 0.0
    return dcg(rel, k) / idcg


def reciprocal_rank(rel):
    """Reciprocal rank: 1 / position of the first relevant item (1-indexed), or 0."""
    for i, r in enumerate(rel):
        if r > 0:
            return 1.0 / (i + 1)
    return 0.0


def mrr(queries):
    """Mean Reciprocal Rank over a list of per-query relevance lists."""
    if not queries:
        return 0.0
    return sum(reciprocal_rank(q) for q in queries) / len(queries)


def precision_at_k(rel, k):
    """Fraction of top-K retrieved items that are relevant (rel > 0)."""
    top = rel[:k]
    if not top:
        return 0.0
    return sum(1 for r in top if r > 0) / len(top)


def recall_at_k(rel, k):
    """Fraction of all relevant items (in rel) that appear in the top-K."""
    n_relevant = sum(1 for r in rel if r > 0)
    if n_relevant == 0:
        return 0.0
    return sum(1 for r in rel[:k] if r > 0) / n_relevant


def average_precision(rel, k=None):
    """Average Precision at K.

    AP@K = (1 / min(|R|, K)) * sum_{i=1}^{K} P@i * is_relevant(i)

    where |R| is the total number of relevant items in rel and is_relevant(i)
    is 1 when the item at rank i has rel > 0, else 0.
    """
    n = len(rel) if k is None else min(k, len(rel))
    n_relevant = sum(1 for r in rel if r > 0)
    if n_relevant == 0:
        return 0.0
    norm = min(n_relevant, n)
    running_rel = 0
    ap = 0.0
    for i in range(n):
        if rel[i] > 0:
            running_rel += 1
            ap += running_rel / (i + 1)
    return ap / norm
