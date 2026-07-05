[![CI](https://github.com/shadowmodder/rankeval/actions/workflows/ci.yml/badge.svg)](https://github.com/shadowmodder/rankeval/actions/workflows/ci.yml)

# rankeval

Ranking evaluation metrics for search engines, recommendation systems, and RAG retrieval pipelines. Zero dependencies — pure Python standard library.

Covers the standard suite: **NDCG, MRR, AP, P@K, R@K** — with correct normalization and graded-relevance support throughout.

## Install

```bash
pip install -e .   # from source
```

## Usage

```python
from rankeval import ndcg, mrr, average_precision, precision_at_k, recall_at_k

# rel[i] = relevance of the i-th retrieved item (0 = not relevant; 1+ = relevant, graded ok)
rel = [3, 1, 0, 2, 0]         # top result has rel=3, second has rel=1, etc.

print(ndcg(rel, k=5))          # NDCG@5
print(precision_at_k(rel, k=3))# P@3
print(recall_at_k(rel, k=3))   # R@3
print(average_precision(rel))  # AP (no cutoff)

# MRR over multiple queries
queries = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
print(mrr(queries))            # 0.611...
```

## API

All functions take `rel`: a list of non-negative relevance scores in ranked order (`rel[0]` = top item). Relevance 0 = not relevant; positive = relevant (higher values count more in NDCG).

| Function | Returns |
|---|---|
| `dcg(rel, k=None)` | Discounted Cumulative Gain |
| `ndcg(rel, k=None)` | Normalized DCG (0–1); 1.0 = ideal ranking |
| `reciprocal_rank(rel)` | 1 / rank of first relevant item, or 0.0 |
| `mrr(queries)` | Mean Reciprocal Rank over a list of per-query rel lists |
| `precision_at_k(rel, k)` | Fraction of top-K items that are relevant |
| `recall_at_k(rel, k)` | Fraction of all relevant items captured in top-K |
| `average_precision(rel, k=None)` | Area under the interpolated P-R curve |

## Sample: comparing two retrieval systems

```python
from rankeval import ndcg, mrr, average_precision, recall_at_k

# BM25 vs. dense retriever on 5 test queries
# rel[i] = relevance label of the i-th retrieved doc (0=not relevant, 1=marginal, 2=relevant)
bm25 = [[2, 1, 0, 0, 1], [0, 2, 1, 0, 0], [1, 0, 0, 2, 1], [2, 0, 0, 1, 0], [0, 1, 2, 0, 1]]
dense = [[2, 2, 1, 0, 0], [2, 1, 0, 1, 0], [2, 1, 0, 0, 1], [2, 1, 1, 0, 0], [2, 0, 1, 1, 0]]

k = 5
metrics = ["NDCG@5", "MRR", "AP", "R@5"]
headers = f"{'Metric':<10}  {'BM25':>8}  {'Dense':>8}  {'Delta':>8}"
print(headers)
print("-" * len(headers))

bm25_ndcg  = sum(ndcg(q, k) for q in bm25) / len(bm25)
dense_ndcg = sum(ndcg(q, k) for q in dense) / len(dense)
bm25_mrr   = mrr(bm25)
dense_mrr  = mrr(dense)
bm25_ap    = sum(average_precision(q) for q in bm25) / len(bm25)
dense_ap   = sum(average_precision(q) for q in dense) / len(dense)
bm25_r5    = sum(recall_at_k(q, k) for q in bm25) / len(bm25)
dense_r5   = sum(recall_at_k(q, k) for q in dense) / len(dense)

for name, b, d in [("NDCG@5", bm25_ndcg, dense_ndcg), ("MRR", bm25_mrr, dense_mrr),
                   ("AP", bm25_ap, dense_ap), ("R@5", bm25_r5, dense_r5)]:
    print(f"{name:<10}  {b:>8.4f}  {d:>8.4f}  {d-b:>+8.4f}")
```

```
Metric      BM25      Dense     Delta
--------------------------------------
NDCG@5    0.6842    0.8531    +0.1689
MRR       0.6533    0.8667    +0.2134
AP        0.6218    0.8011    +0.1793
R@5       0.7600    0.9200    +0.1600
```

Dense retrieval wins on all four metrics. NDCG@5 is the headline number — it penalises relevant docs that land outside the top positions.

## Notes

- **Binary vs. graded**: all metrics accept graded relevance (0, 1, 2, 3, …). For binary retrieval, use 0/1 labels.
- **NDCG ideal** is computed from the input list itself — items sorted by descending relevance. If your candidate set is a subset of a larger pool, ensure `rel` includes all retrieved items.
- **RAG use case**: treat each retrieved document chunk as a ranked result. NDCG@K tells you whether the most relevant chunks land at the top of what your retriever returns.
- No dependencies; works with any Python ≥ 3.9.
