"""rankeval: ranking evaluation metrics for search, recommendation, and RAG."""
from .core import dcg, ndcg, reciprocal_rank, mrr, average_precision, precision_at_k, recall_at_k

__all__ = ["dcg", "ndcg", "reciprocal_rank", "mrr", "average_precision",
           "precision_at_k", "recall_at_k"]
__version__ = "0.1.0"
