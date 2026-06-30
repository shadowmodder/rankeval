import math
from rankeval import dcg, ndcg, reciprocal_rank, mrr, average_precision, precision_at_k, recall_at_k


def test_dcg_binary():
    # [1, 0, 1]: (2^1-1)/log2(2) + 0 + (2^1-1)/log2(4) = 1.0 + 0.0 + 0.5 = 1.5
    assert abs(dcg([1, 0, 1]) - 1.5) < 1e-9


def test_dcg_graded():
    # [3, 2, 1]: 7/log2(2) + 3/log2(3) + 1/log2(4)
    expected = 7.0 / math.log2(2) + 3.0 / math.log2(3) + 1.0 / math.log2(4)
    assert abs(dcg([3, 2, 1]) - expected) < 1e-9


def test_dcg_at_k():
    # Only first item counts at k=1
    assert abs(dcg([1, 1, 1], k=1) - 1.0) < 1e-9


def test_ndcg_perfect():
    # Already sorted descending → NDCG = 1.0
    assert abs(ndcg([3, 2, 1]) - 1.0) < 1e-9


def test_ndcg_reversed():
    # Worst possible ranking for graded relevance
    assert ndcg([1, 2, 3]) < 1.0


def test_ndcg_zero_relevant():
    assert ndcg([0, 0, 0]) == 0.0


def test_ndcg_at_k_top_first():
    # Most relevant item is first → NDCG@1 = 1.0
    assert abs(ndcg([1, 0, 0], k=1) - 1.0) < 1e-9


def test_ndcg_at_k_top_second():
    # Most relevant item missed at k=1
    assert ndcg([0, 1, 0], k=1) == 0.0


def test_reciprocal_rank_first():
    assert reciprocal_rank([1, 0, 0]) == 1.0


def test_reciprocal_rank_second():
    assert abs(reciprocal_rank([0, 1, 0]) - 0.5) < 1e-9


def test_reciprocal_rank_none():
    assert reciprocal_rank([0, 0, 0]) == 0.0


def test_mrr_basic():
    # first relevant at rank 1 and rank 2 → MRR = (1 + 0.5) / 2 = 0.75
    assert abs(mrr([[1, 0], [0, 1]]) - 0.75) < 1e-9


def test_mrr_empty():
    assert mrr([]) == 0.0


def test_precision_at_k():
    assert precision_at_k([1, 1, 0, 0], k=2) == 1.0
    assert precision_at_k([1, 1, 0, 0], k=4) == 0.5
    assert precision_at_k([0, 0, 0, 0], k=2) == 0.0


def test_recall_at_k():
    # [1, 0, 1, 0]: 2 relevant. k=1 → 0.5; k=3 → 1.0
    assert recall_at_k([1, 0, 1, 0], k=1) == 0.5
    assert recall_at_k([1, 0, 1, 0], k=3) == 1.0
    assert recall_at_k([0, 0], k=2) == 0.0


def test_average_precision_perfect():
    # All relevant at top → AP = 1.0
    assert abs(average_precision([1, 1, 1]) - 1.0) < 1e-9


def test_average_precision_mixed():
    # [1, 0, 1]: 2 relevant; AP = (P@1 + P@3) / 2 = (1/1 + 2/3) / 2
    expected = (1.0 + 2.0 / 3.0) / 2.0
    assert abs(average_precision([1, 0, 1]) - expected) < 1e-9


def test_average_precision_no_relevant():
    assert average_precision([0, 0, 0]) == 0.0


def test_average_precision_at_k():
    # Only consider top-2 of [1, 0, 1]
    result = average_precision([1, 0, 1], k=2)
    # 1 relevant in top-2 of 2 possible; AP = P@1 * 1 / min(2,2) = 1/1 / 2 ... wait
    # n=2, n_relevant=2, norm=min(2,2)=2; running_rel at i=0: 1, ap += 1/1; i=1: rel=0 → skip
    # ap = 1.0 / 2 = 0.5
    assert abs(result - 0.5) < 1e-9
