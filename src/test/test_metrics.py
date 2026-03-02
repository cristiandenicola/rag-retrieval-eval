from rag_eval.metrics import precision_at_k, recall_at_k, mrr_at_k, ndcg_at_k, evaluate_query, aggregate

def test_precision_recall_mrr():
    relevant = {"d2", "d4"}
    retrieved = ["d1", "d2", "d3", "d4"]

    assert precision_at_k(relevant, retrieved, 1) == 0.0
    assert precision_at_k(relevant, retrieved, 2) == 0.5  # {d2} in top2
    assert recall_at_k(relevant, retrieved, 2) == 0.5      # 1 hit out of 2 relevant
    assert mrr_at_k(relevant, retrieved, 10) == 1.0 / 2.0  # first relevant at rank 2

def test_ndcg_binary():
    relevant = {"d2", "d4"}
    retrieved = ["d1", "d2", "d3", "d4"]  # relevant at ranks 2 and 4

    # k=2: only d2 is in top2. DCG = 1/log2(3)
    # IDCG for 2 relevant within k=2: 1/log2(2) + 1/log2(3) = 1 + 1/log2(3)
    import math
    dcg = 1.0 / math.log2(2 + 1)  # rank 2 -> log2(3)
    idcg = 1.0 / math.log2(1 + 1) + 1.0 / math.log2(2 + 1)  # 1 + 1/log2(3)
    expected = dcg / idcg

    assert abs(ndcg_at_k(relevant, retrieved, 2) - expected) < 1e-9

def test_evaluate_and_aggregate():
    ms = evaluate_query("q1", ["d1"], ["d2","d1"], ks=[1,2])
    agg = aggregate(ms)

    assert agg[1]["precision"] == 0.0
    assert agg[2]["precision"] == 0.5
    assert agg[2]["mrr"] == 1.0 / 2.0
    assert "ndcg" in agg[2]