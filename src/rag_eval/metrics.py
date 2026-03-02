from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Dict, List, Set


def precision_at_k(relevant: Set[str], retrieved: Sequence[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be > 0")
    topk = retrieved[:k]
    if not topk:
        return 0.0
    hits = sum(1 for rid in topk if rid in relevant)
    return hits / float(k)


def recall_at_k(relevant: Set[str], retrieved: Sequence[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be > 0")
    if not relevant:
        # Convention: if no ground-truth relevant docs, recall undefined -> return 0
        return 0.0
    topk = retrieved[:k]
    hits = sum(1 for rid in topk if rid in relevant)
    return hits / float(len(relevant))


def mrr_at_k(relevant: Set[str], retrieved: Sequence[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be > 0")
    topk = retrieved[:k]
    for i, rid in enumerate(topk, start=1):
        if rid in relevant:
            return 1.0 / float(i)
    return 0.0


@dataclass(frozen=True)
class QueryMetrics:
    query_id: str
    k: int
    precision: float
    recall: float
    mrr: float


def evaluate_query(query_id: str, relevant_ids: Iterable[str], retrieved_ids: Sequence[str], ks: Sequence[int]) -> List[QueryMetrics]:
    rel = set(relevant_ids)
    out: List[QueryMetrics] = []
    for k in ks:
        out.append(
            QueryMetrics(
                query_id=query_id,
                k=k,
                precision=precision_at_k(rel, retrieved_ids, k),
                recall=recall_at_k(rel, retrieved_ids, k),
                mrr=mrr_at_k(rel, retrieved_ids, k),
            )
        )
    return out


def aggregate(all_metrics: Sequence[QueryMetrics]) -> Dict[int, Dict[str, float]]:
    """
    Returns: {k: {"precision": avg, "recall": avg, "mrr": avg}}
    """
    by_k: Dict[int, List[QueryMetrics]] = {}
    for m in all_metrics:
        by_k.setdefault(m.k, []).append(m)

    agg: Dict[int, Dict[str, float]] = {}
    for k, items in by_k.items():
        n = len(items)
        if n == 0:
            continue
        agg[k] = {
            "precision": sum(x.precision for x in items) / n,
            "recall": sum(x.recall for x in items) / n,
            "mrr": sum(x.mrr for x in items) / n,
            "n_queries": float(n),
        }
    return agg