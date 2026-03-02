from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import List, Dict, Any

from rag_eval.io import load_corpus, load_queries, write_json
from rag_eval.metrics import evaluate_query, aggregate, QueryMetrics
from rag_eval.retrievers.factory import build_retriever
from rag_eval.normalize import collapse_to_doc_level, normalize_relevant_ids_for_doc_eval

try:
    import matplotlib.pyplot as plt
except ImportError: 
    plt = None


def _format_table(agg: Dict[int, Dict[str, float]], ks: List[int]) -> str:
    headers = ["k", "precision", "recall", "mrr", "ndcg"]
    rows = []
    for k in ks:
        a = agg.get(k, {})
        rows.append([
            str(k),
            f"{a.get('precision', 0.0):.4f}",
            f"{a.get('recall', 0.0):.4f}",
            f"{a.get('mrr', 0.0):.4f}",
            f"{a.get('ndcg', 0.0):.4f}",
        ])

    colw = [max(len(headers[i]), max(len(r[i]) for r in rows)) for i in range(len(headers))]
    def fmt_row(r): return " | ".join(r[i].ljust(colw[i]) for i in range(len(headers)))

    line = "-+-".join("-" * w for w in colw)
    out = [fmt_row(headers), line]
    out += [fmt_row(r) for r in rows]
    return "\n".join(out)


def _maybe_plot(agg: Dict[int, Dict[str, float]], plot_path: str) -> None:
    if plt is None:
        raise RuntimeError("Plot requested but matplotlib is not installed. Install it or run without --plot.")
    if not agg:
        raise RuntimeError("No metrics were computed; agg is empty.")

    ks_sorted = sorted(agg.keys())
    precisions = [agg[k]["precision"] for k in ks_sorted]
    recalls = [agg[k]["recall"] for k in ks_sorted]
    ndcgs = [agg[k]["ndcg"] for k in ks_sorted]

    plt.figure()
    plt.plot(ks_sorted, precisions, marker="o", label="Precision@k")
    plt.plot(ks_sorted, recalls, marker="o", label="Recall@k")
    plt.plot(ks_sorted, ndcgs, marker="o", label="nDCG@k")
    plt.xlabel("k")
    plt.ylabel("Score")
    plt.title("Retrieval Metrics vs k")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate retrieval quality with standard IR metrics.")
    p.add_argument("--corpus", required=True, help="Path to corpus JSONL (doc_id, text, optional metadata).")
    p.add_argument("--queries", required=True, help="Path to queries JSONL (query_id, query, relevant_ids).")

    p.add_argument("--retriever", default="bm25", choices=["bm25"], help="Retriever backend.")
    p.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10], help="Cutoffs k for metrics.")
    p.add_argument("--topk", type=int, default=10, help="How many docs to retrieve per query (>= max(ks)).")
    p.add_argument("--out", default="results.json", help="Output JSON path.")

    p.add_argument("--plot", default=None, help="Optional path to save a metrics plot (e.g. report.png).")

    p.add_argument("--eval-mode", choices=["chunk", "doc"], default="chunk",
                   help="Evaluate on chunk IDs or collapse to doc IDs using metadata.parent_doc (if available).")

    p.add_argument("--bm25-k1", type=float, default=1.2)
    p.add_argument("--bm25-b", type=float, default=0.75)

    args = p.parse_args()

    ks = sorted(set(args.ks))
    if args.topk < max(ks):
        raise ValueError("--topk must be >= max(--ks)")

    corpus = load_corpus(args.corpus)
    queries = load_queries(args.queries)

    retriever = build_retriever(
        args.retriever,
        corpus,
        bm25_k1=args.bm25_k1,
        bm25_b=args.bm25_b,
    )

    all_metrics: List[QueryMetrics] = []
    per_query: List[Dict[str, Any]] = []

    for q in queries:
        retrieved = retriever.retrieve(q.query, top_k=args.topk)
        retrieved_ids = [doc_id for doc_id, _score in retrieved]

        relevant_ids = q.relevant_ids

        if args.eval_mode == "doc":
            retrieved_ids = collapse_to_doc_level(retrieved_ids, corpus)
            relevant_ids = normalize_relevant_ids_for_doc_eval(relevant_ids, corpus)

        q_metrics = evaluate_query(q.query_id, relevant_ids, retrieved_ids, ks=ks)
        all_metrics.extend(q_metrics)

        per_query.append({
            "query_id": q.query_id,
            "query": q.query,
            "relevant_ids": relevant_ids,
            "retrieved": [{"doc_id": doc_id, "score": float(score)} for doc_id, score in retrieved],
            "eval_mode": args.eval_mode,
            "metrics": [
                {"k": m.k, "precision": m.precision, "recall": m.recall, "mrr": m.mrr, "ndcg": m.ndcg}
                for m in q_metrics
            ],
        })

    agg = aggregate(all_metrics)

    if args.plot:
        _maybe_plot(agg, args.plot)
        print(f"Saved: {args.plot}")

    print(_format_table(agg, ks))

    payload = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "retriever": args.retriever,
            "eval_mode": args.eval_mode,
            "bm25": {"k1": args.bm25_k1, "b": args.bm25_b},
            "ks": ks,
            "topk": args.topk,
            "corpus_path": args.corpus,
            "queries_path": args.queries,
        },
        "aggregate": agg,
        "per_query": per_query,
    }
    write_json(args.out, payload)
    print(f"\nSaved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())