from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import List, Dict, Any

from rag_eval.io import load_corpus, load_queries, write_json
from rag_eval.metrics import evaluate_query, aggregate, QueryMetrics
from rag_eval.retrievers.factory import build_retriever
from typing import Sequence, List
from rag_eval.io import CorpusDoc

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

#TODO
def collapse_to_doc_level(retrieved_ids: Sequence[str], corpus: Sequence[CorpusDoc]) -> List[str]:
    # Placeholder: currently doc_id == chunk_id
    # Future: map chunk_id -> parent_doc (e.g. metadata.parent_doc) and deduplicate preserving order
    return list(retrieved_ids)

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


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate retrieval quality (BM25 baseline) with standard metrics.")
    p.add_argument("--corpus", required=True, help="Path to corpus JSONL (doc_id, text).")
    p.add_argument("--queries", required=True, help="Path to queries JSONL (query_id, query, relevant_ids).")
    p.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10], help="Cutoffs k for metrics.")
    p.add_argument("--topk", type=int, default=10, help="How many docs to retrieve per query (>= max(ks)).")
    p.add_argument("--out", default="results.json", help="Output JSON path.")
    p.add_argument("--plot", default=None, help="Path to save a metrics plot (e.g. report.png). If omitted, no plot.")
    p.add_argument("--retriever", default="bm25", choices=["bm25"], help="Retriever backend.")
    p.add_argument("--per-query-out", default=None, help="Optional path to save per-query breakdown JSON.")
    p.add_argument("--eval-mode", default="doc", choices=["doc", "chunk"], help="Evaluate at doc-level or chunk-level.")
    p.add_argument("--bm25-k1", type=float, default=1.2)
    p.add_argument("--bm25-b", type=float, default=0.75)

    args = p.parse_args()

    ks = sorted(set(args.ks))
    if args.topk < max(ks):
        raise ValueError("--topk must be >= max(--ks)")

    corpus = load_corpus(args.corpus)
    queries = load_queries(args.queries)

    retriever = build_retriever(args.retriever, corpus, bm25_k1=args.bm25_k1, bm25_b=args.bm25_b)

    all_metrics: List[QueryMetrics] = []
    per_query: List[Dict[str, Any]] = []

    for q in queries:
        retrieved = retriever.retrieve(q.query, top_k=args.topk)
        retrieved_ids = [doc_id for doc_id, _score in retrieved]

        q_metrics = evaluate_query(q.query_id, q.relevant_ids, retrieved_ids, ks=ks)
        all_metrics.extend(q_metrics)

        per_query.append({
            "query_id": q.query_id,
            "query": q.query,
            "relevant_ids": q.relevant_ids,
            "retrieved": [{"doc_id": doc_id, "score": float(score)} for doc_id, score in retrieved],
            "metrics": [
                {"k": m.k, "precision": m.precision, "recall": m.recall, "mrr": m.mrr, "ndcg": m.ndcg}
                for m in q_metrics
            ],
        })

    agg = aggregate(all_metrics)

    if not agg:
        raise RuntimeError("No metrics were computed (agg is empty). Check input files.")
    
    if args.eval_mode == "doc":
        retrieved_ids = collapse_to_doc_level(retrieved_ids, corpus) # identity func

    if args.plot:
        if plt is None:
            raise RuntimeError("Plot requested but matplotlib is not installed. Install it or run without --plot.")
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
        plt.savefig(args.plot)
        plt.close()
        print(f"Saved: {args.plot}")

    print(_format_table(agg, ks))
    payload = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "retriever": "bm25",
            "bm25": {"k1": args.bm25_k1, "b": args.bm25_b},
            "ks": ks,
            "topk": args.topk,
            "corpus_path": args.corpus,
            "queries_path": args.queries,
        },
        "aggregate": agg,
        "per_query": per_query,
    }

    if args.per_query_out:
        write_json(args.per_query_out, {"per_query": per_query})
        print(f"Saved: {args.per_query_out}")

    write_json(args.out, payload)
    print(f"\nSaved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())