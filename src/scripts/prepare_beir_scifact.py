"""
Download BEIR SciFact and convert it to rag-eval JSONL format.

Outputs:
  data/beir_scifact_subset/corpus.jsonl
  data/beir_scifact_subset/queries.jsonl
"""

import os
import json
from beir import util
from beir.datasets.data_loader import GenericDataLoader


OUTPUT_DIR = "data/beir_scifact_subset"
MAX_QUERIES = 5000


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Downloading SciFact dataset...")
    dataset_url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
    data_path = util.download_and_unzip(dataset_url, "data")

    print("Loading dataset...")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    # Limit number of queries
    selected_queries = list(queries.keys())[:MAX_QUERIES]

    # Build set of relevant doc_ids for selected queries
    relevant_docs = set()
    for qid in selected_queries:
        for doc_id in qrels[qid]:
            relevant_docs.add(doc_id)

    print(f"Selected {len(selected_queries)} queries")
    print(f"Corpus size (relevant subset): {len(relevant_docs)} documents")

    # Write corpus.jsonl
    corpus_path = os.path.join(OUTPUT_DIR, "corpus.jsonl")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for doc_id in relevant_docs:
            doc = corpus[doc_id]
            text = doc["title"] + " " + doc["text"]
            json.dump({"doc_id": doc_id, "text": text}, f)
            f.write("\n")

    # Write queries.jsonl
    queries_path = os.path.join(OUTPUT_DIR, "queries.jsonl")
    with open(queries_path, "w", encoding="utf-8") as f:
        for qid in selected_queries:
            rel_ids = list(qrels[qid].keys())
            json.dump({
                "query_id": qid,
                "query": queries[qid],
                "relevant_ids": rel_ids
            }, f)
            f.write("\n")

    print("Done.")
    print(f"Corpus written to: {corpus_path}")
    print(f"Queries written to: {queries_path}")


if __name__ == "__main__":
    main()