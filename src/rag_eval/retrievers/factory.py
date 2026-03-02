from __future__ import annotations

from typing import Sequence
from rag_eval.io import CorpusDoc
from rag_eval.retrievers.bm25 import BM25Retriever, BM25Config

def build_retriever(name: str, corpus: Sequence[CorpusDoc], **kwargs):
    name = name.lower().strip()
    if name == "bm25":
        return BM25Retriever(
            corpus,
            BM25Config(
                k1=float(kwargs.get("bm25_k1", 1.2)),
                b=float(kwargs.get("bm25_b", 0.75)),
            ),
        )
    raise ValueError(f"Unknown retriever: {name}")