from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from rag_eval.io import CorpusDoc


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass
class BM25Config:
    k1: float = 1.2
    b: float = 0.75


class BM25Retriever:
    """
    BM25 classico su doc (no chunk). Ritorna doc_id ordinati per score desc.
    """
    def __init__(self, corpus: Sequence[CorpusDoc], config: BM25Config | None = None):
        self.corpus = list(corpus)
        self.cfg = config or BM25Config()

        # Precompute
        self.doc_ids: List[str] = [d.doc_id for d in self.corpus]
        self.doc_tokens: List[List[str]] = [tokenize(d.text) for d in self.corpus]
        self.doc_lens: List[int] = [len(toks) for toks in self.doc_tokens]

        self.N = len(self.corpus)
        if self.N == 0:
            raise ValueError("Corpus is empty")

        self.avgdl = sum(self.doc_lens) / float(self.N) if self.N else 0.0

        # term freq per doc + document frequency
        self.tf: List[Dict[str, int]] = []
        df: Dict[str, int] = {}

        for toks in self.doc_tokens:
            freq: Dict[str, int] = {}
            for t in toks:
                freq[t] = freq.get(t, 0) + 1
            self.tf.append(freq)
            for t in freq.keys():
                df[t] = df.get(t, 0) + 1

        # IDF with BM25+1 smoothing-ish
        self.idf: Dict[str, float] = {}
        for term, dfi in df.items():
            # classic BM25 idf: log( (N - df + 0.5) / (df + 0.5) + 1 )
            self.idf[term] = math.log(((self.N - dfi + 0.5) / (dfi + 0.5)) + 1.0)

    def retrieve(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        if top_k <= 0:
            return []

        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        scores = [0.0] * self.N
        k1 = self.cfg.k1
        b = self.cfg.b

        for i in range(self.N):
            doc_tf = self.tf[i]
            dl = self.doc_lens[i]
            denom_norm = k1 * (1.0 - b + b * (dl / self.avgdl)) if self.avgdl > 0 else k1

            s = 0.0
            for t in q_tokens:
                if t not in doc_tf:
                    continue
                tf = doc_tf[t]
                idf = self.idf.get(t, 0.0)
                num = tf * (k1 + 1.0)
                den = tf + denom_norm
                s += idf * (num / den)
            scores[i] = s

        # top-k
        pairs = [(self.doc_ids[i], scores[i]) for i in range(self.N)]
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:top_k]