from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Any


@dataclass(frozen=True)
class CorpusDoc:
    doc_id: str
    text: str
    metadata: Dict[str, Any] | None = None


@dataclass(frozen=True)
class QueryItem:
    query_id: str
    query: str
    relevant_ids: List[str]


def read_jsonl(path: str) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {e}") from e


def load_corpus(path: str) -> List[CorpusDoc]:
    docs: List[CorpusDoc] = []
    for obj in read_jsonl(path):
        if "doc_id" not in obj or "text" not in obj:
            raise ValueError("Corpus JSONL requires fields: doc_id, text")
        docs.append(CorpusDoc(doc_id=str(obj["doc_id"]), text=str(obj["text"]), metadata=obj.get("metadata")))
    if not docs:
        raise ValueError("Empty corpus")
    return docs


def load_queries(path: str) -> List[QueryItem]:
    items: List[QueryItem] = []
    for obj in read_jsonl(path):
        for key in ("query_id", "query", "relevant_ids"):
            if key not in obj:
                raise ValueError(f"Queries JSONL requires fields: query_id, query, relevant_ids. Missing: {key}")
        rel = obj["relevant_ids"]
        if not isinstance(rel, list):
            raise ValueError("relevant_ids must be a list")
        items.append(QueryItem(
            query_id=str(obj["query_id"]),
            query=str(obj["query"]),
            relevant_ids=[str(x) for x in rel],
        ))
    if not items:
        raise ValueError("Empty queries")
    return items


def write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)