from __future__ import annotations

from typing import Dict, List, Sequence, Set
from rag_eval.io import CorpusDoc


def build_id_maps(corpus: Sequence[CorpusDoc]) -> tuple[Dict[str, str], Set[str]]:
    """
    Returns:
      chunk_to_doc: maps doc_id (or chunk_id) -> parent_doc if available, else itself
      parent_docs: set of all known parent_doc values found in corpus metadata
    """
    chunk_to_doc: Dict[str, str] = {}
    parent_docs: Set[str] = set()

    for d in corpus:
        parent = None
        if d.metadata and isinstance(d.metadata, dict):
            parent = d.metadata.get("parent_doc") or d.metadata.get("doc_id")
        if parent is not None:
            parent_docs.add(str(parent))
            chunk_to_doc[d.doc_id] = str(parent)
        else:
            chunk_to_doc[d.doc_id] = d.doc_id

    return chunk_to_doc, parent_docs


def collapse_to_doc_level(ids: Sequence[str], corpus: Sequence[CorpusDoc]) -> List[str]:
    """
    Maps ids (chunk_ids) -> parent_doc and deduplicates preserving order.
    If no parent_doc mapping exists, ids remain unchanged.
    """
    chunk_to_doc, _parent_docs = build_id_maps(corpus)
    seen = set()
    out: List[str] = []
    for x in ids:
        doc = chunk_to_doc.get(x, x)
        if doc in seen:
            continue
        seen.add(doc)
        out.append(doc)
    return out


def normalize_relevant_ids_for_doc_eval(relevant_ids: Sequence[str], corpus: Sequence[CorpusDoc]) -> List[str]:
    """
    Ground truth can be either:
      - already doc-level (e.g. "d1")
      - chunk-level (e.g. "d1#c2")

    If a relevant id matches a known parent_doc, keep it.
    Otherwise, try mapping chunk_id -> parent_doc.
    Deduplicate preserving order.
    """
    chunk_to_doc, parent_docs = build_id_maps(corpus)

    mapped: List[str] = []
    for rid in relevant_ids:
        if rid in parent_docs:
            mapped.append(rid)  # already doc-level
        else:
            mapped.append(chunk_to_doc.get(rid, rid))

    seen = set()
    out: List[str] = []
    for x in mapped:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out