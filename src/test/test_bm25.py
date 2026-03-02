from rag_eval.io import CorpusDoc
from rag_eval.retrievers import BM25Retriever

def test_bm25_simple_ranking():
    corpus = [
        CorpusDoc(doc_id="d1", text="apple banana"),
        CorpusDoc(doc_id="d2", text="apple apple apple"),
        CorpusDoc(doc_id="d3", text="banana orange"),
    ]

    retriever = BM25Retriever(corpus)
    results = retriever.retrieve("apple", top_k=3)

    ranked_ids = [doc_id for doc_id, _ in results]

    # d2 should rank first because of higher tf
    assert ranked_ids[0] == "d2"
    assert "d1" in ranked_ids