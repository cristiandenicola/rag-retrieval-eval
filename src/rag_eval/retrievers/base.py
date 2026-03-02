from __future__ import annotations
from typing import Protocol, List, Tuple

class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        ...