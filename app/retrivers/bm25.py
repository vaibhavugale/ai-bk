from typing import List, Any
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

class BM25:
    @classmethod
    def from_documents(cls, docs: List[Document], k: int):
        return BM25Retriever.from_documents(docs,k=k)
    