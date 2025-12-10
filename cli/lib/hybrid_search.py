import os

from .inverted_index import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import BM25_K1, BM25_B


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        # I cannot get the index_path from the InvertedIndex class as it is not an attribute
        # I will assume the path is "cache/index.pkl"
        if not os.path.exists("cache/index.pkl"):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, BM25_K1, BM25_B, limit)

    def weighted_search(self, query, alpha, limit=5):
        raise NotImplementedError("Weighted hybrid search is not implemented yet.")

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet. ")
