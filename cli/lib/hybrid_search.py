import os
import numpy as np

from .inverted_index import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import BM25_K1, BM25_B


def _normalize_scores(scores):
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [1.0] * len(scores)
    return [(score - min_score) / (max_score - min_score) for score in scores]


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, BM25_K1, BM25_B, limit)

    def weighted_search(self, query, alpha, limit=5):
        bm25_results = self._bm25_search(query, limit=limit * 500)
        semantic_results = self.semantic_search.search(query, limit=limit * 500)

        bm25_scores = [score for _, score in bm25_results]
        semantic_scores = [result["score"] for result in semantic_results]

        if not bm25_scores:
            normalized_bm25_scores = []
        else:
            normalized_bm25_scores = _normalize_scores(bm25_scores)

        if not semantic_scores:
            normalized_semantic_scores = []
        else:
            normalized_semantic_scores = _normalize_scores(semantic_scores)

        combined_results = {}
        for i, (doc, _) in enumerate(bm25_results):
            doc_id = doc["id"]
            if doc_id not in combined_results:
                combined_results[doc_id] = {
                    "doc": doc,
                    "bm25_score": 0,
                    "semantic_score": 0,
                }
            combined_results[doc_id]["bm25_score"] = normalized_bm25_scores[i]

        for i, result in enumerate(semantic_results):
            doc_id = result["id"]
            if doc_id not in combined_results:
                combined_results[doc_id] = {
                    "doc": self.semantic_search.document_map[doc_id],
                    "bm25_score": 0,
                    "semantic_score": 0,
                }
            combined_results[doc_id]["semantic_score"] = normalized_semantic_scores[i]

        for doc_id, scores in combined_results.items():
            scores["hybrid_score"] = (1 - alpha) * scores[
                "bm25_score"
            ] + alpha * scores["semantic_score"]

        sorted_results = sorted(
            combined_results.values(), key=lambda x: x["hybrid_score"], reverse=True
        )

        return sorted_results[:limit]
    def rrf_search(self, query, k, limit=10):
        bm25_results = self._bm25_search(query, limit=limit * 500)
        semantic_results = self.semantic_search.search(query, limit=limit * 500)

        combined_results = {}

        for i, (doc, _) in enumerate(bm25_results):
            doc_id = doc["id"]
            if doc_id not in combined_results:
                combined_results[doc_id] = {
                    "doc": doc,
                    "bm25_rank": None,
                    "semantic_rank": None,
                    "rrf_score": 0,
                }
            combined_results[doc_id]["bm25_rank"] = i + 1

        for i, result in enumerate(semantic_results):
            doc_id = result["id"]
            if doc_id not in combined_results:
                combined_results[doc_id] = {
                    "doc": self.semantic_search.document_map[doc_id],
                    "bm25_rank": None,
                    "semantic_rank": None,
                    "rrf_score": 0,
                }
            combined_results[doc_id]["semantic_rank"] = i + 1
        
        from .search_utils import rrf_score

        for doc_id, ranks in combined_results.items():
            score = 0
            if ranks["bm25_rank"] is not None:
                score += rrf_score(ranks["bm25_rank"], k)
            if ranks["semantic_rank"] is not None:
                score += rrf_score(ranks["semantic_rank"], k)
            ranks["rrf_score"] = score
        
        sorted_results = sorted(
            combined_results.values(), key=lambda x: x["rrf_score"], reverse=True
        )

        return sorted_results[:limit]
