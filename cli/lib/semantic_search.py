import json
import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from .search_utils import load_movies, PROJECT_ROOT


class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text: str):
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty.")
        return self.model.encode([text])[0]

    def build_embeddings(self, documents):
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        movie_strings = [f"{doc['title']}: {doc['description']}" for doc in documents]
        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)

        cache_dir = os.path.join(PROJECT_ROOT, "cache")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        np.save(os.path.join(cache_dir, "movie_embeddings.npy"), self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        embedding_path = os.path.join(PROJECT_ROOT, "cache", "movie_embeddings.npy")
        if os.path.exists(embedding_path):
            self.embeddings = np.load(embedding_path)
            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)

    def search(self, query: str, limit: int = 5):
        if self.embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)
        similarities = [
            (cosine_similarity(query_embedding, doc_embedding), doc)
            for doc, doc_embedding in zip(self.documents, self.embeddings)
        ]

        sorted_similarities = sorted(similarities, key=lambda x: x[0], reverse=True)

        results = []
        for score, doc in sorted_similarities[:limit]:
            results.append(
                {
                    "score": score,
                    "title": doc["title"],
                    "description": doc["description"],
                }
            )
        return results


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        for i, doc in enumerate(documents):
            self.document_map[doc["id"]] = doc

        all_chunks = []
        chunk_metadata = []

        for i, doc in enumerate(self.documents):
            if not doc["description"]:
                continue
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", doc["description"]) if s.strip()]
            doc_chunks = []
            max_chunk_size = 4
            overlap = 1
            step = max_chunk_size - overlap
            for j in range(0, len(sentences), step):
                chunk = " ".join(sentences[j : j + max_chunk_size])
                if chunk:
                    doc_chunks.append(chunk)

            for j, chunk in enumerate(doc_chunks):
                all_chunks.append(chunk)
                chunk_metadata.append(
                    {
                        "movie_idx": i,
                        "chunk_idx": j,
                        "total_chunks": len(doc_chunks),
                    }
                )

        self.chunk_embeddings = self.model.encode(
            all_chunks, show_progress_bar=True
        )
        self.chunk_metadata = chunk_metadata

        cache_dir = os.path.join(PROJECT_ROOT, "cache")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        np.save(os.path.join(cache_dir, "chunk_embeddings.npy"), self.chunk_embeddings)

        with open(os.path.join(cache_dir, "chunk_metadata.json"), "w") as f:
            json.dump(
                {"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2
            )

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        embedding_path = os.path.join(PROJECT_ROOT, "cache", "chunk_embeddings.npy")
        metadata_path = os.path.join(PROJECT_ROOT, "cache", "chunk_metadata.json")

        if os.path.exists(embedding_path) and os.path.exists(metadata_path):
            self.chunk_embeddings = np.load(embedding_path)
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                self.chunk_metadata = metadata["chunks"]
            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def verify_model():
    search = SemanticSearch()
    model = search.model
    print(f"Model loaded: {model}")
    print(f"Max sequence length: {model.max_seq_length}")


def embed_text(text: str):
    search = SemanticSearch()
    embedding = search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    search = SemanticSearch()
    documents = load_movies()
    embeddings = search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")


def embed_query_text(query: str):
    search = SemanticSearch()
    embedding = search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")
