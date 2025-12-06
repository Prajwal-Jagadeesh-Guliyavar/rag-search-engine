import os
import pickle
from collections import Counter
from .search_utils import load_movies, tokenize_text, PROJECT_ROOT


class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}

    def __add_document(self, doc_id: int, text: str):
        tokens = tokenize_text(text)
        self.term_frequencies[doc_id] = Counter(tokens)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

    def get_documents(self, term: str) -> list[int]:
        from .search_utils import stemmer # Import stemmer locally to avoid circular dependency
        term = stemmer.stem(term.lower())
        if term not in self.index:
            return []
        
        return sorted(list(self.index[term]))

    def get_tf(self, doc_id: int, term: str) -> int:
        tokens = tokenize_text(term)
        if len(tokens) > 1:
            raise ValueError("More than one token in term")

        if not tokens:
            return 0
        
        token = tokens[0]
        if doc_id not in self.term_frequencies:
            return 0
        
        return self.term_frequencies[doc_id].get(token, 0)

    def get_bm25_idf(self, term: str) -> float:
        import math
        tokens = tokenize_text(term)
        if len(tokens) > 1:
            raise ValueError("More than one token in term")
        
        if not tokens:
            return 0.0

        token = tokens[0]
        N = len(self.docmap)
        df = len(self.index.get(token, []))
        
        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def get_bm25_tf(self, doc_id: int, term: str, k1: float) -> float:
        tf = self.get_tf(doc_id, term)
        return (tf * (k1 + 1)) / (tf + k1)

    def build(self):
        movies = load_movies()
        for movie in movies:
            self.docmap[movie["id"]] = movie
            text_to_index = f"{movie['title']} {movie['description']}"
            self.__add_document(movie["id"], text_to_index)

    def save(self):
        cache_dir = os.path.join(PROJECT_ROOT, "cache")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        index_path = os.path.join(cache_dir, "index.pkl")
        with open(index_path, "wb") as f:
            pickle.dump(self.index, f)

        docmap_path = os.path.join(cache_dir, "docmap.pkl")
        with open(docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)

        term_frequencies_path = os.path.join(cache_dir, "term_frequencies.pkl")
        with open(term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self):
        cache_dir = os.path.join(PROJECT_ROOT, "cache")
        index_path = os.path.join(cache_dir, "index.pkl")
        docmap_path = os.path.join(cache_dir, "docmap.pkl")
        term_frequencies_path = os.path.join(cache_dir, "term_frequencies.pkl")

        try:
            with open(index_path, "rb") as f:
                self.index = pickle.load(f)

            with open(docmap_path, "rb") as f:
                self.docmap = pickle.load(f)

            with open(term_frequencies_path, "rb") as f:
                self.term_frequencies = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                "Index files not found. Please run the build command."
            )
