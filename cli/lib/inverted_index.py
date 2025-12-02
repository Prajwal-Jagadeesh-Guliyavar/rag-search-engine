import os
import pickle
from .search_utils import load_movies, tokenize_text, PROJECT_ROOT


class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}

    def __add_document(self, doc_id: int, text: str):
        tokens = tokenize_text(text)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

    def get_documents(self, term: str) -> list[int]:
        term = term.lower()
        if term not in self.index:
            return []
        
        return sorted(list(self.index[term]))

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

    def load(self):
        cache_dir = os.path.join(PROJECT_ROOT, "cache")
        index_path = os.path.join(cache_dir, "index.pkl")
        docmap_path = os.path.join(cache_dir, "docmap.pkl")

        try:
            with open(index_path, "rb") as f:
                self.index = pickle.load(f)

            with open(docmap_path, "rb") as f:
                self.docmap = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                "Index files not found. Please run the build command."
            )
