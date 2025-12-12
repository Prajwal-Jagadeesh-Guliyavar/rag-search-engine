import json
import os
import string
from nltk.stem import PorterStemmer

DEFAULT_SEARCH_LIMIT = 5
BM25_K1 = 1.5
BM25_B = 0.75

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOP_WORD_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")


def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]


def load_stopwords() -> set[str]:
    with open(STOP_WORD_PATH, "r") as f:
        return set(f.read().splitlines())


STOP_WORDS = load_stopwords()
stemmer = PorterStemmer()


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token and token not in STOP_WORDS:
            stemmed_token = stemmer.stem(token)
            valid_tokens.append(stemmed_token)
    return valid_tokens


def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)