import string
from nltk.stem import PorterStemmer
from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies, load_stopwords

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


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies() or []
    results: list[dict] = []

    query_tokens = tokenize_text(query)

    if not query_tokens:
        return []

    for movie in movies:
        title = movie.get("title", "")
        title_tokens = tokenize_text(title)

        if has_matching_token(query_tokens, title_tokens):
            results.append(movie)

    results.sort(key=lambda m: m.get("id", 0))
    return results[:limit]