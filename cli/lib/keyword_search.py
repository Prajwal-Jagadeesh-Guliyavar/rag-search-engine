import string
import unicodedata

from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies

###################################################################################################


def preprocess_text(text: str) -> str:
    if text is None:
        return ""

    text = text.lower()
    cleaned = []

    for ch in text:
        if not unicodedata.category(ch).startswith("P"):
            cleaned.append(ch)
    return "".join(cleaned)


###################################################################################################


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []

    preprocessed_query = preprocess_text(query)

    for movie in movies:

        preprocessed_title = preprocess_text(movie["title"])
        if preprocessed_query in preprocessed_title:
            results.append(movie)

            if len(results) >= limit:
                break

    return results


###################################################################################################
