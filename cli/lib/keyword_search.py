import string

from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies

###################################################################################################


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


###################################################################################################


def tokenize(text: str) -> list[dict]:
    temp = []
    for t in text.split():
        if t:
            temp.append(t)
    return temp


###################################################################################################


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies() or []
    results = []

    processed_query = preprocess_text(query)
    query_tokens = tokenize(processed_query)
    if not query_tokens:
        return []

    for movie in movies:
        title = movie.get("title", "")
        processed_title = preprocess_text(title)
        title_tokens = tokenize(processed_title)

        match_found = False
        for q_tok in query_tokens:
            for t_tok in title_tokens:
                if q_tok in t_tok:
                    match_found = True
                    break
            if match_found:
                break

        if match_found:
            results.append(movie)
            if len(results) >= limit:
                break

    results.sort(key=lambda m: m.get("id", 0))
    return results[:limit]


###################################################################################################
