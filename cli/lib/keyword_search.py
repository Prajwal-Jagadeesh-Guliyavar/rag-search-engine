from .search_utils import DEFAULT_SEARCH_LIMIT, tokenize_text
from .inverted_index import InvertedIndex


def search_command(
    query: str, index: InvertedIndex, limit: int = DEFAULT_SEARCH_LIMIT
) -> list[dict]:
    query_tokens = tokenize_text(query)
    if not query_tokens:
        return []

    results = set()
    for token in query_tokens:
        doc_ids = index.get_documents(token)
        for doc_id in doc_ids:
            results.add(doc_id)
            if len(results) >= limit:
                break
        if len(results) >= limit:
            break
    
    return [index.docmap[doc_id] for doc_id in sorted(list(results))][:limit]


def bm25_idf_command(term: str) -> float:
    try:
        index = InvertedIndex()
        index.load()
    except FileNotFoundError as e:
        print(e)
        return 0.0

    return index.get_bm25_idf(term)


def bm25_tf_command(doc_id: int, term: str, k1: float) -> float:
    try:
        index = InvertedIndex()
        index.load()
    except FileNotFoundError as e:
        print(e)
        return 0.0

    return index.get_bm25_tf(doc_id, term, k1)