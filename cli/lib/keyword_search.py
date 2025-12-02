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