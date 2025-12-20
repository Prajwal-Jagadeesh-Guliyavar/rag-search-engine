import os
import json
from time import sleep
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import CrossEncoder

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

model = genai.GenerativeModel(model_name)

cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")

def llm_rerank_individual(query: str, documents: list[dict], limit: int)-> list[dict]:
    scored_docs = []

    for doc in documents:
        prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("document", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Give me ONLY the number in your response, no other text or explanation.

Score:"""
        response = model.generate_content(prompt)
        score_text = (response.text or "").strip()
        score = int(score_text)
        scored_docs.append({**doc, "individual_score": score})
        sleep(3)

    scored_docs.sort(key=lambda x:x["individual_score"], reverse=True)
    return scored_docs[:limit]


def llm_rank_batch(query: str, documnets: list[dict], limit: int=5) -> list[dict]:
    if not documnets:
        return []

    doc_map = {}
    doc_list = []

    for doc in documnets:
        doc_id = doc["doc_id"]
        doc_map[doc_id] = doc
        doc_list.append(f"{doc_id}: {doc.get('title', '')} - {doc.get('document', '')[:200]}")

    doc_list_str = "\n".join(doc_list)

    prompt = f"""Rank these movies by relevance to the search query.

Query: "{query}"

Movies:
{doc_list_str}

Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

[75, 12, 34, 2, 1]
"""

    response = model.generate_content(prompt)
    ranking_text = (response.text or "").strip()

    parsed_ids = json.load(ranking_text)
    reranked = []

    for i, doc_id in enumerate(parsed_ids):
        if doc_id in doc_map:
            reranked.append({**doc_map[doc_id], "batch_rank": i + 1})


def cross_encoder_rerank(query: str, documents: list[dict], limit: int = 5) -> list[dict]:
    pairs = []
    for doc in documents:
        pairs.append([query, f"{doc.get('title', '')} - {doc.get('document', '')}"])

    scores = cross_encoder.predict(pairs)

    for doc, score in zip(documents, scores):
        doc["crossencoder_score"] = float(score)

    documents.sort(key=lambda x: x["crossencoder_score"], reverse=True)
    return documents[:limit]


def rerank(query: str, documents: list[dict], method: str="batch", limit:int=5) -> list[dict]:
    if method == "individual":
        return llm_rerank_individual(query, documents, limit)
    if method == "batch":
        return llm_rank_batch(query, documents, limit)
    if method == "cross_encoder":
        return cross_encoder_rerank(query, documents, limit)
    else :
        return documents[:limit]