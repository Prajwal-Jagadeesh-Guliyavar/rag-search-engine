import json
import os
import string
from nltk.stem import PorterStemmer
from dotenv import load_dotenv
import google.generativeai as genai

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


def enhance_query_with_gemini(query: str, method: str) -> str:
    """
    Enhance movie search query using Gemini API based on the specified method.

    Args:
        query: Original search query
        method: Enhancement method ('spell' or 'rewrite')

    Returns:
        Enhanced query
    """
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    genai.configure(api_key=api_key)

    model_name = os.environ.get("GEMINI_MODEL", "gemini-pro")
    model = genai.GenerativeModel(model_name)

    if method == "rewrite":
        prompt = f"""Rewrite this movie search query to be more specific and searchable.

Original: "{query}"

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep it concise (under 10 words)
- It should be a google style search query that's very specific
- Don't use boolean logic

Examples:

- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

Rewritten query:"""
    elif method == "spell":
        prompt = f"""Fix any spelling errors in this movie search query.
Only correct obvious typos. Don't change correctly spelled words.
Query: "{query}"
If no errors, return the original query.
Corrected:"""
    else:
        # If an unknown method is provided, return the original query
        return query

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return query # Return original query in case of API error