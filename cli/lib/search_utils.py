import json
import os

DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOP_WORDS_PATH = os.path.join(PROJECT_ROOT,"data","stopwords.txt")


def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]


def load_stopwords()->set:
    words_set = set()
    with open(STOP_WORDS_PATH, "r") as f:
        content = f.read()
        for word in content.splitlines():
            stripped = word.strip()
            if stripped:
                words_set.add(stripped)
    return words_set