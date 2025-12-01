import argparse
import json
from pathlib import Path


def load_movies():
    data_path = Path(__file__).parent.parent / "data" / "movies.json"
    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["movies"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
