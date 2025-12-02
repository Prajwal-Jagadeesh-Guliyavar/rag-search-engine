import argparse

from lib.keyword_search import search_command
from lib.inverted_index import InvertedIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build the inverted index")

    args = parser.parse_args()

    match args.command:
        case "search":
            query = args.query
            print(f"Searching for: {query}")

            results = search_command(query)

            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']}")
        
        case "build":
            print("Building inverted index...")
            index = InvertedIndex()
            index.build()
            index.save()
            print("Inverted index built and saved.")
            docs = index.get_documents("merida")
            if docs:
                print(f"First document for token 'merida' = {docs[0]}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
