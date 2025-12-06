import argparse
import math

from lib.keyword_search import search_command
from lib.inverted_index import InvertedIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build the inverted index")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to get frequency for")

    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency")
    idf_parser.add_argument("term", type=str, help="Term to get IDF for")

    args = parser.parse_args()

    match args.command:
        case "search":
            query = args.query
            print(f"Searching for: {query}")

            try:
                index = InvertedIndex()
                index.load()
            except FileNotFoundError as e:
                print(e)
                return

            results = search_command(query, index)

            for i, res in enumerate(results, 1):
                print(f"{i}. [ID: {res['id']}] {res['title']}")

        case "build":
            print("Building inverted index...")
            index = InvertedIndex()
            index.build()
            index.save()
            print("Inverted index built and saved.")

        case "tf":
            doc_id = args.doc_id
            term = args.term

            try:
                index = InvertedIndex()
                index.load()
            except FileNotFoundError as e:
                print(e)
                return

            try:
                tf = index.get_tf(doc_id, term)
                print(tf)
            except ValueError as e:
                print(e)

        case "idf":
            term = args.term

            try:
                index = InvertedIndex()
                index.load()
            except FileNotFoundError as e:
                print(e)
                return

            total_doc_count = len(index.docmap)
            term_match_doc_count = len(index.get_documents(term))
            print(f"Total doc count: {total_doc_count}")
            print(f"Term '{term}' match doc count: {term_match_doc_count}")
            idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
