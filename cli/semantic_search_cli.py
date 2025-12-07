import argparse
from lib.semantic_search import (
    embed_query_text,
    embed_text,
    verify_embeddings,
    verify_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser(
        "verify", help="Verify the semantic search model"
    )

    embed_text_parser = subparsers.add_parser("embed_text", help="Embed a string")
    embed_text_parser.add_argument("text", type=str, help="Text to embed")

    verify_embeddings_parser = subparsers.add_parser(
        "verify_embeddings", help="Verify the movie embeddings"
    )

    embed_query_parser = subparsers.add_parser("embedquery", help="Embed a query")
    embed_query_parser.add_argument("query", type=str, help="Query to embed")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()

        case "embed_text":
            embed_text(args.text)

        case "verify_embeddings":
            verify_embeddings()

        case "embedquery":
            embed_query_text(args.query)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
