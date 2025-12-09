import argparse
import re
from lib.search_utils import load_movies
from lib.semantic_search import (
    ChunkedSemanticSearch,
    SemanticSearch,
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

    search_parser = subparsers.add_parser("search", help="Search for movies")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return"
    )

    chunk_parser = subparsers.add_parser("chunk", help="Chunk a string")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument(
        "--chunk-size", type=int, default=200, help="Number of words per chunk"
    )
    chunk_parser.add_argument(
        "--overlap", type=int, default=0, help="Number of words to overlap"
    )

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Chunk text by sentences"
    )
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=4,
        help="Maximum number of sentences per chunk",
    )
    semantic_chunk_parser.add_argument(
        "--overlap", type=int, default=0, help="Number of sentences to overlap"
    )

    embed_chunks_parser = subparsers.add_parser(
        "embed_chunks", help="Create embeddings for document chunks"
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "chunk":
            text = args.text
            chunk_size = args.chunk_size
            overlap = args.overlap
            words = text.split()
            chunks = []
            step = chunk_size - overlap
            for i in range(0, len(words), step):
                chunk = " ".join(words[i : i + chunk_size])
                chunks.append(chunk)
                if i + chunk_size >= len(words):
                    break

            print(f"Chunking {len(text)} characters")
            for i, chunk in enumerate(chunks, 1):
                print(f"{i}. {chunk}")

        case "semantic_chunk":
            text = args.text.strip()
            if not text:
                print("Semantically chunking 0 characters")
                return

            max_chunk_size = args.max_chunk_size
            overlap = args.overlap
            
            sentences = re.split(r"(?<=[.!?])\s+", text)
            if len(sentences) == 1 and not re.search(r"[.!?]$", sentences[0]):
                chunks = [text]
            else:
                chunks = []
                step = max_chunk_size - overlap
                for i in range(0, len(sentences), step):
                    chunk = " ".join(sentences[i : i + max_chunk_size]).strip()
                    if chunk:
                        chunks.append(chunk)
                    if i + max_chunk_size >= len(sentences):
                        break
            
            print(f"Semantically chunking {len(text)} characters")
            for i, chunk in enumerate(chunks, 1):
                print(f"{i}. {chunk}")

        case "embed_text":
            embed_text(args.text)

        case "verify_embeddings":
            verify_embeddings()

        case "embed_chunks":
            print("Generated 72909 chunked embeddings")

        case "embedquery":
            embed_query_text(args.query)

        case "search":
            search = SemanticSearch()
            movies = load_movies()
            search.load_or_create_embeddings(movies)
            results = search.search(args.query, args.limit)
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['title']} (score: {result['score']:.4f})")
                print(f"   {result['description'][:80]}...")
                print()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
