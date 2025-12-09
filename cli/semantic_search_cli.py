import argparse

from lib.semantic_search import (
    search_chunked_command,
    embed_chunks_command,
    chunk_text,
    semantic_search,
    semantic_chunk_text,
    embed_query_text,
    verify_model,
    embed_text,
    verify_embeddings
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify that the embedding model is loaded")

    #embedding text
    single_embed_parser = subparsers.add_parser(
        "embed_text", help="Generate an embedding for a single text"
    )
    single_embed_parser.add_argument("text", type=str, help="Text to embed")

    #verify embeddings
    subparsers.add_parser(
        "verify_embeddings", help="Verify embeddings for the movie dataset"
    )

    #query embeddings
    embed_query_parser = subparsers.add_parser(
        "embedquery", help="Generate an embedding for a search query"
    )
    embed_query_parser.add_argument("query", type=str, help="Query to embed")

    #Semantic Search
    search_parser = subparsers.add_parser(
        "search", help="Search for movies using semantic search"
    )
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return"
    )

    #Fixed size Chunking
    chunk_parser = subparsers.add_parser(
        "chunk", help="Split text into fixed-size chunks with optional overlap"
    )
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument(
        "--chunk-size", type=int, default=200, help="Size of each chunk in words"
    )
    #Overlap in Fixed size Chunking
    chunk_parser.add_argument(
        "--overlap", type=int, default=0, help="Number of words to overlap between chunks",
    )

    #semantic chunks
    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Split text on sentence boundaries to preserve meaning"
    )
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size",
        type=int, default=4, help="Maximum size of each chunk in sentences",
    )
    semantic_chunk_parser.add_argument(
        "--overlap", type=int, default=0, help="Number of sentences to overlap between chunks",
    )

    #embedding semantic chunks
    subparsers.add_parser(
        "embed_chunks", help="Generate embeddings for chunked documents"
    )
    
    #search semantic chunks
    search_chunked_parser = subparsers.add_parser(
        "search_chunked", help="Search using chunked embeddings"
    )
    search_chunked_parser.add_argument("query", type=str, help="Search query")
    search_chunked_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return"
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "search_chunked":
            from lib.search_utils import load_movies
            from lib.semantic_search import ChunkedSemanticSearch
            search = ChunkedSemanticSearch()
            movies = load_movies()
            search.load_or_create_chunk_embeddings(movies)
            results = search.search_chunks(args.query, args.limit)
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result['title']} (score: {result['score']:.4f})")
                print(f"   {result['document']}...")

        case "embed_text":
            embed_text(args.text)

        case "verify_embeddings":
            verify_embeddings()

        case "embedquery":
            embed_query_text(args.query)

        case "search":
            semantic_search(args.query, args.limit)

        case "chunk":
            chunk_text(args.text, args.chunk_size, args.overlap)

        case "semantic_chunk":
            semantic_chunk_text(args.text, args.max_chunk_size, args.overlap)

        case "embed_chunks":
            embeddings = embed_chunks_command()
            print(f"Generated {len(embeddings)} chunked embeddings")

        case "search_chunked":
            result = search_chunked_command(args.query, args.limit)
            print(f"Query: {result['query']}")
            print("Results:")
            for i, res in enumerate(result["results"], 1):
                print(f"\n{i}. {res['title']} (score: {res['score']:.4f})")
                print(f"   {res['document']}...")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
