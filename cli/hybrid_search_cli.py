import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser(
        "normalize", help="Normalize a list of scores using min-max normalization"
    )
    normalize_parser.add_argument(
        "scores", type=float, nargs="*", default=[], help="List of scores to normalize"
    )

    weighted_search_parser = subparsers.add_parser(
        "weighted-search", help="Perform a weighted hybrid search"
    )
    weighted_search_parser.add_argument("query", type=str, help="Search query")
    weighted_search_parser.add_argument(
        "--alpha", type=float, default=0.5, help="Weight for semantic search"
    )
    weighted_search_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return"
    )

    rrf_search_parser = subparsers.add_parser(
        "rrf-search", help="Perform a hybrid search with Reciprocal Rank Fusion"
    )
    rrf_search_parser.add_argument("query", type=str, help="Search query")
    rrf_search_parser.add_argument(
        "-k", type=int, default=60, help="Ranking constant for RRF"
    )
    rrf_search_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return"
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            scores = args.scores
            if not scores:
                return

            min_score = min(scores)
            max_score = max(scores)

            if min_score == max_score:
                for _ in scores:
                    print(f"* {1.0:.4f}")
            else:
                normalized_scores = [
                    (score - min_score) / (max_score - min_score) for score in scores
                ]
                for score in normalized_scores:
                    print(f"* {score:.4f}")

        case "weighted-search":
            from lib.hybrid_search import HybridSearch
            from lib.search_utils import load_movies

            search = HybridSearch(load_movies())
            results = search.weighted_search(args.query, args.alpha, args.limit)
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['doc']['title']}")
                print(f"   Hybrid Score: {result['hybrid_score']:.3f}")
                print(
                    f"   BM25: {result['bm25_score']:.3f}, Semantic: {result['semantic_score']:.3f}"
                )
                print(f"   {result['doc']['description'][:80]}...")
        
        case "rrf-search":
            from lib.hybrid_search import HybridSearch
            from lib.search_utils import load_movies

            search = HybridSearch(load_movies())
            results = search.rrf_search(args.query, args.k, args.limit)
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['doc']['title']}")
                print(f"   RRF Score: {result['rrf_score']:.3f}")
                bm25_rank = result['bm25_rank'] if result['bm25_rank'] is not None else 'N/A'
                semantic_rank = result['semantic_rank'] if result['semantic_rank'] is not None else 'N/A'
                print(f"   BM25 Rank: {bm25_rank}, Semantic Rank: {semantic_rank}")
                print(f"   {result['doc']['description'][:80]}...")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
