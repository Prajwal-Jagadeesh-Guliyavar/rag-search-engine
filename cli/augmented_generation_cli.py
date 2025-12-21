import argparse
import json
import random

# Mock function to simulate Gemini API call for RAG response generation
def call_gemini_api(prompt: str) -> str:
    """
    Simulates a call to the Gemini API to generate an answer based on the prompt.
    This is a placeholder and would be replaced by an actual API call in a real application.
    """
    print("Calling Gemini API (simulated)...")
    # Basic simulation: extract query and document titles for a somewhat context-aware response.
    # In a real scenario, the LLM would process the full prompt.

    # A very simple mock response generator
    if "dinosaurs" in prompt.lower():
        return "Hoopla has several exciting movies featuring dinosaurs that action fans will love! Classics like 'Jurassic Park' offer thrilling adventures with realistic dinosaur encounters, while films like 'The Good Dinosaur' provide a more heartwarming, family-friendly take on prehistoric creatures. For those who enjoy action mixed with prehistoric settings, exploring titles that combine these elements can lead to a great viewing experience on Hoopla."
    elif "action" in prompt.lower():
        return "Hoopla offers a wide selection of action movies! You can find everything from high-octane thrillers to epic adventures. Browse our action category to discover new releases and timeless classics that will keep you on the edge of your seat."
    else:
        return "Hoopla has a great selection of movies to suit your needs. Please check our catalog for more details!"

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # New parser for summarize command
    summarize_parser = subparsers.add_parser(
        "summarize", help="Synthesize search results into a summary"
    )
    summarize_parser.add_argument("query", type=str, help="Search query for summarization")
    summarize_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of search results to consider for summarization (default: 5)",
    )

    # Existing rag parser
    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    args = parser.parse_args()

    match args.command:
        # New case for summarize command
        case "summarize":
            query = args.query
            limit = args.limit
            print(f"Summarize Query: {query}, Limit: {limit}\n")

            # 1. Load movies data
            from lib.search_utils import load_movies
            from lib.hybrid_search import HybridSearch
            
            all_movies_data = load_movies()
            search_instance = HybridSearch(all_movies_data)

            # 2. Perform RRF search
            print("Performing RRF search...")
            # Pass the limit argument to rrf_search
            rrf_results = search_instance.rrf_search(query, k=60, limit=limit) 
            
            # Format search results for printing
            search_results_output = []
            for i, result in enumerate(rrf_results):
                title = result.get("doc", {}).get("title", "N/A")
                search_results_output.append(f"  - {title}")

            # Construct the specific prompt for summarization
            results_string = "\n".join(search_results_output)
            
            # The prompt provided by the user:
            summarize_prompt_template = f"""
Provide information useful to this query by synthesizing information from multiple search results in detail.
The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
This should be tailored to Hoopla users. Hoopla is a movie streaming service.
Query: {query}
Search Results:
{results_string}
Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
"""

            # Call the simulated Gemini API
            summary_response = call_gemini_api(summarize_prompt_template)

            # 3. Print the results in the specified format
            print("Search Results:")
            for line in search_results_output:
                print(line)

            print("\nLLM Summary:")
            print(summary_response)

        # Existing case for rag command
        case "rag":
            query = args.query
            print(f"RAG Query: {query}\n")

            # 1. Load movies data
            from lib.search_utils import load_movies
            from lib.hybrid_search import HybridSearch
            
            all_movies_data = load_movies()
            search_instance = HybridSearch(all_movies_data)

            # 2. Perform RRF search for top 5 results
            print("Performing RRF search...")
            # Assuming rrf_search returns results in a format similar to HybridSearch.weighted_search
            # Each result is expected to be a dictionary with a 'doc' key containing 'title' and 'description'.
            # We limit to top 5 for the RAG context.
            rrf_results = search_instance.rrf_search(query, k=60, limit=5) 
            
            # Format search results for printing and for LLM prompt
            search_results_output = []
            docs_for_llm = []
            for i, result in enumerate(rrf_results):
                title = result.get("doc", {}).get("title", "N/A")
                search_results_output.append(f"  - {title}")
                
                # Format document for LLM prompt
                description = result.get("doc", {}).get("description", "")
                docs_for_llm.append(f"Document {i+1}: {title}\nDescription: {description[:200]}...\n") # Truncate description

            print("Search Results:")
            for line in search_results_output:
                print(line)

            docs = "\n".join(docs_for_llm)

            # 3. Prompt Gemini API with query, search results, and instructions
            # Note: This is a simulated call to Gemini API.
            system_prompt_template = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Documents:
{docs}

Provide a comprehensive answer that addresses the query:"""

            # Call the simulated Gemini API
            rag_response = call_gemini_api(system_prompt_template)

            # 4. Print the RAG response
            print("\nRAG Response:")
            print(rag_response)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
