import os
from typing import Optional
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")
model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash") # Added a common fallback model name

if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

genai.configure(api_key=api_key)

try:
    model = genai.GenerativeModel(model_name)
except Exception as e:
    print(f"Error initializing model '{model_name}': {e}")
    raise e


def spell_correct(query: str) -> str:
    """
    Fixes spelling errors in a movie search query using the Gemini model.
    """
    prompt = f"""Fix any spelling errors in this movie search query.

Only correct obvious typos. Don't change correctly spelled words.

Query: "{query}"

If no errors, return the original query.
Corrected:"""

    try:
        response = model.generate_content(prompt)

        corrected = (response.text or "").strip().strip('"')

        if corrected:
            return corrected
        else:
            return query
        
    except Exception as e:
        print(f"An error occurred during spell correction: {e}")
        return query


def enhance_query(query: str, method: Optional[str] = None) -> str:
    match method:
        case "spell":
            return spell_correct(query)
        case _:
            return query
