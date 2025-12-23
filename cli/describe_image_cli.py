import argparse
import mimetypes
import os

# Updated import for google.genai (recommended package)
import google.genai as genai
# Correct import for Part in the new package
from google.genai.types import Part
from dotenv import load_dotenv # Import dotenv

# Load environment variables from .env file
load_dotenv()

# The google.genai library automatically picks up the API key from the GOOGLE_API_KEY
# environment variable. We ensure GEMINI_API_KEY is set as GOOGLE_API_KEY.
gemini_api_key = os.getenv("GEMINI_API_KEY")
if gemini_api_key:
    os.environ["GOOGLE_API_KEY"] = gemini_api_key
else:
    print("Warning: GEMINI_API_KEY not found in .env file or environment variables.")
    print("Please ensure GOOGLE_API_KEY is set for the library to work.")

def main():
    parser = argparse.ArgumentParser(description="Rewrite text query based on image content using Gemini.")
    parser.add_argument("--image", required=True, help="Path to the image file.")
    parser.add_argument("--query", required=True, help="Text query to rewrite.")
    args = parser.parse_args()

    # --- Image Handling ---
    if not os.path.exists(args.image):
        print(f"Error: Image file not found at {args.image}")
        return

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    try:
        with open(args.image, "rb") as f:
            img_data = f.read()
    except IOError as e:
        print(f"Error reading image file: {e}")
        return

    # --- Gemini Setup ---
    # Use a currently supported multimodal model name.
    # 'gemini-pro-vision' has been deprecated.
    # 'gemini-2.5-flash' is a supported model based on recent lists and previous success.
    model_name_to_use = "gemini-2.5-flash"

    print(f"Initializing client and preparing to use model: {model_name_to_use}...")
    try:
        # Instantiate the client. The API key is expected to be automatically read from
        # the GOOGLE_API_KEY environment variable by the google.genai library.
        client = genai.Client()
        print("Google GenAI client initialized successfully.")

    except Exception as e:
        print(f"\nAn error occurred during client initialization: {e}")
        print("Please ensure:")
        print("1. The GEMINI_API_KEY is correctly set in your .env file, and it has been loaded.")
        print("2. The 'google-genai' and 'python-dotenv' libraries are installed (`pip install google-genai python-dotenv`).")
        print("3. The API key is valid and the library can connect to the Gemini service.")
        return # Exit main function if client initialization fails


    # --- Gemini System Prompt and Request ---
    system_prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary"""

    # --- Build Request Parts ---
    parts = [
        system_prompt,
        # Use Part.from_bytes for multimodal input
        Part.from_bytes(data=img_data, mime_type=mime),
        args.query.strip(),
    ]

    # --- Send Request and Print Output ---
    try:
        # Use the client to generate content, passing the model name and contents directly.
        # The client.models.generate_content method is used here.
        response = client.models.generate_content(
            model=model_name_to_use,
            contents=parts
        )

        print(f"\nRewritten query: {response.text.strip()}")
        if response.usage_metadata is not None:
            print(f"Total tokens:    {response.usage_metadata.total_token_count}")
        else:
            print("Total tokens:    N/A (metadata not available)")

    except Exception as e:
        print(f"\nAn error occurred during Gemini API call: {e}")

if __name__ == "__main__":
    main()