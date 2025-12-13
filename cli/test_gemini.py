import os

from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-flash-lite")

if __name__ == "__main__":
    prompt = "Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum."
    response = model.generate_content(contents=prompt)
    assert response.usage_metadata is not None
    print(f"Prompt Tokens: {response.usage_metadata.prompt_token_count}")
    print(f"Response Tokens: {response.usage_metadata.candidates_token_count}")
    print(response.text)