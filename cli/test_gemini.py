import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
print(f"Using key {api_key[:6]}...")

genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-2.5-flash-lite")

response = model.generate_content(
    "Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum."
)

print(response.text)

print(f"Prompt Tokens: {response.usage_metadata.prompt_token_count}")
print(f"Response Tokens: {response.usage_metadata.candidates_token_count}")

