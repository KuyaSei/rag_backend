# # ==================================
# # Ask Groq LLM
# # ==================================
from groq import Groq
import os
from pathlib import Path
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")

def ask_llm(prompt: str):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    response = client.chat.completions.create(
        model='llama-3.3-70b-versatile', #llama-3.1-8-instant
        messages=[
            {"role": "system", "content": "你是一個專業營養助理。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
    )

    return response.choices[0].message.content.strip()
