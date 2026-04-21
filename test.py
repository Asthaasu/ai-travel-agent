from dotenv import load_dotenv
import os

load_dotenv()

print("OpenAI:", os.getenv("OPENAI_API_KEY"))
