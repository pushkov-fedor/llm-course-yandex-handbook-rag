import os

from dotenv import load_dotenv

load_dotenv()

EMBEDDER_MODEL = "BAAI/bge-m3"
INDEX_FILE = "index.faiss"
LLM_MODEL = "mistralai/mistral-small-3.1-24b-instruct"
JUDGE_MODEL = "deepseek/deepseek-v3.2"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"