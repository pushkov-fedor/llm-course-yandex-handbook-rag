import os

from dotenv import load_dotenv

load_dotenv()

EMBEDDER_MODEL = "intfloat/multilingual-e5-base"
INDEX_FILE = "index.faiss"
LLM_MODEL = "mistral-medium-latest"
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")