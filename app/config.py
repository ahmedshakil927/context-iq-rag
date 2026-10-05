import os

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.1:8b"
# Optional hosted provider (off by default). Llama 3.1 8B is not offered on Groq any more, so a hosted
# run uses a different model and its eval numbers are not comparable with the local ones.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")   # "ollama" | "groq"
GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_URL = "https://api.groq.com/openai/v1"
ACTIVE_MODEL = GROQ_MODEL if LLM_PROVIDER == "groq" else LLM_MODEL
OLLAMA_URL = "http://localhost:11434"
CHUNK_SIZE = 800      # characters per chunk
CHUNK_OVERLAP = 150
TOP_K = 8
MIN_SCORE = 0.45      # best vector similarity below this -> "I don't know"
KEYWORD_GATE = True    # also accept a question whose key words appear in one chunk
KW_COVER = 0.5         # share of the question's content words that must appear in a single chunk
SEARCH_MODE = "hybrid"  # "vector" or "hybrid" (vector + BM25 keyword search)
STORE_PATH = "data/store.json"
