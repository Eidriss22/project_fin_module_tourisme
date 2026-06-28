"""Central configuration: env loading, model names, paths, hyperparameters.

All other modules should import from here rather than reading env vars directly.
This keeps the system reproducible and makes the rubric-graded
"justification of choices" easy to point to in the report.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root regardless of where the script is invoked from.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


# --- API keys ---------------------------------------------------------------
GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY: str | None = os.getenv("TAVILY_API_KEY")
LANGSMITH_API_KEY: str | None = os.getenv("LANGSMITH_API_KEY")

# Enable LangSmith tracing if a key is present.
if LANGSMITH_API_KEY:
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", "tourism-agentic-rag")


# --- LLM ---------------------------------------------------------------------
# Provider toggle: "ollama" (local, free, no rate limit) or "groq" (hosted).
# Local Ollama avoids the Groq free-tier daily token cap (100k TPD on 70B).
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama").lower()

# Ollama local model. llama3.2:latest is the 3B variant (~2GB), fast on CPU.
# Use "llama3.1:latest" (8B) for better reasoning/structured-output reliability
# at the cost of higher latency.
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Groq fallback model (used when LLM_PROVIDER=groq).
GROQ_MODEL: str = "llama-3.3-70b-versatile"

# Active chat model name (exposed for logging/reporting).
CHAT_MODEL: str = OLLAMA_MODEL if LLM_PROVIDER == "ollama" else GROQ_MODEL
CHAT_TEMPERATURE: float = 0.0  # deterministic for graders/routers


# --- Embeddings --------------------------------------------------------------
# Multilingual MiniLM: small (~120MB), fast on CPU, supports FR/EN/AR which
# matches a Moroccan tourism corpus. We trade a few accuracy points for
# practicality on a student laptop. Swap to BAAI/bge-m3 if you have GPU.
EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DEVICE: str = "cpu"


# --- Paths -------------------------------------------------------------------
DATA_DIR: Path = PROJECT_ROOT / "data" / "corpus"
CHROMA_DIR: Path = PROJECT_ROOT / "chroma_db"
EVAL_DIR: Path = PROJECT_ROOT / "eval"
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"

COLLECTION_NAME: str = "tourism_morocco"


# --- Chunking ----------------------------------------------------------------
CHUNK_SIZE: int = 800       # ~600 tokens — fits well in 70B context
CHUNK_OVERLAP: int = 120    # 15% overlap preserves cross-chunk coreference


# --- Retrieval ---------------------------------------------------------------
RETRIEVER_K: int = 5
RETRIEVER_FETCH_K: int = 20  # for MMR (diversity)
USE_MMR: bool = True


# --- Graph control -----------------------------------------------------------
MAX_ITERATIONS: int = 3      # cap on rewrite_query loops
HALLUCINATION_MAX_RETRIES: int = 2  # cap on regeneration retries


def assert_keys() -> None:
    """Raise early with a clear message if required keys are missing.

    With LLM_PROVIDER=ollama only TAVILY_API_KEY is required (and only if the
    websearch fallback fires). GROQ_API_KEY is only needed when LLM_PROVIDER=groq.
    """
    required: list[tuple[str, str | None]] = []
    if LLM_PROVIDER == "groq":
        required.append(("GROQ_API_KEY", GROQ_API_KEY))
    # Tavily is optional — the graph only calls it on the websearch fallback path.
    # We warn but don't fail if missing.
    missing = [name for name, val in required if not val]
    if missing:
        raise RuntimeError(
            f"Missing required API key(s): {', '.join(missing)}. "
            f"Add them to {PROJECT_ROOT / '.env'}."
        )
