"""LLM and embedding factories.

Single source of truth for model instantiation. All nodes import their
LLM from here so that swapping providers/models is a one-line change.

The active provider is chosen via ``config.LLM_PROVIDER`` (``"ollama"`` or
``"groq"``). Ollama runs models locally (no rate limit, no API key) — useful
when the Groq free-tier daily quota is exhausted.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import (
    CHAT_TEMPERATURE,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)


def _build_chat_llm(temperature: float, max_tokens: int) -> BaseChatModel:
    """Construct a chat model for the active provider."""
    if LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama

        # ``format="json"`` is intentionally NOT set globally — nodes that need
        # structured output call ``.with_structured_output(...)``, which wraps
        # the call with JSON parsing. Forcing JSON globally would break the
        # generate node's free-form answer.
        return ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=temperature,
            num_predict=max_tokens,
        )
    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=GROQ_MODEL,
            temperature=temperature,
            api_key=GROQ_API_KEY,
            max_tokens=max_tokens,
        )
    raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER!r}. Use 'ollama' or 'groq'.")


@lru_cache(maxsize=1)
def get_chat_llm(temperature: float | None = None) -> BaseChatModel:
    """Return a singleton chat model (used by graders/routers/hallucination check).

    Both Ollama and Groq support ``.bind_tools()`` and ``.with_structured_output()``.
    Llama 3.2 (Ollama) supports tool calling natively; structured outputs use
    JSON mode under the hood.
    """
    return _build_chat_llm(
        temperature=CHAT_TEMPERATURE if temperature is None else temperature,
        max_tokens=1024,
    )


def get_generation_llm() -> BaseChatModel:
    """Slightly higher temperature for the final answer-generation node."""
    return _build_chat_llm(temperature=0.2, max_tokens=2048)


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Return a singleton multilingual embedding model.

    See [[config]] for the model rationale (FR/EN/AR support, CPU-friendly).
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True},  # cosine-friendly
    )
