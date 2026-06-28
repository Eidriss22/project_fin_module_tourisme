"""Tools exposed to the agent: vector retrieval + web search fallback.

We define tools with ``@tool`` so they can be ``bind_tools()``-ed onto the
Groq LLM if needed; the graph itself calls them directly via the
retrieve / websearch nodes (we deliberately avoid ``create_agent``).
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from src.config import (
    RETRIEVER_FETCH_K,
    RETRIEVER_K,
    TAVILY_API_KEY,
    USE_MMR,
)
from src.ingest import get_vectorstore


# --- Vector retrieval -------------------------------------------------------

def _build_retriever():
    """Return a configured retriever from the persistent Chroma store."""
    vs = get_vectorstore()
    if USE_MMR:
        return vs.as_retriever(
            search_type="mmr",
            search_kwargs={"k": RETRIEVER_K, "fetch_k": RETRIEVER_FETCH_K},
        )
    return vs.as_retriever(search_kwargs={"k": RETRIEVER_K})


@tool("retrieve_tourism_docs")
def retrieve_tourism_docs(query: str) -> list[Document]:
    """Search the local Moroccan-tourism knowledge base.

    Use this tool for questions about Moroccan tourist sites, riads, regional
    travel info, visa rules, cultural heritage, gastronomy, transport, etc.

    Args:
        query: User question, preferably rewritten for retrieval clarity.

    Returns:
        Up to ``RETRIEVER_K`` :class:`Document` objects with text + metadata.
    """
    retriever = _build_retriever()
    return retriever.invoke(query)


# --- Web search fallback ----------------------------------------------------

def _build_tavily() -> TavilySearch:
    """Tavily client tuned for short fact-finding queries."""
    return TavilySearch(
        max_results=4,
        topic="general",
        search_depth="basic",
        include_answer=False,
        tavily_api_key=TAVILY_API_KEY,
    )


@tool("web_search")
def web_search(query: str) -> list[Document]:
    """Search the public web via Tavily.

    Use this when the local corpus is insufficient (e.g., recent news,
    current prices, current opening hours, events).
    """
    raw = _build_tavily().invoke({"query": query})
    # Tavily returns either a dict with "results" or a raw list depending on version.
    items = raw.get("results", raw) if isinstance(raw, dict) else raw
    docs: list[Document] = []
    for r in items or []:
        docs.append(
            Document(
                page_content=r.get("content", "") or r.get("snippet", ""),
                metadata={
                    "source": r.get("url", "web"),
                    "title": r.get("title", ""),
                    "origin": "tavily",
                },
            )
        )
    return docs
