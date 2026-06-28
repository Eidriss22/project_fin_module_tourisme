"""Graph state schema.

Kept minimal — every field is read or written by at least one node. Extra
fields confuse LangGraph Studio's UI and add no value to grading.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from langchain_core.documents import Document


# Routes the router can emit from ``route_question``.
RouteDecision = Literal["retrieve", "websearch", "generate"]


class AgentState(TypedDict, total=False):
    """Shared state passed between nodes.

    All keys are optional (``total=False``) so partial-update nodes work
    without having to know about unrelated fields.
    """

    # --- Input / user question ---
    question: str            # original user question (immutable)
    rewritten_question: str  # current query used for retrieval (mutable)

    # --- Retrieved context ---
    documents: list[Document]

    # --- Output ---
    generation: str

    # --- Control flow ---
    route: RouteDecision     # routing decision from route_question
    iterations: int          # # of times the query has been rewritten
    web_search_used: bool    # whether we already fell back to the web
    grounded: bool           # set by hallucination_check (True = answer is grounded)
    halluc_retries: int      # # of regenerations after a hallucination flag
