"""Graph nodes for the CRAG-flavored agentic RAG workflow.

Each function takes the current ``AgentState`` and returns a partial dict
update. Conditional routers return a string label (the name of the next
node) and **do not** mutate state.

Structured outputs (router, grader, hallucination check) use
``with_structured_output`` so Groq returns JSON we can trust.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.config import HALLUCINATION_MAX_RETRIES, MAX_ITERATIONS
from src.llms import get_chat_llm, get_generation_llm
from src.state import AgentState
from src.tools import retrieve_tourism_docs, web_search


# =============================================================================
# Structured output schemas
# =============================================================================
class _Route(BaseModel):
    """Decision from the router: where should the question be sent first?"""

    datasource: Literal["retrieve", "websearch", "generate"] = Field(
        description=(
            "retrieve -> use the local Moroccan-tourism vector store; "
            "websearch -> use Tavily web search (real-time / out-of-scope); "
            "generate -> answer directly from the LLM (small talk / no facts needed)."
        )
    )


class _GradeDoc(BaseModel):
    """Binary relevance grade for a single retrieved document."""

    relevant: Literal["yes", "no"] = Field(
        description="'yes' if the document helps answer the question, else 'no'."
    )


class _Hallucination(BaseModel):
    """Whether the generated answer is grounded in the supplied context."""

    grounded: Literal["yes", "no"] = Field(
        description="'yes' if every claim in the answer is supported by the documents."
    )


# =============================================================================
# 1. Router
# =============================================================================
_ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Tu es un routeur pour un agent RAG sur le tourisme marocain. "
            "Choisis la source la plus adaptee:\n"
            "- 'retrieve' pour toute question factuelle sur le tourisme au Maroc "
            "(sites, riads, regions, culture, gastronomie, transport, visas).\n"
            "- 'websearch' si la question concerne l'actualite, des prix recents, "
            "des horaires actuels ou un sujet visiblement hors corpus.\n"
            "- 'generate' uniquement pour les salutations / questions meta "
            "(\"qui es-tu ?\", \"merci\").",
        ),
        ("human", "{question}"),
    ]
)


def route_question(state: AgentState) -> dict:
    """Pick the entry branch: local retrieval, web search, or direct answer."""
    chain = _ROUTER_PROMPT | get_chat_llm().with_structured_output(_Route)
    decision: _Route = chain.invoke({"question": state["question"]})
    return {
        "route": decision.datasource,
        "rewritten_question": state["question"],
        "iterations": 0,
        "web_search_used": False,
        "halluc_retries": 0,
    }


def from_route(state: AgentState) -> Literal["retrieve", "websearch", "generate"]:
    """Conditional-edge dispatcher reading ``state['route']``."""
    return state["route"]


# =============================================================================
# 2. Retrieve
# =============================================================================
def retrieve(state: AgentState) -> dict:
    """Pull top-k chunks from the local Chroma vector store."""
    query = state.get("rewritten_question") or state["question"]
    docs = retrieve_tourism_docs.invoke(query)
    return {"documents": docs}


# =============================================================================
# 3. Grade documents
# =============================================================================
_GRADER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Tu es un evaluateur de pertinence pour un systeme RAG. "
            "Reponds 'yes' si le document est utile pour repondre a la question, "
            "sinon 'no'. Sois indulgent: un lien thematique suffit.",
        ),
        ("human", "Question:\n{question}\n\nDocument:\n{document}"),
    ]
)


def grade_documents(state: AgentState) -> dict:
    """Keep only the documents the grader marks as relevant."""
    question = state.get("rewritten_question") or state["question"]
    grader = _GRADER_PROMPT | get_chat_llm().with_structured_output(_GradeDoc)

    kept: list[Document] = []
    for doc in state.get("documents", []):
        try:
            verdict: _GradeDoc = grader.invoke(
                {"question": question, "document": doc.page_content[:1500]}
            )
            if verdict.relevant == "yes":
                kept.append(doc)
        except Exception:
            # If the grader fails on one doc, keep it conservatively.
            kept.append(doc)
    return {"documents": kept}


# =============================================================================
# 4. Decide next step after grading
# =============================================================================
def decide_next(
    state: AgentState,
) -> Literal["generate", "rewrite_query", "websearch"]:
    """CRAG-style decision after grading.

    - If we have at least one relevant doc -> generate.
    - Else if we still have rewrite budget -> rewrite_query.
    - Else fall back to the web (unless already used) -> websearch / generate.
    """
    has_docs = bool(state.get("documents"))
    iterations = state.get("iterations", 0)

    if has_docs:
        return "generate"

    if iterations < MAX_ITERATIONS:
        return "rewrite_query"

    if not state.get("web_search_used", False):
        return "websearch"

    # Budget exhausted; let the LLM answer with whatever we have (or nothing).
    return "generate"


# =============================================================================
# 5. Rewrite query
# =============================================================================
_REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Reformule la question de l'utilisateur pour ameliorer la recherche "
            "vectorielle: explicite les entites, ajoute des synonymes utiles "
            "(francais, anglais, arabe translitere). Renvoie UNE SEULE phrase, "
            "sans guillemets.",
        ),
        ("human", "Question originale: {question}"),
    ]
)


def rewrite_query(state: AgentState) -> dict:
    """Rewrite the question and bump the iteration counter."""
    base = state.get("rewritten_question") or state["question"]
    chain = _REWRITE_PROMPT | get_chat_llm()
    new_q = chain.invoke({"question": base}).content.strip()
    return {
        "rewritten_question": new_q,
        "iterations": state.get("iterations", 0) + 1,
    }


# =============================================================================
# 6. Web search fallback
# =============================================================================
def websearch(state: AgentState) -> dict:
    """Call Tavily and merge results into ``documents``."""
    query = state.get("rewritten_question") or state["question"]
    web_docs = web_search.invoke(query)
    merged = list(state.get("documents") or []) + list(web_docs)
    return {"documents": merged, "web_search_used": True}


# =============================================================================
# 7. Generate
# =============================================================================
_GENERATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Tu es un assistant expert du tourisme marocain. Reponds de maniere "
            "claire, concise et FACTUELLE en t'appuyant uniquement sur le "
            "contexte fourni. Si le contexte est insuffisant, dis-le "
            "explicitement. Cite les sources entre crochets [source].",
        ),
        (
            "human",
            "Question:\n{question}\n\nContexte:\n{context}\n\nReponds en francais.",
        ),
    ]
)


def _format_context(docs: list[Document]) -> str:
    """Numbered, source-tagged context for grounded answers."""
    if not docs:
        return "(aucun document disponible)"
    blocks = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "inconnu")
        blocks.append(f"[{i}] (source: {src})\n{d.page_content}")
    return "\n\n".join(blocks)


def generate(state: AgentState) -> dict:
    """Produce the final answer from the (graded) context."""
    chain = _GENERATE_PROMPT | get_generation_llm()
    answer = chain.invoke(
        {
            "question": state["question"],
            "context": _format_context(state.get("documents") or []),
        }
    ).content
    return {"generation": answer}


# =============================================================================
# 8. Hallucination check
# =============================================================================
_HALLUC_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Tu es un verificateur de fidelite. Reponds 'yes' si chaque "
            "affirmation de la reponse est SOUTENUE par les documents fournis, "
            "sinon 'no'. Une reponse honnete du type 'je ne sais pas' est 'yes'.",
        ),
        ("human", "Documents:\n{context}\n\nReponse:\n{generation}"),
    ]
)


def hallucination_check(state: AgentState) -> dict:
    """Tag the run as ``grounded`` or ``hallucinated`` and bump retries."""
    chain = _HALLUC_PROMPT | get_chat_llm().with_structured_output(_Hallucination)
    try:
        verdict: _Hallucination = chain.invoke(
            {
                "context": _format_context(state.get("documents") or []),
                "generation": state.get("generation", ""),
            }
        )
        grounded = verdict.grounded == "yes"
    except Exception:
        # On verifier failure, assume grounded to avoid infinite loops.
        grounded = True
    return {
        "grounded": grounded,
        "halluc_retries": state.get("halluc_retries", 0) + (0 if grounded else 1),
    }


def from_halluc(state: AgentState) -> Literal["end", "rewrite_query"]:
    """Loop back once if hallucinated and retries remain, else END."""
    if state.get("grounded", True):
        return "end"
    if state.get("halluc_retries", 0) >= HALLUCINATION_MAX_RETRIES:
        return "end"
    return "rewrite_query"
