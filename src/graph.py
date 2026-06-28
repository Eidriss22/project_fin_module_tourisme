"""LangGraph wiring for the Moroccan-tourism agentic RAG agent.

This module exposes ``graph`` as a compiled, module-level object so that
LangGraph Studio (``langgraph dev``) can pick it up via ``langgraph.json``.

CRAG-flavored topology:

::

    START -> route_question
    route_question --[router]--> {retrieve | websearch | generate}
    retrieve -> grade_documents -> decide_next
    decide_next --> {generate | rewrite_query | websearch}
    rewrite_query -> retrieve   (loop, capped by state['iterations'] < 3)
    websearch -> generate
    generate -> hallucination_check
    hallucination_check --> {END | rewrite_query}   (capped by halluc_retries)
"""

from __future__ import annotations

from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from src.config import ARTIFACTS_DIR, assert_keys
from src.nodes import (
    decide_next,
    from_halluc,
    from_route,
    generate,
    grade_documents,
    hallucination_check,
    retrieve,
    rewrite_query,
    route_question,
    websearch,
)
from src.state import AgentState


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def build_graph() -> StateGraph:
    """Assemble the StateGraph (without compiling)."""
    g = StateGraph(AgentState)

    # Nodes
    g.add_node("route_question", route_question)
    g.add_node("retrieve", retrieve)
    g.add_node("grade_documents", grade_documents)
    g.add_node("rewrite_query", rewrite_query)
    g.add_node("websearch", websearch)
    g.add_node("generate", generate)
    g.add_node("hallucination_check", hallucination_check)

    # Entry
    g.add_edge(START, "route_question")

    # Router -> three possible branches
    g.add_conditional_edges(
        "route_question",
        from_route,
        {
            "retrieve": "retrieve",
            "websearch": "websearch",
            "generate": "generate",
        },
    )

    # Retrieval -> grading -> decision
    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges(
        "grade_documents",
        decide_next,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "websearch": "websearch",
        },
    )

    # Rewrite loops back to retrieval; web fallback feeds the generator.
    g.add_edge("rewrite_query", "retrieve")
    g.add_edge("websearch", "generate")

    # Post-generation: verify grounding, optionally loop.
    g.add_edge("generate", "hallucination_check")
    g.add_conditional_edges(
        "hallucination_check",
        from_halluc,
        {
            "end": END,
            "rewrite_query": "rewrite_query",
        },
    )
    return g


# Module-level compiled graphs.
#
# - ``graph``: NO checkpointer. This is the object consumed by
#   ``langgraph.json`` and loaded by ``langgraph dev``. The LangGraph API
#   refuses graphs that ship a custom checkpointer (it manages persistence
#   itself).
# - ``graph_with_memory``: same topology but compiled with an
#   ``InMemorySaver`` checkpointer. Used by ``evaluate.py``, the notebook
#   and ``run_once`` so that ``thread_id`` provides short-term memory.
graph = build_graph().compile()
graph_with_memory = build_graph().compile(checkpointer=InMemorySaver())


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------
def _draw_mermaid(out: Path = ARTIFACTS_DIR / "graph.png") -> None:
    """Save a mermaid PNG of the graph. Requires internet (mermaid.ink)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        png = graph.get_graph(xray=True).draw_mermaid_png()
        out.write_bytes(png)
        print(f"[viz] Wrote {out}")
    except Exception as e:  # pragma: no cover - network/pyppeteer dependent
        # Fallback: write the mermaid source so the user can render it elsewhere.
        mmd = graph.get_graph(xray=True).draw_mermaid()
        mmd_path = out.with_suffix(".mmd")
        mmd_path.write_text(mmd, encoding="utf-8")
        print(
            f"[viz] PNG rendering unavailable ({e!r}). "
            f"Wrote mermaid source to {mmd_path} — paste it into https://mermaid.live to view."
        )


def run_once(question: str, thread_id: str = "demo") -> str:
    """Invoke the graph end-to-end and return the final answer.

    Uses ``graph_with_memory`` so that ``thread_id`` enables short-term
    memory across invocations.
    """
    assert_keys()
    result = graph_with_memory.invoke(
        {"question": question},
        config={"configurable": {"thread_id": thread_id}},
    )
    return result.get("generation", "")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run / visualize the tourism agent.")
    parser.add_argument("--draw", action="store_true", help="Render the graph to artifacts/graph.png")
    parser.add_argument(
        "--ask",
        type=str,
        default="Quels sont les sites incontournables a visiter a Fes ?",
        help="Sample question to run end-to-end.",
    )
    args = parser.parse_args()

    if args.draw:
        _draw_mermaid()

    print(f"\n[ask] {args.ask}\n")
    answer = run_once(args.ask)
    print(f"[answer]\n{answer}\n")
