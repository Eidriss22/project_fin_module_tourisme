# Agentic RAG — Tourisme Marocain

Projet de fin de module (Master IIBDCC). Système RAG agentique construit avec
**LangGraph** (sans `create_agent`), propulsé par **Groq** (`llama-3.3-70b-versatile`),
embeddings **HuggingFace** multilingues locaux, vector store **Chroma**, et
recherche web **Tavily** en fallback.

## Architecture (CRAG-flavored)

```
              ┌──────────────┐
START ──────▶ │ route_question│──┐
              └──────────────┘  │
                       │        │
        ┌──────────────┼────────┼────────────┐
        ▼              ▼        ▼            ▼
    retrieve       websearch  generate    (direct)
        │              │        ▲
        ▼              │        │
  grade_documents      │        │
        │              │        │
        ▼              │        │
   decide_next ────────┘        │
   │   │   │                    │
   │   │   └──▶ rewrite_query ──┐
   │   │           │            │
   │   └──▶ generate            │
   │           ▲                │
   │           └────────────────┘
   ▼
hallucination_check ──▶ {END | rewrite_query (capped)}
```

- **Iteration cap**: `state['iterations']` is bumped on every rewrite and
  bounded by `MAX_ITERATIONS = 3` (see `src/config.py`).
- **Hallucination cap**: `state['halluc_retries']` is bounded by
  `HALLUCINATION_MAX_RETRIES = 2`.
- **Checkpointing**: `InMemorySaver` (thread-id support for replay / Studio).

## Setup

```bash
# 1. Install (uv recommended; pip works too)
uv sync                       # or: pip install -e .

# 2. Configure secrets — .env must contain GROQ_API_KEY and TAVILY_API_KEY
#    (LANGSMITH_API_KEY is optional but enables tracing.)

# 3. Drop tourism documents into data/corpus/  (see data/corpus/README.md)

# 4. Build the vector index
python -m src.ingest --reset

# 5. (Optional) Render the graph diagram
python -m src.graph --draw       # writes artifacts/graph.png (or .mmd)

# 6. Ask a question
python -m src.graph --ask "Quels sont les sites incontournables a Chefchaouen ?"
```

## LangGraph Studio (optional but nice)

```bash
uv run langgraph dev          # opens the UI; reads ./langgraph.json
```

## Evaluation

1. Edit `eval/questions.json` to add 10 simple + 10 complex questions.
2. Run:
   ```bash
   python -m src.evaluate
   ```
3. Results land in `eval/results/results.csv` plus a console summary.

## Project layout

```
.
├── data/corpus/         # drop PDFs / TXT / MD here
├── chroma_db/           # persistent vector store (gitignored)
├── eval/
│   ├── questions.json   # 20 evaluation questions
│   └── results/         # CSV outputs
├── artifacts/           # graph.png / graph.mmd
├── src/
│   ├── config.py        # env, paths, hyperparameters
│   ├── llms.py          # ChatGroq + HuggingFace embeddings factories
│   ├── ingest.py        # PDF → chunks → Chroma
│   ├── tools.py         # @tool retriever + Tavily web search
│   ├── state.py         # AgentState TypedDict
│   ├── nodes.py         # all node functions + structured-output schemas
│   ├── graph.py         # StateGraph builder + module-level `graph`
│   └── evaluate.py      # eval harness
├── langgraph.json       # LangGraph Studio config
└── pyproject.toml
```

## Design choices (for the report)

| Choix                                   | Justification                                                                                |
|-----------------------------------------|----------------------------------------------------------------------------------------------|
| `llama-3.3-70b-versatile`               | Meilleur modèle Groq pour reasoning + tool use, 128k contexte, latence < 1s/token.           |
| `paraphrase-multilingual-MiniLM-L12-v2` | Petit (~120MB), CPU-friendly, multilingue FR/EN/AR (cohérent avec un corpus marocain).       |
| Chroma persistant                       | Léger, sans serveur, parfait pour un projet académique.                                      |
| LangGraph (sans `create_agent`)         | Topologie explicite → contrôle fin du flux agentique (exigé par le sujet).                   |
| Sorties structurées Pydantic            | Décisions de routage / grading / hallucination déterministes (JSON validé).                  |
| InMemorySaver                           | Permet le rewind / replay dans LangGraph Studio sans dépendance externe.                     |
