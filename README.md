# Agentic RAG — Tourisme Marocain

Projet de fin de module (Master IIBDCC). Système **Agentic RAG** construit avec **LangGraph** (sans `create_agent`), architecture **CRAG (Corrective RAG)**, propulsé par **Ollama** en local (`llama3.2:latest`) ou **Groq** (`llama-3.3-70b-versatile`) en option. Embeddings **HuggingFace** multilingues, vector store **Chroma** persistant, recherche web **Tavily** en fallback.

> Encadrante : Prof. RETAL Sara — Master IIBDCC (SMA & IAD)

## Architecture (CRAG-flavored)

```
START → route_question → {retrieve | websearch | generate}
retrieve → grade_documents → decide_next
decide_next → {generate | rewrite_query | websearch}
rewrite_query → retrieve   (boucle bornée par MAX_ITERATIONS = 3)
websearch → generate
generate → hallucination_check → {END | rewrite_query}   (borné par HALLUCINATION_MAX_RETRIES = 2)
```

- **Tool-calling manuel** : pas de `create_agent`. Les nœuds `retrieve` et `websearch` invoquent directement les `@tool` de `src/tools.py`.
- **Décisions agentiques structurées** : router, grader et hallucination-check renvoient du JSON validé via `llm.with_structured_output(PydanticModel)`.
- **Deux graphes compilés** exposés depuis `src/graph.py` :
  - `graph` — sans checkpointer, consommé par `langgraph dev` / Studio.
  - `graph_with_memory` — avec `InMemorySaver`, utilisé par `evaluate.py`, le notebook et `run_once()`.

Rendu Mermaid : `artifacts/graph.png` (généré par le notebook §2 ou `python -m src.graph --draw`).

## Setup

```bash
# 1. Dépendances
uv sync

# 2. Secrets — créer .env à la racine
#    GROQ_API_KEY=...          # requis uniquement si LLM_PROVIDER=groq
#    TAVILY_API_KEY=...        # requis pour la fallback web
#    LANGSMITH_API_KEY=...     # optionnel, active le tracing

# 3. Modèle local Ollama (provider par défaut)
ollama pull llama3.2:latest

# 4. Construire l'index vectoriel à partir de data/corpus/
uv run python -m src.ingest --reset

# 5. Tester une question
uv run python -m src.graph --ask "Quels sont les sites UNESCO du Maroc ?"
```

### Basculer Ollama ↔ Groq

```bash
LLM_PROVIDER=groq   uv run python -m src.graph --ask "..."   # cloud, plus rapide, quota TPD
LLM_PROVIDER=ollama uv run python -m src.graph --ask "..."   # local, gratuit (défaut)
```

## Utilisation

### Notebook de démonstration (entrée principale)

```bash
uv run jupyter notebook notebooks/demo.ipynb
```

Le notebook enchaîne : initialisation → diagramme Mermaid → état du corpus → requête simple → requête complexe → évaluation 20 questions → tableaux et graphiques.

### CLI

```bash
uv run python -m src.graph --draw          # rend artifacts/graph.png
uv run python -m src.graph --ask "..."     # invocation unique
uv run python -m src.evaluate              # 20 questions → eval/results/results.csv
```

### LangGraph Studio

```bash
uv run langgraph dev
```

Ouvre `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`. Si Chrome bloque (mixed-content HTTPS → HTTP), autorise *Insecure content* sur smith.langchain.com, ou utilise `--tunnel` + ajoute `*.trycloudflare.com` aux domaines autorisés.

## Évaluation

20 questions versionnées dans `eval/questions.json` (10 simples + 10 complexes, toutes ancrées dans le corpus). Pour chacune, on capture latence, itérations, sources retrouvées, usage de la fallback web et retries hallucination → `eval/results/results.csv` + agrégats dans `summary_by_type.csv`.

| Type | N | Médiane latence | Itérations moy. | Web search | Hallu. retries |
|---|---|---|---|---|---|
| simple | 10 | ~18 s | 0.6 | 1/10 | 1.0 |
| complex | 10 | ~35 s | 0.5 | 0/10 | 0.6 |

(Valeurs typiques sur Ollama `llama3.2:3B` ; varient selon la machine.)

## Rapport et démo

- **Rapport PDF 4 pages** : `report/rapport.pdf`.
- **Vidéo de démonstration** : 2 min, scénarisée pour le notebook (voir `report/` ou demander le script).

## Project layout

```
.
├── data/corpus/                # documents ingérés (régions, culture, pratique, ...)
├── chroma_db/                  # vector store persistant (gitignored)
├── notebooks/
│   └── demo.ipynb              # entrée démo + éval principale
├── eval/
│   ├── questions.json          # 20 questions de référence
│   └── results/                # results.csv, summary_by_type.csv, latency_per_question.png
├── artifacts/
│   └── graph.png               # diagramme Mermaid du graphe
├── report/
│   └── rapport.pdf             # rapport individuel (4 pages)
├── src/
│   ├── config.py               # env, providers, paths, hyperparamètres
│   ├── llms.py                 # factories Ollama/Groq + embeddings HF
│   ├── ingest.py               # chargement + chunking + persistance Chroma
│   ├── tools.py                # @tool retriever + @tool web search
│   ├── state.py                # AgentState (TypedDict)
│   ├── nodes.py                # 8 nœuds + schémas Pydantic
│   ├── graph.py                # StateGraph + graph / graph_with_memory
│   └── evaluate.py             # harnais CLI (doublon du notebook §6)
├── langgraph.json              # config LangGraph Studio
└── pyproject.toml
```

## Choix techniques (résumé pour le rapport)

| Choix | Justification |
|---|---|
| **Ollama `llama3.2:latest`** (défaut) | Modèle local 3B, pas de quota API, reproductibilité hors-ligne. |
| **Groq `llama-3.3-70b-versatile`** (option) | Latence faible et qualité supérieure si quota TPD disponible. |
| **`paraphrase-multilingual-MiniLM-L12-v2`** | 384 dims, multilingue FR/EN/AR, ~120 Mo, CPU-friendly. |
| **Chroma persistant** | Léger, sans serveur, parfait pour un projet académique. |
| **LangGraph (sans `create_agent`)** | Topologie explicite, contrôle fin du flux agentique (exigé par l'énoncé). |
| **Sorties structurées Pydantic** | Décisions router/grader/hallucination déterministes (JSON validé). |
| **CRAG** | Boucle de correction (grade → rewrite → retrieve) + fallback web pour les cas hors-corpus + vérification d'ancrage final. |
| **InMemorySaver (variante `_with_memory`)** | `thread_id` ⇒ mémoire à court terme par fil ; tenu séparément du graphe Studio (l'API LangGraph rejette les checkpointers custom). |

## Troubleshooting

- **`uv run langgraph dev` ne démarre pas** : vérifie qu'aucun checkpointer custom n'est compilé dans le `graph` exposé par `langgraph.json` (la LangGraph API les rejette). Le `graph` de ce projet est compilé sans checkpointer ; utilise `graph_with_memory` côté Python.
- **Studio "Failed to fetch"** : navigateur bloque le mixed-content (HTTPS → HTTP). Autorise insecure content (Chrome) ou utilise `--tunnel`.
- **Groq 429 TPD limit** : passe à Ollama (`LLM_PROVIDER=ollama`) ou attends le reset journalier UTC.
- **Embeddings hangs au premier run** : Hugging Face télécharge ~120 Mo une fois ; les runs suivants démarrent en <10 s.
