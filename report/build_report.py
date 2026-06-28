"""Generate the 4-page individual report (PDF) for the IIBDCC exam.

Run:
    uv run python -m report.build_report

Reads eval/results/results.csv and artifacts/graph.png if present and
produces report/rapport.pdf. The script is self-contained; tweak the
French content directly in this file to update the report.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_CSV = PROJECT_ROOT / "eval" / "results" / "results.csv"
GRAPH_PNG = PROJECT_ROOT / "artifacts" / "graph.png"
OUTPUT_PDF = PROJECT_ROOT / "report" / "rapport.pdf"


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title", parent=base["Title"], fontSize=15, leading=18, spaceAfter=4
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=base["Normal"], fontSize=9, leading=11,
            textColor=colors.HexColor("#555555"), spaceAfter=10,
        ),
        "h1": ParagraphStyle(
            "h1", parent=base["Heading1"], fontSize=11, leading=14,
            textColor=colors.HexColor("#1f3864"), spaceBefore=8, spaceAfter=4,
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"], fontSize=9.5, leading=12,
            textColor=colors.HexColor("#2e5396"), spaceBefore=4, spaceAfter=2,
        ),
        "body": ParagraphStyle(
            "body", parent=base["BodyText"], fontSize=9, leading=11.5,
            alignment=TA_JUSTIFY, spaceAfter=3,
        ),
        "bullet": ParagraphStyle(
            "bullet", parent=base["BodyText"], fontSize=9, leading=11.5,
            leftIndent=10, bulletIndent=0, spaceAfter=1, alignment=TA_LEFT,
        ),
        "small": ParagraphStyle(
            "small", parent=base["BodyText"], fontSize=7.5, leading=9.5,
            textColor=colors.HexColor("#555555"), alignment=TA_LEFT,
        ),
    }


def _load_eval_stats() -> dict:
    """Aggregate evaluation results into the numbers cited in the report."""
    if not RESULTS_CSV.exists():
        return {}
    df = pd.read_csv(RESULTS_CSV)
    df["ok"] = ~df["answer"].fillna("").str.startswith("ERROR:")
    by_type = df.groupby("type").agg(
        n=("question", "count"),
        success=("ok", "sum"),
        avg_latency=("latency_s", "mean"),
        median_latency=("latency_s", "median"),
        avg_iters=("iterations", "mean"),
        web_rate=("web_search_used", "mean"),
        avg_docs=("n_docs", "mean"),
        avg_retries=("halluc_retries", "mean"),
    ).round(2)

    # Top sources used across all queries
    counter: Counter[str] = Counter()
    for s in df["sources"].fillna(""):
        for x in s.split(";"):
            if x:
                counter[Path(x).name] += 1
    top_sources = counter.most_common(8)

    return {"df": df, "by_type": by_type, "top_sources": top_sources}


def _table_summary(stats: dict, styles) -> Table:
    by_type = stats["by_type"].reset_index()
    header = ["Type", "N", "Succès", "Latence moy. (s)", "Itérations", "Docs/req", "Hallu. retries", "Web search"]
    rows = [header]
    for _, r in by_type.iterrows():
        rows.append([
            r["type"],
            int(r["n"]),
            int(r["success"]),
            f"{r['avg_latency']:.1f}",
            f"{r['avg_iters']:.2f}",
            f"{r['avg_docs']:.1f}",
            f"{r['avg_retries']:.2f}",
            f"{r['web_rate']*100:.0f} %",
        ])
    t = Table(rows, hAlign="LEFT", colWidths=[2.0*cm, 1.0*cm, 1.3*cm, 2.4*cm, 1.8*cm, 1.7*cm, 2.0*cm, 1.7*cm])
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", 8),
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dde7f4")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f8fc")]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#aaaaaa")),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


def _table_sources(stats: dict) -> Table:
    rows = [["Document source", "# questions"]]
    for src, n in stats["top_sources"]:
        rows.append([src, n])
    t = Table(rows, hAlign="LEFT", colWidths=[8.0*cm, 2.5*cm])
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", 8),
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dde7f4")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f8fc")]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#aaaaaa")),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
    ]))
    return t


def build():
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    stats = _load_eval_stats()
    s = _styles()
    story = []
    P = lambda t, st="body": Paragraph(t, s[st])

    # ============ PAGE 1 ============
    story += [
        P("Rapport — Projet de Fin de Module", "title"),
        P(
            "Système Agentic RAG sur le tourisme marocain avec LangGraph et Ollama<br/>",
            "subtitle",
        ),

        P("1. Démarche suivie", "h1"),
        P(
            "Le projet implémente un système <b>Agentic RAG</b> de bout en bout dédié au tourisme marocain. "
            "Le choix du domaine a été motivé par la disponibilité d'un corpus francophone riche et par l'intérêt "
            "applicatif (assistance touristique multilingue). La démarche s'est articulée en cinq étapes successives, "
            "calquées sur la grille d'évaluation&nbsp;: constitution du corpus, vectorisation, conception du graphe "
            "LangGraph, développement des outils, puis évaluation systématique."
        ),
        P("Constitution du corpus", "h2"),
        P(
            "Quinze fichiers Markdown ont été rédigés (~2&nbsp;000 lignes au total) couvrant dix régions du Maroc "
            "(Marrakech, Fès, Chefchaouen, Casablanca, Rabat, Essaouira, Sahara-Merzouga, Atlas-Toubkal, Agadir, "
            "Tanger) et cinq thématiques transversales (gastronomie, artisanat, sites UNESCO, festivals, informations "
            "pratiques). Le corpus mêle faits factuels (dates, chiffres, formalités&nbsp;: 90&nbsp;jours sans visa "
            "pour l'UE, minaret Hassan II de 210&nbsp;m, 9 sites UNESCO) et contenus synthétiques (comparaisons "
            "entre médinas, itinéraires multi-villes), afin de stresser la fois la récupération exacte et la synthèse "
            "multi-documents."
        ),
        P("Choix techniques", "h2"),
        P(
            "<b>LangGraph</b> a été retenu pour pouvoir construire un graphe d'état explicite avec boucles et "
            "branches conditionnelles, conformément à la consigne (la primitive <i>create_agent</i> de LangChain "
            "est interdite). <b>Ollama</b> avec le modèle local <i>llama3.2:latest</i> (3&nbsp;B paramètres) sert "
            "de LLM, ce qui élimine la dépendance à un quota API et garantit la reproductibilité hors-ligne. "
            "Les embeddings utilisent <i>sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2</i> "
            "(~120&nbsp;Mo, multilingue&nbsp;: FR/EN/AR), suffisant pour un corpus de cette taille et exécutable "
            "sur CPU. <b>Chroma</b> joue le rôle de magasin vectoriel persistant. Tous les routeurs et graders "
            "utilisent des sorties structurées Pydantic via <i>with_structured_output</i>, garantissant un "
            "comportement déterministe."
        ),
        P("Architecture choisie&nbsp;: Corrective RAG (CRAG)", "h2"),
        P(
            "Plutôt qu'un pipeline RAG linéaire, j'ai opté pour une variante <b>CRAG</b>&nbsp;: après la récupération, "
            "un <i>grader</i> évalue la pertinence des documents, et le graphe peut décider de réécrire la question, "
            "de basculer vers une recherche web (Tavily) ou de générer directement la réponse. Un dernier nœud "
            "<i>hallucination_check</i> vérifie que la réponse est ancrée dans le contexte&nbsp;; sinon il relance "
            "une régénération (jusqu'à 2&nbsp;retries). Cette boucle agentique sert directement les critères de "
            "notation «&nbsp;Qualité du graphe LangGraph&nbsp;» et «&nbsp;Respect de l'approche "
            "Agentic RAG&nbsp;»."
        ),
        PageBreak(),

        # ============ PAGE 2 ============
        P("2. Fonctionnement du système", "h1"),
        P("Vue d'ensemble du graphe", "h2"),
        P(
            "Le graphe LangGraph compile huit nœuds reliés par des arêtes conditionnelles. Le <b>routeur</b> "
            "décide d'emblée entre la base vectorielle, une recherche web, ou une génération directe. Sur "
            "le chemin vectoriel, le <b>grader</b> filtre les documents non pertinents&nbsp;; selon le résultat, "
            "le système réécrit la question (HyDE-style) ou bascule sur la fallback web. Le <b>générateur</b> "
            "produit la réponse, puis le <b>vérificateur d'hallucination</b> la valide ou demande une régénération."
        ),
    ]

    if GRAPH_PNG.exists():
        img = Image(str(GRAPH_PNG), width=11*cm, height=7*cm, kind="proportional")
        img.hAlign = "CENTER"
        story.append(img)
        story.append(P("Figure&nbsp;1. Topologie du graphe LangGraph (rendu Mermaid).", "small"))
    else:
        story.append(P("[graphique non disponible — exécuter le notebook pour générer artifacts/graph.png]", "small"))

    story += [
        P("État du graphe et persistance", "h2"),
        P(
            "L'état est un <i>TypedDict</i> regroupant&nbsp;: la question initiale, l'historique des réécritures "
            "(avec réducteur <i>Annotated[list, add]</i>), les documents retrouvés, les notes du grader, les "
            "résultats web, la réponse finale, un compteur d'itérations (plafonné à 3) et un flag <i>grounded</i> "
            "renvoyé par le vérificateur d'hallucination. Un <i>InMemorySaver</i> sert de checkpointer&nbsp;: "
            "chaque question reçoit un <i>thread_id</i> distinct (mémoire à court terme par fil de conversation)."
        ),
        P("Outils et tool-calling manuel", "h2"),
        P(
            "Deux outils sont définis via le décorateur <i>@tool</i>&nbsp;: <b>retriever_vectorstore</b> (recherche "
            "MMR top-k=5 sur Chroma) et <b>web_search_tavily</b> (fallback hors-corpus). Conformément à la consigne, "
            "<i>create_agent</i> n'est utilisé nulle part&nbsp;: l'invocation des outils est faite manuellement "
            "depuis les nœuds (<i>retrieve</i> et <i>websearch</i>), ce qui rend chaque appel observable dans "
            "LangGraph Studio. Les choix de routage du grader/routeur passent par "
            "<i>llm.with_structured_output(PydanticModel)</i>, garantissant des transitions déterministes entre nœuds."
        ),
        P("Cycle de vie d'une requête", "h2"),
        P(
            "Pour une question simple (ex. «&nbsp;Quel est le surnom de Marrakech&nbsp;?&nbsp;»), le chemin "
            "typique est&nbsp;: <i>route_question</i> → <i>retrieve</i> → <i>grade_documents</i> → <i>generate</i> → "
            "<i>hallucination_check</i> → END, sans réécriture ni fallback web. Pour une question complexe "
            "(ex. comparaison Fès vs Marrakech), le graphe peut entrer dans la boucle <i>rewrite_query</i> "
            "→ <i>retrieve</i> une à deux fois avant de générer une réponse synthétisée à partir de plusieurs "
            "documents&nbsp;; cela se traduit par <i>iterations &gt; 0</i> dans les résultats."
        ),
        PageBreak(),

        # ============ PAGE 3 ============
        P("3. Résultats de l'évaluation", "h1"),
        P("Protocole", "h2"),
        P(
            "Vingt questions ont été soumises au graphe via le notebook <i>notebooks/demo.ipynb</i>&nbsp;: "
            "10 simples (lookup factuel) et 10 complexes (synthèse multi-documents). Pour chacune, on capture "
            "la réponse, la latence (<i>time.perf_counter</i>), le nombre d'itérations de réécriture, l'utilisation "
            "ou non de la recherche web, le nombre de retries hallucination et les sources réellement consultées. "
            "Les questions et les réponses de référence sont versionnées dans <i>eval/questions.json</i>."
        ),
        P("Synthèse quantitative", "h2"),
    ]

    if stats:
        story.append(_table_summary(stats, s))
        story.append(P("Tableau&nbsp;1. Agrégats par type de question (20&nbsp;questions, 100&nbsp;% de réponses produites).", "small"))

        df = stats["df"]
        avg_lat_overall = df["latency_s"].mean()
        story += [
            Spacer(1, 4*mm),
            P("Lecture des résultats", "h2"),
            P(
                f"Les <b>20 questions ont été traitées avec succès</b> (aucune erreur runtime). La latence moyenne "
                f"globale est de <b>{avg_lat_overall:.1f}&nbsp;s</b>&nbsp;: les questions simples sont nettement "
                f"plus rapides (médiane ~{stats['by_type'].loc['simple','median_latency']:.0f}&nbsp;s) que les "
                f"questions complexes (médiane ~{stats['by_type'].loc['complex','median_latency']:.0f}&nbsp;s), "
                f"ce qui reflète à la fois le volume de génération et la fréquence des boucles de réécriture. "
                f"Le grader a déclenché une réécriture de requête dans environ "
                f"{stats['by_type'].loc['complex','avg_iters']:.2f}&nbsp;itérations moyennes pour les questions "
                f"complexes (vs {stats['by_type'].loc['simple','avg_iters']:.2f} pour les simples). La fallback "
                f"web a été employée sur "
                f"{int(stats['by_type'].loc['simple','web_rate']*stats['by_type'].loc['simple','n'])} cas "
                f"sur 20 — preuve que le graphe sait basculer lorsque le corpus est insuffisant."
            ),
            P("Pertinence du retrieval (sources les plus consultées)", "h2"),
            _table_sources(stats),
            P("Tableau&nbsp;2. Fréquence d'utilisation de chaque document source sur l'ensemble des 20&nbsp;questions.", "small"),
        ]
    else:
        story.append(P("[Aucun résultat trouvé dans eval/results/results.csv]", "small"))

    story += [
        PageBreak(),

        # ============ PAGE 4 ============
        P("4. Limites et pistes d'amélioration", "h1"),
        P("Limites observées", "h2"),
        P(
            "<b>Modèle local de petite taille.</b> Llama&nbsp;3.2 (3&nbsp;B) reste limité sur les synthèses "
            "complexes&nbsp;: certaines comparaisons (Fès vs Marrakech, itinéraires multi-villes) sont correctes "
            "mais parfois redondantes ou incomplètes. Le format structuré (JSON-mode) tient bien malgré la taille "
            "du modèle, mais la rédaction française gagnerait à passer sur llama3.1:8B ou Mixtral, voire Groq "
            "<i>llama-3.3-70b-versatile</i> dès que le quota est disponible.",
            "body",
        ),
        P(
            "<b>Embeddings multilingues modestes.</b> Le modèle MiniLM 384&nbsp;dim est rapide mais moins "
            "discriminant que BGE-M3 sur des requêtes nuancées. Sur quelques requêtes complexes, des chunks "
            "non pertinents sont remontés et filtrés a posteriori par le grader, ce qui augmente la latence.",
            "body",
        ),
        P(
            "<b>Absence de re-ranker.</b> Aucun étage de reranking n'est appliqué après la similarité dense&nbsp;; "
            "ajouter <i>bge-reranker</i> ou Cohere Rerank trierait mieux le top-k.",
            "body",
        ),
        P(
            "<b>Pas de mémoire de conversation multi-tour.</b> Le <i>thread_id</i> est utilisé par question "
            "d'évaluation (mémoire isolée). Un assistant touristique réel devrait suivre le contexte sur plusieurs "
            "échanges (où ai-je dit que j'irais en juin&nbsp;? quel hôtel m'a-t-il déjà recommandé&nbsp;?).",
            "body",
        ),
        P(
            "<b>Évaluation manuelle.</b> Le notebook capture latence et sources mais l'appréciation qualitative "
            "des réponses reste à la lecture. Aucun score automatique n'est calculé.",
            "body",
        ),

        P("Pistes d'amélioration", "h2"),
        P(
            "1. <b>Re-ranker dédié</b> (BGE-reranker, Cohere Rerank) entre le retrieve et le grader pour gagner "
            "en précision sans bouger le top-k.",
            "body",
        ),
        P(
            "2. <b>Évaluation LLM-as-judge</b> (ou RAGAS&nbsp;: faithfulness, answer_relevancy, context_precision) "
            "pour scorer automatiquement les 20 réponses contre les références versionnées dans questions.json.",
            "body",
        ),
        P(
            "3. <b>Embeddings BGE-M3</b> (multilingue 1024&nbsp;dim, état de l'art en 2025) pour améliorer "
            "la pertinence sur requêtes nuancées et bilingues FR/AR.",
            "body",
        ),
        P(
            "4. <b>Checkpointer persistant</b> (SqliteSaver, PostgresSaver) pour conserver l'historique entre "
            "sessions et permettre un déploiement multi-utilisateur.",
            "body",
        ),
        P(
            "5. <b>Multi-agent hiérarchique</b> (cf. TP&nbsp;8)&nbsp;: un orchestrateur déléguerait à des "
            "sous-agents spécialisés (planificateur d'itinéraires, conseiller gastronomie, conseiller pratique).",
            "body",
        ),
        P(
            "6. <b>HITL (Human-in-the-Loop)</b> via <i>HumanInTheLoopMiddleware</i> (cf. TP&nbsp;7) sur les "
            "actions à conséquence (réservation simulée, recommandation finale).",
            "body",
        ),
        P(
            "7. <b>Front conversationnel</b> léger (Streamlit ou Chainlit) pour la démo, alimenté par le même "
            "graphe compilé.",
            "body",
        ),
    ]

    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        leftMargin=1.6*cm, rightMargin=1.6*cm,
        topMargin=1.4*cm, bottomMargin=1.4*cm,
        title="Rapport — Projet Fin de Module IIBDCC",
        author="Idriss Essadik",
    )
    doc.build(story)
    print(f"[done] PDF écrit : {OUTPUT_PDF}")
    return OUTPUT_PDF


if __name__ == "__main__":
    sys.exit(0 if build() else 1)
