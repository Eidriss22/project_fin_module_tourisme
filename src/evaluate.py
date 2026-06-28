"""Evaluation harness.

Reads ``eval/questions.json`` (a list of ``{"question": ..., "type": ...,
"reference": ...}`` items), runs them through the agent, and writes
``eval/results/results.csv`` plus a console summary table.

The exam requires 10 simple + 10 complex questions; this script is a
scaffold so the user can drop those in and re-run.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from src.config import EVAL_DIR, assert_keys
from src.graph import graph_with_memory as graph


def _load_questions(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Question file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array.")
    return data


def _run_one(item: dict, thread_id: str) -> dict:
    """Run a single question and capture latency / final answer / path."""
    t0 = time.perf_counter()
    final_state = graph.invoke(
        {"question": item["question"]},
        config={"configurable": {"thread_id": thread_id}},
    )
    dt = time.perf_counter() - t0
    docs = final_state.get("documents") or []
    return {
        "question": item["question"],
        "type": item.get("type", "unknown"),
        "reference": item.get("reference", ""),
        "answer": final_state.get("generation", ""),
        "iterations": final_state.get("iterations", 0),
        "web_search_used": final_state.get("web_search_used", False),
        "halluc_retries": final_state.get("halluc_retries", 0),
        "n_docs": len(docs),
        "sources": ";".join(
            sorted({d.metadata.get("source", "?") for d in docs})
        ),
        "latency_s": round(dt, 2),
    }


def run_eval(questions_path: Path, out_dir: Path) -> pd.DataFrame:
    assert_keys()
    out_dir.mkdir(parents=True, exist_ok=True)
    items = _load_questions(questions_path)
    if not items:
        raise RuntimeError(
            f"No questions in {questions_path}. Add 10 simple + 10 complex items "
            f"(see file header comment for the format)."
        )

    rows: list[dict] = []
    for i, item in enumerate(items, start=1):
        print(f"[{i:02d}/{len(items)}] ({item.get('type','?')}) {item['question']}")
        try:
            rows.append(_run_one(item, thread_id=f"eval-{i}"))
        except Exception as e:
            print(f"  [!] Failure: {e}")
            rows.append(
                {
                    "question": item["question"],
                    "type": item.get("type", "unknown"),
                    "reference": item.get("reference", ""),
                    "answer": f"ERROR: {e}",
                    "iterations": 0,
                    "web_search_used": False,
                    "halluc_retries": 0,
                    "n_docs": 0,
                    "sources": "",
                    "latency_s": 0.0,
                }
            )

    df = pd.DataFrame(rows)
    csv_path = out_dir / "results.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\n[done] Wrote {csv_path}")

    # Console summary by question type.
    if not df.empty:
        summary = (
            df.groupby("type")
            .agg(
                count=("question", "count"),
                avg_latency_s=("latency_s", "mean"),
                avg_iters=("iterations", "mean"),
                web_search_rate=("web_search_used", "mean"),
                avg_docs=("n_docs", "mean"),
            )
            .round(2)
        )
        print("\n--- Summary by question type ---")
        print(summary.to_string())
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the eval suite.")
    parser.add_argument(
        "--questions",
        type=Path,
        default=EVAL_DIR / "questions.json",
        help="Path to the questions JSON file.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=EVAL_DIR / "results",
        help="Directory to write results into.",
    )
    args = parser.parse_args()
    run_eval(args.questions, args.out)


if __name__ == "__main__":
    main()
