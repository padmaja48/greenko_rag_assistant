"""Evaluate retrieval and abstention behavior on the bundled datasets.

The expanded dataset is intentionally synthetic and source-labelled. It is used
to exercise the complete indexing and retrieval path, not to claim production
accuracy or domain safety.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# Allow running as `python -m backend.evaluate` or `python backend/evaluate.py`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.rag import MIN_ANSWER_SCORE, pipeline  # noqa: E402


MANUAL_REVIEW_EXAMPLES = [
    {
        "question": "What should be checked for inverter derating?",
        "expected_source": "solar_inverter_sop_test.md",
        "review_focus": "Answer should mention temperature, ventilation, filters, fans, and related inverter evidence.",
        "review_result": "Pass: retrieved evidence is from the solar inverter SOP and contains actionable inspection steps.",
    },
    {
        "question": "How should gearbox vibration be investigated?",
        "expected_source": "wind_turbine_maintenance_test.md",
        "review_focus": "Answer should describe vibration trends/frequencies, operating condition, sensor checks, and escalation.",
        "review_result": "Pass: retrieved evidence is from the wind maintenance guide and supports a procedural answer.",
    },
    {
        "question": "How do I bake a chocolate cake?",
        "expected_source": None,
        "review_focus": "The assistant should abstain instead of inventing an operations answer.",
        "review_result": "Pass: the assistant returns the insufficient-evidence response with no citation.",
    },
]


def _rank_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    if not total:
        return {
            "top1_source_accuracy": 0.0,
            "top3_source_accuracy": 0.0,
            "mean_reciprocal_rank": 0.0,
            "abstentions": 0,
            "total": 0,
        }

    top1_hits = sum(row["top1_hit"] for row in rows)
    top3_hits = sum(row["top3_hit"] for row in rows)
    reciprocal_ranks = [row["reciprocal_rank"] for row in rows]
    return {
        "top1_source_accuracy": round(top1_hits / total, 3),
        "top3_source_accuracy": round(top3_hits / total, 3),
        "mean_reciprocal_rank": round(sum(reciprocal_ranks) / total, 3),
        "abstentions": sum(row["abstained"] for row in rows),
        "total": total,
    }


def evaluate_retrieval(csv_path: Path) -> dict[str, Any]:
    df = pd.read_csv(csv_path).dropna(subset=["question", "expected_source"])
    rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        question = str(row["question"])
        expected = str(row["expected_source"])
        retrieved = pipeline.retrieve(question, top_k=4)
        sources = [item["source"] for item in retrieved]
        rank = sources.index(expected) + 1 if expected in sources else None
        top_score = retrieved[0]["score"] if retrieved else 0.0
        rows.append(
            {
                "question": question,
                "expected_source": expected,
                "top_source": sources[0] if sources else None,
                "top_score": top_score,
                "top1_hit": bool(rank == 1),
                "top3_hit": bool(rank is not None and rank <= 3),
                "rank": rank,
                "reciprocal_rank": round(1 / rank, 3) if rank else 0.0,
                "abstained": bool(not retrieved or top_score < MIN_ANSWER_SCORE),
                "retrieved_sources": sources,
            }
        )

    metrics = _rank_metrics(rows)
    return {
        "dataset": csv_path.name,
        "metric_scope": "synthetic source-level retrieval benchmark",
        "metrics": metrics,
        "details": rows,
    }


def evaluate_off_topic(csv_path: Path) -> dict[str, Any]:
    df = pd.read_csv(csv_path).dropna(subset=["question"])
    rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        question = str(row["question"])
        result = pipeline.ask(question)
        check = result["hallucination_check"]
        abstained = bool(check.get("abstained"))
        rows.append(
            {
                "question": question,
                "expected_abstain": True,
                "abstained": abstained,
                "correct": abstained,
                "citations": result.get("citations", []),
                "answer_preview": result.get("answer", "")[:180],
            }
        )

    correct = sum(row["correct"] for row in rows)
    total = len(rows)
    return {
        "dataset": csv_path.name,
        "metric": "off_topic_abstention_accuracy",
        "accuracy": round(correct / total, 3) if total else 0.0,
        "abstained": sum(row["abstained"] for row in rows),
        "total": total,
        "details": rows,
    }


def build_report() -> dict[str, Any]:
    smoke = evaluate_retrieval(ROOT / "data" / "questions.csv")
    expanded = evaluate_retrieval(ROOT / "data" / "questions_test.csv")
    off_topic = evaluate_off_topic(ROOT / "data" / "off_topic_questions.csv")
    return {
        "metric_scope": (
            "Synthetic/source-level retrieval benchmark. Results are not a production "
            "accuracy or safety guarantee."
        ),
        "smoke_test": smoke,
        "expanded_evaluation": expanded,
        "off_topic_evaluation": off_topic,
        "manual_review_examples": MANUAL_REVIEW_EXAMPLES,
    }


def main() -> None:
    report = build_report()
    out_path = ROOT / "evaluation_results.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
