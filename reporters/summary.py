"""Export evaluation results to JSON and CSV, and print a summary table."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from runners.batch_runner import EvalRecord


class Reporter:
    """Formats and exports evaluation results."""

    @staticmethod
    def print_summary(summary: dict[str, Any]) -> None:
        """Print a human-readable summary table to stdout."""
        print("\n" + "=" * 60)
        print(f"  EVALUATION SUMMARY  —  {summary['total_examples']} examples")
        print("=" * 60)
        for name, stats in summary["evals"].items():
            bar = _progress_bar(stats["mean_score"])
            print(
                f"  {name:<20}  score: {stats['mean_score']:.3f}  {bar}  "
                f"pass: {stats['pass_rate']*100:.1f}%  (n={stats['n']})"
            )
        if summary["errors"]:
            print(f"\n  ⚠️   {summary['errors']} example(s) had errors")
        print("=" * 60 + "\n")

    @staticmethod
    def to_json(records: list[EvalRecord], path: str | Path) -> None:
        """Write all evaluation records to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in records], f, indent=2)
        print(f"  Results saved → {path}")

    @staticmethod
    def to_csv(records: list[EvalRecord], path: str | Path) -> None:
        """Write a flat CSV with one row per (example, evaluator) pair."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for record in records:
            base = {"id": record.example_id, "duration_s": record.duration_s, "error": record.error or ""}
            if not record.results:
                rows.append(base)
                continue
            for result in record.results:
                row = {**base, **result.to_dict()}
                rows.append(row)

        if not rows:
            return

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys(), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Results saved → {path}")


def _progress_bar(score: float, width: int = 20) -> str:
    filled = round(score * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"
