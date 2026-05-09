"""Run one or more evaluators over a dataset of examples."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from evals.judge import EvalResult, LLMJudge


@dataclass
class EvalRecord:
    """A single row in the evaluation output."""
    example_id: str
    results: list[EvalResult]
    error: str | None = None
    duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.example_id,
            "duration_s": round(self.duration_s, 3),
            "error": self.error,
            "evals": [r.to_dict() for r in self.results],
        }


class BatchRunner:
    """
    Runs a list of evaluators over a dataset of examples.

    Each example is a dict whose keys are passed as kwargs to each evaluator.
    Evaluators that don't recognise a key simply won't receive it (they use
    explicit keyword arguments, so extra keys are ignored).

    Usage:
        runner = BatchRunner(evaluators=[FaithfulnessEval(), CorrectnessEval()])
        records = runner.run(dataset)
        summary = runner.summarise(records)
    """

    def __init__(
        self,
        evaluators: list[LLMJudge],
        sleep_between_calls: float = 0.5,
    ):
        """
        Args:
            evaluators: List of LLMJudge instances to run on each example.
            sleep_between_calls: Seconds to wait between API calls to avoid
                                 rate-limit errors (default 0.5s).
        """
        self.evaluators = evaluators
        self.sleep = sleep_between_calls

    def run(self, dataset: list[dict[str, Any]]) -> list[EvalRecord]:
        """
        Evaluate every example in the dataset.

        Args:
            dataset: List of dicts. Each dict must contain the keys expected
                     by the configured evaluators (e.g. question, context,
                     answer, reference).

        Returns:
            List of EvalRecord — one per example.
        """
        records: list[EvalRecord] = []
        total = len(dataset)

        for i, example in enumerate(dataset):
            example_id = str(example.get("id", i))
            print(f"  [{i+1}/{total}] Evaluating example {example_id}...")

            results: list[EvalResult] = []
            error: str | None = None
            start = time.monotonic()

            for evaluator in self.evaluators:
                try:
                    # Pass only the kwargs the evaluator's evaluate() accepts
                    import inspect
                    sig = inspect.signature(evaluator.evaluate)
                    valid_keys = {
                        p for p in sig.parameters
                        if p not in ("self", "kwargs")
                    }
                    kwargs = {k: v for k, v in example.items() if k in valid_keys}
                    result = evaluator.evaluate(**kwargs)
                    results.append(result)
                    if self.sleep > 0:
                        time.sleep(self.sleep)
                except Exception as exc:
                    error = f"{evaluator.name}: {exc}"
                    print(f"    ⚠️  {error}")

            duration = time.monotonic() - start
            records.append(EvalRecord(
                example_id=example_id,
                results=results,
                error=error,
                duration_s=duration,
            ))

        return records

    def summarise(self, records: list[EvalRecord]) -> dict[str, Any]:
        """
        Compute aggregate statistics across all records.

        Returns:
            Dict with per-eval mean score, pass rate, and overall stats.
        """
        from collections import defaultdict
        scores: dict[str, list[float]] = defaultdict(list)
        passed: dict[str, list[bool]] = defaultdict(list)

        for record in records:
            for result in record.results:
                scores[result.eval_name].append(result.score)
                passed[result.eval_name].append(result.passed)

        summary: dict[str, Any] = {
            "total_examples": len(records),
            "errors": sum(1 for r in records if r.error),
            "evals": {},
        }
        for name in scores:
            s = scores[name]
            p = passed[name]
            summary["evals"][name] = {
                "mean_score": round(sum(s) / len(s), 4) if s else 0,
                "pass_rate": round(sum(p) / len(p), 4) if p else 0,
                "n": len(s),
            }
        return summary
