"""Base class for LLM-as-judge evaluators."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import google.generativeai as genai


@dataclass
class EvalResult:
    """The output of a single evaluation."""
    eval_name: str
    score: float              # 0.0 – 1.0
    passed: bool              # score >= threshold
    reasoning: str            # judge's explanation
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "eval_name": self.eval_name,
            "score": round(self.score, 4),
            "passed": self.passed,
            "reasoning": self.reasoning,
            **self.metadata,
        }


class LLMJudge:
    """
    Base class for LLM-as-judge evaluators.

    Uses Gemini Pro to score LLM outputs against a structured rubric.
    Subclasses implement `build_prompt()` and `parse_score()` for their
    specific evaluation criteria.

    Why LLM-as-judge?
    -----------------
    Rule-based metrics (ROUGE, BLEU) miss semantic correctness. An LLM judge
    can evaluate whether an answer is *meaningfully* correct, faithful to a
    source document, or contains hallucinations — the same way a human reviewer
    would, at scale.
    """

    name: str = "base"
    threshold: float = 0.7     # minimum score to count as "passed"

    def __init__(
        self,
        model_name: str = "gemini-pro",
        api_key: str | None = None,
        temperature: float = 0.0,
    ):
        key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError(
                "No API key provided. Set GOOGLE_API_KEY or pass api_key."
            )
        genai.configure(api_key=key)
        self.model = genai.GenerativeModel(
            model_name,
            generation_config=genai.GenerationConfig(temperature=temperature),
        )

    def build_prompt(self, **kwargs) -> str:
        """Return the judge prompt for this evaluator. Subclasses must implement."""
        raise NotImplementedError

    def parse_score(self, response_text: str) -> tuple[float, str]:
        """
        Extract (score, reasoning) from the judge's response.

        Expected response format (enforced by prompt):
            {"score": 0.85, "reasoning": "The answer correctly references..."}
        """
        text = response_text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        try:
            data = json.loads(text)
            score = float(data.get("score", 0.0))
            reasoning = str(data.get("reasoning", ""))
            return max(0.0, min(1.0, score)), reasoning
        except (json.JSONDecodeError, ValueError):
            # Fallback: look for a bare float on the first line
            first_line = response_text.strip().split("\n")[0]
            try:
                score = float(first_line)
                return max(0.0, min(1.0, score)), response_text
            except ValueError:
                return 0.0, f"Could not parse score from: {response_text[:200]}"

    def evaluate(self, **kwargs) -> EvalResult:
        """Run the evaluation and return an EvalResult."""
        prompt = self.build_prompt(**kwargs)
        response = self.model.generate_content(prompt)
        score, reasoning = self.parse_score(response.text)
        return EvalResult(
            eval_name=self.name,
            score=score,
            passed=score >= self.threshold,
            reasoning=reasoning,
        )
