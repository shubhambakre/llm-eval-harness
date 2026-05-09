"""
Correctness evaluator — measures whether an LLM answer actually answers
the question correctly, compared to a reference (ground-truth) answer.

Unlike faithfulness (which only checks grounding in context),
correctness checks whether the answer is *right* — semantically aligned
with the ground truth. Useful for QA benchmarks and regression testing.
"""

from __future__ import annotations

from .judge import EvalResult, LLMJudge

_PROMPT = """\
You are an expert correctness judge. Compare an AI-generated answer to a
reference (ground-truth) answer and score how correct the AI answer is.

Correctness means: the AI answer conveys the same key facts and conclusions
as the reference answer. Minor phrasing differences are fine. Penalise:
  - Missing key facts present in the reference
  - Factually wrong claims
  - Answers that technically respond but miss the point of the question

Question: {question}

Reference answer (ground truth):
\"\"\"
{reference}
\"\"\"

AI-generated answer:
\"\"\"
{answer}
\"\"\"

Score correctness from 0.0 to 1.0:
  1.0 = AI answer conveys all key facts from the reference, no errors
  0.5 = AI answer is partially correct, missing some key points
  0.0 = AI answer is wrong, irrelevant, or contradicts the reference

Respond with ONLY valid JSON:
{{"score": <float 0.0-1.0>, "reasoning": "<one or two sentences>"}}
"""


class CorrectnessEval(LLMJudge):
    """
    Evaluates whether an LLM answer is correct relative to a reference answer.

    Use this for regression testing (does a new model version answer as well
    as the previous one?) and for QA dataset benchmarking.
    """

    name = "correctness"
    threshold = 0.7

    def build_prompt(self, *, question: str, reference: str, answer: str) -> str:
        return _PROMPT.format(question=question, reference=reference, answer=answer)

    def evaluate(self, *, question: str, reference: str, answer: str) -> EvalResult:
        result = super().evaluate(question=question, reference=reference, answer=answer)
        result.metadata = {
            "question_preview": question[:120],
            "reference_preview": reference[:120],
        }
        return result
