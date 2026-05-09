"""
Faithfulness evaluator — measures whether an LLM answer is grounded in
the provided context, with no hallucinated or fabricated claims.

This is the single most important eval for RAG systems: a high-scoring
answer is one where every factual claim can be traced back to a source
chunk. An unfaithful answer may be fluent and plausible but wrong.
"""

from __future__ import annotations

from .judge import EvalResult, LLMJudge

_PROMPT = """\
You are a strict faithfulness judge. Your job is to determine whether an
AI-generated answer is fully grounded in the provided context documents.

Faithfulness means: every factual claim in the answer can be directly
supported by the context. An answer is UNFAITHFUL if it:
  - Introduces facts not present in the context
  - Makes inferences that go beyond what the context supports
  - Contradicts the context
  - Fabricates numbers, names, dates, or events

Context documents:
\"\"\"
{context}
\"\"\"

Question asked: {question}

AI-generated answer:
\"\"\"
{answer}
\"\"\"

Score the faithfulness from 0.0 to 1.0:
  1.0 = every claim is directly supported by the context
  0.5 = some claims are supported, some are hallucinated or inferred
  0.0 = the answer contradicts or ignores the context entirely

Respond with ONLY valid JSON in this exact format:
{{"score": <float 0.0-1.0>, "reasoning": "<one or two sentences explaining the score>"}}
"""


class FaithfulnessEval(LLMJudge):
    """
    Evaluates whether an LLM answer is faithful to the retrieved context.

    Primarily used for RAG system evaluation. A score below 0.7 suggests
    the model is hallucinating facts not present in the source documents.
    """

    name = "faithfulness"
    threshold = 0.7

    def build_prompt(self, *, question: str, context: str, answer: str) -> str:
        return _PROMPT.format(question=question, context=context, answer=answer)

    def evaluate(self, *, question: str, context: str, answer: str) -> EvalResult:
        result = super().evaluate(question=question, context=context, answer=answer)
        result.metadata = {
            "question_preview": question[:120],
            "answer_preview": answer[:120],
        }
        return result
