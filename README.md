# llm-eval-harness

A lightweight Python framework for evaluating LLM outputs using the **LLM-as-judge** pattern. Measures faithfulness, correctness, and hallucination on any question-answering or RAG system — with batch evaluation, JSON/CSV export, and a summary scorecard.

---

## Why evaluation matters

Calling an LLM and getting a fluent response is easy. Knowing whether the response is *correct, grounded, and trustworthy* at scale requires systematic evaluation. This harness automates that: give it a dataset of (question, context, answer) triples and it returns scores, pass/fail flags, reasoning, and aggregate statistics.

---

## Architecture

```
dataset (JSON)
      │
      ▼
┌─────────────────────────────┐
│  BatchRunner                │  iterates examples, handles errors
│  (runners/batch_runner.py)  │
└─────────────┬───────────────┘
              │  for each example × evaluator
              ▼
┌─────────────────────────────┐
│  LLMJudge (base class)      │  builds prompt, calls Gemini Pro,
│  (evals/judge.py)           │  parses JSON score + reasoning
└──────┬──────────┬───────────┘
       │          │
       ▼          ▼
┌──────────┐  ┌─────────────┐
│ Faithful-│  │ Correctness │   (add your own by subclassing LLMJudge)
│ nessEval │  │ Eval        │
└──────────┘  └─────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Reporter                   │  prints scorecard, writes JSON + CSV
│  (reporters/summary.py)     │
└─────────────────────────────┘
```

---

## Evaluators

### `FaithfulnessEval`
Scores whether every claim in an LLM answer is directly supported by the retrieved context documents. Critical for RAG pipelines — a faithful answer introduces no hallucinated facts.

**Input:** `question`, `context`, `answer`
**Score:** 1.0 = fully grounded · 0.0 = contradicts or ignores context

### `CorrectnessEval`
Scores whether an LLM answer conveys the same key facts as a reference (ground-truth) answer. Use for regression testing across model versions or for QA dataset benchmarks.

**Input:** `question`, `reference`, `answer`
**Score:** 1.0 = all key facts correct · 0.0 = wrong or irrelevant

### Adding your own
Subclass `LLMJudge`, set `name` and `threshold`, implement `build_prompt()`. That's it.

```python
class ConcisernessEval(LLMJudge):
    name = "conciseness"
    threshold = 0.6

    def build_prompt(self, *, question, answer):
        return f"Score how concisely this answers the question...\nQ: {question}\nA: {answer}"
```

---

## Quick Start

```bash
git clone https://github.com/shubhambakre/llm-eval-harness.git
cd llm-eval-harness
pip install -r requirements.txt

export GOOGLE_API_KEY=your-gemini-api-key
python examples/eval_rag_outputs.py
```

Example output:

```
Loaded 3 examples from sample_data.json

Running evaluations...
  [1/3] Evaluating example ex_001...
  [2/3] Evaluating example ex_002...
  [3/3] Evaluating example ex_003...

============================================================
  EVALUATION SUMMARY  —  3 examples
============================================================
  faithfulness          score: 0.717  [██████████████░░░░░░]  pass: 66.7%  (n=3)
  correctness           score: 0.783  [███████████████░░░░░]  pass: 66.7%  (n=3)
============================================================

  Results saved → results/eval_results.json
  Results saved → results/eval_results.csv
```

---

## Dataset Format

JSON array of examples. Keys vary by evaluator:

```json
[
  {
    "id": "ex_001",
    "question": "What is the return policy for electronics?",
    "context": "Electronics may be returned within 30 days...",
    "answer": "You can return electronics within 30 days...",
    "reference": "The return window is 30 days from purchase..."
  }
]
```

- `FaithfulnessEval` uses: `question`, `context`, `answer`
- `CorrectnessEval` uses: `question`, `reference`, `answer`
- `BatchRunner` automatically routes the right keys to each evaluator

---

## Output

**JSON** (`results/eval_results.json`): full per-example results with scores, reasoning, and metadata.

**CSV** (`results/eval_results.csv`): flat table, one row per (example, evaluator) — ready for analysis in pandas or Excel.

**Scorecard** (stdout): mean score, visual bar, pass rate per evaluator.

---

## Project Structure

```
llm-eval-harness/
├── evals/
│   ├── judge.py           # LLMJudge base class — prompt → Gemini → score parser
│   ├── faithfulness.py    # FaithfulnessEval — grounding check for RAG systems
│   └── correctness.py     # CorrectnessEval — semantic match vs. ground truth
├── runners/
│   └── batch_runner.py    # BatchRunner — iterate dataset, collect EvalRecords
├── reporters/
│   └── summary.py         # Reporter — scorecard, JSON export, CSV export
├── examples/
│   ├── eval_rag_outputs.py  # End-to-end example script
│   └── sample_data.json     # 3 sample (question, context, answer, reference) triples
├── requirements.txt
└── .env.example
```

---

## Design Notes

**LLM-as-judge** scores on semantics, not surface text. ROUGE and BLEU penalise correct answers that use different wording. A Gemini Pro judge reads the answer the way a human reviewer would.

**JSON-structured scores** make the judge's output machine-parseable and deterministic (`temperature=0`). The harness falls back gracefully if the model returns an unexpected format.

**Extensible by design.** Every evaluator is a single class with two methods. Adding a conciseness, toxicity, or domain-specific evaluator takes ~15 lines.

**Fail-fast in CI.** `eval_rag_outputs.py` exits with code 1 if any evaluator's pass rate drops below threshold — ready to wire into a GitHub Actions pipeline to catch regressions before they ship.

---

*Stack: Python · Google Gemini Pro · LLM-as-judge*
