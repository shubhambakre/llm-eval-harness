#!/usr/bin/env python3
"""
Example: evaluate RAG pipeline outputs for faithfulness and correctness.

Run from the repo root:
    export GOOGLE_API_KEY=your-key
    python examples/eval_rag_outputs.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals import FaithfulnessEval, CorrectnessEval
from runners import BatchRunner
from reporters import Reporter

def main():
    # Load sample dataset
    data_path = Path(__file__).parent / "sample_data.json"
    with open(data_path) as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} examples from {data_path.name}\n")

    # Configure evaluators
    evaluators = [
        FaithfulnessEval(),   # Is the answer grounded in the retrieved context?
        CorrectnessEval(),    # Is the answer semantically correct vs ground truth?
    ]

    # Run batch evaluation
    runner = BatchRunner(evaluators=evaluators, sleep_between_calls=0.5)
    print("Running evaluations...")
    records = runner.run(dataset)

    # Summarise
    summary = runner.summarise(records)
    Reporter.print_summary(summary)

    # Export results
    out_dir = Path("results")
    Reporter.to_json(records, out_dir / "eval_results.json")
    Reporter.to_csv(records, out_dir / "eval_results.csv")

    # Exit non-zero if any eval is below threshold
    for name, stats in summary["evals"].items():
        if stats["pass_rate"] < 0.7:
            print(f"⚠️  {name} pass rate {stats['pass_rate']:.1%} is below 70% threshold")
            sys.exit(1)

    print("✅  All evaluations passed.")

if __name__ == "__main__":
    main()
