"""
main.py - entry point for v0.5

CSC8208 Multi-Agent Hallucination Detection Framework — Newcastle University
v0.5: ground truth fact checking across 5 questions, 10 rounds, no injection

runs each question through:
  1. 10-round discussion (clean prompt — agents just re-evaluate)
  2. ground truth fact checking per agent per round
  3. hallucination heatmap saved as PNG
  4. full results saved as JSON

usage:
  python main.py
"""

import os        # directory creation
import json      # saving results
import datetime  # timestamps
import re        # slug generation

from config       import QUESTIONS, TOTAL_ROUNDS
from experiment   import run_experiment
from ground_truth import evaluate_experiment, print_evaluation
from visualizer   import plot_hallucination_heatmap

# results directory — auto-created
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def make_timestamp():
    # readable timestamp — e.g. 07Mar2026_02-30PM
    return datetime.datetime.now().strftime("%d%b%Y_%I-%M%p")


def make_slug(question, index):
    # short folder-safe id from question — e.g. q1_who_invented_telephone
    words = re.sub(r"[^a-z0-9 ]", "", question.lower()).split()
    return f"q{index + 1}_{'_'.join(words[:4])}"


def run_question(q_config, index, run_ts):
    question = q_config["question"]   # the question string
    facts    = q_config["facts"]      # the ground truth facts dict
    slug     = make_slug(question, index)

    print(f"\n{'='*55}")
    print(f"  QUESTION {index + 1}: {question}")
    print(f"{'='*55}")

    # --- step 1: run the discussion ---
    experiment_results = run_experiment(question, total_rounds=TOTAL_ROUNDS)

    # --- step 2: check ground truth per agent per round ---
    # pass canonical answer + question so NLI can run alongside keyword check
    evaluation = evaluate_experiment(
        experiment_results,
        facts,
        canonical_answer=q_config["answer"],
        question=question,
    )

    # print summary table to terminal
    print_evaluation(evaluation, question)

    # --- step 3: save full results to json ---
    ts        = make_timestamp()
    json_path = os.path.join(RESULTS_DIR, f"{slug}_{run_ts}.json")

    # combine experiment data and ground truth evaluation into one json
    output = {
        "question":          question,
        "ground_truth":      q_config["answer"],
        "facts_checked":     list(facts.keys()),
        "total_rounds":      TOTAL_ROUNDS,
        "experiment":        experiment_results,
        "ground_truth_eval": evaluation
    }

    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  [json] saved: {json_path}")

    # --- step 4: plot hallucination heatmap ---
    plot_hallucination_heatmap(evaluation, question, RESULTS_DIR, slug, run_ts)


def main():
    run_ts = make_timestamp()   # one timestamp for the whole run

    print("=" * 55)
    print("  CSC8208 — Multi-Agent Hallucination Detection v0.5")
    print("  Ground Truth Fact Checking — No Injection")
    print(f"  questions : {len(QUESTIONS)}")
    print(f"  rounds    : {TOTAL_ROUNDS}")
    print(f"  timestamp : {run_ts}")
    print("=" * 55)

    # run all 5 questions in sequence
    for i, q_config in enumerate(QUESTIONS):
        run_question(q_config, i, run_ts)

    print(f"\n{'='*55}")
    print(f"  ALL DONE — results saved to: {RESULTS_DIR}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
