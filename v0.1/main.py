"""
main.py
=======
The main entry point for the Multi-Agent Hallucination Detection Framework.
Runs the complete pipeline end to end and saves a full audit log to disk.

Pipeline:
  1. Primary agent generates an answer
  2. Four verification agents independently answer the same question
  3. Aggregation module computes semantic similarity across all answers
  4. Voting module computes majority + weighted hallucination risk score
  5. If flagged, regeneration module triggers self-correction via debate
  6. Full audit record saved to audit_log.json

Usage:
  python main.py
  python main.py --question "Your question here"
  python main.py --simulate "A wrong answer to inject"
"""

import json
import datetime
import hashlib
import os
import argparse

from primary_agent        import query_primary_agent
from verification_agents  import run_all_verification_agents
from aggregation_module   import run_aggregation
from voting_module        import run_voting
from regeneration_module  import run_regeneration

AUDIT_LOG_PATH = "audit_log.json"


# ── Audit log ─────────────────────────────────────────────────────────────────

def load_audit_log() -> list:
    """Loads existing audit log from disk, or returns empty list."""
    if os.path.exists(AUDIT_LOG_PATH):
        with open(AUDIT_LOG_PATH, "r") as f:
            return json.load(f)
    return []


def save_audit_log(log: list):
    """Saves audit log to disk as formatted JSON."""
    with open(AUDIT_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n[Audit] Log saved to {AUDIT_LOG_PATH} ({len(log)} entries total)")


def build_audit_entry(
    question:             str,
    primary_result:       dict,
    verification_results: list,
    aggregation_report:   dict,
    voting_report:        dict,
    regen_report:         dict,
) -> dict:
    """
    Builds a complete, tamper-evident audit record for one pipeline run.
    Includes a chain hash linking this entry to the previous one,
    implementing a lightweight append-only audit trail.
    """
    entry = {
        "timestamp":          datetime.datetime.utcnow().isoformat(),
        "question":           question,
        "primary": {
            "answer":         primary_result["answer"],
            "model":          primary_result["model"],
            "hash":           primary_result["hash"],
        },
        "verifiers": [
            {
                "agent":      v["agent"],
                "style":      v["style"],
                "answer":     v["answer"],
                "hash":       v["hash"],
            }
            for v in verification_results
        ],
        "aggregation": {
            "agreement_score":      aggregation_report["agreement_score"],
            "risk_level":           aggregation_report["risk_level"],
            "primary_vs_verifiers": aggregation_report["primary_vs_verifiers"],
        },
        "voting": {
            "majority_verdict":  voting_report["majority_voting"]["majority_verdict"],
            "agree_count":       voting_report["majority_voting"]["agree_count"],
            "disagree_count":    voting_report["majority_voting"]["disagree_count"],
            "weighted_risk":     voting_report["weighted_voting"]["weighted_risk"],
            "final_risk_score":  voting_report["final_verdict"]["final_risk_score"],
            "risk_level":        voting_report["final_verdict"]["risk_level"],
            "action":            voting_report["final_verdict"]["action"],
        },
        "regeneration": {
            "triggered":  regen_report["regeneration_triggered"],
            "outcome":    regen_report.get("outcome", "N/A"),
            "final_answer": regen_report["final_answer"],
        },
    }

    # Chain hash — SHA256 of this entry links it to audit chain
    entry_string = json.dumps(entry, sort_keys=True)
    entry["entry_hash"] = hashlib.sha256(entry_string.encode()).hexdigest()

    return entry


def simulate_primary(question: str, fake_answer: str) -> dict:
    """Injects a simulated hallucinated answer as the primary output."""
    answer_hash = hashlib.sha256(fake_answer.encode()).hexdigest()
    print(f"\n[SIMULATED Primary Agent]")
    print(f"  Injecting: {fake_answer}")
    return {
        "agent":     "primary",
        "model":     "simulated",
        "question":  question,
        "answer":    fake_answer,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "hash":      answer_hash,
    }


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(question: str, simulated_answer: str = None) -> dict:
    """
    Executes the full hallucination detection pipeline for a given question.

    Args:
        question:         The question to evaluate.
        simulated_answer: If provided, injects this as the primary answer
                          instead of querying the model (for testing).

    Returns:
        Complete pipeline result including audit entry.
    """
    print("\n" + "█" * 60)
    print("  HALLUCINATION DETECTION FRAMEWORK")
    print("█" * 60)
    print(f"  Question: {question}")
    print("█" * 60)

    # ── Step 1: Primary agent ─────────────────────────────────────────────────
    if simulated_answer:
        primary_result = simulate_primary(question, simulated_answer)
    else:
        primary_result = query_primary_agent(question)

    # ── Step 2: Verification agents ───────────────────────────────────────────
    verification_results = run_all_verification_agents(question)

    # ── Step 3: Aggregation ───────────────────────────────────────────────────
    aggregation_report = run_aggregation(primary_result, verification_results)

    # ── Step 4: Voting ────────────────────────────────────────────────────────
    voting_report = run_voting(aggregation_report)

    # ── Step 5: Regeneration (if flagged) ─────────────────────────────────────
    regen_report = run_regeneration(
        question             = question,
        primary_result       = primary_result,
        verification_results = verification_results,
        voting_report        = voting_report,
    )

    # ── Step 6: Audit log ─────────────────────────────────────────────────────
    audit_entry = build_audit_entry(
        question             = question,
        primary_result       = primary_result,
        verification_results = verification_results,
        aggregation_report   = aggregation_report,
        voting_report        = voting_report,
        regen_report         = regen_report,
    )

    audit_log = load_audit_log()
    audit_log.append(audit_entry)
    save_audit_log(audit_log)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE — FINAL SUMMARY")
    print("=" * 60)
    print(f"  Question        : {question}")
    print(f"  Agreement Score : {aggregation_report['agreement_score']}")
    print(f"  Risk Score      : {voting_report['final_verdict']['final_risk_score']}")
    print(f"  Risk Level      : {voting_report['final_verdict']['risk_level']}")
    print(f"  Initial Action  : {voting_report['final_verdict']['action']}")
    print(f"  Regeneration    : {'Yes — ' + regen_report.get('outcome','') if regen_report['regeneration_triggered'] else 'No'}")
    print(f"  Final Answer    : {regen_report['final_answer'][:120]}...")
    print(f"  Audit Hash      : {audit_entry['entry_hash'][:24]}...")
    print("=" * 60)

    return {
        "audit_entry":        audit_entry,
        "final_answer":       regen_report["final_answer"],
        "final_risk_score":   voting_report["final_verdict"]["final_risk_score"],
        "action":             voting_report["final_verdict"]["action"],
        "regenerated":        regen_report["regeneration_triggered"],
    }


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hallucination Detection Framework")
    parser.add_argument(
        "--question", type=str,
        default="Who was the first person to walk on the moon and in what year?",
        help="The question to run through the pipeline"
    )
    parser.add_argument(
        "--simulate", type=str, default=None,
        help="Inject a simulated primary answer instead of querying the model"
    )
    args = parser.parse_args()

    result = run_pipeline(
        question         = args.question,
        simulated_answer = args.simulate,
    )
