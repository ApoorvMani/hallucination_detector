"""
regeneration_module.py
======================
The Regeneration Module is triggered when the voting module returns
FLAG or REGENERATE. It implements a single-round debate mechanism:

  1. The primary agent's original answer is shown back to it
  2. All verification agents' answers are presented as evidence
  3. The primary agent is asked to re-evaluate and correct itself

This extends the multiagent debate paradigm (Du et al., 2023) with a
key architectural difference: agents generated their answers independently
BEFORE this stage. The primary agent is only exposed to other answers
AFTER the hallucination risk has already been quantified. This prevents
the echo-chamber convergence risk present in standard debate systems.

If the regenerated answer scores lower risk than the original, it replaces
it. If not, the original is kept and the discrepancy is logged.

Run standalone:  python regeneration_module.py
"""

import ollama
import hashlib
import datetime
import json
from aggregation_module import run_aggregation


def build_debate_prompt(original_answer: str, verifier_answers: list[dict]) -> str:
    """
    Constructs the re-evaluation prompt shown to the primary agent.

    The prompt presents:
      - The agent's own original answer
      - Each verifier's independent answer
      - An instruction to reconsider and correct if needed

    Args:
        original_answer:  The primary agent's first response.
        verifier_answers: List of result dicts from verification agents.

    Returns:
        A formatted string prompt for re-evaluation.
    """
    verifier_block = ""
    for i, v in enumerate(verifier_answers, 1):
        verifier_block += f"\n  Agent {i} ({v['style']} style): {v['answer']}"

    prompt = f"""You previously answered a question as follows:

YOUR ORIGINAL ANSWER:
{original_answer}

OTHER INDEPENDENT AGENTS answered the same question as follows:
{verifier_block}

Carefully compare your answer with the other agents' answers.
- If your answer is correct and consistent with theirs, restate it confidently.
- If your answer contains errors, correct them now based on the evidence above.
- If there is genuine uncertainty or disagreement, acknowledge it explicitly.

Provide your final, corrected answer below. Be concise and factual."""

    return prompt


def run_regeneration(
    question:             str,
    primary_result:       dict,
    verification_results: list,
    voting_report:        dict,
    model:                str = "mistral"
) -> dict:
    """
    Full regeneration pipeline. Only runs if the voting action is FLAG or REGENERATE.
    Compares the regenerated answer's risk score against the original.

    Args:
        question:             Original question string.
        primary_result:       Output from primary_agent.py
        verification_results: Output list from verification_agents.py
        voting_report:        Output from voting_module.py
        model:                Ollama model to use.

    Returns:
        Dict containing original result, regenerated result, and comparison.
    """
    action = voting_report["final_verdict"]["action"]

    # Only regenerate if flagged — skip if already accepted
    if action == "ACCEPT":
        print("\n[Regeneration Module] Action is ACCEPT — no regeneration needed.")
        return {
            "regeneration_triggered": False,
            "reason": "Original output accepted by voting module",
            "original_risk": voting_report["final_verdict"]["final_risk_score"],
            "final_answer": primary_result["answer"],
        }

    print(f"\n[Regeneration Module] Action is {action} — triggering re-evaluation...")
    print(f"  Original risk score: {voting_report['final_verdict']['final_risk_score']}")

    # Build the debate prompt
    debate_prompt = build_debate_prompt(
        original_answer  = primary_result["answer"],
        verifier_answers = verification_results,
    )

    print("\n  [Debate Prompt sent to primary agent]:")
    print("  " + "\n  ".join(debate_prompt.split("\n")[:6]) + "\n  ...")

    # Send to primary agent for re-evaluation
    print("\n  [Primary Agent] Re-evaluating...")
    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a self-correcting factual assistant. "
                    "When shown evidence that your previous answer may be wrong, "
                    "you correct yourself honestly and concisely."
                )
            },
            {
                "role": "user",
                "content": debate_prompt,
            }
        ]
    )

    regenerated_answer = response["message"]["content"].strip()
    regen_hash = hashlib.sha256(regenerated_answer.encode()).hexdigest()

    print(f"\n  Regenerated Answer: {regenerated_answer[:200]}...")
    print(f"  Hash: {regen_hash[:16]}...")

    # Score the regenerated answer using the same verification results
    regen_primary = {
        "agent":     "primary_regenerated",
        "model":     model,
        "question":  question,
        "answer":    regenerated_answer,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "hash":      regen_hash,
    }

    print("\n  [Scoring regenerated answer against verification agents...]")
    regen_aggregation = run_aggregation(regen_primary, verification_results)

    regen_risk = regen_aggregation["agreement_score"]
    original_risk = voting_report["final_verdict"]["final_risk_score"]

    # Compare: did regeneration improve things?
    improved = regen_aggregation["agreement_score"] > voting_report.get(
        "aggregation_agreement", regen_aggregation["agreement_score"] - 0.01
    )

    if regen_aggregation["risk_level"] in ["LOW"] or \
       regen_aggregation["agreement_score"] > 0.85:
        final_answer = regenerated_answer
        outcome = "IMPROVED"
        print(f"\n  ✅ Regeneration improved output — using regenerated answer")
    else:
        final_answer = primary_result["answer"]
        outcome = "NO_IMPROVEMENT"
        print(f"\n  ⚠️  Regeneration did not clearly improve — flagging for human review")

    print(f"  Original agreement : {voting_report['final_verdict']['final_risk_score']:.4f} risk")
    print(f"  Regenerated agreement: {regen_aggregation['agreement_score']:.4f} consensus")
    print(f"  Outcome: {outcome}")

    return {
        "regeneration_triggered":    True,
        "trigger_action":            action,
        "original_answer":           primary_result["answer"],
        "original_risk_score":       voting_report["final_verdict"]["final_risk_score"],
        "regenerated_answer":        regenerated_answer,
        "regenerated_agreement":     regen_aggregation["agreement_score"],
        "regenerated_risk_level":    regen_aggregation["risk_level"],
        "outcome":                   outcome,
        "final_answer":              final_answer,
        "audit": {
            "timestamp":        datetime.datetime.utcnow().isoformat(),
            "question":         question,
            "trigger":          action,
            "original_hash":    primary_result["hash"],
            "regenerated_hash": regen_hash,
            "outcome":          outcome,
        }
    }


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from primary_agent import query_primary_agent
    from verification_agents import run_all_verification_agents
    from aggregation_module import run_aggregation
    from voting_module import run_voting
    import hashlib
    import datetime

    question = "Who was the first person to walk on the moon and in what year?"

    # Simulate a hallucinated primary answer
    fake_answer = "Buzz Aldrin was the first person to walk on the moon in 1971 during the Apollo 14 mission."
    primary_result = {
        "agent":     "primary",
        "model":     "simulated",
        "question":  question,
        "answer":    fake_answer,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "hash":      hashlib.sha256(fake_answer.encode()).hexdigest(),
    }

    print(f"\n[Test] Injected hallucinated answer: {fake_answer}")

    # Run verification, aggregation, voting
    verification_results = run_all_verification_agents(question)
    aggregation_report   = run_aggregation(primary_result, verification_results)
    voting_report        = run_voting(aggregation_report)

    # Now trigger regeneration
    regen_report = run_regeneration(
        question             = question,
        primary_result       = primary_result,
        verification_results = verification_results,
        voting_report        = voting_report,
    )

    print("\n" + "=" * 60)
    print("REGENERATION REPORT:")
    print("=" * 60)
    print(f"  Triggered        : {regen_report['regeneration_triggered']}")
    print(f"  Original Answer  : {regen_report['original_answer']}")
    print(f"  Original Risk    : {regen_report['original_risk_score']}")
    print(f"  Regenerated      : {regen_report['regenerated_answer'][:150]}...")
    print(f"  New Agreement    : {regen_report['regenerated_agreement']}")
    print(f"  Outcome          : {regen_report['outcome']}")
    print(f"\n  FINAL ANSWER     : {regen_report['final_answer']}")