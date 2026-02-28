"""
test_hallucination.py
=====================
Tests the full pipeline using a SIMULATED hallucinating primary agent.

Why simulate?
  Modern LLMs like Mistral are trained to refuse or express uncertainty
  on clearly unanswerable questions. To properly evaluate a hallucination
  detection system we need controlled, known hallucinations — i.e. we
  know the correct answer and we inject a wrong one as the primary output.

  This mirrors the evaluation methodology used in SelfCheckGPT (Manakul et
  al., 2023) which uses the WikiBio dataset where ground truth is known.

Three test scenarios are run:
  1. TRUE POSITIVE  — primary agent gives a wrong confident answer
                      system should detect HIGH risk
  2. TRUE NEGATIVE  — primary agent gives the correct answer
                      system should detect LOW risk
  3. PARTIAL        — primary agent gives a half-right answer
                      system should detect MODERATE risk

Run: python test_hallucination.py
"""

import datetime
import hashlib
from verification_agents import run_all_verification_agents
from aggregation_module import run_aggregation
from voting_module import run_voting


def fake_primary(question: str, fake_answer: str) -> dict:
    """
    Simulates a hallucinating primary agent by injecting a known wrong answer.
    The structure matches exactly what query_primary_agent() returns so the
    rest of the pipeline is unaffected.
    """
    answer_hash = hashlib.sha256(fake_answer.encode()).hexdigest()
    print(f"\n[SIMULATED Primary Agent] Injecting answer:")
    print(f"  Question : {question}")
    print(f"  Answer   : {fake_answer}")
    print(f"  Hash     : {answer_hash[:16]}...")
    return {
        "agent":     "primary",
        "model":     "simulated",
        "question":  question,
        "answer":    fake_answer,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "hash":      answer_hash,
    }


def run_test(label, question, primary_answer):
    print("\n" + "=" * 60)
    print(f"TEST: {label}")
    print("=" * 60)

    p = fake_primary(question, primary_answer)
    v = run_all_verification_agents(question)
    a = run_aggregation(p, v)
    r = run_voting(a)

    print(f"\n  -- SUMMARY ----------------------------------------")
    print(f"  Test             : {label}")
    print(f"  Injected Answer  : {primary_answer[:80]}...")
    print(f"  Agreement Score  : {a['agreement_score']}")
    print(f"  Final Risk Score : {r['final_verdict']['final_risk_score']}")
    print(f"  Risk Level       : {r['final_verdict']['risk_level']}")
    print(f"  Action           : {r['final_verdict']['action']}")
    print(f"  {r['final_verdict']['label']}")
    print(f"  ---------------------------------------------------")

    return {
        "test":       label,
        "agreement":  a["agreement_score"],
        "risk_score": r["final_verdict"]["final_risk_score"],
        "risk_level": r["final_verdict"]["risk_level"],
        "action":     r["final_verdict"]["action"],
    }


if __name__ == "__main__":

    results = []

    # Test 1: HALLUCINATED answer — Buzz Aldrin, wrong year
    results.append(run_test(
        label          = "TRUE POSITIVE - Hallucinated Answer",
        question       = "Who was the first person to walk on the moon and in what year?",
        primary_answer = "Buzz Aldrin was the first person to walk on the moon in 1971 during the Apollo 14 mission.",
    ))

    # Test 2: CORRECT answer — should be accepted
    results.append(run_test(
        label          = "TRUE NEGATIVE - Correct Answer",
        question       = "Who was the first person to walk on the moon and in what year?",
        primary_answer = "Neil Armstrong was the first person to walk on the moon on July 20, 1969, during the Apollo 11 mission.",
    ))

    # Test 3: PARTIAL hallucination — right person, wrong year
    results.append(run_test(
        label          = "PARTIAL - Half-Right Answer",
        question       = "Who was the first person to walk on the moon and in what year?",
        primary_answer = "Neil Armstrong was the first person to walk on the moon in 1971.",
    ))

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"{'Test':<38} {'Risk Score':<12} {'Level':<12} {'Action'}")
    print("-" * 60)
    for r in results:
        print(f"{r['test']:<38} {r['risk_score']:<12} {r['risk_level']:<12} {r['action']}")

    print("\nSave this output — it is your Table 1 in the final report.")