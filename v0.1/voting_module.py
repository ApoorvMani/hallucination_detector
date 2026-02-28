"""
voting_module.py
================
The Voting Module is the final decision layer of the hallucination detection
framework. It receives the aggregation report and applies two independent
voting mechanisms to produce a final hallucination risk score:

  1. Majority Voting   — each agent casts a binary vote (agree/disagree
                         with the primary), the majority determines the verdict.

  2. Weighted Voting   — agents are assigned weights based on how consistently
                         reliable they have been in past queries. Higher-trust
                         agents carry more influence over the final decision.

The final risk score combines both mechanisms, producing a value between
0.0 (no hallucination risk) and 1.0 (certain hallucination).

Academic grounding:
  - Majority voting: standard ensemble learning (Lam & Suen, 1997)
  - Weighted voting: reputation-based consensus (CSC8208 Module Handbook, §3.2.2)
  - Combined scoring: mirrors Byzantine fault-tolerant voting in distributed systems

Run standalone:  python voting_module.py
"""

import json
import datetime
import hashlib


# ── Agent reputation weights ──────────────────────────────────────────────────
# These represent how much we trust each agent's vote.
# In a production system these would update dynamically based on historical
# accuracy. Here we initialise with slight variation to demonstrate the
# weighted voting mechanism. In your evaluation, you can test what happens
# when one agent is set to a very low weight (simulating a faulty agent).

AGENT_WEIGHTS = {
    "primary":    1.0,   # Primary agent (we are checking this one)
    "verifier_1": 1.0,   # Direct style — generally precise
    "verifier_2": 0.9,   # Academic style — sometimes over-elaborates
    "verifier_3": 1.0,   # Cautious style — reliable, flags uncertainty
    "verifier_4": 0.8,   # Analytical style — concise but may oversimplify
}

# Threshold: similarity score below this = agent DISAGREES with primary
DISAGREEMENT_THRESHOLD = 0.75


def majority_vote(primary_vs_verifiers: dict) -> dict:
    """
    Each verifier casts a binary vote: AGREE (1) or DISAGREE (0).
    A verifier agrees if its similarity to the primary exceeds the threshold.

    Majority verdict:
      - More AGREE votes    → output is likely factual
      - More DISAGREE votes → output is likely hallucinated

    Args:
        primary_vs_verifiers: Dict of {agent_id: similarity_score}

    Returns:
        Dict with individual votes, agree/disagree counts, and majority verdict.
    """
    votes = {}
    agree_count    = 0
    disagree_count = 0

    for agent_id, sim_score in primary_vs_verifiers.items():
        if sim_score >= DISAGREEMENT_THRESHOLD:
            votes[agent_id] = {"vote": "AGREE", "similarity": sim_score}
            agree_count += 1
        else:
            votes[agent_id] = {"vote": "DISAGREE", "similarity": sim_score}
            disagree_count += 1

    total = agree_count + disagree_count
    majority_verdict = "AGREE" if agree_count >= (total / 2) else "DISAGREE"

    # Hallucination risk from majority: 0.0 if all agree, 1.0 if all disagree
    majority_risk = round(disagree_count / total, 4) if total > 0 else 0.0

    return {
        "votes":            votes,
        "agree_count":      agree_count,
        "disagree_count":   disagree_count,
        "majority_verdict": majority_verdict,
        "majority_risk":    majority_risk,
    }


def weighted_vote(primary_vs_verifiers: dict) -> dict:
    """
    Each verifier's vote is scaled by its assigned trust weight.
    A high-trust agent that disagrees has more impact than a low-trust agent.

    Weighted risk = sum(weight * disagreement) / sum(weights)
    Where disagreement = 1 - similarity_score

    Args:
        primary_vs_verifiers: Dict of {agent_id: similarity_score}

    Returns:
        Dict with weighted contributions and final weighted risk score.
    """
    weighted_contributions = {}
    total_weight      = 0.0
    weighted_risk_sum = 0.0

    for agent_id, sim_score in primary_vs_verifiers.items():
        weight       = AGENT_WEIGHTS.get(agent_id, 1.0)
        disagreement = 1.0 - sim_score          # 0.0 = perfect agreement, 1.0 = total disagreement
        contribution = weight * disagreement

        weighted_contributions[agent_id] = {
            "similarity":    round(sim_score, 4),
            "weight":        weight,
            "disagreement":  round(disagreement, 4),
            "contribution":  round(contribution, 4),
        }

        total_weight      += weight
        weighted_risk_sum += contribution

    weighted_risk = round(weighted_risk_sum / total_weight, 4) if total_weight > 0 else 0.0

    return {
        "agent_weights":          AGENT_WEIGHTS,
        "weighted_contributions": weighted_contributions,
        "weighted_risk":          weighted_risk,
    }


def compute_final_score(majority_risk: float, weighted_risk: float,
                        agreement_score: float) -> dict:
    """
    Combines majority risk, weighted risk, and agreement score into a
    single final hallucination risk score between 0.0 and 1.0.

    Formula:
      final_risk = 0.4 * majority_risk + 0.4 * weighted_risk + 0.2 * (1 - agreement_score)

    Rationale for weights:
      - Majority and weighted voting each contribute 40% — they are the
        primary detection mechanisms and are complementary.
      - Agreement score contributes 20% — it acts as a global calibration
        signal that captures overall inter-agent divergence beyond pairwise votes.

    Risk thresholds:
      0.0  – 0.20  →  LOW risk     →  ACCEPT
      0.20 – 0.45  →  MODERATE risk →  FLAG
      0.45 – 1.0   →  HIGH risk    →  REGENERATE

    Args:
        majority_risk:   Float from majority_vote()
        weighted_risk:   Float from weighted_vote()
        agreement_score: Float overall agreement from aggregation_module()

    Returns:
        Dict with final score, risk level, and recommended action.
    """
    final_risk = round(
        0.4 * majority_risk +
        0.4 * weighted_risk +
        0.2 * (1.0 - agreement_score),
        4
    )

    if final_risk <= 0.20:
        risk_level = "LOW"
        action     = "ACCEPT"
        label      = "✅ Output accepted — high inter-agent consensus"
    elif final_risk <= 0.45:
        risk_level = "MODERATE"
        action     = "FLAG"
        label      = "⚠️  Output flagged — partial disagreement detected"
    else:
        risk_level = "HIGH"
        action     = "REGENERATE"
        label      = "🚨 Output rejected — significant hallucination risk"

    return {
        "final_risk_score": final_risk,
        "risk_level":       risk_level,
        "action":           action,
        "label":            label,
        "formula": {
            "majority_risk_component":   round(0.4 * majority_risk, 4),
            "weighted_risk_component":   round(0.4 * weighted_risk, 4),
            "agreement_risk_component":  round(0.2 * (1.0 - agreement_score), 4),
        },
    }


def run_voting(aggregation_report: dict) -> dict:
    """
    Full voting pipeline. Runs majority voting, weighted voting, and
    combines them into a final hallucination verdict with audit metadata.

    Args:
        aggregation_report: Output dict from aggregation_module.run_aggregation()

    Returns:
        Complete voting report including all intermediate results.
    """
    print("\n[Voting Module] Starting voting process...")

    primary_vs_verifiers = aggregation_report["primary_vs_verifiers"]
    agreement_score      = aggregation_report["agreement_score"]

    # Step 1: Majority vote
    print("  Running majority vote...")
    majority_result = majority_vote(primary_vs_verifiers)
    print(f"  Majority verdict : {majority_result['majority_verdict']} "
          f"({majority_result['agree_count']} agree, "
          f"{majority_result['disagree_count']} disagree)")

    # Step 2: Weighted vote
    print("  Running weighted vote...")
    weighted_result = weighted_vote(primary_vs_verifiers)
    print(f"  Weighted risk    : {weighted_result['weighted_risk']:.4f}")

    # Step 3: Final combined score
    final = compute_final_score(
        majority_result["majority_risk"],
        weighted_result["weighted_risk"],
        agreement_score,
    )

    print(f"\n  ── FINAL VERDICT ──────────────────────────────")
    print(f"  Final Risk Score : {final['final_risk_score']:.4f}")
    print(f"  Risk Level       : {final['risk_level']}")
    print(f"  Action           : {final['action']}")
    print(f"  {final['label']}")
    print(f"  ───────────────────────────────────────────────")

    # Audit entry — every voting decision is logged with a hash
    audit_entry = {
        "timestamp":      datetime.datetime.utcnow().isoformat(),
        "question":       aggregation_report["question"],
        "final_risk":     final["final_risk_score"],
        "action":         final["action"],
    }
    audit_hash = hashlib.sha256(json.dumps(audit_entry).encode()).hexdigest()

    return {
        "question":         aggregation_report["question"],
        "majority_voting":  majority_result,
        "weighted_voting":  weighted_result,
        "final_verdict":    final,
        "audit": {
            "entry": audit_entry,
            "hash":  audit_hash,
        },
    }


# ── Quick test using full pipeline ────────────────────────────────────────────
if __name__ == "__main__":
    from primary_agent import query_primary_agent
    from verification_agents import run_all_verification_agents
    from aggregation_module import run_aggregation

    question = "Who was the first person to walk on the moon, and in what year?"

    primary_result       = query_primary_agent(question)
    verification_results = run_all_verification_agents(question)
    aggregation_report   = run_aggregation(primary_result, verification_results)
    voting_report        = run_voting(aggregation_report)

    print("\n" + "=" * 60)
    print("FULL VOTING REPORT (JSON):")
    print("=" * 60)
    print(json.dumps(voting_report, indent=2))