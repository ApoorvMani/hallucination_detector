"""
aggregation_module.py
=====================
The Aggregation Module is the analytical core of the hallucination detection
framework. It takes the primary agent's answer and all verification agent
answers, converts them into semantic vector embeddings, and computes pairwise
cosine similarity scores.

The key insight (from Manakul et al., 2023 - SelfCheckGPT):
  - High similarity across agents  →  likely factual, consistent knowledge
  - Low similarity across agents   →  likely hallucinated, unstable output

Semantic similarity is used instead of exact string matching because two
answers can say the same thing in completely different words. For example:
  "Neil Armstrong in 1969" and "Armstrong walked on the moon on July 20, 1969"
  are semantically similar but lexically very different.

Run standalone:  python aggregation_module.py
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

# ── Load the embedding model once (downloaded on first run, ~90MB) ───────────
# all-MiniLM-L6-v2 is fast, lightweight, and well-suited for semantic similarity
print("[Aggregation] Loading sentence embedding model...")
EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
print("[Aggregation] Model ready.")


def embed_answers(answers: list[str]) -> np.ndarray:
    """
    Converts a list of text answers into semantic vector embeddings.

    Args:
        answers: List of answer strings from all agents.

    Returns:
        2D numpy array of shape (num_answers, embedding_dim).
    """
    return EMBEDDER.encode(answers, convert_to_numpy=True)


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Computes pairwise cosine similarity between all agent embeddings.

    Cosine similarity ranges from -1 to 1:
        1.0  = identical meaning
        0.0  = completely unrelated
       -1.0  = opposite meaning (rare in practice)

    Args:
        embeddings: 2D array of shape (num_agents, embedding_dim).

    Returns:
        Square matrix of shape (num_agents, num_agents).
    """
    return cosine_similarity(embeddings)


def compute_agreement_score(similarity_matrix: np.ndarray) -> float:
    """
    Derives a single agreement score from the similarity matrix.

    We take the mean of all off-diagonal values (i.e., all pairwise
    similarities excluding self-similarity which is always 1.0).

    A score close to 1.0 means all agents strongly agree.
    A score close to 0.0 means agents are producing very different answers.

    Args:
        similarity_matrix: Square pairwise similarity matrix.

    Returns:
        Float between 0 and 1 representing overall inter-agent agreement.
    """
    n = similarity_matrix.shape[0]
    # Extract upper triangle (excluding diagonal) to avoid double counting
    upper_triangle_indices = np.triu_indices(n, k=1)
    pairwise_scores = similarity_matrix[upper_triangle_indices]
    return float(np.mean(pairwise_scores))


def interpret_agreement(score: float) -> dict:
    """
    Translates a raw agreement score into a human-readable risk label.

    Thresholds based on SelfCheckGPT's empirical findings and
    standard practice in consistency-based hallucination detection:
        >= 0.85  →  HIGH agreement   →  LOW hallucination risk
        >= 0.65  →  MEDIUM agreement →  MODERATE hallucination risk
        <  0.65  →  LOW agreement    →  HIGH hallucination risk

    Args:
        score: Float agreement score from compute_agreement_score().

    Returns:
        Dict with risk_level, label, and recommended action.
    """
    if score >= 0.85:
        return {
            "risk_level": "LOW",
            "label":      "✅ Agents strongly agree — output is likely factual",
            "action":     "ACCEPT",
            "color":      "green",
        }
    elif score >= 0.65:
        return {
            "risk_level": "MODERATE",
            "label":      "⚠️  Partial agreement — output may contain inaccuracies",
            "action":     "FLAG",
            "color":      "yellow",
        }
    else:
        return {
            "risk_level": "HIGH",
            "label":      "🚨 Agents disagree significantly — likely hallucination",
            "action":     "REGENERATE",
            "color":      "red",
        }


def run_aggregation(primary_result: dict, verification_results: list) -> dict:
    """
    Main aggregation pipeline. Takes primary + verification results and
    produces a full similarity analysis with hallucination risk assessment.

    Args:
        primary_result:       Output dict from primary_agent.py
        verification_results: List of output dicts from verification_agents.py

    Returns:
        Full aggregation report as a structured dict.
    """
    print("\n[Aggregation Module] Starting similarity analysis...")

    # Collect all answers in order: primary first, then verifiers
    all_results  = [primary_result] + verification_results
    agent_ids    = [r["agent"] for r in all_results]
    answers      = [r["answer"] for r in all_results]

    print(f"  Comparing {len(answers)} agent answers using sentence embeddings...")

    # Step 1: Embed all answers into vector space
    embeddings = embed_answers(answers)

    # Step 2: Compute pairwise cosine similarity
    sim_matrix = compute_similarity_matrix(embeddings)

    # Step 3: Derive single agreement score
    agreement_score = compute_agreement_score(sim_matrix)

    # Step 4: Interpret risk level
    risk = interpret_agreement(agreement_score)

    # Step 5: Build similarity breakdown (primary vs each verifier)
    primary_vs_verifiers = {}
    for i, result in enumerate(verification_results):
        # primary is index 0, verifiers start at index 1
        sim_score = float(sim_matrix[0][i + 1])
        primary_vs_verifiers[result["agent"]] = round(sim_score, 4)

    print(f"\n  Agreement Score : {agreement_score:.4f}")
    print(f"  Risk Level      : {risk['risk_level']}")
    print(f"  Decision        : {risk['action']}")
    print(f"  {risk['label']}")

    print("\n  Primary Agent vs Each Verifier:")
    for agent_id, score in primary_vs_verifiers.items():
        bar = "█" * int(score * 20)
        print(f"    {agent_id}: {score:.4f}  {bar}")

    return {
        "question":              primary_result["question"],
        "num_agents":            len(all_results),
        "agreement_score":       round(agreement_score, 4),
        "risk_level":            risk["risk_level"],
        "action":                risk["action"],
        "risk_label":            risk["label"],
        "primary_vs_verifiers":  primary_vs_verifiers,
        "similarity_matrix":     sim_matrix.tolist(),
        "agent_order":           agent_ids,
    }


# ── Quick test using live agents ──────────────────────────────────────────────
if __name__ == "__main__":
    from primary_agent import query_primary_agent
    from verification_agents import run_all_verification_agents

    question = "Who was the first person to walk on the moon, and in what year?"

    # Run primary agent
    primary_result = query_primary_agent(question)

    # Run all verification agents
    verification_results = run_all_verification_agents(question)

    # Run aggregation
    report = run_aggregation(primary_result, verification_results)

    print("\n" + "=" * 60)
    print("FULL AGGREGATION REPORT (JSON):")
    print("=" * 60)
    # Print without the full matrix to keep output clean
    display = {k: v for k, v in report.items() if k != "similarity_matrix"}
    print(json.dumps(display, indent=2))