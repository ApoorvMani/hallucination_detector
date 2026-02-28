"""
verification_agents.py
======================
Four independent verification agents query the same question as the primary
agent but with slightly different prompt styles. This decorrelates their
outputs — meaning if they all agree, we have high confidence. If they
diverge, that divergence is our hallucination signal.

This implements the multi-sampling principle from:
  Manakul et al. (2023) SelfCheckGPT — arXiv:2303.08896
  Du et al. (2023) Improving Factuality via Multiagent Debate — arXiv:2305.14325

Run standalone:  python verification_agents.py
"""

import ollama
import hashlib
import datetime


# ── Four different prompt styles ─────────────────────────────────────────────
# Each agent receives the same question but framed differently.
# This is intentional: it reduces the chance that all agents make the
# same systematic error due to identical prompt phrasing.

AGENT_CONFIGS = [
    {
        "id": "verifier_1",
        "style": "direct",
        "system": (
            "You are a precise fact-checking assistant. "
            "Answer the question in one or two sentences using only confirmed facts. "
            "Do not include opinions or speculation."
        ),
    },
    {
        "id": "verifier_2",
        "style": "academic",
        "system": (
            "You are an academic research assistant with expertise in world history and science. "
            "Provide a concise, accurate answer to the question. "
            "If you are uncertain about any detail, explicitly say so."
        ),
    },
    {
        "id": "verifier_3",
        "style": "cautious",
        "system": (
            "You are a highly cautious assistant that only states facts it is certain about. "
            "Answer the question briefly. "
            "If there is any doubt about a fact, flag it clearly with the phrase: [UNCERTAIN]"
        ),
    },
    {
        "id": "verifier_4",
        "style": "analytical",
        "system": (
            "You are an analytical assistant. Answer the question factually and concisely. "
            "Structure your answer as: Fact: [answer]. "
            "Do not elaborate beyond what is directly asked."
        ),
    },
]


def query_verification_agent(question: str, agent_config: dict, model: str = "mistral") -> dict:
    """
    Sends a question to a single verification agent using its assigned prompt style.

    Args:
        question:     The question to verify.
        agent_config: Dict containing agent id, style label, and system prompt.
        model:        Ollama model to use.

    Returns:
        Structured dict with agent id, answer, hash, and timestamp.
    """
    print(f"\n  [{agent_config['id']}] ({agent_config['style']} style) querying {model}...")

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": agent_config["system"]},
            {"role": "user",   "content": question},
        ]
    )

    answer = response["message"]["content"].strip()
    answer_hash = hashlib.sha256(answer.encode()).hexdigest()

    print(f"  [{agent_config['id']}] Answer: {answer[:120]}{'...' if len(answer) > 120 else ''}")

    return {
        "agent":     agent_config["id"],
        "style":     agent_config["style"],
        "model":     model,
        "question":  question,
        "answer":    answer,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "hash":      answer_hash,
    }


def run_all_verification_agents(question: str, model: str = "mistral") -> list:
    """
    Runs all four verification agents against the same question.

    Args:
        question: The question to verify.
        model:    Ollama model to use.

    Returns:
        List of result dicts, one per agent.
    """
    print(f"\n[Verification Layer] Running {len(AGENT_CONFIGS)} agents on question:")
    print(f"  \"{question}\"")
    print("-" * 60)

    results = []
    for config in AGENT_CONFIGS:
        result = query_verification_agent(question, config, model)
        results.append(result)

    print("\n[Verification Layer] All agents completed.")
    return results


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    test_question = "Who was the first person to walk on the moon, and in what year?"

    results = run_all_verification_agents(test_question)

    print("\n" + "=" * 60)
    print("ALL VERIFICATION AGENT RESULTS:")
    print("=" * 60)
    for r in results:
        print(f"\n[{r['agent']}] ({r['style']})")
        print(f"  Answer : {r['answer']}")
        print(f"  Hash   : {r['hash'][:16]}...")