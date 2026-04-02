"""
ground_truth.py - checks agent answers against known correct facts

this runs AFTER the discussion — agents never see ground truth.
two methods run in parallel:
  1. keyword matching  — fast, rule-based (original v0.5 method)
  2. NLI (DeBERTa)     — semantic, model-based (supervisor algorithm)

for keyword: missing any fact keyword = hallucinating.
for NLI:     contradiction vs ground_truth canonical answer = hallucinating.
"""

from nli_detector import detect_hallucination


def check_facts(answer, facts):
    """
    Check how many key facts are present in the agent's answer.

    answer  — the agent's answer string
    facts   — dict of {fact_name: [keyword_list]} from config

    returns:
      fact_results  — {fact_name: True/False} — which facts were found
      hallucinating — True if ANY fact is missing
    """

    answer_lower = answer.lower()   # lowercase once for all comparisons

    fact_results = {}

    for fact_name, keywords in facts.items():
        # fact is present if ANY keyword from the list appears in the answer
        fact_results[fact_name] = any(kw.lower() in answer_lower for kw in keywords)

    # hallucinating if any single fact is missing from the answer
    hallucinating = not all(fact_results.values())

    return fact_results, hallucinating


def evaluate_experiment(results, facts, canonical_answer=None, question=None):
    """
    Run ground truth checking across all rounds and all agents.
    Runs both keyword and NLI checks if canonical_answer is provided.

    results          — experiment results dict from experiment.py
    facts            — the facts dict for this question from config
    canonical_answer — the correct answer string (for NLI); optional
    question         — the original question string (for NLI tracing); optional

    returns a list of per-round evaluations:
    [
      {
        "round": 1,
        "agents": {
          "agent_0": {
            "fact_results":        {...},
            "hallucinating":       True/False,   # keyword verdict
            "nli":                 {...} or None  # NLI result if canonical_answer given
          },
          ...
        }
      },
      ...
    ]
    """

    evaluation = []

    for round_data in results["rounds"]:
        round_eval = {"round": round_data["round"], "agents": {}}

        for agent_id, agent_data in round_data["agents"].items():
            answer = agent_data["answer"]   # this agent's answer this round

            # --- method 1: keyword matching ---
            fact_results, hallucinating_kw = check_facts(answer, facts)

            # --- method 2: NLI (DeBERTa) --- only if canonical answer provided
            nli_result = None
            if canonical_answer:
                prompt = question if question else ""
                nli_result = detect_hallucination(prompt, answer, canonical_answer)

            round_eval["agents"][agent_id] = {
                "fact_results":  fact_results,       # keyword: which facts were found
                "hallucinating": hallucinating_kw,   # keyword verdict
                "nli":           nli_result          # NLI verdict (or None)
            }

        evaluation.append(round_eval)

    return evaluation


def print_evaluation(evaluation, question):
    # print a clean summary of hallucination status per agent per round
    # shows both keyword (KW) and NLI results side by side
    print(f"\n[GROUND TRUTH] {question}")

    agent_ids  = list(evaluation[0]["agents"].keys())
    has_nli    = evaluation[0]["agents"][agent_ids[0]]["nli"] is not None

    col_w = 22 if has_nli else 14   # wider columns when showing NLI too

    print("-" * (8 + col_w * len(agent_ids) + 4))
    header = f"  {'round':<8}" + "".join(f"{aid:<{col_w}}" for aid in agent_ids)
    print(header)

    for round_eval in evaluation:
        row = f"  {round_eval['round']:<8}"
        for agent_id in agent_ids:
            data       = round_eval["agents"][agent_id]
            kw_status  = "H" if data["hallucinating"] else "ok"

            if has_nli and data["nli"]:
                nli = data["nli"]
                h   = nli["hallucinating"]
                nli_status = "H" if h is True else ("ok" if h is False else "?")
                cell = f"kw:{kw_status} nli:{nli_status}({nli['confidence']:.2f})"
            else:
                cell = "HALLUCINATING" if data["hallucinating"] else "correct"

            row += f"{cell:<{col_w}}"
        print(row)
