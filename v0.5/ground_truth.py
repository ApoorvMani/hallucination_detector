"""
ground_truth.py - checks agent answers against known correct facts

this runs AFTER the discussion — agents never see ground truth.
for each agent per round: check if their answer contains the key facts.
missing any fact = hallucinating for that round.
"""


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


def evaluate_experiment(results, facts):
    """
    Run ground truth checking across all rounds and all agents.

    results — experiment results dict from experiment.py
    facts   — the facts dict for this question from config

    returns a list of per-round evaluations:
    [
      {
        "round": 1,
        "agents": {
          "agent_0": {"fact_results": {...}, "hallucinating": True/False},
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
            answer = agent_data["answer"]   # this agents answer this round

            # check which facts are present in this answer
            fact_results, hallucinating = check_facts(answer, facts)

            round_eval["agents"][agent_id] = {
                "fact_results": fact_results,       # which facts were found
                "hallucinating": hallucinating      # overall hallucination verdict
            }

        evaluation.append(round_eval)

    return evaluation


def print_evaluation(evaluation, question):
    # print a clean summary of hallucination status per agent per round
    print(f"\n[GROUND TRUTH] {question}")
    print("-" * 55)

    # header row
    agent_ids = list(evaluation[0]["agents"].keys())
    header    = f"  {'round':<8}" + "".join(f"{aid:<14}" for aid in agent_ids)
    print(header)

    for round_eval in evaluation:
        row = f"  {round_eval['round']:<8}"
        for agent_id in agent_ids:
            status = "HALLUCINATING" if round_eval["agents"][agent_id]["hallucinating"] else "correct"
            row   += f"{status:<14}"
        print(row)
