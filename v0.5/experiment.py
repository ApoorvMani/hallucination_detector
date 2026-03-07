"""
experiment.py - runs the multi-round discussion for one question

agents just re-evaluate their answer each round — no mention of hallucination.
ground truth checking happens separately in ground_truth.py after the run.
"""

import ollama   # local model calls
import re       # parsing model responses

from config import AGENTS, TOPOLOGY, SYSTEM_PROMPT, TOTAL_ROUNDS


def build_round1_prompt(question):
    # round 1 — just ask the question, cold start, no context
    return f"Question: {question}"


def build_discussion_prompt(own_answer, neighbour_answers):
    # show the agent its own previous answer
    prompt  = f"Here is your previous answer: {own_answer}\n\n"

    # show each neighbours answer — agents just see text, no framing
    prompt += "Here are other agents' answers:\n"
    for nid, answer in neighbour_answers.items():
        prompt += f"[{nid}]: {answer}\n"

    # simple re-evaluate instruction — no hallucination mention anywhere
    prompt += "\nRe-evaluate your answer. If you are wrong, correct it.\n"

    # ask for answer only — clean structured output
    prompt += "\nFormat your response exactly like this:\n"
    prompt += "ANSWER: [your updated answer]\n"

    return prompt


def parse_answer(raw):
    # extract just the answer text after ANSWER:
    match = re.search(r"ANSWER:\s*(.+)", raw, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else raw.strip()


def query_model(model, prompt, temperature):
    # call ollama and return the raw response text
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        options={"temperature": temperature}
    )
    return response["message"]["content"].strip()


def run_experiment(question, total_rounds=None):
    # fall back to config rounds if not specified
    n         = total_rounds if total_rounds is not None else TOTAL_ROUNDS
    agent_ids = list(AGENTS.keys())

    # results holds every round — answers and metadata
    results = {
        "question":     question,
        "total_rounds": n,
        "rounds":       []
    }

    current_answers = {}   # latest answer per agent, updated each round

    # --- round 1 — cold start ---
    print(f"\n{'='*50}")
    print("ROUND 1 — initial answers")
    print(f"{'='*50}")

    round1 = {"round": 1, "agents": {}}

    for agent_id in agent_ids:
        agent  = AGENTS[agent_id]
        prompt = build_round1_prompt(question)

        print(f"  querying {agent_id}...")
        raw    = query_model(agent["model"], prompt, agent["temperature"])

        current_answers[agent_id] = raw   # store as round 1 baseline

        round1["agents"][agent_id] = {
            "answer":     raw,
            "word_count": len(raw.split()),
            "changed":    False   # round 1 is baseline — nothing to compare against
        }
        print(f"  {agent_id}: {raw[:100]}...")

    results["rounds"].append(round1)

    # --- rounds 2 to n — discussion phase ---
    for round_num in range(2, n + 1):
        print(f"\n{'='*50}")
        print(f"ROUND {round_num} — discussion")
        print(f"{'='*50}")

        round_data  = {"round": round_num, "agents": {}}
        new_answers = dict(current_answers)   # all agents read from previous round answers

        for agent_id in agent_ids:
            agent      = AGENTS[agent_id]
            neighbours = TOPOLOGY[agent_id]   # who this agent can see

            # build neighbour answers dict for this agent
            neighbour_answers = {nid: current_answers[nid] for nid in neighbours}

            prompt = build_discussion_prompt(current_answers[agent_id], neighbour_answers)

            print(f"  querying {agent_id}...")
            raw    = query_model(agent["model"], prompt, agent["temperature"])
            answer = parse_answer(raw)

            # track whether this agent changed its answer this round
            changed = answer.strip() != current_answers[agent_id].strip()

            new_answers[agent_id] = answer   # update for next round

            round_data["agents"][agent_id] = {
                "answer":     answer,
                "word_count": len(answer.split()),
                "changed":    changed
            }
            print(f"  {agent_id}: {'CHANGED' if changed else 'kept'}")

        current_answers = new_answers   # swap in new answers
        results["rounds"].append(round_data)

    return results
