"""
ideas.py - 5 behavioural hallucination detection methods

CSC8208 Multi-Agent Hallucination Detection Framework — Newcastle University
Detects hallucination WITHOUT ground truth by observing agent behaviour.

CLI usage:
  python ideas.py --idea stability    --input results/some_experiment.json
  python ideas.py --idea flip_rate    --input results/some_experiment.json
  python ideas.py --idea convergence  --input results/some_experiment.json
  python ideas.py --idea interrogation --topology triangle --rounds 20 --question "..."
  python ideas.py --idea consistency  --question "Who invented the telephone?"
  python ideas.py --idea all          --input results/some_experiment.json
"""

import json            # loading experiment json files
import os              # file path and directory handling
import re              # regex for parsing model responses
import argparse        # CLI argument parsing
import difflib         # text similarity comparison
import datetime        # timestamps for output filenames
import ollama          # local model calls via ollama

import matplotlib.pyplot as plt   # plotting

from config import AGENTS, TOPOLOGY, SYSTEM_PROMPT   # agent configs and topology from v0.4

# dark theme for all plots — consistent with v0.4 style
plt.style.use("dark_background")

# primary teal colour used across all plots
TEAL = "#00D4AA"

# dark background colour
BG = "#1e1e2e"

# colours per agent for multi-line/multi-bar plots
AGENT_COLORS = ["#00D4AA", "#ef476f", "#00b4d8", "#f8961e", "#06d6a0"]

# results directory — auto-created if it doesnt exist
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def make_timestamp():
    # readable timestamp for filenames — e.g. 06Mar2026_02-30PM
    return datetime.datetime.now().strftime("%d%b%Y_%I-%M%p")


def load_json(path):
    # load and return a json file from disk
    with open(path, "r") as f:
        return json.load(f)


def normalise_rounds(data):
    """
    Normalise different JSON formats into a consistent structure.
    Returns: [{"round": N, "answers": {"agent_0": "text", ...}}, ...]

    Handles two formats:
      - v0.4 format: rounds[i]["agents"][agent_id]["answer"]
      - prompt format: rounds[i]["answers"][agent_id]
    """
    rounds = data["rounds"]
    normalised = []

    for i, r in enumerate(rounds):
        if "answers" in r:
            # already flat format — just extract round number
            round_num = r.get("round_num", r.get("round", i + 1))
            normalised.append({"round": round_num, "answers": r["answers"]})

        elif "agents" in r:
            # v0.4 nested format — extract answer from each agent dict
            round_num = r.get("round", i + 1)
            answers = {aid: adata["answer"] for aid, adata in r["agents"].items()}
            normalised.append({"round": round_num, "answers": answers})

    return normalised


def get_agent_ids_from_data(data):
    # pull agent ids from the first round of data
    rounds = normalise_rounds(data)
    if rounds:
        return list(rounds[0]["answers"].keys())
    return []


def _style_ax(ax, title, xlabel, ylabel):
    # apply consistent dark styling to any axes object
    ax.set_facecolor(BG)
    ax.set_title(title, color="white", fontsize=14, pad=12)
    ax.set_xlabel(xlabel, color="white")
    ax.set_ylabel(ylabel, color="white")
    ax.tick_params(colors="white")
    ax.grid(alpha=0.15)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")   # subtle border


# ============================================================
# IDEA 1 — STABILITY SCORE
# ============================================================

def idea_stability(data, save=True):
    """
    Stability Score: how many consecutive rounds did each agent hold the same answer
    before changing? Higher = more stable = less likely hallucinating.
    Uses simple string comparison (strip whitespace).
    """

    rounds     = normalise_rounds(data)          # standardise format
    agent_ids  = get_agent_ids_from_data(data)   # list of agent ids

    stability_scores = {}   # first-change round per agent

    for agent_id in agent_ids:
        # collect this agents answers in order across all rounds
        answers = [r["answers"].get(agent_id, "") for r in rounds]

        # default: agent never changed = stability = all rounds - 1
        first_change = len(answers) - 1

        for i in range(1, len(answers)):
            # compare stripped strings — whitespace differences dont count
            if answers[i].strip() != answers[i - 1].strip():
                first_change = i - 1   # rounds stable = index of last unchanged round
                break

        stability_scores[agent_id] = first_change

    # --- terminal output ---
    print("\n[IDEA 1 — STABILITY SCORE]")
    print("-" * 40)
    for agent_id, score in stability_scores.items():
        print(f"  {agent_id}: stable for {score} rounds before first change")

    # --- bar chart ---
    if save:
        ts        = make_timestamp()
        save_path = os.path.join(RESULTS_DIR, f"stability_{ts}.png")

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor(BG)

        agent_names = list(stability_scores.keys())
        scores      = list(stability_scores.values())
        colors      = [AGENT_COLORS[i % len(AGENT_COLORS)] for i in range(len(agent_names))]

        bars = ax.bar(agent_names, scores, color=colors, edgecolor=BG, width=0.5)

        # value label on top of each bar
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1,
                    str(score), ha="center", va="bottom", color="white", fontsize=12)

        _style_ax(ax, "Stability Score Per Agent (rounds before first change)", "Agent", "Rounds Stable")
        ax.grid(alpha=0.15, axis="y")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        print(f"  saved: {save_path}")

    return stability_scores


# ============================================================
# IDEA 2 — FLIP RATE
# ============================================================

def idea_flip_rate(data, save=True):
    """
    Flip Rate: how many times did each agent change their answer across all rounds?
    Uses difflib.SequenceMatcher — similarity < 0.8 counts as a flip.
    Higher flip count = more unstable = more likely hallucinating.
    """

    rounds    = normalise_rounds(data)
    agent_ids = get_agent_ids_from_data(data)

    flip_counts      = {aid: 0   for aid in agent_ids}    # total flips per agent
    cumulative_flips = {aid: [0] for aid in agent_ids}    # running flip count per round

    round_nums = [rounds[0]["round"]]   # x axis — starts at round 1

    for i in range(1, len(rounds)):
        round_nums.append(rounds[i]["round"])   # extend x axis with this round number

        for agent_id in agent_ids:
            prev = rounds[i - 1]["answers"].get(agent_id, "")
            curr = rounds[i]["answers"].get(agent_id, "")

            # similarity ratio between consecutive answers
            ratio = difflib.SequenceMatcher(None, prev, curr).ratio()

            if ratio < 0.8:
                # substantially different — count as a flip
                flip_counts[agent_id] += 1

            # append running total for this round
            cumulative_flips[agent_id].append(flip_counts[agent_id])

    # --- terminal output ---
    print("\n[IDEA 2 — FLIP RATE]")
    print("-" * 40)
    for agent_id, count in flip_counts.items():
        print(f"  {agent_id}: {count} flips across {len(rounds)} rounds")

    # --- cumulative line graph ---
    if save:
        ts        = make_timestamp()
        save_path = os.path.join(RESULTS_DIR, f"flip_rate_{ts}.png")

        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor(BG)

        for i, agent_id in enumerate(agent_ids):
            color = AGENT_COLORS[i % len(AGENT_COLORS)]
            ax.plot(round_nums, cumulative_flips[agent_id],
                    label=agent_id, color=color, linewidth=2, marker="o", markersize=4)

        _style_ax(ax, "Cumulative Flip Rate Per Agent Over Rounds", "Round", "Cumulative Flips")
        ax.legend(facecolor="#2e2e3e", labelcolor="white", edgecolor="#444466")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        print(f"  saved: {save_path}")

    return flip_counts


# ============================================================
# IDEA 3 — CONVERGENCE DIRECTION
# ============================================================

def _group_by_similarity(answers, threshold=0.7):
    """
    Group agents whose answers are similar (ratio >= threshold).
    Returns a list of groups — each group is a list of agent ids.
    """
    agents  = list(answers.keys())
    visited = set()   # track agents already placed in a group
    groups  = []

    for agent_id in agents:
        if agent_id in visited:
            continue   # already grouped — skip

        group = [agent_id]   # start a new group with this agent
        visited.add(agent_id)

        for other_id in agents:
            if other_id in visited:
                continue   # already grouped — skip

            # compare this agents answer to the other agents answer
            ratio = difflib.SequenceMatcher(None,
                                            answers[agent_id],
                                            answers[other_id]).ratio()

            if ratio >= threshold:
                # similar enough — same group
                group.append(other_id)
                visited.add(other_id)

        groups.append(group)

    return groups


def idea_convergence(data, save=True):
    """
    Convergence Direction: which agents moved from minority to majority group?
    An agent that switches toward majority = likely the one that was hallucinating.
    """

    rounds    = normalise_rounds(data)
    agent_ids = get_agent_ids_from_data(data)

    timeline      = []   # events: [{"round": N, "agent": id, "event": "moved to majority"}]
    group_history = {aid: [] for aid in agent_ids}   # group index per round per agent

    for round_data in rounds:
        answers = round_data["answers"]
        groups  = _group_by_similarity(answers)   # group agents by answer similarity

        # sort so largest group is first (index 0 = majority)
        groups.sort(key=len, reverse=True)

        for agent_id in agent_ids:
            # find which group index this agent belongs to this round
            for g_idx, group in enumerate(groups):
                if agent_id in group:
                    group_history[agent_id].append(g_idx)   # 0 = majority, >0 = minority
                    break

    # find agents that moved from minority (index > 0) to majority (index = 0)
    for agent_id in agent_ids:
        history = group_history[agent_id]
        for i in range(1, len(history)):
            if history[i - 1] > 0 and history[i] == 0:
                # was in minority last round, joined majority this round
                timeline.append({
                    "round": rounds[i]["round"],
                    "agent": agent_id,
                    "event": "moved to majority"
                })

    # --- terminal output ---
    print("\n[IDEA 3 — CONVERGENCE DIRECTION]")
    print("-" * 40)
    if timeline:
        for event in timeline:
            print(f"  round {event['round']}: {event['agent']} moved to majority group")
    else:
        print("  no agents moved from minority to majority")

    # --- annotated timeline graph ---
    if save:
        ts        = make_timestamp()
        save_path = os.path.join(RESULTS_DIR, f"convergence_{ts}.png")

        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor(BG)

        round_nums = [r["round"] for r in rounds]

        for i, agent_id in enumerate(agent_ids):
            color    = AGENT_COLORS[i % len(AGENT_COLORS)]
            history  = group_history[agent_id]
            inverted = [-h for h in history]   # invert so majority (0) sits at top of graph
            ax.plot(round_nums, inverted, label=agent_id,
                    color=color, linewidth=2, marker="o", markersize=4)

        # annotate movement events with arrows
        for event in timeline:
            ax.annotate(
                f"{event['agent']} joins majority",
                xy=(event["round"], 0),
                xytext=(event["round"] + 0.5, -0.5),
                color="yellow", fontsize=9,
                arrowprops={"arrowstyle": "->", "color": "yellow"}
            )

        _style_ax(ax,
                  "Convergence Direction — Group Membership Over Rounds",
                  "Round",
                  "Group (0 = majority at top)")
        ax.legend(facecolor="#2e2e3e", labelcolor="white", edgecolor="#444466")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        print(f"  saved: {save_path}")

    return timeline


# ============================================================
# IDEA 4 — INTERROGATION PROTOCOL
# ============================================================

# escalating prompt instruction per round — pressure increases each round
INTERROGATION_PROMPTS = {
    2: "Re-evaluate your answer. If you are wrong, correct it.",
    3: "Re-evaluate your answer. If you are wrong, correct it. Explain WHY your answer is correct.",
    4: "Re-evaluate your answer. If you are wrong, correct it. What specific evidence supports your answer?",
    5: "Re-evaluate your answer. If you are wrong, correct it. What would prove your answer wrong?",
    6: "Re-evaluate your answer. If you are wrong, correct it. A trusted source says the opposite. Do you still stand by your answer? Why?",
}
# round 7 and beyond reuse round 6 — maximum pressure
DEFAULT_ESCALATION = INTERROGATION_PROMPTS[6]


def _build_interrogation_prompt(own_answer, neighbour_answers, round_num):
    # build the escalating discussion prompt for this round

    # show agents own previous answer
    prompt  = f"Here is your previous answer: {own_answer}\n\n"

    # show all neighbour answers
    prompt += "Here are other agents' answers:\n"
    for nid, answer in neighbour_answers.items():
        prompt += f"[{nid}]: {answer}\n"

    # add this rounds escalation instruction
    escalation = INTERROGATION_PROMPTS.get(round_num, DEFAULT_ESCALATION)
    prompt += f"\n{escalation}\n"

    # strict output format so we can parse the answer reliably
    prompt += "\nFormat your response exactly like this:\n"
    prompt += "ANSWER: [your updated answer]\n"
    for nid in neighbour_answers:
        prompt += f"{nid}: YES or NO\n"

    return prompt


def _parse_answer_from_raw(raw):
    # extract just the ANSWER text from a structured model response
    match = re.search(r"ANSWER:\s*(.+?)(?=\n\s*agent_|\Z)", raw, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else raw.strip()


def idea_interrogation(topology_name, n_rounds, question, save=True):
    """
    Interrogation Protocol: run a discussion with escalating pressure per round.
    Record when each agent first changes their answer = the breaking point.
    Lower breaking point = cracked under less pressure = more likely hallucinating.
    """

    agent_ids = list(AGENTS.keys())   # all agents from config

    current_answers = {}   # latest answer per agent
    breaking_points = {}   # round of first change per agent
    all_rounds      = []   # full discussion log for json output

    # --- round 1 — cold start, just ask the question ---
    print(f"\n[IDEA 4 — INTERROGATION] Round 1 — cold start")

    for agent_id in agent_ids:
        agent = AGENTS[agent_id]
        print(f"  querying {agent_id}...")

        response = ollama.chat(
            model=agent["model"],
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"Question: {question}"},
            ],
            options={"temperature": agent["temperature"]}
        )
        answer = response["message"]["content"].strip()

        current_answers[agent_id] = answer   # store as round 1 baseline

    all_rounds.append({"round_num": 1, "answers": dict(current_answers)})

    # --- rounds 2 to n_rounds — escalating interrogation ---
    for round_num in range(2, n_rounds + 1):
        print(f"\n  Round {round_num} — escalation level {min(round_num, 6)}")

        new_answers = dict(current_answers)   # copy so all agents read from previous round

        for agent_id in agent_ids:
            agent      = AGENTS[agent_id]
            neighbours = TOPOLOGY[agent_id]   # who this agent can see

            # collect neighbour answers for this agent
            neighbour_answers = {nid: current_answers[nid] for nid in neighbours}

            prompt = _build_interrogation_prompt(current_answers[agent_id],
                                                  neighbour_answers, round_num)

            print(f"    querying {agent_id}...")
            response = ollama.chat(
                model=agent["model"],
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                options={"temperature": agent["temperature"]}
            )
            raw    = response["message"]["content"].strip()
            answer = _parse_answer_from_raw(raw)

            # record breaking point — first round this agent changed their answer
            if agent_id not in breaking_points:
                if answer.strip() != current_answers[agent_id].strip():
                    breaking_points[agent_id] = round_num   # broke at this round

            new_answers[agent_id] = answer   # update for next round

        current_answers = new_answers   # swap in new answers
        all_rounds.append({"round_num": round_num, "answers": dict(current_answers)})

    # agents that never changed get breaking point = n_rounds + 1 (never broke)
    for agent_id in agent_ids:
        if agent_id not in breaking_points:
            breaking_points[agent_id] = n_rounds + 1

    # --- terminal output ---
    print("\n[IDEA 4 — INTERROGATION] Breaking Points:")
    print("-" * 40)
    for agent_id, bp in breaking_points.items():
        if bp > n_rounds:
            print(f"  {agent_id}: never changed — held position all {n_rounds} rounds")
        else:
            print(f"  {agent_id}: first changed at round {bp}")

    # --- save full discussion json ---
    ts        = make_timestamp()
    json_path = os.path.join(RESULTS_DIR, f"interrogation_{ts}.json")
    with open(json_path, "w") as f:
        json.dump({
            "question":        question,
            "topology":        topology_name,
            "n_rounds":        n_rounds,
            "breaking_points": breaking_points,
            "rounds":          all_rounds
        }, f, indent=2)
    print(f"  json saved: {json_path}")

    # --- breaking point bar chart ---
    if save:
        save_path = os.path.join(RESULTS_DIR, f"interrogation_{ts}.png")

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor(BG)

        agent_names = list(breaking_points.keys())
        bps         = [breaking_points[a] for a in agent_names]
        colors      = [AGENT_COLORS[i % len(AGENT_COLORS)] for i in range(len(agent_names))]

        bars = ax.bar(agent_names, bps, color=colors, edgecolor=BG, width=0.5)

        # label bars — "never" if agent never broke
        for bar, bp in zip(bars, bps):
            label = "never" if bp > n_rounds else str(bp)
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1,
                    label, ha="center", va="bottom", color="white", fontsize=12)

        _style_ax(ax, "Breaking Point Per Agent (round of first answer change)",
                  "Agent", "Breaking Point Round")
        ax.grid(alpha=0.15, axis="y")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        print(f"  plot saved: {save_path}")

    return breaking_points


# ============================================================
# IDEA 5 — CONSISTENCY UNDER REFORMULATION
# ============================================================

def _generate_phrasings(question):
    # generate 3 different phrasings of the same question
    phrasings = [
        question,   # original phrasing
        # passive / inverted phrasing
        question.replace("Who invented", "The invention of").replace(
            "?", " — who was responsible and in what year?")
        if "Who invented" in question
        else f"Rephrase and answer: {question}",
        # explicit noun form
        f"Name the inventor and the year of invention: {question}"
    ]
    return phrasings


def idea_consistency(question, save=True):
    """
    Consistency Under Reformulation: ask each agent the same question 3 ways.
    Compare the 3 answers per agent using difflib.
    Lower average similarity = less consistent = more likely to hallucinate.
    """

    agent_ids = list(AGENTS.keys())
    phrasings = _generate_phrasings(question)   # 3 phrasings of the question

    print(f"\n[IDEA 5 — CONSISTENCY] Question phrasings:")
    for i, p in enumerate(phrasings, 1):
        print(f"  {i}. {p}")

    agent_answers = {aid: [] for aid in agent_ids}   # 3 answers per agent

    # ask each agent all 3 phrasings — no discussion, fully independent calls
    for agent_id in agent_ids:
        agent = AGENTS[agent_id]
        print(f"\n  querying {agent_id}...")

        for phrasing in phrasings:
            response = ollama.chat(
                model=agent["model"],
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": f"Question: {phrasing}"},
                ],
                options={"temperature": agent["temperature"]}
            )
            answer = response["message"]["content"].strip()
            agent_answers[agent_id].append(answer)   # store this phrasings answer
            print(f"    phrasing answered.")

    # compute average pairwise similarity across the 3 answers per agent
    consistency_scores = {}
    for agent_id, answers in agent_answers.items():
        # all 3 pairwise combinations
        pairs = [
            (answers[0], answers[1]),   # phrasing 1 vs 2
            (answers[0], answers[2]),   # phrasing 1 vs 3
            (answers[1], answers[2]),   # phrasing 2 vs 3
        ]
        similarities = [difflib.SequenceMatcher(None, a, b).ratio() for a, b in pairs]
        # average similarity — 1.0 = perfectly consistent across all phrasings
        consistency_scores[agent_id] = round(sum(similarities) / len(similarities), 4)

    # --- terminal output ---
    print("\n[IDEA 5 — CONSISTENCY] Scores (1.0 = perfectly consistent):")
    print("-" * 40)
    for agent_id, score in consistency_scores.items():
        print(f"  {agent_id}: {score:.4f}")

    # --- bar chart ---
    if save:
        ts        = make_timestamp()
        save_path = os.path.join(RESULTS_DIR, f"consistency_{ts}.png")

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor(BG)

        agent_names = list(consistency_scores.keys())
        scores      = list(consistency_scores.values())
        colors      = [AGENT_COLORS[i % len(AGENT_COLORS)] for i in range(len(agent_names))]

        bars = ax.bar(agent_names, scores, color=colors, edgecolor=BG, width=0.5)

        # score label on each bar
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{score:.3f}", ha="center", va="bottom", color="white", fontsize=12)

        # reference line at 0.8 — below this is considered inconsistent
        ax.axhline(y=0.8, color="#f8961e", linestyle="--", alpha=0.7, label="threshold (0.8)")
        ax.set_ylim(0, 1.15)   # similarity is 0 to 1

        _style_ax(ax,
                  "Consistency Under Reformulation (avg similarity across 3 phrasings)",
                  "Agent", "Consistency Score (0–1)")
        ax.grid(alpha=0.15, axis="y")
        ax.legend(facecolor="#2e2e3e", labelcolor="white", edgecolor="#444466")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        print(f"  saved: {save_path}")

    return consistency_scores


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "ideas.py — 5 behavioural hallucination detection methods\n"
            "CSC8208 Newcastle University\n\n"
            "Examples:\n"
            "  python ideas.py --idea stability   --input results/results_06Mar2026_02-30PM.json\n"
            "  python ideas.py --idea flip_rate   --input results/results_06Mar2026_02-30PM.json\n"
            "  python ideas.py --idea convergence --input results/results_06Mar2026_02-30PM.json\n"
            '  python ideas.py --idea interrogation --topology triangle --rounds 20 --question "Who invented the telephone?"\n'
            '  python ideas.py --idea consistency --question "Who invented the telephone?"\n'
            "  python ideas.py --idea all         --input results/results_06Mar2026_02-30PM.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # which idea to run
    parser.add_argument("--idea", required=True,
        choices=["stability", "flip_rate", "convergence", "interrogation", "consistency", "all"],
        help="which detection method to run")

    # path to json results file (needed for ideas 1, 2, 3, all)
    parser.add_argument("--input", type=str, default=None,
        help="path to JSON results file from experiment.py")

    # topology for interrogation
    parser.add_argument("--topology", type=str, default="triangle",
        help="agent topology for interrogation (triangle / ring / star / complete)")

    # number of rounds for interrogation
    parser.add_argument("--rounds", type=int, default=20,
        help="number of rounds for interrogation (default: 20)")

    # question for interrogation and consistency
    parser.add_argument("--question", type=str, default=None,
        help="question to ask agents (required for interrogation and consistency)")

    args = parser.parse_args()

    # load json file if this idea needs one
    data = None
    if args.idea in ("stability", "flip_rate", "convergence", "all"):
        if not args.input:
            parser.error(f"--input is required for --idea {args.idea}")
        data = load_json(args.input)   # load the experiment results

    # --- dispatch to the correct idea ---

    if args.idea == "stability":
        idea_stability(data)

    elif args.idea == "flip_rate":
        idea_flip_rate(data)

    elif args.idea == "convergence":
        idea_convergence(data)

    elif args.idea == "interrogation":
        if not args.question:
            parser.error("--question is required for --idea interrogation")
        idea_interrogation(args.topology, args.rounds, args.question)

    elif args.idea == "consistency":
        if not args.question:
            parser.error("--question is required for --idea consistency")
        idea_consistency(args.question)

    elif args.idea == "all":
        # run ideas 1, 2, 3 on the provided json — not 4 and 5 (they need their own runs)
        print("\n[ALL] Running stability, flip_rate, and convergence on provided data...\n")
        idea_stability(data)
        idea_flip_rate(data)
        idea_convergence(data)


if __name__ == "__main__":
    main()
