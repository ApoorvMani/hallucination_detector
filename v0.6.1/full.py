"""
experiment.py - multi-agent discussion experiment
CSC8208 Multi-Agent Hallucination Detection — Newcastle University
v0.7.0: CSV question iteration — runs full experiment for each question in CSV

Setup:
  - 3 agents, all llama3.2 via ollama
  - triangle topology: every agent sees every other agent's answer
  - temperature 0.5 for all agents
  - 5 rounds per question
  - iterates through all questions in questions.csv

Usage:
  python experiment.py
"""

import os        # for creating the results directory
import json      # for saving results to file
import csv       # for reading questions from CSV
import ollama    # for querying llama3.2 locally


# --- config ---

AGENTS = ["agent_0", "agent_1", "agent_2"]   # three agents in the experiment

MODEL        = "llama3.2"   # local model via ollama
TEMPERATURE  = 0.5          # same temperature for all agents
TOTAL_ROUNDS = 5            # run for 5 rounds per question

# triangle topology: every agent sees every other agent
TOPOLOGY = {
    "agent_0": ["agent_1", "agent_2"],
    "agent_1": ["agent_0", "agent_2"],
    "agent_2": ["agent_0", "agent_1"],
}

# --- CSV config ---
QUESTIONS_FILE   = os.path.join(os.path.dirname(__file__), "questions.csv")
QUESTION_COLUMN  = "questions"   # <-- change this if your CSV column is named differently

# where to save results
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


# --- CSV loader ---

def load_questions(filepath, column):
    """Load all questions from a CSV file. Returns a list of question strings."""
    questions = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row.get(column, "").strip()
            if q:                        # skip blank rows
                questions.append(q)
    return questions


# --- prompt builders ---

def build_round1_prompt(question):
    """Build the cold-start prompt for round 1 — no context, just the question."""
    prompt  = f"Question: {question}\n"
    prompt += "Answer accurately and concisely."
    return prompt


def build_discussion_prompt(own_answer, neighbour_answers):
    """Build the discussion prompt for rounds 2+ — show own answer and neighbours' answers."""
    prompt  = f"Here is your previous answer: {own_answer}\n\n"
    prompt += "Here are other agents' answers:\n"
    for agent_id, answer in neighbour_answers.items():
        prompt += f"[{agent_id}]: {answer}\n"
    prompt += "\nRe-evaluate your answer. If you are wrong, correct it."
    return prompt


# --- model call ---

def query_agent(agent_id, prompt):
    """Send a prompt to the model via ollama and return the response text."""
    print(f"  [Round] {agent_id} responding...")
    response = ollama.generate(
        model=MODEL,
        prompt=prompt,
        options={"temperature": TEMPERATURE}
    )
    return response["response"].strip()


# --- single question experiment ---

def run_experiment(question):
    """Run the full multi-round discussion experiment for a single question."""

    results = {
        "question":     question,
        "model":        MODEL,
        "temperature":  TEMPERATURE,
        "total_rounds": TOTAL_ROUNDS,
        "rounds":       []
    }

    current_answers = {}

    # ------------------------------------------------------------------ round 1
    print(f"\n{'='*50}")
    print(f"[Round 1] cold start — asking all agents independently")
    print(f"{'='*50}")

    round1_data = {"round": 1, "agents": {}}

    for agent_id in AGENTS:
        prompt = build_round1_prompt(question)
        answer = query_agent(agent_id, prompt)
        current_answers[agent_id] = answer
        round1_data["agents"][agent_id] = {
            "answer":     answer,
            "word_count": len(answer.split()),
        }

    results["rounds"].append(round1_data)

    # ------------------------------------------------------------------ rounds 2+
    for round_num in range(2, TOTAL_ROUNDS + 1):
        print(f"\n{'='*50}")
        print(f"[Round {round_num}] discussion phase")
        print(f"{'='*50}")

        round_data = {"round": round_num, "agents": {}}
        new_answers = {}

        for agent_id in AGENTS:
            neighbours        = TOPOLOGY[agent_id]
            neighbour_answers = {nid: current_answers[nid] for nid in neighbours}
            own_answer        = current_answers[agent_id]
            prompt            = build_discussion_prompt(own_answer, neighbour_answers)
            answer            = query_agent(agent_id, prompt)

            new_answers[agent_id] = answer
            round_data["agents"][agent_id] = {
                "answer":     answer,
                "word_count": len(answer.split()),
            }

        current_answers = new_answers
        results["rounds"].append(round_data)

    return results


# --- save helpers ---

def save_results(results, results_file):
    """Save the full results dict to a JSON file."""
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  [saved] {results_file}")


def generate_annotation_template(results, annotation_file):
    """
    Generate a blank annotation file for manual ground truth checking.
    Only written if the file doesn't already exist.
    """
    if os.path.exists(annotation_file):
        print(f"  [skip] annotation file already exists: {annotation_file}")
        return

    template = {
        "question":     results["question"],
        "instructions": (
            "For each agent per round: set hallucinating to true or false. "
            "null means you haven't checked it yet."
        ),
        "rounds": []
    }

    for round_data in results["rounds"]:
        round_entry = {"round": round_data["round"], "agents": {}}
        for agent_id in round_data["agents"]:
            answer = round_data["agents"][agent_id]["answer"]
            round_entry["agents"][agent_id] = {
                "answer":        answer,
                "hallucinating": None
            }
        template["rounds"].append(round_entry)

    with open(annotation_file, "w") as f:
        json.dump(template, f, indent=2)

    print(f"  [template] annotation file created: {annotation_file}")


def print_summary(results):
    """Print a clean per-round per-agent summary to the terminal."""
    print(f"\n{'='*55}")
    print(f"  SUMMARY — {results['question']}")
    print(f"  model: {results['model']}  |  rounds: {results['total_rounds']}")
    print(f"{'='*55}")

    for round_data in results["rounds"]:
        print(f"\n  --- Round {round_data['round']} ---")
        for agent_id, data in round_data["agents"].items():
            answer     = data["answer"]
            word_count = data["word_count"]
            preview    = answer[:120].replace("\n", " ")
            dots       = "..." if len(answer) > 120 else ""
            print(f"  {agent_id} ({word_count}w): {preview}{dots}")


# --- entry point ---

if __name__ == "__main__":

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # load all questions from CSV
    questions = load_questions(QUESTIONS_FILE, QUESTION_COLUMN)
    total_questions = len(questions)

    print("=" * 55)
    print("  CSC8208 — Multi-Agent Discussion Experiment v0.7.0")
    print(f"  questions file : {QUESTIONS_FILE}")
    print(f"  total questions: {total_questions}")
    print(f"  agents         : {', '.join(AGENTS)}")
    print(f"  rounds/question: {TOTAL_ROUNDS}")
    print("=" * 55)

    for q_idx, question in enumerate(questions, start=1):

        print(f"\n{'#'*55}")
        print(f"  QUESTION {q_idx}/{total_questions}: {question}")
        print(f"{'#'*55}")

        # unique filenames per question using its index
        safe_idx        = str(q_idx).zfill(3)   # e.g. 001, 002, ...
        results_file    = os.path.join(RESULTS_DIR, f"q{safe_idx}_results.json")
        annotation_file = os.path.join(RESULTS_DIR, f"q{safe_idx}_annotations.json")

        # skip if already completed (allows resuming after a crash)
        if os.path.exists(results_file):
            print(f"  [skip] results already exist for question {q_idx}, skipping.")
            continue

        results = run_experiment(question)       # run full experiment for this question
        save_results(results, results_file)      # save results JSON
        generate_annotation_template(results, annotation_file)  # blank annotation template
        print_summary(results)                   # terminal summary

    print(f"\n{'='*55}")
    print(f"  ALL DONE — {total_questions} questions processed")
    print(f"  Results saved to: {RESULTS_DIR}")
    print(f"{'='*55}")