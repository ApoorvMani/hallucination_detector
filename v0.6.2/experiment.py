"""
experiment.py - multi-model multi-topology discussion experiment

CSC8208 Multi-Agent Hallucination Detection — Newcastle University
v0.6.2: Group 5 collab adaptation — 50 nodes, 3 models, random graph topology

Key differences from v0.6.1:
  - 3 models instead of 1 (llama3.2, qwen2.5, mistral cycling across 50 nodes)
  - configurable graph topologies: random, complete, star, ring
  - CSV output with ground_truth + blank Hallucination column for manual annotation
  - saves every SAVE_INTERVAL rounds (not just final)
  - uses ollama library (import ollama) consistent with rest of the project
  - ThreadPoolExecutor parallelises all node calls within each round
  - timer shows elapsed time, per-round time, and ETA throughout

Usage:
  ollama serve                # must be running first
  python experiment.py
  # then open results/pipeline_output.csv, fill in Hallucination column (yes/no)
  # then run: python plot.py
"""

import os
import csv
import time
import random
import ollama
import numpy as np
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed


# --- config ---

QUESTION     = "Who won the nobel prize in mathematics in 2007?"
GROUND_TRUTH = "No one"       # correct answer — There is no Nobel Prize in mathematics

ROUNDS        = 50            # discussion rounds — use 100 for full experiment, 10 for quick run
SAVE_INTERVAL = 5              # save a snapshot to CSV every N rounds
PROMPT_FAMILY = "B"            # A = each node re-evaluates own answer vs neighbours
                               # B = merge all neighbours into a single improved answer
GRAPH_TYPE    = "random"       # random | complete | star | ring
EDGE_PROB     = 0.03           # used only for random topology

WORKERS      = 1               # parallel threads for ThreadPoolExecutor

RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "results")
CSV_FILE     = os.path.join(RESULTS_DIR, "pipeline_output.csv")

# 50 nodes cycling llama3.2 / qwen2.5 / mistral
NODE_CONFIGS = [
    {"id": i + 1, "model": ["llama3.2", "qwen2.5", "mistral"][i % 3]}
    for i in range(50)
]


# --- timer helpers ---

def fmt_time(seconds):
    # format seconds as mm:ss for display
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


def print_progress(label, node_id, model, call_time, elapsed, done, total):
    # one-line progress update printed after each node completes
    pct = done / total * 100
    print(f"  {label} | Node {node_id:>2} ({model:<8}) | {call_time:.1f}s | "
          f"elapsed {fmt_time(elapsed)} | {done}/{total} ({pct:.0f}%)")


# --- model call ---

def call_ollama(model, prompt):
    # send prompt to locally running Ollama and return the response text
    response = ollama.generate(model=model, prompt=prompt)
    return response["response"].strip()


# --- node ---

class Node:
    def __init__(self, node_id, model, prompt_family):
        self.node_id        = node_id
        self.model          = model
        self.prompt_family  = prompt_family
        self.current_answer = ""

    def generate(self, question):
        # cold start — just answer the question independently
        prompt = (
            f"You are Node {self.node_id} (model: {self.model}). "
            f"Answer this question: {question}"
        )
        self.current_answer = call_ollama(self.model, prompt)
        return self.current_answer

    def evaluate(self, question, neighbour_answers, round_num):
        # build the formatted list of neighbour answers
        formatted = "\n".join(
            f"{nid}: {ans}" for nid, ans in neighbour_answers.items()
        )

        if self.prompt_family == "A" or round_num == 1:
            # re-evaluate own answer in light of neighbours
            prompt = (
                f"You are Node {self.node_id} (model: {self.model}). "
                f"The original question was: {question}\n"
                f"- Here is your current answer: {self.current_answer}\n"
                f"- Here are your neighbours' answers:\n{formatted}\n"
                f"- Re-evaluate your answer. If you are wrong, correct it.\n"
                f"- Respond with your final answer only, nothing else."
            )
        else:
            # merge all neighbours into a single improved answer
            prompt = (
                f'Here are multiple agents\' answers to "{question}":\n'
                f"{formatted}\n"
                f"- Merge into a single improved answer."
            )

        self.current_answer = call_ollama(self.model, prompt)
        return self.current_answer


# --- topologies ---

def complete_mesh_graph(node_ids):
    # every node connected to every other node
    return {nid: [other for other in node_ids if other != nid] for nid in node_ids}


def star_graph(node_ids):
    # first node is the hub, all others connect only to the hub
    hub, leaves = node_ids[0], node_ids[1:]
    graph = {hub: list(leaves)}
    for leaf in leaves:
        graph[leaf] = [hub]
    return graph


def ring_graph(node_ids):
    # each node connected to previous and next in the list
    n = len(node_ids)
    return {
        node_ids[i]: [node_ids[(i - 1) % n], node_ids[(i + 1) % n]]
        for i in range(n)
    }


def random_graph(node_ids, edge_probability):
    # Erdos-Renyi random graph — each edge exists with edge_probability
    graph = {nid: [] for nid in node_ids}
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            if random.random() < edge_probability:
                graph[node_ids[i]].append(node_ids[j])
                graph[node_ids[j]].append(node_ids[i])
    return graph


def get_largest_component(graph, edge_probability, n):
    # compute expected component size and filter graph to largest connected component
    c = edge_probability * (n - 1)
    sizes = [
        len(nx.node_connected_component(nx.fast_gnp_random_graph(n, c / (n - 1)), 1))
        for _ in range(1000)
    ]
    avg = np.mean(sizes)
    print(f"  c value: {c:.2f}")
    print(f"  average component size (1000 trials): {avg:.2f}")
    print(f"  {'giant component expected' if c >= 1 else f'theoretical expectation: {1/(1-c):.2f}'}")

    G = nx.Graph(graph)
    largest = max(nx.connected_components(G), key=len)
    print(f"  total nodes: {len(graph)} | in largest component: {len(largest)} | isolated: {len(graph) - len(largest)}")
    return list(largest)


# --- parallel broadcast (round 0) ---

def broadcast(nodes, question, exp_start):
    # fire all node.generate() calls in parallel — WORKERS threads at a time
    all_answers = {}
    total = len(nodes)
    done  = 0

    def _generate(node):
        t0  = time.time()
        ans = node.generate(question)
        return node, ans, time.time() - t0

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_generate, node): node for node in nodes}
        for future in as_completed(futures):
            node, ans, call_time = future.result()
            all_answers[f"Node {node.node_id}"] = ans
            done += 1
            print_progress("Round 0", node.node_id, node.model,
                           call_time, time.time() - exp_start, done, total)

    return all_answers


# --- parallel evaluate (rounds 1+) ---

def evaluate_round(nodes, graph, all_answers, question, round_num, exp_start, round_start):
    # fire all node.evaluate() calls in parallel — WORKERS threads at a time
    # each node reads from all_answers (previous round, read-only) and writes only to itself
    new_answers = {}
    total = len(nodes)
    done  = 0

    def _evaluate(node):
        neighbour_ids     = graph[node.node_id]
        neighbour_answers = {
            f"Node {nid}": all_answers[f"Node {nid}"]
            for nid in neighbour_ids
        }
        t0  = time.time()
        ans = node.evaluate(question, neighbour_answers, round_num)
        return node, ans, time.time() - t0

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_evaluate, node): node for node in nodes}
        for future in as_completed(futures):
            node, ans, call_time = future.result()
            new_answers[f"Node {node.node_id}"] = ans
            done += 1
            print_progress(f"Round {round_num}", node.node_id, node.model,
                           call_time, time.time() - exp_start, done, total)

    return new_answers


# --- pipeline ---

def run_pipeline():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    exp_start = time.time()   # experiment-wide start time

    # build nodes
    nodes    = [Node(cfg["id"], cfg["model"], PROMPT_FAMILY) for cfg in NODE_CONFIGS]
    node_ids = [node.node_id for node in nodes]

    # build graph
    if GRAPH_TYPE == "complete":
        graph = complete_mesh_graph(node_ids)
    elif GRAPH_TYPE == "star":
        graph = star_graph(node_ids)
    elif GRAPH_TYPE == "ring":
        graph = ring_graph(node_ids)
    elif GRAPH_TYPE == "random":
        graph = random_graph(node_ids, EDGE_PROB)
        active_ids = get_largest_component(graph, EDGE_PROB, len(node_ids))
        nodes  = [n for n in nodes if n.node_id in active_ids]      # drop isolated nodes
        graph  = {k: v for k, v in graph.items() if k in active_ids}
    else:
        raise ValueError(f"Unknown graph type: {GRAPH_TYPE}")

    print(f"\n  graph ({GRAPH_TYPE}): {len(nodes)} active nodes | workers: {WORKERS}")

    rows          = []    # all CSV rows, appended throughout
    round_times   = []    # track time per round for ETA

    # --- round 0: cold start ---
    print(f"\n{'='*55}")
    print("  Round 0 — cold start (all nodes answer independently)")
    print(f"{'='*55}")

    r0_start    = time.time()
    all_answers = broadcast(nodes, QUESTION, exp_start)
    r0_time     = time.time() - r0_start
    print(f"\n  Round 0 done in {fmt_time(r0_time)}")

    for node in nodes:
        rows.append({
            "round":         0,
            "node":          node.node_id,
            "model":         node.model,
            "answer":        all_answers[f"Node {node.node_id}"],
            "ground_truth":  GROUND_TRUTH,
            "Hallucination": "",        # blank — fill in manually after running
        })

    # --- rounds 1 to ROUNDS ---
    for round_num in range(1, ROUNDS + 1):
        round_start = time.time()

        # ETA estimate — average of completed rounds so far
        if round_times:
            avg_round  = sum(round_times) / len(round_times)
            remaining  = avg_round * (ROUNDS - round_num + 1)
            eta_str    = f"ETA ~{fmt_time(remaining)}"
        else:
            eta_str = "ETA calculating..."

        print(f"\n{'='*55}")
        print(f"  Round {round_num}/{ROUNDS} | elapsed {fmt_time(time.time() - exp_start)} | {eta_str}")
        print(f"{'='*55}")

        all_answers = evaluate_round(
            nodes, graph, all_answers, QUESTION, round_num, exp_start, round_start
        )

        round_time = time.time() - round_start
        round_times.append(round_time)
        print(f"\n  Round {round_num} done in {fmt_time(round_time)}")

        # snapshot every SAVE_INTERVAL rounds
        if round_num % SAVE_INTERVAL == 0:
            for node in nodes:
                rows.append({
                    "round":         round_num,
                    "node":          node.node_id,
                    "model":         node.model,
                    "answer":        node.current_answer,
                    "ground_truth":  GROUND_TRUTH,
                    "Hallucination": "",
                })

    # --- save CSV ---
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["round", "node", "model", "answer", "ground_truth", "Hallucination"]
        )
        writer.writeheader()
        writer.writerows(rows)

    total_time = time.time() - exp_start
    print(f"\n  [saved] {CSV_FILE}")
    print(f"  total time: {fmt_time(total_time)}")
    print(f"  Open the CSV, fill in the Hallucination column (yes / no), then run plot.py")


# --- entry point ---

if __name__ == "__main__":
    print("=" * 55)
    print("  CSC8208 — Multi-Agent Experiment v0.6.2")
    print(f"  question     : {QUESTION}")
    print(f"  ground truth : {GROUND_TRUTH}")
    print(f"  nodes        : {len(NODE_CONFIGS)}")
    print(f"  topology     : {GRAPH_TYPE}")
    print(f"  rounds       : {ROUNDS}")
    print(f"  prompt family: {PROMPT_FAMILY}")
    print(f"  workers      : {WORKERS}")
    print("=" * 55)

    run_pipeline()

    print(f"\n{'='*55}")
    print("  DONE")
    print(f"{'='*55}")
