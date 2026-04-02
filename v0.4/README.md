# Multi-Agent Hallucination Detection — v0.3
## CSC8208 — Newcastle University — MSc Cybersecurity 2025/2026

---

## What Is This?

This project studies whether LLMs can catch each other's hallucinations through multi-round discussion. Instead of using ground truth to detect hallucination, we observe **how agents behave** during discussion — the behaviour itself reveals who is hallucinating.

Think of it like a room of 5 people debating a fact. The person who made something up will:
- Change their answer quickly when challenged
- Flip back and forth
- Not be able to hold their position under pressure

The person who knows the truth will:
- Stay consistent
- Hold firm when challenged
- Others will converge toward their answer

We don't need to know the truth ourselves. We just watch the debate.

---

## How It Works

### The Discussion Engine

1. **Round 1:** All agents answer a factual question independently. No agent sees any other agent's answer.

2. **Hallucination Injection (optional):** One agent's answer is replaced with a known wrong answer. This is for controlled experiments — we plant a hallucination and see if the group catches it.

3. **Round 2 to N:** Each agent receives this prompt:

```
Here is your previous answer: [their own previous answer]

Here are other agents' answers:
[agent_0]: [their answer]
[agent_1]: [their answer]

Re-evaluate your answer. If you are wrong, correct it.
```

Which agents' answers they see depends on the **topology** (sharing rules):
- **Triangle:** everyone sees everyone (3 agents)
- **Ring:** see left and right neighbour only
- **Star:** everyone talks through one center agent
- **Complete:** everyone sees everyone

4. **After all rounds:** We analyse the recorded answers using the 5 detection ideas below.

---

## The 5 Behavioural Hallucination Detection Ideas

These are 5 different ways to detect hallucination WITHOUT needing ground truth. Each one looks at a different aspect of how agents behave during discussion.

### Idea 1 — Stability Score

**Plain English:** If you change your answer quickly when someone disagrees, you probably made it up. If you hold firm, you're probably right.

**Technical:** For each agent, count how many consecutive rounds they maintained the same core answer before changing. Higher stability = lower hallucination risk.

**Metric:** Number of consecutive rounds before first answer change.

**Example:**
```
agent_0: held "Bell" for 100 rounds → stability = 100 → probably true
agent_2: held "Edison" for 2 rounds → stability = 2  → probably hallucinated
```

**Plot:** Bar chart — stability score per agent.

---

### Idea 2 — Flip Rate

**Plain English:** Someone who keeps changing their mind back and forth is unreliable. Someone who changes once and stays is just correcting themselves.

**Technical:** Count the total number of times each agent changes its core claim across all rounds. High flip count = hallucination signal.

**Metric:** Total number of answer changes across all rounds.

**Example:**
```
agent_0: changed 0 times → flip_rate = 0 → stable, likely correct
agent_1: changed 1 time  → flip_rate = 1 → corrected once, fine
agent_2: changed 5 times → flip_rate = 5 → unstable, likely hallucinating
```

**Plot:** Line graph — cumulative flips per agent over rounds.

---

### Idea 3 — Convergence Direction

**Plain English:** When people disagree, someone has to change their mind. The person who changes = the one who was probably wrong. The majority doesn't move — the minority does.

**Technical:** When agents disagree, track WHO changes their answer to match the majority. The agent that moved away from its original position toward the group consensus was the one hallucinating.

**Metric:** Boolean — did this agent move toward majority or stay put?

**Example:**
```
Round 1: agent_0="Bell", agent_1="Bell", agent_2="Edison" (2 vs 1)
Round 3: agent_2 changes to "Bell"
→ agent_2 moved toward majority → agent_2 was hallucinating
```

**Plot:** Shows which agent moved and when — annotated timeline.

---

### Idea 4 — Interrogation Protocol

**Plain English:** Ask harder and harder follow-up questions. True facts survive interrogation. Made-up facts collapse because you can't provide consistent evidence for something that isn't real.

**Technical:** Instead of the same prompt every round, escalate the questioning:

```
Round 2: "Re-evaluate your answer."
Round 3: "Explain WHY your answer is correct."
Round 4: "What evidence supports your answer?"
Round 5: "What would prove your answer wrong?"
Round 6: "A trusted source says [opposite]. Do you still stand by your answer?"
```

**Metric:** The round at which an agent changes its answer = the "breaking point." Lower breaking point = less confident = more likely hallucinating.

**Plot:** Breaking point per agent — when did they crack under pressure?

---

### Idea 5 — Consistency Under Reformulation

**Plain English:** Ask the same question three different ways. If someone gives different answers to the same question, they're making it up. Real knowledge doesn't change based on how you ask.

**Technical:** In Round 1, ask the same question with 3 different phrasings. Compare the 3 answers for each agent. Inconsistency = hallucination signal.

```
Phrasing 1: "Who invented the telephone?"
Phrasing 2: "The telephone was invented by whom and when?"
Phrasing 3: "Name the inventor of the telephone and the year of invention."
```

**Metric:** Consistency score — how similar are the 3 answers from the same agent?

**Plot:** Consistency score per agent — bar chart.

---

## Combined Hallucination Risk Score

All 5 ideas can be combined into a single formula:

```
hallucination_risk = w1 * (1/stability) + w2 * flip_rate + w3 * moved_to_majority + w4 * (1/breaking_point) + w5 * inconsistency
```

Where w1 through w5 are weights. Higher score = more likely hallucinating.

This is **Behavioural Hallucination Detection** — detecting hallucination not from the content of the answer, but from how the answer behaves during multi-agent discussion.

---

## File Structure

```
v0.3/
  agent_registry.py    — defines agents (model, temperature, system prompt)
  ollama_client.py     — sends prompts to local Ollama, returns responses
  topology_manager.py  — defines sharing rules (triangle, ring, star, complete)
  discussion.py        — runs multi-round discussion, records all answers
  ideas.py             — implements all 5 detection ideas as CLI commands
  plotter.py           — generates beautiful dark-themed graphs
  results/             — JSON logs and PNG graphs from experiments
  README.md            — this file
```

---

## How To Run

```bash
# Setup
ollama serve                    # start Ollama
ollama pull llama3.2            # download the model
pip install networkx matplotlib requests

# Run discussion (100 rounds, triangle topology, inject hallucination)
python discussion.py --topology triangle --rounds 100 --inject agent_2 "Thomas Edison invented the telephone in 1877."

# Analyse with different ideas
python ideas.py --idea stability --input results/latest.json
python ideas.py --idea flip_rate --input results/latest.json
python ideas.py --idea convergence --input results/latest.json
python ideas.py --idea interrogation --topology triangle --rounds 20
python ideas.py --idea consistency --question "Who invented the telephone?"

# Run all ideas at once
python ideas.py --idea all --input results/latest.json
```

---

## Experiments Planned

| Experiment | Topology | Agents | Model | Rounds | Injection |
|-----------|----------|--------|-------|--------|-----------|
| 1 | Triangle | 3 | all llama3.2 | 100 | agent_2 = Edison |
| 2 | Star | 3 | all llama3.2 | 100 | agent_2 = Edison |
| 3 | Triangle | 3 | mixed models | 100 | agent_2 = Edison |
| 4 | Triangle | 3 | all llama3.2 | 100 | no injection (control) |
| 5 | Triangle | 3 | all llama3.2 | 100 | 2 agents injected (majority wrong) |

---

## Team

Alan Jacob | Apoorv Mani | Chris George | Ramya Varunsegar | Shreya Patil | Shreyas Shetty

CSC8208 — Research Methods & Group Project in Security and Resilience
Newcastle University — School of Computing — MSc Cybersecurity — 2025/2026
Deadline: 20 March 2026
