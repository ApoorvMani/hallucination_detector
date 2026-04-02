# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

CSC8208 — Multi-Agent Systems, Newcastle University (MSc Cybersecurity).
Research framework for detecting hallucinations in LLM agents through multi-agent discussion, without requiring ground truth.

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) running locally with `llama3.2` pulled

```bash
ollama pull llama3.2
ollama serve          # must be running before any experiment
pip install -r requirements.txt
```

## Running Experiments

Each version is self-contained. Run from within the version directory:

```bash
# v0.4 — single experiment (100 rounds, 3 agents)
cd v0.4 && python main.py

# v0.4 — batch all 5 detection ideas across 3 questions
cd v0.4 && python run_all.py

# v0.4 — run a specific detection idea on saved JSON results
cd v0.4 && python ideas.py

# v0.5 — ground truth fact checking (10 rounds, 5 questions)
cd v0.5 && python main.py

# v0.6.1 — CSV-driven batch (5 rounds per question)
cd v0.6.1 && python experiment.py

# v0.6.2 — Group 5 collab (50 nodes, 3 models, random topology)
cd v0.6.2 && python experiment.py   # generates results/pipeline_output.csv
# annotate the CSV (fill Hallucination column: yes / no), then:
cd v0.6.2 && python plot.py         # generates results/hallucination_heatmap.png
```

Results are saved to `results/` inside each version directory (gitignored).

## Architecture

The codebase is a **versioned research progression** — each version is a standalone experiment, not a refactor of the previous. Do not consolidate versions.

### Version history and core ideas

| Version | Core idea | Key files |
|---------|-----------|-----------|
| `v0.1` | Baseline: cosine similarity + voting | `main.py`, `aggregation_module.py`, `voting_module.py` |
| `v0.2` | 12-stage pipeline: 3 independent detection layers fused with weights (cosine 0.25 / NLI 0.35 / judge 0.40) + blockchain audit | `main.py`, `cosine_layer.py`, `nli_layer.py`, `judge_layer.py`, `fusion_module.py`, `blockchain_logger.py` |
| `v0.3` | Multi-agent discussion with convergence/influence/deviation tracking | `main.py`, `discussion.py`, `topology_manager.py`, `convergence_tracker.py` |
| `v0.4` | Pure behavioural detection — agents discuss naturally, we observe patterns | `config.py`, `experiment.py`, `ideas.py` |
| `v0.5` | Ground truth validation via dual keyword + NLI (DeBERTa) fact checking | `ground_truth.py`, `nli_detector.py` |
| `v0.6.1` | CSV-driven batch with manual annotation support | `experiment.py`, `questions.csv`, `results/q00N/` |
| `v0.6.2` | Group 5 collab — 50 nodes, 3 models (llama3.2/qwen2.5/mistral), configurable topology, CSV + heatmap | `experiment.py`, `plot.py`, `results/pipeline_output.csv` |

### v0.6.1 pipeline (current active version)

```
questions.csv → experiment.py: 5 rounds per question, saves JSON per question
    → results/q00N/: raw JSON + blank annotations.json template per question
    → plot.py: visualisations from saved results
    → full.py: end-to-end pipeline combining all steps
```

Manual annotation workflow: run `experiment.py`, then fill in `annotations.json` (true/false per answer), then run `plot.py`.

### v0.4 pipeline

```
config.py (QUESTION, TOTAL_ROUNDS=100, AGENTS, TOPOLOGY)
    → experiment.py: Round 1 cold start → Rounds 2–100 re-evaluate prompt
    → main.py: saves JSON + calls visualizer.py (3 dark-themed PNG plots)
    → ideas.py: 5 behavioural metrics run independently on the JSON output
```

**5 behavioural detection metrics** (`v0.4/ideas.py`):
1. **Stability Score** — rounds before first answer change (more stable → more likely true)
2. **Flip Rate** — total answer changes (more flips → more likely hallucinating)
3. **Convergence Direction** — who moved toward majority (mover → suspected hallucinator)
4. **Interrogation Protocol** — break point under escalating pressure (breaks sooner → less confident)
5. **Consistency Under Reformulation** — same question, 3 phrasings; inconsistency = hallucination signal

### Critical design decision (v0.4)

Agents are **never told** they are being evaluated for hallucination. The prompt is a clean re-evaluation:
```
Here is your previous answer: X
Here are other agents' answers: ...
Re-evaluate your answer. If you are wrong, correct it.
```
Detection comes entirely from observing behavioural patterns — not from agent self-judgment. This is the core research contribution.

### v0.5 ground truth checking

`ground_truth.py` runs **after** discussion, never during. Agents never see ground truth labels. Two methods run in parallel:
- Keyword matching (fast, rule-based)
- NLI via DeBERTa (semantic, model-based)

### v0.2 fusion weights

Cosine: 0.25 / NLI: 0.35 / Judge: 0.40 — do not change without understanding `fusion_module.py` and `cross_validation.py`.

## Code Style

Follow the commenting style used in `v0.4/` — this is the house style for the project:

- module docstring at the top: `filename.py - one line description`, then a blank line, then a few lines of context
- inline comments after code, lowercase, no full stops
- comments explain *why*, not what — e.g. `# no verdict parsing — agents dont judge each other, we detect from behaviour`
- use em dashes (`—`) in comments and docstrings for clarity
- no docstrings on individual functions — a short inline comment above suffices

Example:
```python
"""
experiment.py - runs the full multi-round experiment

round 1 is a cold start — agents just answer the question independently
rounds 2 onwards — each agent sees its neighbours answers and re-evaluates
"""

def build_discussion_prompt(own_answer, neighbour_answers):
    # start with this agents own previous answer
    prompt = f"Here is your previous answer: {own_answer}\n\n"
    ...
```

## Git

Never add `Co-Authored-By` or any AI attribution lines to commit messages.

## Key Files

- `CHANGES.md` — decision log explaining *why* each version exists (read this before modifying any version)
- `v0.6.1/questions.csv` — question bank driving the current experiments
- `v0.6.1/experiment.py` — main entry point for the active version
- `v0.4/config.py` — single source of truth pattern to follow when adding config to any version
- `v0.4/ideas.py` — standalone CLI tool; all 5 metrics work on answer text alone
- `v0.2/blockchain_logger.py` — local SHA-256 chain hash audit log (not Ethereum)
- `v0.5/ground_truth.py` — dual fact-checking implementation
