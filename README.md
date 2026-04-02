# Multi-Agent LLM Hallucination Detection

**CSC8208 — Research Methods & Group Project in Security and Resilience**
**Newcastle University — School of Computing — MSc Cybersecurity — 2025/2026**

Alan Jacob · Apoorv Mani · Chris George · Ramya Varunsegar · Shreya Patil · Shreyas Shetty

---

## Overview

This repository documents the full research progression for CSC8208 Group 5. The work studies how hallucinations form, propagate, and get corrected across networks of LLM agents — evolving from a single-agent baseline to a 50-node multi-model topology experiment with Solidity blockchain audit infrastructure.

Two peer-written research papers were produced, both accepted for submission:

- **Paper 1** (submitted 2 Mar 2026): Three-layer detection pipeline with SHA-256 blockchain audit — implemented in `v0.2/`
- **Paper 2** (final paper): Hallucination propagation dynamics on Erdős-Rényi random graphs with Solidity smart contract reputation system — implemented in `v0.6.2/` and `Hallucination_LLM_G5.ipynb` (Google Colab)

---

## Papers

### Paper 1 — Three-Layer Pipeline with Immutable Audit Infrastructure

> **A Three-Layer Multi-Agent Pipeline for Hallucination Detection with Immutable Audit Infrastructure**
> Alan Jacob, Apoorv Mani, Chris George, Ramya Varunsegar, Shreya Patil, Shreyas Shetty — Newcastle University, 2026
> Submitted: 2 March 2026 — Turnitin similarity: 13%

**Core claim:** Existing hallucination detection methods share a fundamental structural gap — none produce tamper-evident, persistent records of their detection decisions. This paper addresses that gap with a 10-agent pipeline whose every execution is committed as an immutable SHA-256 chained block.

**Architecture:**
- 1 primary agent (Mistral, temp 0.5) + 9 independent verification agents across 5 model families: Llama3.2, Qwen2.5, DeepSeek-R1:1.5b, Gemma2:2b, Phi3, and 3 Mistral temperature variants
- Agent_09 permanently flagged as Byzantine adversary in the registry
- **Layer 1 — Cosine Similarity** (weight 0.25): `all-MiniLM-L6-v2` sentence embeddings, 10×10 similarity matrix; efficient but blind to numeric errors (1969 vs. 1971)
- **Layer 2 — NLI Contradiction** (weight 0.35): `cross-encoder/nli-deberta-v3-small` classifies each primary–verifier pair as ENTAILMENT / NEUTRAL / CONTRADICTION; catches numeric inconsistencies Layer 1 misses
- **Layer 3 — Consensus Panel Judge** (weight 0.40): every agent evaluates every other across 4 dimensions (factual accuracy, hallucination score, completeness, reasoning quality, each 0–10); 90 independent evaluations per run; generalises Zheng et al. from 1-judge to 9-judge consensus panel
- **Cross-validation module**: layer-agreement pattern classifies hallucination type — NLI-only = factual contradiction, cosine-only = semantic divergence, judge-only = reasoning flaw, all-three = confirmed hallucination
- **Weighted adaptive fusion**: R = 0.25·R₁ + 0.35·R₂ + 0.40·R₃ (weights adjusted when 2-layer agreement is found)
- **Decision engine**: ACCEPT (R < 0.20) / FLAG (0.20 ≤ R < 0.45) / REGENERATE (R ≥ 0.45); low-confidence override downgrades REGENERATE→FLAG when only one layer fires
- **Context-aware regeneration loop**: on REGENERATE, primary agent receives all verifier responses, panel scores/justifications, hallucination type, and layer-by-layer risk; all three layers re-score the corrected answer; improvement delta measured per run
- **SHA-256 blockchain audit logger**: every execution commits an immutable chained block — question hash, primary answer hash, all layer verdicts, fusion score, final decision, UTC timestamp; `validate_chain()` callable independently; no pipeline execution may complete without a committed block

**Threat model:** Prompt injection · Agent collusion · Byzantine adversary · Score tampering · Audit falsification

→ Implemented in `v0.2/` — see `v0.2/blockchain_logger.py`, `v0.2/fusion_module.py`, `v0.2/judge_layer.py`

---

### Paper 2 — Hallucination Propagation Dynamics (Final Paper)

> **Dynamics of Hallucination Propagation in Multi-Agent LLM Collaboration on Graph Topologies with Penalty-Based Enforcement**
> Alan Jacob, Apoorv Mani, Chris George, Ramya Varunsegar, Shreya Patil, Shreyas Shetty — Newcastle University, 2026

**Core claim:** Existing detection methods suffer from an infinite regress of trust — when an LLM evaluates another LLM's output and itself hallucinates, the verdict is compromised. This work bypasses the recursive trust problem entirely by studying the *dynamics* of hallucination propagation and self-correction through behavioural observation, without any automated detection layer.

**Architecture:**
- 50-node Erdős-Rényi random graph experiment, 3 LLM architectures (Llama 3.2 3B, Qwen 2.5 7B, Mistral 7B) cycling across nodes in balanced rotation
- Graph densities tested: p = 0.03 (c ≈ 1.47, sparse), p = 0.07 (c ≈ 3.43, moderate), p = 0.12 (c ≈ 5.88, dense); giant component extracted via NetworkX, isolated nodes excluded
- Round 0: independent inference (no neighbour influence); Rounds 1–50: each agent merges topological neighbours' answers; no agent is ever told it is being evaluated for hallucination
- Answers recorded at Round 0 and every 5th round; 21 experimental runs across 3 question categories (Factual, Math, Decode) × 3 edge probabilities
- Ground truth validated manually by full research team — deliberately chosen over automated detection to avoid the recursive trust problem
- **Solidity smart contract** (`HallucinationAudit.sol`, v0.8.20) deployed on local Ethereum test chain via `Web3(EthereumTesterProvider())`; compiled with `py-solc-x`
- Per-agent `AuditEntry` struct: round, hashed node ID, hashed model name, hashed answer, hashed ground truth, hallucination flag, penalty applied, timestamp
- Reputation scoring: `reputation = clean_count / total_records × 100`; each hallucination event incurs 10-point penalty, cumulative across runs
- `recordAudit()` restricted via `onlyOwner` modifier; `getAuditCount()` and `getAuditEntry()` view functions for independent verification
- SHA-256 file-level integrity hash computed over full CSV after each pipeline execution

**Threat model (adversarial):** Direct user input injection · Result tampering (factual falsification, selective label editing, user negligence) · Prompt family switching · Colluding adversarial agents

**Threat model (non-adversarial):** Iterative error propagation · Consensus bias · Context contamination · Convergence failure

**Metrics:**
- Hallucination Rate = hallucinated nodes / 50
- Shannon entropy (binary): H_b = −p·log₂(p) − (1−p)·log₂(1−p)
- Answer-distribution entropy (multi-class): H_a = −Σ pᵢ·log₂(pᵢ)
- H(t) decay rate ΔH as convergence efficiency measure

→ Implemented in `v0.6.2/` and `Hallucination_LLM_G5.ipynb` (Colab)

---

## Experimental Results

### Shannon Entropy Decay — 21 Trials

| Phase | Rounds | Mean Entropy | Observation |
|-------|--------|-------------|-------------|
| Initial stochasticity | 0 | ≈ 0.74 | Baseline disorder — 50 uncoordinated models |
| Rapid convergence | 1–15 | 0.74 → ≈ 0.22 | Communication protocol activates strong alignment |
| Steady-state consensus | 15–50 | ≈ 0.22 → ~0 | Network reliably reaches and holds singular state |

Narrowing standard deviation across trials confirms convergence is a robust property of the architecture, not an artefact of specific random initialisations.

### Blockchain Audit Results — All 21 Experimental Conditions

| Prompt Category | Edge p | Nodes | Total Penalty | Hall. % | Mean Rep. |
|----------------|--------|-------|--------------|---------|-----------|
| **Factual P1** — misdirection riddle (GT: "All 12 months") | | | | | |
| Factual P1 | 0.03 | 19 | 160 | 7.7% | 92.35 |
| Factual P1 | 0.07 | 47 | 390 | 7.5% | 92.46 |
| Factual P1 | 0.12 | 50 | 340 | 6.2% | 93.82 |
| **Math P1** — algebra word problem (GT: 12) | | | | | |
| Math P1 | 0.03 | 41 | 170 | 3.8% | 96.23 |
| **Math P1** | **0.07** | **50** | **150** | **2.7%** | **97.27** |
| Math P1 | 0.12 | 50 | 260 | 4.7% | 95.27 |
| **Decode P1** — ROT-13 cipher | | | | | |
| Decode P1 | 0.03 | 19 | 2,090 | 100.0% | 0.00 |
| Decode P1 | 0.07 | 48 | 5,280 | 100.0% | 0.00 |
| Decode P1 | 0.12 | 50 | 5,500 | 100.0% | 0.00 |

**Key findings:**
- Denser topologies accelerate convergence toward factual accuracy (Factual P1: 7.7% → 6.2% as p increases)
- Math P1 at p=0.07 achieves the best result across all conditions: 2.7% hallucination rate, 97.27 mean reputation
- Decode P1 (ROT-13) shows 100% hallucination at all densities — peer consensus cannot correct procedural execution failures; network topology has no corrective effect on tasks requiring exact deterministic computation
- The architecture of interaction is the dominant factor in system stability; even when individual nodes produced divergent data, collective structural pressure filtered stochastic errors

**Sample blockchain audit record (Factual P1, p=0.03, 209 total records):**

| Node Idx | Rd | Hallucinated | Pen. | Rep. |
|----------|----|-----------||------|------|
| 0 | 0 | False | 0 | 100.00 |
| 4 | 0 | True | 10 | 81.82 |
| 208 | 50 | False | 0 | 100.00 |

16 hallucinations total across 19 nodes × 11 sampled rounds = 209 records. Row 4 triggered the smart-contract penalty engine.

---

## Version History

| Version | Date | Core idea | Status |
|---------|------|-----------|--------|
| `v0.1` | Jan 2026 | Baseline: cosine similarity + voting, SHA-256 response hashing; known limit: cannot detect numeric hallucinations | Superseded |
| `v0.2` | Feb 2026 | 12-stage detection pipeline: cosine / NLI / consensus panel judge, weighted fusion, blockchain audit, Byzantine adversary, regeneration loop | **Paper 1** |
| `v0.3` | Feb 2026 | Multi-agent discussion, configurable topologies (mesh/ring/star), convergence + influence + deviation tracking; limitation: structured judge prompt contaminated observations | Superseded |
| `v0.4` | Mar 2026 | Pure behavioural detection — agents discuss naturally, 100 rounds, 5 detection metrics observing patterns post-hoc | Core research contribution |
| `v0.5` | Mar 2026 | Ground truth validation: 5 questions, 10 rounds, dual keyword + NLI (DeBERTa) fact checking, hallucination heatmaps | Validated methodology |
| `v0.6.1` | Mar 2026 | CSV-driven batch: ROT13/Caesar cipher challenges, 5 rounds per question, manual annotation workflow | Intermediate |
| `v0.6.2` | Mar 2026 | Final local experiment: 50 nodes, 3 models, Erdős-Rényi random graph, Solidity blockchain | **Paper 2 (local)** |
| `Hallucination_LLM_G5.ipynb` | Mar 2026 | Final experiment on Google Colab with GPU acceleration — produced all published results | **Paper 2 (Colab)** |

---

## v0.4 — Behavioural Detection Methods

The core research contribution: detecting hallucination **without ground truth, NLI, or embeddings** — purely from observed behavioural patterns during multi-agent discussion.

Agents are never told they are being evaluated. The prompt is only:

```
Here is your previous answer: X
Here are other agents' answers: ...
Re-evaluate your answer. If you are wrong, correct it.
```

**5 behavioural detection metrics** (`v0.4/ideas.py`):

| # | Metric | Signal | Example |
|---|--------|--------|---------|
| 1 | Stability Score | Rounds before first answer change — high stability → likely correct | agent_0 held "Bell" 100 rounds → stable; agent_2 changed after 2 → suspect |
| 2 | Flip Rate | Total answer changes — high flips → hallucination signal | 0 changes = stable; 5 changes = highly unstable |
| 3 | Convergence Direction | Who moved toward majority — mover was likely wrong | agent_2 changed from "Edison" to "Bell" → agent_2 hallucinated |
| 4 | Interrogation Protocol | Round of first change under escalating pressure — earlier break = less confident | Re-evaluate → Explain why → What evidence? → What would disprove this? |
| 5 | Consistency Under Reformulation | Same question, 3 phrasings — real knowledge is phrasing-invariant | "Who invented the telephone?" vs "The telephone was invented by whom?" |

Combined risk: `hallucination_risk = w₁·(1/stability) + w₂·flip_rate + w₃·moved_to_majority + w₄·(1/breaking_point) + w₅·inconsistency`

---

## Software Artifacts

| Artifact | Location | Description |
|----------|----------|-------------|
| `Hallucination_LLM_G5.ipynb` | `CSC8208_G5/Software Artifacts/` | Main Google Colab notebook — 57 cells, full pipeline producing all Paper 2 results |
| `HallucinationAudit.sol` | `CSC8208_G5/Software Artifacts/` | Solidity smart contract (v0.8.20) — `AuditEntry` struct, `NodeStatus`, `recordAudit()`, `onlyOwner` |
| `colab_hallucination_blockchain_setup.py` | `CSC8208_G5/Software Artifacts/` | Blockchain orchestration — hashes fields, calls `recordAudit()` per CSV row |
| `FinalOutput/` | `CSC8208_G5/Software Artifacts/FinalOutput/` | All CSVs and heatmap PNGs from 21 experimental runs |
| `v0.6.2/experiment.py` | `hallucination_detector/v0.6.2/` | Local version of the same pipeline |
| `v0.6.2/plot.py` | `hallucination_detector/v0.6.2/` | Heatmap generation from annotated CSV |
| `v0.2/blockchain_logger.py` | `hallucination_detector/v0.2/` | Local SHA-256 chain hash (no Ethereum) |
| `v0.2/fusion_module.py` | `hallucination_detector/v0.2/` | Weighted adaptive fusion — do not change weights without reading `cross_validation.py` |
| `v0.4/ideas.py` | `hallucination_detector/v0.4/` | 5 behavioural detection metrics — standalone CLI |
| `v0.5/ground_truth.py` | `hallucination_detector/v0.5/` | Dual fact-checking: keyword + NLI (DeBERTa) |

---

## Setup

**Requirements:** Python 3.10+, [Ollama](https://ollama.com) running locally.

```bash
# pull models used across experiments
ollama pull llama3.2
ollama pull qwen2.5
ollama pull mistral

# start ollama
ollama serve

# install dependencies
pip install -r requirements.txt
```

For v0.6.2 blockchain audit: additionally requires `web3`, `py-solc-x`:
```bash
pip install web3 py-solc-x
python -c "from solcx import install_solc; install_solc('0.8.20')"
```

For the Colab notebook: open `Hallucination_LLM_G5.ipynb` in Google Colab — the setup cell installs Ollama and pulls all models automatically.

---

## Running

```bash
# v0.6.2 — local version of final experiment (50 nodes, 3 models, Erdős-Rényi)
cd v0.6.2 && python experiment.py
# fill in Hallucination column in results/pipeline_output.csv (yes/no)
cd v0.6.2 && python plot.py          # → results/hallucination_heatmap.png

# v0.6.1 — CSV-driven batch (5 rounds per question)
cd v0.6.1 && python experiment.py

# v0.5 — ground truth fact checking (10 rounds, 5 questions)
cd v0.5 && python main.py

# v0.4 — behavioural detection, 100 rounds, single question
cd v0.4 && python main.py

# v0.4 — batch all 5 detection ideas across 3 questions
cd v0.4 && python run_all.py

# v0.2 — full 12-stage pipeline with blockchain audit
cd v0.2 && python main.py
```

---

## Output Structure

### v0.6.2 / Colab

```
results/
  pipeline_output.csv         — round, node_id, model, answer, ground_truth, hallucination label
  hallucination_heatmap.png   — agent × round heatmap (green = correct, red = hallucinated)
  blockchain_audit_output.csv — per-record ledger: node hash, model hash, answer hash,
                                  ground truth hash, hallucination flag, penalty, timestamp,
                                  reputation score, immutable audit row hash
```

### v0.5

```
results/
  qN_slug_timestamp.json   — discussion results + ground truth evaluation
  qN_slug_timestamp.png    — hallucination heatmap — agent × round
```

### v0.4

```
results/
  results_YYYYMMDD_HHMMSS.json    — full raw data — all rounds, all answers
  hallucination_votes_*.png       — how many agents flagged each agent per round
  word_counts_*.png               — answer length per agent per round
  answer_changes_*.png            — answer change frequency per agent
```

---

## Pipeline Diagrams

### v0.6.2

```
Question (QUESTION string or questions.csv)
    → Round 0: independent inference — no neighbour influence, cold start
    → Rounds 1–50: each node merges topological neighbours' answers (Prompt Family B)
    → CSV snapshot every 5 rounds + Round 0
    → Manual annotation: Hallucination column (yes/no) per row
    → plot.py: hallucination_heatmap.png
    → blockchain orchestration:
        hash(node_id) · hash(model) · hash(answer) · hash(ground_truth)
        → recordAudit() on HallucinationAudit.sol (local Ethereum test chain)
        → NodeStatus updated: clean_count / hallucination_count / totalPenalties
        → reputation = clean_count / total_records × 100
```

### v0.2

```
Input prompt
    → Primary Agent (Mistral, temp 0.5) → initial response
    → 9 Verification Agents (isolated — no inter-agent interaction at generation)
    → Layer 1: Cosine similarity — all-MiniLM-L6-v2, 10×10 matrix, mean primary-to-verifiers
    → Layer 2: NLI contradiction — cross-encoder/nli-deberta-v3-small per primary-verifier pair
    → Layer 3: Consensus panel judge — 90 evaluations, 4 dimensions each
    → Cross-validation: layer agreement → hallucination type classification
    → Weighted fusion: R = 0.25·R₁ + 0.35·R₂ + 0.40·R₃
    → Decision engine: ACCEPT / FLAG / REGENERATE
    → Regeneration loop (if REGENERATE): structured correction prompt + re-scoring
    → SHA-256 blockchain audit logger — every execution, no exceptions
```

---

## Research Questions

- Do agents naturally converge on the correct answer through peer review?
- Does a hallucination in one agent get identified and corrected by its neighbours?
- Can hallucination be detected from behaviour alone — without ground truth, NLI, or embeddings?
- How does network topology (sparse/moderate/dense Erdős-Rényi) affect hallucination propagation speed and correction rate?
- Does architectural diversity (mixing Llama, Qwen, Mistral) improve collective factual accuracy?
- Can a Solidity blockchain reputation system derived from multi-agent behavioural signals provide tamper-evident trust scores for individual agents?

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `CHANGES.md` | Decision log — why each version exists, what changed and why |
| `v0.2/blockchain_logger.py` | Local SHA-256 chained audit (no Ethereum, no external infrastructure) |
| `v0.2/fusion_module.py` | Weighted adaptive fusion — weights cosine 0.25 / NLI 0.35 / judge 0.40 |
| `v0.4/config.py` | Single source of truth pattern for all experiment configuration |
| `v0.4/ideas.py` | 5 behavioural detection metrics — standalone CLI, operates on answer text only |
| `v0.5/ground_truth.py` | Dual fact-checking: keyword matching + NLI (DeBERTa) run in parallel |
| `v0.6.2/experiment.py` | Local final experiment entry point |
| `v0.6.2/plot.py` | Heatmap visualisation from annotated CSV |
| `HallucinationAudit.sol` | Solidity smart contract — `recordAudit()`, `NodeStatus`, `AuditEntry` |
| `Hallucination_LLM_G5.ipynb` | Complete Colab notebook — 57 cells, produced all published experimental results |
