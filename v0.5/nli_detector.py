"""
nli_detector.py - DeBERTa NLI-based hallucination detection

CSC8208 Multi-Agent Hallucination Detection Framework — Newcastle University
v0.5: ground truth NLI checking — detects factual contradictions

Algorithm (from supervisor S. Nagaraja):
  Input:  prompt, response, ground_truth
  Output: hallucinating (True/False), label, confidence (0-1)

  1. Concatenate ground_truth + [SEP] + response
  2. Tokenise into subword tokens
  3. Pass through DeBERTa encoder (cross-attention across all tokens)
  4. Final layer outputs logits: [contradiction, entailment, neutral]
  5. Softmax → probabilities
  6. argmax → label, max → confidence
  7. CONTRADICTION → hallucinating=True
     ENTAILMENT    → hallucinating=False
     NEUTRAL       → hallucinating=Uncertain (returned as None)
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# cross-encoder/nli-deberta-v3-base:
#   pre-trained DeBERTa fine-tuned on NLI (MNLI, SNLI, FEVER-NLI)
#   label order: ['contradiction', 'entailment', 'neutral']
MODEL_NAME = "cross-encoder/nli-deberta-v3-base"

# label indices as the model outputs them
LABEL_CONTRADICTION = 0
LABEL_ENTAILMENT    = 1
LABEL_NEUTRAL       = 2
LABEL_NAMES         = {0: "contradiction", 1: "entailment", 2: "neutral"}

# load once at import time — reused across all calls
_tokenizer = None
_model     = None


def _load_model():
    """Load tokenizer and model on first call, then cache."""
    global _tokenizer, _model
    if _tokenizer is None:
        print(f"  [nli] loading {MODEL_NAME}...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        _model.eval()
        print("  [nli] model ready")
    return _tokenizer, _model


def detect_hallucination(prompt, response, ground_truth):
    """
    Detect whether a model response contradicts the ground truth.

    Parameters
    ----------
    prompt       : str  — the original question asked to the agent
    response     : str  — the agent's answer to check
    ground_truth : str  — the known correct answer

    Returns
    -------
    dict with keys:
      hallucinating : bool or None  — True=contradiction, False=entailment, None=neutral
      label         : str           — 'contradiction' | 'entailment' | 'neutral'
      confidence    : float         — probability of the winning label (0-1)
      probabilities : dict          — full {label: prob} breakdown
    """
    tokenizer, model = _load_model()

    # --- step 1: build input pair ---
    # premise = ground_truth (what we know is true)
    # hypothesis = response  (what the agent claims)
    # NLI asks: does the premise entail / contradict / be neutral to the hypothesis?
    premise    = ground_truth.strip()
    hypothesis = response.strip()

    # --- step 2 & 3: tokenise and encode ---
    # AutoTokenizer for cross-encoder NLI handles [SEP] placement automatically
    inputs = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )

    # --- step 3: forward pass through DeBERTa ---
    with torch.no_grad():
        logits = model(**inputs).logits   # shape: [1, 3] → [contradiction, entailment, neutral]

    # --- step 5: softmax → probabilities ---
    probs = torch.softmax(logits, dim=-1).squeeze()   # [3]

    p_contradiction = probs[LABEL_CONTRADICTION].item()
    p_entailment    = probs[LABEL_ENTAILMENT].item()
    p_neutral       = probs[LABEL_NEUTRAL].item()

    # --- step 6: argmax → winning label ---
    label_idx  = torch.argmax(probs).item()
    label      = LABEL_NAMES[label_idx]
    confidence = probs[label_idx].item()

    # --- step 7: classify ---
    if label == "contradiction":
        hallucinating = True
    elif label == "entailment":
        hallucinating = False
    else:  # neutral — uncertain
        hallucinating = None

    return {
        "hallucinating": hallucinating,
        "label":         label,
        "confidence":    round(confidence, 4),
        "probabilities": {
            "contradiction": round(p_contradiction, 4),
            "entailment":    round(p_entailment, 4),
            "neutral":       round(p_neutral, 4),
        },
        # stored for logging — prompt not used in NLI itself but useful for tracing
        "prompt":        prompt,
        "response":      response,
        "ground_truth":  ground_truth,
    }


def evaluate_single(prompt, response, ground_truth, verbose=True):
    """
    Convenience wrapper — runs detect_hallucination and prints a summary.

    Returns the same dict as detect_hallucination.
    """
    result = detect_hallucination(prompt, response, ground_truth)

    if verbose:
        status = {True: "HALLUCINATING", False: "correct", None: "UNCERTAIN"}[result["hallucinating"]]
        print(f"\n  [nli] prompt     : {prompt[:80]}")
        print(f"  [nli] label      : {result['label'].upper()} ({status})")
        print(f"  [nli] confidence : {result['confidence']:.4f}")
        print(f"  [nli] probs      : "
              f"contradiction={result['probabilities']['contradiction']:.3f}  "
              f"entailment={result['probabilities']['entailment']:.3f}  "
              f"neutral={result['probabilities']['neutral']:.3f}")

    return result
