"""
auto_annotate.py - automatic hallucination annotation for pipeline_output.csv

CSC8208 Multi-Agent Hallucination Detection — Newcastle University
v0.6.2: reads the CSV produced by experiment.py and fills in the Hallucination
column automatically by checking each answer against the ground truth

Detection logic (two-pass):
  1. keyword match — fast, checks for correct-answer signals and wrong-answer signals
  2. LLM judge    — for ambiguous cases, asks llama3.2 to decide yes/no

Writes results to:
  results/pipeline_output_annotated.csv   — filled Hallucination column
  results/annotation_report.txt           — per-row decisions + confidence

Usage:
  python auto_annotate.py
"""

import os
import csv
import ollama


RESULTS_DIR       = os.path.join(os.path.dirname(__file__), "results")
CSV_IN            = os.path.join(RESULTS_DIR, "pipeline_output.csv")
CSV_OUT           = os.path.join(RESULTS_DIR, "pipeline_output_annotated.csv")
REPORT_FILE       = os.path.join(RESULTS_DIR, "annotation_report.txt")

JUDGE_MODEL       = "llama3.2"   # model used for ambiguous cases
GROUND_TRUTH      = "no one"     # must match what experiment.py used


# --- keyword banks ---
# these are checked first — faster and transparent

# phrases that suggest the answer is CORRECT (no hallucination)
CORRECT_SIGNALS = [
    "no nobel prize in mathematics",
    "no nobel prize for mathematics",
    "there is no nobel prize in math",
    "there is no nobel prize for math",
    "mathematics does not have a nobel",
    "math does not have a nobel",
    "no such prize",
    "does not exist",
    "doesn't exist",
    "not awarded",
    "no one won",
    "nobody won",
    "no winner",
    "no prize",
    "no one",
    "no such award",
    "fields medal",       # correct alternative — agents often mention Fields Medal
    "fields' medal",
    "there is no",
    "does not award",
    "not a nobel",
    "nobel prize is not given",
    "not given in mathematics",
    "not given for mathematics",
]

# phrases that suggest the answer is WRONG (hallucinating a winner)
WRONG_SIGNALS = [
    "won the nobel prize",
    "awarded the nobel prize",
    "received the nobel prize",
    "was awarded",
    "was given",
    "the winner was",
    "the recipient was",
    "was honored",
    "was recognised",
    "was recognized",
    "laureate",
]


# --- keyword classifier ---

def keyword_classify(answer):
    """
    Returns:
      "no"        — clearly correct, confident keyword match
      "yes"       — clearly hallucinating, confident keyword match
      "ambiguous" — no strong signal, needs LLM judge
    """
    lower = answer.lower()

    correct_hits = [sig for sig in CORRECT_SIGNALS if sig in lower]
    wrong_hits   = [sig for sig in WRONG_SIGNALS   if sig in lower]

    if correct_hits and not wrong_hits:
        return "no", f"keyword correct: {correct_hits[0]}"

    if wrong_hits and not correct_hits:
        return "yes", f"keyword wrong: {wrong_hits[0]}"

    # both or neither — ambiguous
    return "ambiguous", f"correct={correct_hits}, wrong={wrong_hits}"


# --- LLM judge (for ambiguous cases only) ---

def llm_judge(question, ground_truth, answer):
    """Ask llama3.2 to decide yes/no — used only when keywords are inconclusive."""
    prompt = (
        f"Question: {question}\n"
        f"Correct answer: {ground_truth}\n"
        f"Agent answer: {answer}\n\n"
        f"Does the agent answer contain a hallucination — i.e. does it claim someone "
        f"won a prize that does not exist, or name a specific winner?\n"
        f"Reply with exactly one word: YES or NO."
    )
    response = ollama.generate(model=JUDGE_MODEL, prompt=prompt)
    text = response["response"].strip().upper()

    # extract YES/NO from anywhere in the response
    if "YES" in text:
        return "yes", "llm judge: YES"
    if "NO" in text:
        return "no", "llm judge: NO"

    # if model still unclear, default to hallucinating — safer assumption
    return "yes", f"llm judge: unclear response '{text[:40]}' — defaulting yes"


# --- main ---

def annotate():
    if not os.path.exists(CSV_IN):
        print(f"  [error] CSV not found: {CSV_IN}")
        print("  Run experiment.py first, then re-run auto_annotate.py")
        return

    rows          = []
    report_lines  = []
    ambiguous     = 0
    total         = 0
    hallucinating = 0

    with open(CSV_IN, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)

    question = GROUND_TRUTH   # used for LLM judge prompt
    # try to get the actual question from the first row if available
    if raw_rows:
        question = "Who won the nobel prize in mathematics in 2007?"

    print(f"  annotating {len(raw_rows)} rows...")
    print(f"  ground truth: {GROUND_TRUTH}\n")

    for i, row in enumerate(raw_rows):
        answer       = row.get("answer", "")
        ground_truth = row.get("ground_truth", GROUND_TRUTH)
        total       += 1

        # pass 1 — keyword
        verdict, reason = keyword_classify(answer)

        # pass 2 — LLM judge for ambiguous cases
        if verdict == "ambiguous":
            ambiguous      += 1
            verdict, reason = llm_judge(question, ground_truth, answer)
            print(f"  [{i+1}/{len(raw_rows)}] ambiguous → LLM judge → {verdict.upper()}  (node {row['node']}, round {row['round']})")
        else:
            print(f"  [{i+1}/{len(raw_rows)}] {verdict.upper():3}  node {row['node']:>3}  round {row['round']:>3}  — {reason}")

        if verdict == "yes":
            hallucinating += 1

        row["Hallucination"] = verdict
        rows.append(row)

        report_lines.append(
            f"row {i+1:04d} | round {row['round']:>3} | node {row['node']:>3} | "
            f"model {row['model']:<10} | {verdict.upper():3} | {reason}\n"
            f"  answer: {answer[:120].replace(chr(10), ' ')}\n"
        )

    # write annotated CSV
    with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["round", "node", "model", "answer", "ground_truth", "Hallucination"]
        )
        writer.writeheader()
        writer.writerows(rows)

    # write report
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(f"Auto-annotation report — v0.6.2\n")
        f.write(f"ground truth : {GROUND_TRUTH}\n")
        f.write(f"total rows   : {total}\n")
        f.write(f"hallucinating: {hallucinating} ({hallucinating/total*100:.1f}%)\n")
        f.write(f"correct      : {total - hallucinating} ({(total-hallucinating)/total*100:.1f}%)\n")
        f.write(f"ambiguous    : {ambiguous} (sent to LLM judge)\n")
        f.write(f"\n{'='*60}\n\n")
        f.writelines(report_lines)

    print(f"\n  {'='*50}")
    print(f"  total rows   : {total}")
    print(f"  hallucinating: {hallucinating} ({hallucinating/total*100:.1f}%)")
    print(f"  correct      : {total - hallucinating} ({(total-hallucinating)/total*100:.1f}%)")
    print(f"  ambiguous    : {ambiguous} (sent to LLM judge)")
    print(f"\n  [saved] {CSV_OUT}")
    print(f"  [saved] {REPORT_FILE}")
    print(f"\n  Run plot.py — it will use {os.path.basename(CSV_OUT)} automatically")


if __name__ == "__main__":
    print("=" * 55)
    print("  CSC8208 — Auto Annotator v0.6.2")
    print(f"  input  : {CSV_IN}")
    print(f"  output : {CSV_OUT}")
    print("=" * 55)
    print()
    annotate()
    print(f"\n{'='*55}")
    print("  DONE")
    print(f"{'='*55}")
