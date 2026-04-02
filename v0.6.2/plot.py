"""
plot.py - heatmap visualisation from annotated CSV

CSC8208 Multi-Agent Hallucination Detection — Newcastle University
v0.6.2: reads pipeline_output.csv after manual Hallucination annotation

Run this after:
  1. python experiment.py   — generates results/pipeline_output.csv
  2. Open the CSV, fill in Hallucination column (yes / no) for each row
  3. python plot.py         — generates results/hallucination_heatmap.png

Heatmap axes:
  - rows = models (llama3.2, qwen2.5, mistral)
  - columns = round numbers recorded in the CSV
  - cell = max hallucination value across all nodes of that model in that round
  - green = no hallucination | red = hallucination | grey = not annotated
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch


RESULTS_DIR    = os.path.join(os.path.dirname(__file__), "results")
CSV_ANNOTATED  = os.path.join(RESULTS_DIR, "pipeline_output_annotated.csv")
CSV_MANUAL     = os.path.join(RESULTS_DIR, "pipeline_output.csv")
PNG_FILE       = os.path.join(RESULTS_DIR, "hallucination_heatmap.png")

# prefer the auto-annotated CSV, fall back to manually annotated one
CSV_FILE = CSV_ANNOTATED if os.path.exists(CSV_ANNOTATED) else CSV_MANUAL


def load_and_pivot(csv_path):
    df = pd.read_csv(csv_path)

    # normalise the Hallucination column — strip whitespace, lowercase
    df["Hallucination"] = df["Hallucination"].astype(str).str.strip().str.lower()
    df["hall_val"] = df["Hallucination"].map({"yes": 1, "no": 0})   # yes=1, no=0, NaN=unannotated

    # aggregate: for each (model, round) take max — if any node hallucinated, mark red
    pivot = df.pivot_table(index="node", columns="round", values="hall_val", aggfunc="max")
    return pivot


def draw_heatmap(pivot):
    models = pivot.index.tolist()
    rounds = pivot.columns.tolist()
    values = pivot.values   # shape (n_models, n_rounds)

    fig, ax = plt.subplots(figsize=(len(rounds) * 1.8 + 2, len(models) * 1.2 + 2))

    # two-colour map: green (0) → red (1)
    cmap  = mcolors.ListedColormap(["#2ecc71", "#e74c3c"])
    norm  = mcolors.BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    ax.set_facecolor("#888888")   # grey for unannotated cells

    ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")

    # cell labels
    for i in range(len(models)):
        for j in range(len(rounds)):
            val = values[i, j]
            if np.isnan(val):
                label, color = "?", "#333333"
            elif val == 1:
                label, color = "YES", "white"
            else:
                label, color = "NO", "white"
            ax.text(j, i, label, ha="center", va="center", fontsize=11,
                    fontweight="bold", color=color)

    ax.set_xticks(range(len(rounds)))
    ax.set_xticklabels([f"Round {r}" for r in rounds], fontsize=11)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=11)
    ax.set_title(
        "Hallucination Heatmap — v0.6.2\n"
        "Who won the Nobel Prize in Mathematics in 2007?",
        fontsize=13, pad=16
    )

    legend = [
        Patch(facecolor="#2ecc71", label="No Hallucination"),
        Patch(facecolor="#e74c3c", label="Hallucination"),
        Patch(facecolor="#888888", label="Not annotated"),
    ]
    ax.legend(handles=legend, loc="upper right", bbox_to_anchor=(1.22, 1.05), fontsize=10)

    plt.tight_layout()
    plt.savefig(PNG_FILE, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  [saved] {PNG_FILE}")


if __name__ == "__main__":
    if not os.path.exists(CSV_FILE):
        print(f"  [error] CSV not found: {CSV_FILE}")
        print("  Run experiment.py first, annotate the CSV, then re-run plot.py")
    else:
        pivot = load_and_pivot(CSV_FILE)
        draw_heatmap(pivot)
        print("  DONE")
