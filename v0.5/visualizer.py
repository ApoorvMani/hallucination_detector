"""
visualizer.py - plots hallucination status per agent per round as a heatmap grid

rows = agents, columns = rounds
green = correct, red = hallucinating
one plot per question, saved with timestamp to results/
"""

import matplotlib.pyplot as plt    # plotting
import matplotlib.patches as mpt   # for legend patches
import numpy as np                 # grid array
import os                          # file paths
import datetime                    # timestamps

# dark background — consistent with v0.4 style
plt.style.use("dark_background")

BG     = "#1e1e2e"   # plot background
GREEN  = "#06d6a0"   # correct — fact present
RED    = "#ef476f"   # hallucinating — fact missing


def make_timestamp():
    # readable timestamp — e.g. 07Mar2026_02-30PM
    return datetime.datetime.now().strftime("%d%b%Y_%I-%M%p")


def plot_hallucination_heatmap(evaluation, question, results_dir, slug, timestamp):
    """
    Heatmap grid showing hallucination status per agent per round.

    evaluation  — output of ground_truth.evaluate_experiment()
    question    — question string (used in title)
    results_dir — where to save the png
    slug        — short question identifier for filename
    timestamp   — shared timestamp for this run
    """

    agent_ids  = list(evaluation[0]["agents"].keys())   # row labels
    round_nums = [r["round"] for r in evaluation]        # column labels

    n_agents = len(agent_ids)    # number of rows
    n_rounds = len(round_nums)   # number of columns

    # build a 2D grid of values — 0 = correct, 1 = hallucinating
    grid = np.zeros((n_agents, n_rounds))

    for col, round_eval in enumerate(evaluation):
        for row, agent_id in enumerate(agent_ids):
            if round_eval["agents"][agent_id]["hallucinating"]:
                grid[row, col] = 1   # mark as hallucinating

    # --- draw the heatmap ---
    fig, ax = plt.subplots(figsize=(max(10, n_rounds * 0.9), max(4, n_agents * 1.2)))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    for row in range(n_agents):
        for col in range(n_rounds):
            color = RED if grid[row, col] == 1 else GREEN   # red or green per cell

            # draw filled rectangle for this cell
            rect = plt.Rectangle([col, row], 1, 1,
                                  facecolor=color, edgecolor=BG, linewidth=2)
            ax.add_patch(rect)

            # label inside cell — H = hallucinating, ✓ = correct
            label = "H" if grid[row, col] == 1 else "✓"
            ax.text(col + 0.5, row + 0.5, label,
                    ha="center", va="center", color="white",
                    fontsize=12, fontweight="bold")

    # axis labels and ticks
    ax.set_xlim(0, n_rounds)
    ax.set_ylim(0, n_agents)
    ax.set_xticks([c + 0.5 for c in range(n_rounds)])
    ax.set_xticklabels([f"R{r}" for r in round_nums], color="white")
    ax.set_yticks([r + 0.5 for r in range(n_agents)])
    ax.set_yticklabels(agent_ids, color="white")

    # title — truncate long questions
    short_q = question if len(question) <= 60 else question[:57] + "..."
    ax.set_title(f"Hallucination Status — {short_q}",
                 color="white", fontsize=13, pad=12)
    ax.set_xlabel("Round", color="white")
    ax.set_ylabel("Agent", color="white")
    ax.tick_params(colors="white")

    # legend
    legend_patches = [
        mpt.Patch(facecolor=GREEN, label="Correct (all facts present)"),
        mpt.Patch(facecolor=RED,   label="Hallucinating (fact missing)"),
    ]
    ax.legend(handles=legend_patches,
              facecolor="#2e2e3e", labelcolor="white",
              edgecolor="#444466", loc="upper right")

    plt.tight_layout()

    save_path = os.path.join(results_dir, f"heatmap_{slug}_{timestamp}.png")
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  saved: {save_path}")
