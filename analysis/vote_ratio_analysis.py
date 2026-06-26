"""
Vote ratio analysis per model.

Two metrics:
  1. Raw balance   — average 50/50-ness across a model's voted questions
  2. Weighted balance — same, but each question is weighted by its quality
     feedback: +3 upvote, -1 downvote, -3 flag, +1 (default, no feedback)

Balance score per question = 1 - 2*|votes_a/total - 0.5|
  → 1.0 = perfect 50/50, 0.0 = all votes on one side
"""

import sqlite3
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict
from datetime import datetime

DB_PATH   = "../wyr_votes.db"
UP_PATH   = "upvote_questions.json"
BAD_PATH  = "bad_questions.json"
OUT_PATH  = f"graphs/vote_ratio_analysis-{datetime.today().strftime('%Y-%m-%d')}.png"

STOPLIGHT = mcolors.LinearSegmentedColormap.from_list(
    "stoplight", ["#ff0000", "#ffffff", "#00ff00"]
)

WEIGHT_UPVOTE   =  3
WEIGHT_DOWNVOTE = -1
WEIGHT_FLAG     = -3
WEIGHT_DEFAULT  =  1   # unrated questions still count once

# ── load votes ────────────────────────────────────────────────────────────────
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query(
    """
    SELECT mo.question_id, mo.model_name, mo.option_a,
           COALESCE(v.votes_a, 0) AS votes_a,
           COALESCE(v.votes_b, 0) AS votes_b
    FROM model_outputs mo
    LEFT JOIN votes v
           ON mo.question_id = v.question_id
          AND mo.model_name  = v.model_name
    """,
    conn,
)
conn.close()

# normalise legacy model name
df["model_name"] = df["model_name"].replace("llama", "tinyllama")

# filter to questions that actually received votes
df = df[(df["votes_a"] + df["votes_b"]) > 0].copy()
df["total"]   = df["votes_a"] + df["votes_b"]
df["ratio"]   = df["votes_a"] / df["total"]
df["balance"] = 1 - 2 * (df["ratio"] - 0.5).abs()   # 1 = perfect, 0 = one-sided

print(f"Total votes across all questions: {df['total'].sum()}")

# ── load feedback and build quality-weight lookup ────────────────────────────
quality: dict[tuple, int] = defaultdict(lambda: WEIGHT_DEFAULT)

with open(UP_PATH) as fh:
    for rec in json.load(fh):
        key = (rec["MODEL"], rec["optionA"])
        quality[key] += WEIGHT_UPVOTE - WEIGHT_DEFAULT   # add delta over default

with open(BAD_PATH) as fh:
    bad = json.load(fh)
    for rec in bad["downvotes"]:
        key = (rec["MODEL"], rec["optionA"])
        quality[key] += WEIGHT_DOWNVOTE - WEIGHT_DEFAULT

    for rec in bad["flagged"]:
        key = (rec["MODEL"], rec["optionA"])
        quality[key] += WEIGHT_FLAG - WEIGHT_DEFAULT

df["quality_w"] = df.apply(
    lambda r: quality[(r["model_name"], r["option_a"])], axis=1
)

# ── per-model aggregation ─────────────────────────────────────────────────────
EXCLUDE = {"deepseek"}   # match existing analysis filter
models = sorted(m for m in df["model_name"].unique() if m not in EXCLUDE)

raw_scores      = []
weighted_scores = []
n_questions     = []

for model in models:
    mdf = df[df["model_name"] == model]
    n   = len(mdf)

    raw   = mdf["balance"].mean()

    # weighted score: sum(balance * quality_weight) / n_questions
    # negative feedback drags the score down even for balanced questions
    w_score = (mdf["balance"] * mdf["quality_w"]).sum() / n

    raw_scores.append(raw)
    weighted_scores.append(w_score)
    n_questions.append(n)

# ── colour helper ─────────────────────────────────────────────────────────────
def scores_to_colors(scores, vmin=0.0, vmax=1.0):
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    return [STOPLIGHT(norm(s)) for s in scores]

# ── plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(13, 10), constrained_layout=True)
fig.patch.set_facecolor("#1a1a2e")

x    = np.arange(len(models))
bar_w = 0.55

for ax, scores, title, subtitle in [
    (axes[0], raw_scores,
     "Raw Vote Balance per Model",
     "Average 50/50-ness of votes — higher is more balanced"),
    (axes[1], weighted_scores,
     "Quality-Weighted Vote Balance per Model",
     f"Score = Σ(balance × quality_w) / N  ·  weights: upvote +{WEIGHT_UPVOTE}, "
     f"downvote {WEIGHT_DOWNVOTE}, flag {WEIGHT_FLAG}, unrated +{WEIGHT_DEFAULT}"),
]:
    vmin  = min(scores)
    vmax  = max(scores)
    mid   = (vmin + vmax) / 2
    colors = scores_to_colors(scores, vmin=vmin, vmax=vmax)

    bars = ax.bar(x, scores, width=bar_w, color=colors, edgecolor="#333355", linewidth=0.8)

    # value labels
    for bar, score, n in zip(bars, scores, n_questions):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (vmax - vmin) * 0.015,
            f"{score:.3f}\n(n={n})",
            ha="center", va="bottom",
            fontsize=8.5, color="white",
        )

    # 50/50 ideal line (only meaningful for raw)
    if ax is axes[0]:
        ax.axhline(1.0, color="#00ff00", linewidth=1, linestyle="--", alpha=0.5, label="perfect 50/50")
        ax.axhline(0.5, color="#ffffff", linewidth=0.8, linestyle=":", alpha=0.4, label="0.5 balance")

    # colour-bar legend
    sm = plt.cm.ScalarMappable(cmap=STOPLIGHT, norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.02, pad=0.01)
    cb.ax.set_ylabel("balance score", color="white", fontsize=8)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    ax.set_facecolor("#0d0d1e")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=11, color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444466")
    ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=8)
    ax.set_ylabel(subtitle, color="#aaaacc", fontsize=8.5)
    ax.yaxis.label.set_color("#aaaacc")
    ax.grid(axis="y", color="#333355", linewidth=0.5, alpha=0.7)

    if ax is axes[0]:
        ax.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#444466",
                  labelcolor="white", loc="lower right")

fig.suptitle(
    "Vote Balance Analysis by Model\n"
    "Green = closer to 50/50 (more engaging)  ·  Red = lopsided votes",
    color="white", fontsize=14, fontweight="bold", y=1.01,
)

plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {OUT_PATH}")
plt.show()
