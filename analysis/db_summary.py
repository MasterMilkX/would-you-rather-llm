#!/usr/bin/env python3
"""Summary statistics for wyr_votes.db and model_mods.jsonl."""

import json
import sqlite3
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

DB_PATH    = Path(__file__).parent.parent / "wyr_votes.db"
MODS_PATH  = Path(__file__).parent.parent / "model_mods.jsonl"

BAR_WIDTH  = 40   # max bar width for histograms


def bar(count, max_count, width=BAR_WIDTH):
    filled = round(count / max_count * width) if max_count else 0
    return "█" * filled + "░" * (width - filled)


def main():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    # ── Questions ─────────────────────────────────────────────────────────────
    (total_questions,) = cur.execute("SELECT COUNT(*) FROM votes").fetchone()

    # A question is "answered" if any of its vote rows has at least 1 vote cast
    (answered_questions,) = cur.execute(
        "SELECT COUNT(DISTINCT id) FROM votes WHERE votes_a + votes_b > 0"
    ).fetchone()
    unanswered_questions = total_questions - answered_questions

    # ── Votes (per question_id aggregated across all models) ──────────────────
    rows = cur.execute(
        "SELECT id, SUM(votes_a + votes_b) FROM votes GROUP BY id"
    ).fetchall()
    con.close()

    votes_per_question = {qid: total for qid, total in rows}
    total_votes = sum(votes_per_question.values())

    # Histogram: bucket by vote count
    bucket_counts: Counter = Counter()
    for qid in range(1, total_questions + 1):
        v = votes_per_question.get(qid, 0)
        bucket_counts[v] += 1

    max_vote_count = max(bucket_counts.keys(), default=0)

    # ── Moderation flags (model_mods.jsonl) ───────────────────────────────────
    mod_counts: Counter = Counter()
    modded_pairs: set = set()   # (id, model_name) that have any flag

    if MODS_PATH.exists():
        with MODS_PATH.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                mod_counts[d["vote_type"]] += 1
                modded_pairs.add((d["question_id"], d["model_name"]))

    # "neutral" = (question_id, model_name) pairs in votes with no moderation entry
    # Use the votes table as the universe of (question, model) pairs
    con2 = sqlite3.connect(DB_PATH)
    all_pairs = set(
        con2.execute("SELECT id, model_name FROM votes").fetchall()
    )
    con2.close()
    neutral_pairs = len(all_pairs - modded_pairs)

    # ── Print ─────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  WYR Database Summary")
    print("=" * 60)

    print(f"\n{'QUESTIONS':}")
    print(f"  Total questions    : {total_questions:,}")
    print(f"  Answered           : {answered_questions:,}  ({answered_questions/total_questions*100:.1f}%)")
    print(f"  Unanswered         : {unanswered_questions:,}  ({unanswered_questions/total_questions*100:.1f}%)")

    print(f"\n{'VOTES':}")
    print(f"  Total votes cast   : {total_votes:,}")
    if answered_questions:
        print(f"  Avg per answered Q : {total_votes/answered_questions:.2f}")

    print(f"\nHISTOGRAM — votes per question (all {total_questions} questions)")
    print(f"  {'Votes':>6}  {'Count':>5}  {'Bar'}")
    print(f"  {'─'*6}  {'─'*5}  {'─'*BAR_WIDTH}")
    max_bucket = max(bucket_counts.values(), default=1)
    for v in range(0, max_vote_count + 1):
        count = bucket_counts.get(v, 0)
        if count == 0:
            continue
        b = bar(count, max_bucket)
        print(f"  {v:>6}  {count:>5}  {b}  {count}")

    print(f"\n{'MODERATION  (model_mods.jsonl)':}")
    total_mods = sum(mod_counts.values())
    for label in ("UPVOTE", "DOWNVOTE", "FLAG"):
        n = mod_counts.get(label, 0)
        pct = n / total_mods * 100 if total_mods else 0
        print(f"  {label:<10}: {n:>5,}  ({pct:.1f}%)")
    print(f"  {'NEUTRAL':<10}: {neutral_pairs:>5,}  (question-model pairs with no moderation)")
    print(f"  {'TOTAL':<10}: {total_mods:>5,}  moderation events")

    print("\n" + "=" * 60)

    # ── Histogram image ───────────────────────────────────────────────────────
    vote_values = sorted(bucket_counts.keys())
    counts      = [bucket_counts[v] for v in vote_values]
    labels      = [str(v) for v in vote_values]

    fig, ax = plt.subplots(figsize=(8, max(4, len(vote_values) * 0.45)))
    bars = ax.barh(labels, counts, color="#4c72b0", edgecolor="white", height=0.7)

    ax.set_xlabel("Number of questions", labelpad=8)
    ax.set_ylabel("Votes per question", labelpad=8)
    ax.set_title(f"Votes per question — {total_questions} total questions", pad=12)
    ax.invert_yaxis()   # 0 at the top, matching the terminal output

    # value labels on bars
    for bar_patch, count in zip(bars, counts):
        ax.text(
            bar_patch.get_width() + max(counts) * 0.01,
            bar_patch.get_y() + bar_patch.get_height() / 2,
            str(count),
            va="center", ha="left", fontsize=9,
        )

    ax.set_xlim(0, max(counts) * 1.12)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    out_path = Path(__file__).parent / "votes_histogram.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Histogram saved to {out_path}")

    # ── LaTeX table ───────────────────────────────────────────────────────────
    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \begin{tabular}{rr}",
        r"    \toprule",
        r"    Votes per question & \% of questions \\",
        r"    \midrule",
    ]
    for v in vote_values:
        count = bucket_counts[v]
        pct   = count / total_questions * 100
        lines.append(f"    {v} & {pct:.1f}\\% \\\\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption{Distribution of votes per question across all "
        + str(total_questions) + r" questions.}",
        r"  \label{tab:votes-per-question}",
        r"\end{table}",
    ]
    latex = "\n".join(lines)

    tex_path = Path(__file__).parent / "votes_histogram.tex"
    tex_path.write_text(latex + "\n")
    print(f"LaTeX table saved to {tex_path}")
    print("\n" + latex)


if __name__ == "__main__":
    main()
