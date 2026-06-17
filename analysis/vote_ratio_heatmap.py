"""
Vote ratio heatmap analysis — balance score per (model × category).

Outputs two PNG files:
  1. vote_ratio_heatmap_raw-<date>.png       — raw average 50/50-ness
  2. vote_ratio_heatmap_weighted-<date>.png  — quality-weighted average

Each PNG contains two heatmaps: main category (top) and secondary category (bottom).

Balance score per question = 1 − 2·|votes_a/total − 0.5|
  → 1.0 = perfect 50/50, 0.0 = all votes on one side

Quality weights: upvote +3, downvote −1, flag −3, unrated +1
"""

from sentence_transformers import SentenceTransformer
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import sqlite3
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

# ── config ─────────────────────────────────────────────────────────────────────
DB_PATH   = "../wyr_votes.db"
UP_PATH   = "upvote_questions.json"
BAD_PATH  = "bad_questions.json"
TIME      = datetime.today().strftime("%Y-%m-%d")
OUT_PATH  = f"graphs/vote_ratio_heatmap-{TIME}.png"

STOPLIGHT = seaborn.color_palette("blend:#ff0000,#fff,#00ff00", as_cmap=True)

WEIGHT_UPVOTE   =  3
WEIGHT_DOWNVOTE = -1
WEIGHT_FLAG     = -3
WEIGHT_DEFAULT  =  1

CATEGORIES = {
    "animals":      ["animals", "pets", "creatures", "wildlife"],
    "body parts":   ["body parts", "limbs", "organs", "physical features", "arms", "legs"],
    "money":        ["money", "wealth", "finances", "$", "rich", "dollars", "currency"],
    "food / drink": ["food", "drink", "eating", "meals", "cooking", "drinking"],
    "life":         ["life", "survival", "status", "well-being", "death", "dying",
                     "mortality", "afterlife", "living forever"],
    "ability":      ["superpower", "ability", "special ability", "magic power",
                     "power", "skill", "be able to"],
    "curse":        ["curse", "situation", "stuck", "trapped", "bewitched",
                     "inhibited", "rules", "forced"],
    "age / time":   ["age", "time", "youth", "old age", "going back in time"],
    "people":       ["people", "population", "persons", "everybody", "strangers",
                     "neighbors", "friends", "occupation", "job", "career"],
    "locations":    ["location", "countries", "cities", "towns", "world", "space"],
    "sentience":    ["become", "becoming", "inanimate", "transform", "stuck",
                     "mutation", "live as inanimate object", "be a", "be an"],
    "miscellaneous":["miscellaneous", "other", "general", "random", "unrelated"],
}
MISC_THRESH = 0.50
EXCLUDE_MODELS = {"deepseek"}

# ── load ST model and pre-compute category embeddings ──────────────────────────
print("Loading sentence-transformer model…")
ST = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

print("Pre-computing category embeddings…")
cat_embeddings = {cat: ST.encode(phrases) for cat, phrases in CATEGORIES.items()}

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def categorize(option_a, option_b):
    combined = f"{option_a} or {option_b}"
    qe = ST.encode(combined)

    best_cat, best_score = None, -1.0
    second_cat, second_score = None, 0.0

    for cat, embs in cat_embeddings.items():
        score = max(cosine(qe, e) for e in embs)
        if score > best_score:
            second_cat, second_score = best_cat, best_score
            best_cat, best_score = cat, score
        elif score > second_score:
            second_cat, second_score = cat, score

    if best_score < MISC_THRESH:
        second_cat = best_cat
        best_cat   = "miscellaneous"

    return best_cat, second_cat

# ── load voted questions with real text ────────────────────────────────────────
print("Loading data from database…")
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query(
    """
    SELECT mo.question_id, mo.model_name, mo.option_a, mo.option_b,
           v.votes_a, v.votes_b
    FROM model_outputs mo
    JOIN votes v ON mo.question_id = v.question_id
                AND mo.model_name  = v.model_name
    WHERE (v.votes_a + v.votes_b) > 0
      AND mo.option_a NOT LIKE '%option%'
    """,
    conn,
)
conn.close()

df["model_name"] = df["model_name"].replace("llama", "tinyllama")
df = df[~df["model_name"].isin(EXCLUDE_MODELS)].copy()

# ── compute balance score ──────────────────────────────────────────────────────
df["total"]   = df["votes_a"] + df["votes_b"]
df["ratio"]   = df["votes_a"] / df["total"]
df["balance"] = 1 - 2 * (df["ratio"] - 0.5).abs()

# ── load quality-weight lookup ─────────────────────────────────────────────────
quality: dict[tuple, int] = defaultdict(lambda: WEIGHT_DEFAULT)

with open(UP_PATH) as fh:
    for rec in json.load(fh):
        k = (rec["MODEL"], rec["optionA"])
        quality[k] += WEIGHT_UPVOTE - WEIGHT_DEFAULT

with open(BAD_PATH) as fh:
    bad = json.load(fh)
    for rec in bad["downvotes"]:
        k = (rec["MODEL"], rec["optionA"])
        quality[k] += WEIGHT_DOWNVOTE - WEIGHT_DEFAULT
    for rec in bad["flagged"]:
        k = (rec["MODEL"], rec["optionA"])
        quality[k] += WEIGHT_FLAG - WEIGHT_DEFAULT

df["quality_w"] = df.apply(
    lambda r: quality[(r["model_name"], r["option_a"])], axis=1
)

# ── categorise every question ──────────────────────────────────────────────────
print(f"Categorising {len(df)} voted questions…")
main_cats, sec_cats = [], []
for _, row in tqdm(df.iterrows(), total=len(df)):
    m, s = categorize(row["option_a"], row["option_b"])
    main_cats.append(m)
    sec_cats.append(s)

df["main_cat"] = main_cats
df["sec_cat"]  = sec_cats

# ── build aggregate matrices ───────────────────────────────────────────────────
MODELS = sorted(df["model_name"].unique())
CATS   = sorted(CATEGORIES.keys())

def build_matrices(cat_col):
    """Return (balance_df, weight_sum_df).

    balance_df   — average raw 50/50 balance per (category, model); shown as cell text
    weight_sum_df — sum of explicit feedback weights per (category, model); drives cell colour
                    upvote +3, downvote −1, flag −3, unrated questions contribute 0
    """
    balance  = pd.DataFrame(np.nan, index=CATS, columns=MODELS)
    wt_sum   = pd.DataFrame(np.nan, index=CATS, columns=MODELS)

    for (cat, model), grp in df.groupby([cat_col, "model_name"]):
        if cat not in CATS or model not in MODELS:
            continue
        balance.loc[cat, model] = grp["balance"].mean()
        # quality_w == WEIGHT_DEFAULT (1) for unrated → subtract default so unrated → 0
        wt_sum.loc[cat, model]  = (grp["quality_w"] - WEIGHT_DEFAULT).sum()

    return balance, wt_sum

main_bal, main_wt = build_matrices("main_cat")
sec_bal,  sec_wt  = build_matrices("sec_cat")

# ── plotting ───────────────────────────────────────────────────────────────────
def save_heatmap(out_path):
    fig, axes = plt.subplots(2, 1, figsize=(14, 16), constrained_layout=True)
    fig.patch.set_facecolor("#1a1a2e")

    pairs = [
        (axes[0], main_bal, main_wt, "Main Category"),
        (axes[1], sec_bal,  sec_wt,  "Secondary Category"),
    ]

    for ax, bal_m, wt_m, subtitle in pairs:
        # Symmetric colour range centred on 0
        abs_max = max(abs(np.nanmin(wt_m.values)), abs(np.nanmax(wt_m.values)))

        # Build annotation strings from balance values (NaN cells show empty)
        annot = bal_m.map(
            lambda v: f"{v:.2f}" if not np.isnan(v) else ""
        )

        seaborn.heatmap(
            wt_m,
            ax=ax,
            annot=annot,
            fmt="",                     # fmt must be "" when annot is a DataFrame
            cmap=STOPLIGHT,
            center=0,
            vmin=-abs_max,
            vmax=abs_max,
            linewidths=0.4,
            linecolor="#333355",
            cbar_kws={"shrink": 0.8},
        )

        ax.set_facecolor("#0d0d1e")
        ax.set_title(f"Vote Balance by Category × Model — {subtitle}",
                     color="white", fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Model",    color="#aaaacc", fontsize=11)
        ax.set_ylabel("Category", color="#aaaacc", fontsize=11)
        ax.tick_params(colors="white")

        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.set_tick_params(color="white")
        cbar.ax.set_ylabel(
            f"feedback score  (upvote +{WEIGHT_UPVOTE} / downvote {WEIGHT_DOWNVOTE} / flag {WEIGHT_FLAG})",
            color="white", fontsize=8,
        )
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    fig.suptitle(
        "Vote Balance Heatmap\n"
        "Cell colour = sum of feedback weights  ·  Cell value = avg balance ratio\n"
        f"Green = net positive feedback  ·  Red = net negative  ·  White ≈ 0  ·  Black = no voted questions",
        color="white", fontsize=13, fontweight="bold",
    )

    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved → {out_path}")


save_heatmap(OUT_PATH)
print("Done.")
