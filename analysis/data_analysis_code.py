from sentence_transformers import SentenceTransformer
import hdbscan
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import sqlite3
import pandas as pd
from tqdm import tqdm
from datetime import datetime 


# import model
ST_MODEL = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

# add categories
CATEGORIES = {
    "animals":                       ["animals", "pets", "creatures", "wildlife"],
    "body parts":                    ["body parts", "limbs", "organs", "physical features", "arms", "legs"],
    "money":                         ["money", "wealth", "finances", "$", "rich", "dollars", "currency"],
    "food / drink":                  ["food", "drink", "eating", "meals", "cooking", "drinking"],
    "life":                          ["life", "survival", "status", "well-being", "death", "dying", "mortality", "afterlife", "living forever"],
    "ability":                       ["superpower", "ability", "special ability", "magic power", "power", "skill", "be able to"],
    "curse":                         ["curse", "situation", "stuck", "trapped", "bewitched", "inhibited", "rules", "forced"],
    "age / time":                    ["age", "time", "youth", "old age", "going back in time"],
    "people":                        ["people", 'population', 'persons', 'everybody', 'strangers', 'neighbors', 'friends', "occupation", "job", "career"],
    "locations":                     ["location", "countries", "cities", "towns", "world", "space"],
    "sentience":                     ["become", "becoming", "inanimate", "transform", "stuck", "mutation", "live as inanimate object", "be a", "be an"],
    "miscellaneous":                 ["miscellaneous", "other", "general", "random", "unrelated"],
}

MISC_THRESH = 0.50

# embed the categories
print("Pre-computing category embeddings...")
category_embeddings = {
    cat: ST_MODEL.encode(phrases)
    for cat, phrases in CATEGORIES.items()
}

# cosine
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


 
def categorize_pairing(pairing):
    # Combine both options into one text for embedding
    combined = f"{pairing['a']} or {pairing['b']}"
    query_embedding = ST_MODEL.encode(combined)
 
    best_category = None
    best_score = -1
    closest_category = "n/a"
    second_score = 0
 
    for category, phrase_embeddings in category_embeddings.items():
        # Score = max similarity across all representative phrases for this category
        scores = [cosine_similarity(query_embedding, pe) for pe in phrase_embeddings]
        score = max(scores)
        if score > best_score:
            closest_category = best_category
            second_score = best_score
            best_score = score
            best_category = category

        elif second_score < score:
            closest_category = category
            second_score = score

    if best_score < MISC_THRESH:
        closest_category = best_category
        best_category = "miscellaneous"
        
 
    return {**pairing, "main_category": best_category, "secondary_category":closest_category, "score": round(float(best_score), 4)}
 


def vis_data_cat(data, model_name):
    # Count entries by main and secondary category
    category_counts = defaultdict(lambda: defaultdict(int))

    main_cts = {}
    second_cts = {}

    for m in CATEGORIES:
        main_cts[m] = 0
        second_cts[m] = 0

    for entry in data:
        main = entry['main_category']
        secondary = entry['secondary_category']
        main_cts[main] += 1
        second_cts[secondary] += 1

    # For each main category, identify the most common secondary category
    all_categories = sorted(category_counts.keys())
    most_common_secondary = {}
    other_counts = {}

    # Prepare data for stacked bar chart
    blue_values = [main_cts[m] for m in CATEGORIES]
    green_values = [second_cts[m] for m in CATEGORIES]

    # print(blue_values)

    # Create the stacked bar chart
    fig, ax = plt.subplots(figsize=(14, 6))

    x_pos = range(len(CATEGORIES))
    ax.bar(x_pos, blue_values, label='Main', color='blue', alpha=0.8)
    ax.bar(x_pos, green_values, bottom=blue_values, label='Secondary', color='green', alpha=0.8)

    ax.set_xlabel('Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Entries', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name} MODEL THEME (Stacked Bar Chart) - {len(data)}', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(CATEGORIES, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'graphs/{model_name}-[{datetime.now()}].png', dpi=300, bbox_inches='tight')
    plt.show()


# Connect to SQLite database
conn = sqlite3.connect('../wyr_votes.db')
cursor = conn.cursor()

# Load data from database
query = "SELECT model_name, option_a, option_b FROM model_outputs"
df = pd.read_sql_query(query, conn)
conn.close()

ALL_MODELS = sorted(df['model_name'].unique())
print(f"Loaded {len(df)} entries from wyr_votes.db")
print(f"\nUnique models: {ALL_MODELS}")
print(f"\nData sample:")
print(df.head())

# group by model
MODEL_SET = {}

# group each entry
num_entries = 0
for index, entry in df.iterrows():
    if entry['option_a'] == "Option A" or entry['option_b'] == "Option B":
        continue

    model = entry['model_name']

    # change llama to tinyllama
    if model == "llama":
        model = "tinyllama"
        
    if model not in MODEL_SET:
        MODEL_SET[model] = []
    MODEL_SET[model].append(entry)
    num_entries += 1


# categorize each pairing in each model
CAT_PAIRS = {}
with tqdm(total=num_entries) as pbar:
    for model, entries in MODEL_SET.items():
        CAT_PAIRS[model] = []
        for e in entries:
            CAT_PAIRS[model].append(categorize_pairing({'a':e['option_a'], 'b':e['option_b']}))
            pbar.update(1)


# visualize each model
for model, dat in CAT_PAIRS.items():
    vis_data_cat(dat, model)




# get the upvoted, downvoted, flagged
with open("upvote_questions.json", 'r') as ud:
    upvote_dat = json.load(ud)


with open("bad_questions.json", 'r') as bd:
    bad_dat = json.load(bd)
    downvote_dat = bad_dat['downvotes']
    flagged_dat = bad_dat['flagged']


# categorize
upvote_cats = []
for u in upvote_dat:
    c = categorize_pairing({'a':u['optionA'],'b':u['optionB']})
    c['model'] = u['MODEL']
    upvote_cats.append(c)

downvote_cats = []
for u in downvote_dat:
    c = categorize_pairing({'a':u['optionA'],'b':u['optionB']})
    c['model'] = u['MODEL']
    downvote_cats.append(c)

flagged_cats = []
for u in flagged_dat:
    c = categorize_pairing({'a':u['optionA'],'b':u['optionB']})
    c['model'] = u['MODEL']
    flagged_cats.append(c)


# make 2d array for heatmap
# calculate votes with equation
def vote_score_dataframe(ups,downs,flags):
    # x-axis : MODEL
    # y-axis : CATEGORY
    # value = 2*(# upvotes) + -1*(# downvotes) + -3*(# flagged)

    arr_main = np.full((len(CATEGORIES),len(ALL_MODELS)), np.nan)
    arr_sec = np.full((len(CATEGORIES),len(ALL_MODELS)), np.nan)
    y_axis = np.array(CATEGORIES)
    x_axis = np.array(ALL_MODELS)


    # upvotes -> +3
    scoring = [3,-1,-3]
    ds = [ups, downs, flags]
    for i in range(3):
        v = ds[i]
        s = scoring[i]
        for u in v:
            model = u['model']
            m_cat = u['main_category']
            s_cat = u['secondary_category']

            mcell = arr_main[y_axis.index(m_cat)][x_axis.index(model)]
            if not mcell:
                mcell = 0
            mcell += s

            scell = arr_sec[y_axis.index(s_cat)][x_axis.index(model)]
            if not scell:
                scell = 0
            scell += s

    # turn into dataframe
    mdf = pd.DataFrame(arr_main, index=y_axis, columns=x_axis)
    sdf = pd.DataFrame(arr_sec, index=y_axis, columns=x_axis)

    return mdf, sdf



    



