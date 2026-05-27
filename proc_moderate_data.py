import sqlite3
import json
from sync import pull_model_mods


# connect database
DB_PATH = "wyr_votes.db"
db = sqlite3.connect(DB_PATH)
db.row_factory = sqlite3.Row

# grab the votes
MODEL_MODS_PATH = "model_mods.jsonl"
pull_model_mods(MODEL_MODS_PATH)

# grab the model moderation votes
votes = []
with open(MODEL_MODS_PATH, 'r') as jf:
    for l in jf:
        votes.append(json.loads(l))

# only grab positive
upvotes = []
downvotes = []
flags = []
for j in votes:
    d = (j['question_id'], j['model_name'])
    if j['vote_type'] == 'UPVOTE':
        upvotes.append(d)
    elif j['vote_type'] == 'DOWNVOTE':
        downvotes.append(d)
    elif j['vote_type'] == 'FLAG':
        flags.append(d)

# create a new dataset
NEW_DATASET = []
for q, m in upvotes:
    row = db.execute(
        "SELECT option_a, option_b FROM model_outputs WHERE question_id=? AND model_name=?",
        (q, m)
    ).fetchone()
    if row is None:
        continue
    output = {"MODEL":m, "title": "Would you rather", "optionA": row["option_a"], "optionB": row["option_b"]}
    NEW_DATASET.append(output)

NEW_DATA_OUT = "analysis/upvote_questions.json"
with open(NEW_DATA_OUT, 'w') as nd:
    json.dump(NEW_DATASET, nd, indent=3)


# export bad ones
# create a new dataset
DOWN_DATASET = []
FLAG_DATASET = []
for q, m in downvotes:
    row = db.execute(
        "SELECT option_a, option_b FROM model_outputs WHERE question_id=? AND model_name=?",
        (q, m)
    ).fetchone()
    if row is None:
        continue
    output = {"MODEL":m, "optionA": row["option_a"], "optionB": row["option_b"]}
    DOWN_DATASET.append(output)

for q, m in flags:
    row = db.execute(
        "SELECT option_a, option_b FROM model_outputs WHERE question_id=? AND model_name=?",
        (q, m)
    ).fetchone()
    if row is None:
        continue
    output = {"MODEL":m, "optionA": row["option_a"], "optionB": row["option_b"]}
    FLAG_DATASET.append(output)

BAD_DATA_OUT = "analysis/bad_questions.json"
with open(BAD_DATA_OUT, 'w') as nd:
    json.dump({"downvotes":DOWN_DATASET, "flagged":FLAG_DATASET}, nd, indent=3)
    
