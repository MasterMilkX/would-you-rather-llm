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
upvotes = [(j['question_id'], j['model_name']) for j in votes if j['vote_type'] == 'UPVOTE']


# create a new dataset
NEW_DATASET = []
for q, m in upvotes:
    row = db.execute(
        "SELECT option_a, option_b FROM model_outputs WHERE question_id=? AND model_name=?",
        (q, m)
    ).fetchone()
    if row is None:
        continue
    output = {"title": "Would you rather", "optionA": row["option_a"], "optionB": row["option_b"]}
    NEW_DATASET.append(output)

NEW_DATA_OUT = "finetuning-notebooks/upvote_questions.json"
with open(NEW_DATA_OUT, 'w') as nd:
    json.dump(NEW_DATASET, nd, indent=3)