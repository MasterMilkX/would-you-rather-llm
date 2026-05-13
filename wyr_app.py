"""
Would You Rather LLM — Flask App
Y2K themed frontend, 4-model comparison, SQLite voting backend.

Usage:
    pip install flask
    python wyr_app.py

The app serves the frontend at http://localhost:6969
and hits your wyr_server.py API for generation.

SQLite DB: wyr_votes.db (auto-created)
"""

import sqlite3
import json
import random
from flask import Flask, request, jsonify, g
from datetime import datetime

app = Flask(__name__)
DB_PATH = "wyr_votes.db"

# ---------------------------------------------------------------------------
# Point this at your running wyr_server.py instance
# ---------------------------------------------------------------------------
WYR_API_URL = "http://localhost:6969"

# ---------------------------------------------------------------------------
# 4 model "slots" — labels shown to user are randomised per session
# so users don't know which is which. Map these to your actual models.
# ---------------------------------------------------------------------------
MODELS = ["tinyllama", "qwen_0_5b", "qwen_1_5b", "qwen_3b"]
MODEL_LABELS = ["Model A", "Model B", "Model C", "Model D"]

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop("db", None)
    if db:
        db.close()

def init_db():
    with sqlite3.connect(DB_PATH) as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS questions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt      TEXT NOT NULL,
                option_a    TEXT NOT NULL,
                option_b    TEXT NOT NULL,
                model_a     TEXT NOT NULL,
                model_b     TEXT NOT NULL,
                model_c     TEXT NOT NULL,
                model_d     TEXT NOT NULL,
                created_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS votes (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                question_id  INTEGER NOT NULL REFERENCES questions(id),
                chosen_option TEXT NOT NULL,   -- 'A' or 'B'
                preferred_model TEXT NOT NULL, -- 'A','B','C','D'
                voted_at     TEXT NOT NULL
            );
        """)

init_db()

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return app.send_static_file("index.html")

@app.post("/api/generate")
def api_generate():
    """
    Call the WYR generation server for all 4 models, store the question,
    return everything to the frontend with model identities hidden.
    """
    import urllib.request, urllib.error

    results = {}
    order = MODELS[:]
    random.shuffle(order)  # randomise which slot each model fills

    for i, model in enumerate(order):
        label = MODEL_LABELS[i]
        try:
            payload = json.dumps({"model": model, "num": 1}).encode()
            req = urllib.request.Request(
                f"{WYR_API_URL}/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                text = data["questions"][0] if data.get("questions") else ""
        except Exception as e:
            text = f"[Model unavailable: {e}]"

        # Parse "Would you rather X or Y?" / "Would you rather...\nA) X\nB) Y"
        opt_a, opt_b = parse_options(text)
        results[label] = {"raw": text, "option_a": opt_a, "option_b": opt_b, "model": model}

    # Store question (use Model A's options as the canonical question)
    canonical = results["Model A"]
    db = get_db()
    cur = db.execute(
        """INSERT INTO questions
           (prompt, option_a, option_b, model_a, model_b, model_c, model_d, created_at)
           VALUES (?,?,?,?,?,?,?,?)""",
        (
            "Would you rather",
            canonical["option_a"],
            canonical["option_b"],
            results["Model A"]["model"],
            results["Model B"]["model"],
            results["Model C"]["model"],
            results["Model D"]["model"],
            datetime.utcnow().isoformat(),
        )
    )
    db.commit()
    question_id = cur.lastrowid

    return jsonify({
        "question_id": question_id,
        "models": {
            label: {
                "option_a": results[label]["option_a"],
                "option_b": results[label]["option_b"],
            }
            for label in MODEL_LABELS
        }
    })


@app.post("/api/vote")
def api_vote():
    body = request.get_json(silent=True) or {}
    question_id    = body.get("question_id")
    chosen_option  = body.get("chosen_option")   # 'A' or 'B'
    preferred_model = body.get("preferred_model") # 'A','B','C','D'

    if not all([question_id, chosen_option, preferred_model]):
        return jsonify({"error": "Missing fields"}), 400

    db = get_db()
    db.execute(
        "INSERT INTO votes (question_id, chosen_option, preferred_model, voted_at) VALUES (?,?,?,?)",
        (question_id, chosen_option, preferred_model, datetime.utcnow().isoformat())
    )
    db.commit()

    # Return vote tallies for this question
    rows = db.execute(
        "SELECT preferred_model, COUNT(*) as cnt FROM votes WHERE question_id=? GROUP BY preferred_model",
        (question_id,)
    ).fetchall()
    tallies = {r["preferred_model"]: r["cnt"] for r in rows}

    # Reveal which model is which
    q = db.execute("SELECT * FROM questions WHERE id=?", (question_id,)).fetchone()
    reveal = {
        "A": q["model_a"], "B": q["model_b"],
        "C": q["model_c"], "D": q["model_d"],
    } if q else {}

    return jsonify({"tallies": tallies, "reveal": reveal})


@app.get("/api/stats")
def api_stats():
    db = get_db()
    total_q = db.execute("SELECT COUNT(*) as c FROM questions").fetchone()["c"]
    total_v = db.execute("SELECT COUNT(*) as c FROM votes").fetchone()["c"]
    by_model = db.execute(
        """SELECT q.model_a as model, COUNT(*) as cnt FROM votes v
           JOIN questions q ON v.question_id = q.id
           WHERE v.preferred_model = 'A'
           GROUP BY q.model_a"""
    ).fetchall()
    return jsonify({
        "total_questions": total_q,
        "total_votes": total_v,
    })


def parse_options(text):
    """Extract option A and B from generated WYR text."""
    import re
    # Format B: "Would you rather...\nA) foo\nB) bar"
    m = re.search(r"A\)\s*(.+?)\s*\nB\)\s*(.+)", text, re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    # Format A: "Would you rather X or Y?"
    m = re.search(r"[Ww]ould you rather\s+(.+?)\s+or\s+(.+?)[\?.]?$", text, re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    # Fallback
    return text.strip(), "(no option B parsed)"


# ---------------------------------------------------------------------------
# Frontend HTML — Y2K teal theme, inline
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6900, debug=False)
