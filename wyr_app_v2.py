"""
wyr_app.py — Would You Rather LLM Frontend + Vote API

Reads current_question.json (written by generate_job.py) to serve the
active question. Votes are recorded in wyr_votes.db.

Usage:
    pip install flask
    python wyr_app.py
"""

import sqlite3
import json
import os
from flask import Flask, request, jsonify, g, send_from_directory
from datetime import datetime
from sync import append_vote_delta

app   = Flask(__name__)
DB_PATH   = "wyr_votes.db"
JSON_PATH = "current_question.json"

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
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                generated_at TEXT NOT NULL,
                prompt       TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS model_outputs (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                question_id  INTEGER NOT NULL REFERENCES questions(id),
                model_name   TEXT NOT NULL,
                slot         TEXT NOT NULL,
                option_a     TEXT NOT NULL,
                option_b     TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS votes (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                question_id  INTEGER NOT NULL REFERENCES questions(id),
                model_name   TEXT NOT NULL,
                votes_a      INTEGER NOT NULL DEFAULT 0,
                votes_b      INTEGER NOT NULL DEFAULT 0,
                UNIQUE(question_id, model_name)
            );
        """)

init_db()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_current_question():
    """Read the JSON written by generate_job.py."""
    if not os.path.exists(JSON_PATH):
        return None
    with open(JSON_PATH) as f:
        return json.load(f)


def get_vote_totals(db, question_id):
    rows = db.execute(
        "SELECT model_name, votes_a, votes_b FROM votes WHERE question_id=?",
        (question_id,)
    ).fetchall()
    return {r["model_name"]: {"votes_a": r["votes_a"], "votes_b": r["votes_b"]} for r in rows}


def get_slot_to_model(db, question_id):
    rows = db.execute(
        "SELECT slot, model_name FROM model_outputs WHERE question_id=?",
        (question_id,)
    ).fetchall()
    return {r["slot"]: r["model_name"] for r in rows}

# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

STATUS_PATH = "job_status.json"

def load_job_status():
    if not os.path.exists(STATUS_PATH):
        return None
    with open(STATUS_PATH) as f:
        return json.load(f)

@app.get("/health")
def health():
    q = load_current_question()
    s = load_job_status()
    return jsonify({
        "status": "ok",
        "current_question_id": q["question_id"] if q else None,
        "generated_at": q["generated_at"] if q else None,
        "job_status": s,
    })

@app.get("/api/status")
def api_status():
    """Daemon health — last run time, question_id, errors."""
    s = load_job_status()
    if not s:
        return jsonify({"error": "job_status.json not found — has generate_job.py run yet?"}), 404
    # Compute seconds since last run
    try:
        from datetime import timezone
        last = datetime.fromisoformat(s["last_run"])
        age  = (datetime.utcnow() - last).total_seconds()
        s["seconds_since_last_run"] = int(age)
    except Exception:
        pass
    return jsonify(s)


@app.get("/api/question")
def api_question():
    """Return the current question from the JSON file (no model names)."""
    q = load_current_question()
    if not q:
        return jsonify({"error": "No question available yet. Run generate_job.py first."}), 404
    return jsonify(q)


@app.post("/api/vote")
def api_vote():
    """
    Record a vote.

    Body:
        question_id   : int
        slot          : "A" | "B" | "C" | "D"   (the card the user voted on)
        chosen_option : "A" | "B"                (which WYR option they picked)
    """
    body = request.get_json(silent=True) or {}
    question_id    = body.get("question_id")
    slot           = body.get("slot", "").upper()
    chosen_option  = body.get("chosen_option", "").upper()

    if not question_id or slot not in ("A","B","C","D") or chosen_option not in ("A","B"):
        return jsonify({"error": "Required: question_id, slot (A-D), chosen_option (A/B)"}), 400

    db = get_db()

    # Resolve slot -> model_name
    slot_map = get_slot_to_model(db, question_id)
    model_name = slot_map.get(slot)
    if not model_name:
        return jsonify({"error": f"Slot {slot} not found for question {question_id}"}), 404

    # Increment the right vote column
    col = "votes_a" if chosen_option == "A" else "votes_b"
    db.execute(
        f"UPDATE votes SET {col} = {col} + 1 WHERE question_id=? AND model_name=?",
        (question_id, model_name)
    )
    db.commit()

    # Append to delta file so GPU server can pull and merge on next run
    append_vote_delta(question_id, model_name, chosen_option)

    # Return updated totals + reveal model names
    totals = get_vote_totals(db, question_id)
    return jsonify({
        "question_id": question_id,
        "voted_slot": slot,
        "voted_option": chosen_option,
        "totals": totals,           # keyed by actual model name
        "slot_reveal": slot_map,    # reveals which slot = which model
    })


@app.get("/api/results/<int:question_id>")
def api_results(question_id):
    """Get full vote results for any question (reveals model names)."""
    db = get_db()
    totals   = get_vote_totals(db, question_id)
    slot_map = get_slot_to_model(db, question_id)
    if not totals:
        return jsonify({"error": "Question not found"}), 404
    return jsonify({
        "question_id": question_id,
        "totals": totals,
        "slot_reveal": slot_map,
    })

# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4747, debug=False)