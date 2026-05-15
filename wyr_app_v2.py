"""
wyr_app_v2.py — Would You Rather LLM: Web Server

Reads current_question.json (pushed by generate_job.py on the GPU server).
Does NOT require SQLite — all state the web server needs lives in:
  - current_question.json   (question + hidden slot→model map)
  - votes_delta.jsonl       (append-only vote log; GPU server pulls & merges this)

Usage:
    pip install flask
    python wyr_app_v2.py
"""

import json
import os
from collections import defaultdict
from flask import Flask, request, jsonify, g, send_from_directory
from datetime import datetime
from sync import append_vote_delta, add_model_vote

app = Flask(__name__)

JSON_PATH   = "current_question.json"
DELTA_PATH  = "votes_delta.jsonl"
STATUS_PATH = "job_status.json"
MODEL_MOD_PATH = "model_mods.jsonl"   # for thumbs up/down/flag moderation (not implemented yet)

# ---------------------------------------------------------------------------
# Helpers: read current question
# ---------------------------------------------------------------------------

def load_current_question():
    """Load current_question.json. Returns None if missing."""
    if not os.path.exists(JSON_PATH):
        return None
    with open(JSON_PATH) as f:
        return json.load(f)


def get_slot_model_map(question: dict) -> dict:
    """
    Return {slot: model_name} from the private field in current_question.json.
    Falls back to empty dict if the field is missing (old JSON format).
    """
    return question.get("slot_model_map", {})


# ---------------------------------------------------------------------------
# Helpers: vote tallies from delta file (no SQLite needed)
# ---------------------------------------------------------------------------

def get_vote_totals(question_id: int) -> dict:
    """
    Tally votes for question_id by reading votes_delta.jsonl.
    Returns {model_name: {votes_a: N, votes_b: N}}.
    """
    totals = defaultdict(lambda: {"votes_a": 0, "votes_b": 0})
    if not os.path.exists(DELTA_PATH):
        return {}
    with open(DELTA_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("question_id") != question_id:
                    continue
                model = rec.get("model_name")
                opt   = rec.get("chosen_option", "").upper()
                if model and opt in ("A", "B"):
                    col = "votes_a" if opt == "A" else "votes_b"
                    totals[model][col] += 1
            except (json.JSONDecodeError, KeyError):
                continue
    return dict(totals)


# ---------------------------------------------------------------------------
# Helpers: job status
# ---------------------------------------------------------------------------

def load_job_status():
    if not os.path.exists(STATUS_PATH):
        return None
    with open(STATUS_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    q = load_current_question()
    s = load_job_status()
    return jsonify({
        "status": "ok",
        "current_question_id": q["question_id"] if q else None,
        "generated_at":        q["generated_at"] if q else None,
        "job_status":          s,
    })


@app.get("/api/status")
def api_status():
    """Daemon health — last run time, question_id, errors."""
    s = load_job_status()
    if not s:
        return jsonify({"error": "job_status.json not found — has generate_job.py run yet?"}), 404
    try:
        last = datetime.fromisoformat(s["last_run"])
        s["seconds_since_last_run"] = int((datetime.utcnow() - last).total_seconds())
    except Exception:
        pass
    return jsonify(s)


@app.get("/api/question")
def api_question():
    """
    Return the current question for the frontend.
    Only the public 'slots' field is returned — slot_model_map is intentionally excluded.
    """
    q = load_current_question()
    if not q:
        return jsonify({"error": "No question available yet. Run generate_job.py first."}), 404
    return jsonify({
        "question_id":  q["question_id"],
        "generated_at": q["generated_at"],
        "test_mode":    q.get("test_mode", False),
        "slots":        q["slots"],
    })


@app.post("/api/vote")
def api_vote():
    """
    Record a vote.

    Body:
        question_id   : int
        slot          : "A" | "B" | "C" | "D"
        chosen_option : "A" | "B"
    """
    body          = request.get_json(silent=True) or {}
    question_id   = body.get("question_id")
    slot          = body.get("slot", "").upper()
    chosen_option = body.get("chosen_option", "").upper()

    if not question_id or slot not in ("A","B","C","D") or chosen_option not in ("A","B"):
        return jsonify({"error": "Required: question_id, slot (A-D), chosen_option (A/B)"}), 400

    # Load question to get the slot→model mapping
    q = load_current_question()
    if not q or q["question_id"] != question_id:
        return jsonify({"error": f"Question {question_id} not found in current_question.json"}), 404

    slot_map   = get_slot_model_map(q)
    model_name = slot_map.get(slot)
    if not model_name:
        return jsonify({"error": f"Slot {slot} not found — current_question.json may be from an older format. Re-run generate_job.py."}), 404

    # Append to delta file (GPU server pulls this on next run)
    append_vote_delta(question_id, model_name, chosen_option, DELTA_PATH)

    # Tally from delta file and return with reveal
    totals = get_vote_totals(question_id)
    return jsonify({
        "question_id":  question_id,
        "voted_slot":   slot,
        "voted_option": chosen_option,
        "totals":       totals,
        "slot_reveal":  slot_map,
    })


@app.get("/api/results/<int:question_id>")
def api_results(question_id):
    """Vote results for any question (reveals model names)."""
    q = load_current_question()
    slot_map = get_slot_model_map(q) if q and q["question_id"] == question_id else {}
    totals   = get_vote_totals(question_id)
    if not totals and not slot_map:
        return jsonify({"error": "Question not found"}), 404
    return jsonify({
        "question_id": question_id,
        "totals":      totals,
        "slot_reveal": slot_map,
    })


@app.post("/api/moderate")
def api_moderate():
    """
    Record a thumbs up/down/flag for moderation.

    Body:
        question_id : int
        model_name  : str
        vote_type   : "UPVOTE" | "DOWNVOTE" | "FLAG"
    """
    body        = request.get_json(silent=True) or {}
    question_id = body.get("question_id")
    slot  = body.get("slot", "").upper()  # optional, for frontend context but not required for recording the mod vote
    vote_type   = body.get("vote_type", "").upper()

    model_name = get_slot_model_map(load_current_question()).get(slot) if slot else None

    if not question_id or not model_name or vote_type not in ("UPVOTE", "DOWNVOTE", "FLAG"):
        return jsonify({"error": "Required: question_id, slot, vote_type (UPVOTE/DOWNVOTE/FLAG)"}), 400

    # Append to model_mods delta file (not implemented yet)
    add_model_vote(question_id, model_name, vote_type, MODEL_MOD_PATH)

    return jsonify({"status": "ok", "message": f"Recorded {vote_type} for {model_name} on question {question_id}"})


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4747, debug=False)