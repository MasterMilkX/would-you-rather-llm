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
import random
from collections import defaultdict
from flask import Flask, request, jsonify, g, send_from_directory
from datetime import datetime, timezone
from sync import append_vote_delta, add_model_vote

app = Flask(__name__)

JSON_PATH      = "current_question.json"
DELTA_PATH     = "votes_delta.jsonl"
STATUS_PATH    = "job_status.json"
MODEL_MOD_PATH = "model_mods.jsonl"
ARCHIVE_DIR    = "questions_archive"
STATE_PATH     = "server_state.json"

# How long (seconds) before we consider the current question stale and rotate.
# Slightly longer than 1 hour to give the host a window to push without racing.
ROTATION_INTERVAL = 65 * 60

# ---------------------------------------------------------------------------
# Helpers: archive & server state
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _seconds_ago(iso_str: str) -> float:
    """Return how many seconds ago an ISO timestamp was."""
    try:
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds()
    except Exception:
        return float("inf")


def load_server_state() -> dict:
    if not os.path.exists(STATE_PATH):
        return {}
    try:
        with open(STATE_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def save_server_state(state: dict):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def archive_question(q: dict):
    """Save a question dict into questions_archive/{question_id}.json if not already there."""
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    qid = q.get("question_id")
    if qid is None:
        return
    path = os.path.join(ARCHIVE_DIR, f"{qid}.json")
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump(q, f, indent=2)


def list_archived_question_ids() -> list[int]:
    if not os.path.isdir(ARCHIVE_DIR):
        return []
    ids = []
    for fname in os.listdir(ARCHIVE_DIR):
        if fname.endswith(".json"):
            try:
                ids.append(int(fname[:-5]))
            except ValueError:
                pass
    return sorted(ids)


def load_archived_question(qid: int) -> dict | None:
    path = os.path.join(ARCHIVE_DIR, f"{qid}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Helpers: read current question with offline-fallback rotation
# ---------------------------------------------------------------------------

def load_current_question():
    """Load current_question.json. Returns None if missing."""
    if not os.path.exists(JSON_PATH):
        return None
    with open(JSON_PATH) as f:
        return json.load(f)


def get_active_question() -> dict | None:
    """
    Return the question that should be served right now.

    Priority:
    1. If current_question.json has a newer question_id than our state, use it
       (host just pushed a fresh question) and reset state.
    2. If serving_since is within ROTATION_INTERVAL, keep serving the same question.
    3. Otherwise (host is offline / overdue), rotate to an unvisited archived question.
       Falls back to a random archived question if all have been visited.
    4. Last resort: return whatever current_question.json holds even if stale.
    """
    hosted = load_current_question()
    state  = load_server_state()

    hosted_qid = hosted["question_id"] if hosted else None
    state_qid  = state.get("serving_question_id")

    # ── Case 1: host just pushed a fresh question ──────────────────────────
    if hosted_qid is not None and hosted_qid != state_qid:
        # Archive it so it's in the rotation pool later
        archive_question(hosted)
        used = state.get("used_ids", [])
        if hosted_qid not in used:
            used.append(hosted_qid)
        save_server_state({
            "serving_question_id": hosted_qid,
            "serving_since":       _now_iso(),
            "used_ids":            used,
        })
        return hosted

    # ── Case 2: still within rotation window ──────────────────────────────
    serving_since = state.get("serving_since", "")
    if _seconds_ago(serving_since) < ROTATION_INTERVAL:
        # Return whichever question we're currently serving
        if state_qid == hosted_qid:
            return hosted
        q = load_archived_question(state_qid)
        return q if q else hosted

    # ── Case 3: overdue — host is offline, rotate ─────────────────────────
    all_ids  = list_archived_question_ids()
    used_ids = state.get("used_ids", [])

    unvisited = [qid for qid in all_ids if qid not in used_ids]
    if unvisited:
        next_qid = random.choice(unvisited)
    elif all_ids:
        # All have been visited; pick any except the one we just showed
        candidates = [qid for qid in all_ids if qid != state_qid] or all_ids
        next_qid = random.choice(candidates)
    else:
        # No archive at all — stay on whatever we have
        return hosted

    next_q = load_archived_question(next_qid)
    if next_q is None:
        return hosted

    new_used = used_ids + [next_qid]
    save_server_state({
        "serving_question_id": next_qid,
        "serving_since":       _now_iso(),
        "used_ids":            new_used,
    })
    print(f"[rotation] Host offline — rotating to archived question {next_qid} "
          f"(pool: {len(unvisited)} unvisited, {len(all_ids)} total)")
    return next_q


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
    q   = load_current_question()
    s   = load_job_status()
    st  = load_server_state()
    return jsonify({
        "status": "ok",
        "current_question_id":  q["question_id"] if q else None,
        "generated_at":         q["generated_at"] if q else None,
        "serving_question_id":  st.get("serving_question_id"),
        "serving_since":        st.get("serving_since"),
        "archived_count":       len(list_archived_question_ids()),
        "used_ids_count":       len(st.get("used_ids", [])),
        "job_status":           s,
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
    When the host machine is offline, automatically rotates through archived questions.
    """
    q = get_active_question()
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

    if not question_id or slot not in ("A","B","C","D","E","F") or chosen_option not in ("A","B"):
        return jsonify({"error": "Required: question_id, slot (A-D), chosen_option (A/B)"}), 400

    # Load question to get the slot→model mapping.
    # Check current_question.json first, then fall back to archive (covers offline rotation).
    q = load_current_question()
    if not q or q["question_id"] != question_id:
        q = load_archived_question(question_id)
    if not q or q["question_id"] != question_id:
        return jsonify({"error": f"Question {question_id} not found"}), 404

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
    if not q or q["question_id"] != question_id:
        q = load_archived_question(question_id)
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

    model_name = get_slot_model_map(get_active_question()).get(slot) if slot else None

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
    # Seed the archive with whatever question is already on disk so the rotation
    # pool is non-empty from the very first startup.
    _startup_q = load_current_question()
    if _startup_q:
        archive_question(_startup_q)

    app.run(host="0.0.0.0", port=4747, debug=False)