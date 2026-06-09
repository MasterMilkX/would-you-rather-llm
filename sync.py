"""
sync.py — Shared sync utilities for the WYR two-machine pipeline.

Architecture:
    GPU server   runs generate_job.py
                 → pushes current_question.json  to web server via SCP after each run
                 → pulls votes_delta.jsonl        from web server before each run and merges it

    Web server   runs wyr_app_v2.py
                 → appends to votes_delta.jsonl   after each vote
                 → web server DOES NOT need to push; GPU server pulls on its schedule

Both machines import this file for their respective sync calls.
Requires passwordless SSH key auth between the two machines.
"""

import json
import os
import sqlite3
import subprocess
from datetime import datetime

# ---------------------------------------------------------------------------
# Shared config — edit these to match your setup
# ---------------------------------------------------------------------------

# SSH connection to the web server (used by GPU server to push/pull)
WEB_SERVER_USER = "milk"
WEB_SERVER_HOST = "141.166.173.142"
WEB_SERVER_DIR  = "/home/server/APPS/wyr"

# Local paths (same filename on both machines, different directories)
JSON_PATH        = "current_question.json"
DELTA_PATH       = "votes_delta.jsonl"    # on web server: appended to after votes
MODEL_MOD_PATH   = "model_mods.jsonl"       # on web server: appended to after thumbs up/down/flag for moderation
DB_PATH          = "wyr_votes.db"

# Remote paths (as seen from GPU server)
REMOTE_JSON      = f"{WEB_SERVER_USER}@{WEB_SERVER_HOST}:{WEB_SERVER_DIR}/current_question.json"
REMOTE_DELTA     = f"{WEB_SERVER_USER}@{WEB_SERVER_HOST}:{WEB_SERVER_DIR}/votes_delta.jsonl"
REMOTE_DELTA_BAK = f"{WEB_SERVER_USER}@{WEB_SERVER_HOST}:{WEB_SERVER_DIR}/votes_delta.jsonl.bak"
REMOTE_MODEL_MOD = f"{WEB_SERVER_USER}@{WEB_SERVER_HOST}:{WEB_SERVER_DIR}/model_mods.jsonl"


SCP_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=accept-new"]

# ---------------------------------------------------------------------------
# Low-level SCP helpers
# ---------------------------------------------------------------------------

def _scp(src: str, dst: str, label: str) -> bool:
    """Run scp src -> dst. Returns True on success."""
    print(f"  SCP {label}: {src} → {dst}")
    try:
        result = subprocess.run(
            ["scp"] + SCP_OPTS + [src, dst],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            print(f"  ✓ {label} OK")
            return True
        else:
            print(f"  ✗ {label} failed (rc={result.returncode}): {result.stderr.strip()}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  ✗ {label} timed out")
        return False
    except FileNotFoundError:
        print("  ✗ scp not found — is openssh-client installed?")
        return False
    except Exception as e:
        print(f"  ✗ {label} error: {e}")
        return False


def _ssh(command: str, label: str) -> bool:
    """Run a remote SSH command. Returns True on success."""
    print(f"  SSH {label}: {command}")
    try:
        result = subprocess.run(
            ["ssh"] + SCP_OPTS + [f"{WEB_SERVER_USER}@{WEB_SERVER_HOST}", command],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            print(f"  ✓ {label} OK")
            return True
        else:
            # rc=1 on "no such file" is fine for optional clears
            print(f"  ✗ {label} (rc={result.returncode}): {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"  ✗ {label} error: {e}")
        return False


# ---------------------------------------------------------------------------
# GPU server: push question JSON to web server
# ---------------------------------------------------------------------------

def push_question(local_path: str = JSON_PATH) -> bool:
    """
    Called by generate_job.py after writing current_question.json.
    Pushes it to the web server so the frontend picks it up immediately.
    Also archives it under questions_archive/{question_id}.json so the server
    can rotate through past questions when the host machine is offline.
    """
    print("\n[sync] Pushing question JSON to web server ...")
    ok = _scp(local_path, REMOTE_JSON, "push question")

    # Archive a copy keyed by question_id so the server can rotate through past questions
    try:
        with open(local_path) as f:
            q = json.load(f)
        qid = q.get("question_id")
        if qid is not None:
            _ssh(f"mkdir -p {WEB_SERVER_DIR}/questions_archive", "create archive dir")
            remote_archive = (
                f"{WEB_SERVER_USER}@{WEB_SERVER_HOST}:"
                f"{WEB_SERVER_DIR}/questions_archive/{qid}.json"
            )
            _scp(local_path, remote_archive, f"archive question {qid}")
    except Exception as e:
        print(f"  ✗ Archive push failed: {e}")

    return ok


# ---------------------------------------------------------------------------
# GPU server: pull vote deltas from web server and merge into local DB
# ---------------------------------------------------------------------------

def pull_and_merge_votes(local_db_path: str = DB_PATH) -> int:
    """
    Called by generate_job.py at the START of each run (before generating).
    1. SCPs votes_delta.jsonl from the web server to a local temp file
    2. Merges each vote record into the local SQLite DB
    3. Clears the remote delta file atomically (rename to .bak then truncate)

    Returns the number of vote records merged.
    """
    print("\n[sync] Pulling vote deltas from web server ...")

    tmp_delta = "votes_delta_incoming.jsonl"

    # Pull the delta file — if it doesn't exist yet that's fine
    ok = _scp(REMOTE_DELTA, tmp_delta, "pull delta")
    if not ok or not os.path.exists(tmp_delta) or os.path.getsize(tmp_delta) == 0:
        print("  No pending vote deltas.")
        if os.path.exists(tmp_delta):
            os.remove(tmp_delta)
        return 0

    # Read all records before touching the remote file
    records = []
    with open(tmp_delta) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  ✗ Skipping malformed delta line: {e} — {line!r}")

    print(f"  Found {len(records)} vote record(s) to merge")

    if not records:
        os.remove(tmp_delta)
        return 0

    # Atomically clear the remote delta:
    # backup to .bak, then truncate the live file
    # (truncate not delete so the web server's file handle stays valid)
    _ssh(f"cp {WEB_SERVER_DIR}/votes_delta.jsonl {WEB_SERVER_DIR}/votes_delta.jsonl.bak "
         f"&& truncate -s 0 {WEB_SERVER_DIR}/votes_delta.jsonl",
         "clear remote delta")

    # Merge into local DB
    db = sqlite3.connect(local_db_path)
    merged = 0
    skipped = 0
    for rec in records:
        try:
            question_id = rec["question_id"]
            model_name  = rec["model_name"]
            col         = "votes_a" if rec["chosen_option"] == "A" else "votes_b"
            db.execute(
                f"UPDATE votes SET {col} = {col} + 1 "
                f"WHERE question_id = ? AND model_name = ?",
                (question_id, model_name)
            )
            merged += 1
        except Exception as e:
            print(f"  ✗ Failed to merge record {rec}: {e}")
            skipped += 1
    db.commit()
    db.close()

    os.remove(tmp_delta)
    print(f"  ✓ Merged {merged} vote(s), skipped {skipped}")
    return merged


# ---------------------------------------------------------------------------
# Web server: append a vote to the local delta file
# ---------------------------------------------------------------------------

def append_vote_delta(question_id: int, model_name: str, chosen_option: str,
                      delta_path: str = DELTA_PATH):
    """
    Called by wyr_app_v2.py after each successful vote.
    Appends one JSON line to votes_delta.jsonl so the GPU server
    can pull and merge it on the next generation run.

    chosen_option: "A" or "B"
    """
    record = {
        "question_id":   question_id,
        "model_name":    model_name,
        "chosen_option": chosen_option,
        "voted_at":      datetime.utcnow().isoformat(),
    }
    with open(delta_path, "a") as f:
        f.write(json.dumps(record) + "\n")


# append thumbs up/down/flag to delta for moderation
def add_model_vote(question_id: int, model_name: str, vote_type: str, delta_path: str = DELTA_PATH):
    """
    Called by wyr_app_v2.py after a thumbs up/down/flag click.
    Appends one JSON line to votes_delta.jsonl with the feedback.

    vote_type: "UPVOTE", "DOWNVOTE", or "FLAG"
    """
    record = {
        "question_id": question_id,
        "model_name": model_name,
        "vote_type": vote_type,
        "voted_at": datetime.utcnow().isoformat(),
    }
    with open(delta_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def pull_model_mods(local_path: str = MODEL_MOD_PATH) -> int:
    """
    Called by generate_job.py (or manually) to sync model moderation votes
    from the web server to the local machine.

    Pulls model_mods.jsonl from the web server and replaces the local copy.
    The remote file is NOT deleted or truncated — it stays as the authoritative log.

    Returns the number of records in the pulled file (0 if nothing to pull).
    """
    print("\n[sync] Pulling model mods from web server ...")

    tmp_path = local_path + ".incoming"

    ok = _scp(REMOTE_MODEL_MOD, tmp_path, "pull model mods")
    if not ok or not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
        print("  No model mods on server (or file empty).")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return 0

    count = 0
    with open(tmp_path) as f:
        for line in f:
            if line.strip():
                count += 1

    os.replace(tmp_path, local_path)
    print(f"  ✓ Pulled {count} mod record(s) → {local_path}")
    return count


# ---------------------------------------------------------------------------
# GPU server: push unvoted recent questions to the server's archive
# ---------------------------------------------------------------------------

def push_unvoted_archive(db_path: str = DB_PATH, lookback_days: int = 7) -> int:
    """
    Called by generate_job.py after pulling votes each run.
    Finds questions from the last `lookback_days` days with zero total votes,
    reconstructs their JSON payload, and SCPs each to the server's
    questions_archive directory — making them available for offline rotation.

    Questions that received any votes are skipped (they've been served already).
    Returns the number of files pushed.
    """
    print("\n[sync] Checking for unvoted recent questions to archive ...")

    try:
        import sqlite3 as _sqlite3
        db = _sqlite3.connect(db_path)
        rows = db.execute("""
            SELECT q.id, q.generated_at
            FROM questions q
            LEFT JOIN votes v ON v.question_id = q.id
            WHERE q.generated_at >= datetime('now', ? || ' days')
            GROUP BY q.id
            HAVING COALESCE(SUM(v.votes_a + v.votes_b), 0) = 0
            ORDER BY q.id
        """, (f"-{lookback_days}",)).fetchall()

        if not rows:
            print("  No unvoted questions in lookback window.")
            db.close()
            return 0

        print(f"  Found {len(rows)} unvoted question(s) — pushing to server archive ...")
        _ssh(f"mkdir -p {WEB_SERVER_DIR}/questions_archive", "create archive dir")

        pushed = 0
        for (qid, generated_at) in rows:
            outputs = db.execute(
                "SELECT model_name, slot, option_a, option_b "
                "FROM model_outputs WHERE question_id = ? ORDER BY slot",
                (qid,),
            ).fetchall()
            if not outputs:
                continue

            slots = {}
            slot_model_map = {}
            for model_name, slot, option_a, option_b in outputs:
                slots[slot] = {"slot": slot, "option_a": option_a, "option_b": option_b}
                slot_model_map[slot] = model_name

            payload = {
                "question_id":    qid,
                "generated_at":   generated_at,
                "test_mode":      False,
                "slots":          slots,
                "slot_model_map": slot_model_map,
            }

            tmp = f"/tmp/wyr_archive_{qid}.json"
            with open(tmp, "w") as f:
                f.write(json.dumps(payload, indent=2))

            remote = f"{WEB_SERVER_USER}@{WEB_SERVER_HOST}:{WEB_SERVER_DIR}/questions_archive/{qid}.json"
            ok = _scp(tmp, remote, f"archive unvoted q{qid}")
            os.remove(tmp)
            if ok:
                pushed += 1

        db.close()
        print(f"  ✓ Archived {pushed}/{len(rows)} unvoted question(s)")
        return pushed

    except Exception as e:
        print(f"  ✗ push_unvoted_archive failed: {e}")
        return 0
