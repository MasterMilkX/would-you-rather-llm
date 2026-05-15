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
DB_PATH          = "wyr_votes.db"

# Remote paths (as seen from GPU server)
REMOTE_JSON      = f"{WEB_SERVER_USER}@{WEB_SERVER_HOST}:{WEB_SERVER_DIR}/current_question.json"
REMOTE_DELTA     = f"{WEB_SERVER_USER}@{WEB_SERVER_HOST}:{WEB_SERVER_DIR}/votes_delta.jsonl"
REMOTE_DELTA_BAK = f"{WEB_SERVER_USER}@{WEB_SERVER_HOST}:{WEB_SERVER_DIR}/votes_delta.jsonl.bak"

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
    """
    print("\n[sync] Pushing question JSON to web server ...")
    return _scp(local_path, REMOTE_JSON, "push question")


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