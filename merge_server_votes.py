"""
merge_server_votes.py — One-time migration + ongoing sync helper.

Pulls vote data from the web server's wyr_votes.db and votes_delta.jsonl
and merges it into the local GPU server's wyr_votes.db.

Safe to run multiple times — uses INSERT OR IGNORE + additive UPDATE so
re-running won't double-count (it adds to whatever is already there, which
is fine because the web server DB and local DB should have disjoint vote
sources after you update the Discord bot to use votes_delta.jsonl).

Usage:
    python3 merge_server_votes.py
"""

import sqlite3
import json
import subprocess
import os
import sys
from datetime import datetime

LOCAL_DB      = "/home/milk/Desktop/RESEARCH/wyr/wyr_votes.db"
REMOTE_USER   = "milk"
REMOTE_HOST   = "141.166.173.142"
REMOTE_DIR    = "/home/server/APPS/wyr"
REMOTE_DB     = f"{REMOTE_USER}@{REMOTE_HOST}:{REMOTE_DIR}/wyr_votes.db"
REMOTE_DELTA  = f"{REMOTE_USER}@{REMOTE_HOST}:{REMOTE_DIR}/votes_delta.jsonl"
TMP_DB        = "/tmp/wyr_server_votes.db"
TMP_DELTA     = "/tmp/wyr_server_delta.jsonl"
SCP_OPTS      = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                 "-o", "StrictHostKeyChecking=accept-new"]


def scp(src, dst, label):
    print(f"  SCP {label}: {src} → {dst}")
    r = subprocess.run(["scp"] + SCP_OPTS + [src, dst],
                       capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        print(f"  ✗ failed: {r.stderr.strip()}")
        return False
    print(f"  ✓ OK")
    return True


# ---------------------------------------------------------------------------
# Step 1: pull web server DB and merge votes table
# ---------------------------------------------------------------------------

def merge_remote_db():
    print("\n[1] Pulling web server wyr_votes.db ...")
    if not scp(REMOTE_DB, TMP_DB, "web server DB"):
        print("  Skipping DB merge.")
        return 0

    remote = sqlite3.connect(TMP_DB)
    local  = sqlite3.connect(LOCAL_DB)

    rows = remote.execute(
        "SELECT question_id, model_name, votes_a, votes_b FROM votes "
        "WHERE votes_a + votes_b > 0"
    ).fetchall()
    remote.close()
    print(f"  Found {len(rows)} vote rows with data on web server")

    merged = 0
    for question_id, model_name, votes_a, votes_b in rows:
        # Ensure the row exists in local DB
        local.execute(
            "INSERT OR IGNORE INTO votes (question_id, model_name, votes_a, votes_b) "
            "VALUES (?,?,0,0)",
            (question_id, model_name)
        )
        # Check what's already there to avoid double-counting on re-run
        existing = local.execute(
            "SELECT votes_a, votes_b FROM votes WHERE question_id=? AND model_name=?",
            (question_id, model_name)
        ).fetchone()

        # Only add the delta above what's already in the local DB
        # (assumes local DB started at 0 for web-server-only votes)
        add_a = max(0, votes_a - (existing[0] if existing else 0))
        add_b = max(0, votes_b - (existing[1] if existing else 0))

        if add_a > 0 or add_b > 0:
            local.execute(
                "UPDATE votes SET votes_a = votes_a + ?, votes_b = votes_b + ? "
                "WHERE question_id = ? AND model_name = ?",
                (add_a, add_b, question_id, model_name)
            )
            merged += 1

    local.commit()
    local.close()
    os.remove(TMP_DB)
    print(f"  ✓ Merged {merged} vote rows into local DB")
    return merged


# ---------------------------------------------------------------------------
# Step 2: pull votes_delta.jsonl and merge individual vote records
# ---------------------------------------------------------------------------

def merge_remote_delta():
    print("\n[2] Pulling web server votes_delta.jsonl ...")
    if not scp(REMOTE_DELTA, TMP_DELTA, "web server delta"):
        print("  Skipping delta merge.")
        return 0

    if not os.path.exists(TMP_DELTA) or os.path.getsize(TMP_DELTA) == 0:
        print("  Delta file empty — nothing to merge.")
        return 0

    records = []
    with open(TMP_DELTA) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("source") == "discord":
                    continue  # already handled by merge_remote_db
                if "chosen_option" not in rec:
                    continue  # moderation vote, not a A/B vote
                records.append(rec)
            except json.JSONDecodeError:
                pass

    print(f"  Found {len(records)} vote records in delta")
    if not records:
        os.remove(TMP_DELTA)
        return 0

    local = sqlite3.connect(LOCAL_DB)
    merged = 0
    for rec in records:
        try:
            qid   = rec["question_id"]
            model = rec["model_name"]
            col   = "votes_a" if rec["chosen_option"] == "A" else "votes_b"
            local.execute(
                "INSERT OR IGNORE INTO votes (question_id, model_name, votes_a, votes_b) "
                "VALUES (?,?,0,0)", (qid, model)
            )
            local.execute(
                f"UPDATE votes SET {col} = {col} + 1 WHERE question_id=? AND model_name=?",
                (qid, model)
            )
            merged += 1
        except Exception as e:
            print(f"  ✗ Skipping record {rec}: {e}")
    local.commit()
    local.close()
    os.remove(TMP_DELTA)
    print(f"  ✓ Merged {merged} delta records into local DB")
    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"=== WYR Vote Migration — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    db_merged    = merge_remote_db()
    delta_merged = merge_remote_delta()
    print(f"\n✓ Done — {db_merged} DB rows + {delta_merged} delta records merged.")
    print("  Run python3 generate_job.py (or wait for the hourly cron) to push")
    print("  the updated DB back to the web server via the normal sync pipeline.")
