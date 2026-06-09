"""
export_archive.py — Bulk-export unvoted questions from wyr_votes.db into
questions_archive/ and (optionally) push them to the web server.

Run this once before switching OS to pre-populate the server's rotation pool
with the 500+ unvoted questions already in the database.

Usage:
    # Export to local questions_archive/ only:
    python export_archive.py

    # Export locally AND push every file to the web server:
    python export_archive.py --push

    # Preview which questions would be exported (no writes):
    python export_archive.py --dry-run
"""

import argparse
import json
import os
import sqlite3
import tempfile

from sync import (
    _scp, _ssh,
    WEB_SERVER_USER, WEB_SERVER_HOST, WEB_SERVER_DIR,
    DB_PATH,
)

ARCHIVE_DIR = "questions_archive"
REMOTE_ARCHIVE_DIR = f"{WEB_SERVER_DIR}/questions_archive"


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_unvoted_ids(db: sqlite3.Connection) -> list[int]:
    rows = db.execute("""
        SELECT q.id
        FROM questions q
        LEFT JOIN votes v ON v.question_id = q.id
        GROUP BY q.id
        HAVING COALESCE(SUM(v.votes_a + v.votes_b), 0) = 0
        ORDER BY q.id
    """).fetchall()
    return [r[0] for r in rows]


def build_question_json(db: sqlite3.Connection, question_id: int) -> dict | None:
    q_row = db.execute(
        "SELECT id, generated_at FROM questions WHERE id = ?", (question_id,)
    ).fetchone()
    if not q_row:
        return None

    outputs = db.execute(
        "SELECT model_name, slot, option_a, option_b "
        "FROM model_outputs WHERE question_id = ? ORDER BY slot",
        (question_id,),
    ).fetchall()
    if not outputs:
        return None

    slots = {}
    slot_model_map = {}
    for model_name, slot, option_a, option_b in outputs:
        slots[slot] = {"slot": slot, "option_a": option_a, "option_b": option_b}
        slot_model_map[slot] = model_name

    return {
        "question_id":   question_id,
        "generated_at":  q_row[1],
        "test_mode":     False,
        "slots":         slots,
        "slot_model_map": slot_model_map,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Export unvoted WYR questions to archive")
    parser.add_argument("--push",    action="store_true", help="SCP archive files to web server")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing anything")
    parser.add_argument("--db",      default=DB_PATH,    help="Path to wyr_votes.db")
    parser.add_argument("--out",     default=ARCHIVE_DIR, help="Local archive output directory")
    args = parser.parse_args()

    db = sqlite3.connect(args.db)
    unvoted_ids = get_unvoted_ids(db)
    print(f"Found {len(unvoted_ids)} unvoted questions in {args.db}")

    if args.dry_run:
        print("Dry run — no files will be written.")
        for qid in unvoted_ids:
            print(f"  would export question_id={qid}")
        db.close()
        return

    os.makedirs(args.out, exist_ok=True)

    if args.push:
        _ssh(f"mkdir -p {REMOTE_ARCHIVE_DIR}", "create remote archive dir")

    exported = 0
    skipped  = 0
    failed   = 0

    for qid in unvoted_ids:
        local_path = os.path.join(args.out, f"{qid}.json")

        # Build the payload
        payload = build_question_json(db, qid)
        if payload is None:
            print(f"  [{qid}] no model_outputs — skipping")
            failed += 1
            continue

        # Write locally (skip if already present and identical question_id)
        if os.path.exists(local_path):
            skipped += 1
        else:
            with open(local_path, "w") as f:
                json.dump(payload, f, indent=2)
            exported += 1
            print(f"  [{qid}] → {local_path}")

        # Push to server if requested
        if args.push:
            remote_path = f"{WEB_SERVER_USER}@{WEB_SERVER_HOST}:{REMOTE_ARCHIVE_DIR}/{qid}.json"
            _scp(local_path, remote_path, f"push {qid}")

    db.close()

    print(f"\nDone. exported={exported}  already_existed={skipped}  failed={failed}")
    if args.push:
        print(f"Files pushed to {WEB_SERVER_USER}@{WEB_SERVER_HOST}:{REMOTE_ARCHIVE_DIR}/")


if __name__ == "__main__":
    main()
