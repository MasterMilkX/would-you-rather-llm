"""
sync_votes.py — Lightweight vote sync (no GPU required).

Pulls pending votes from the web server and merges them into the local
wyr_votes.db without running the full model generation pipeline.

Run this any time you want votes to show up in the database:
    python3 sync_votes.py

generate_job.py calls this automatically on each hourly run, but you can
use this script to sync on demand between generation runs.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from sync import pull_and_merge_votes, push_discord_delta
from datetime import datetime

DB_PATH              = "/home/milk/Desktop/RESEARCH/wyr/wyr_votes.db"
DISCORD_DELTA_LOCAL  = "/home/milk/Desktop/RESEARCH/wyr/discord_votes_delta.jsonl"

if __name__ == "__main__":
    print(f"=== Vote Sync — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")

    # 1. Pull web-app votes from votes_delta.jsonl on the web server
    web_merged = pull_and_merge_votes(DB_PATH)

    # 2. Pull Discord votes from discord_votes_delta.jsonl on the web server
    discord_merged = push_discord_delta(DISCORD_DELTA_LOCAL, DB_PATH)

    print(f"\n✓ Done — {web_merged} web vote(s) + {discord_merged} Discord vote(s) merged into DB.")
