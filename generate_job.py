"""
generate_job.py — WYR generation job.

Loads each model one at a time (GPU-safe), generates a "Would You Rather"
question, writes results to current_question.json, and inserts a new row
into the SQLite DB with votes initialised at 0.

Usage:
    # Run once (normal hourly cron mode):
    python generate_job.py

    # Test mode: skips model loading, uses dummy questions, runs every 2 min:
    python generate_job.py --test

    # Test mode, custom interval (seconds):
    python generate_job.py --test --interval 30

Schedule hourly via cron (see bottom of file for instructions).
"""

import json
import sqlite3
import random
import re
import os
import gc
import sys
import time
import argparse
import torch
import warnings
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from sync import push_question, pull_and_merge_votes

warnings.filterwarnings("ignore", message=".*datetime.datetime.utcnow.*")


# ---------------------------------------------------------------------------
# Config — update paths to point at your finetuned merged model folders.
# Use a HuggingFace model ID (e.g. "Qwen/Qwen2.5-0.5B") to pull base weights.
# ---------------------------------------------------------------------------

MODELS = {
    "qwen":  "/home/milk/Desktop/RESEARCH/wyr/model_outputs/qwen-wyr-merged",
    "llama": "/home/milk/Desktop/RESEARCH/wyr/model_outputs/llama-wyr-merged",
    "gemma": "/home/milk/Desktop/RESEARCH/wyr/model_outputs/gemma-wyr-merged",
    "smol":  "/home/milk/Desktop/RESEARCH/wyr/model_outputs/smol-wyr-merged",
}

# Shared generation settings
GEN_PARAMS = dict(
    max_new_tokens=80,
    do_sample=True,
    temperature=0.75,
    top_p=0.95,
    repetition_penalty=1.2,
)

PROMPT          = "Would you rather"
DB_PATH         = "/home/milk/Desktop/RESEARCH/wyr/wyr_votes.db"
JSON_PATH       = "/home/milk/Desktop/RESEARCH/wyr/current_question.json"
STATUS_PATH     = "/home/milk/Desktop/RESEARCH/wyr/job_status.json"

# ---------------------------------------------------------------------------
# Test-mode dummy questions — used instead of real models with --test
# ---------------------------------------------------------------------------
TEST_QUESTIONS = [
    ("only be able to whisper for the rest of your life",   "only be able to shout for the rest of your life"),
    ("have fingers as long as your legs",                   "have legs as long as your fingers"),
    ("eat cereal with orange juice instead of milk",        "drink a glass of milk with cereal dissolved in it"),
    ("fight 100 duck-sized horses",                         "fight 1 horse-sized duck"),
    ("always arrive 2 hours early",                         "always arrive 20 minutes late"),
    ("have a rewind button for your life",                  "have a pause button for your life"),
    ("never be able to use a search engine again",          "never be able to use social media again"),
    ("speak every language fluently",                       "play every instrument perfectly"),
    ("live in a world without music",                       "live in a world without films"),
    ("always feel like you're about to sneeze",             "always feel like you need to crack your knuckles"),
    ("only eat food that is room temperature",              "only eat food that is extremely spicy"),
    ("have unlimited battery life on all devices",          "have unlimited free food forever"),
]

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def init_db(db: sqlite3.Connection):
    db.executescript("""
        CREATE TABLE IF NOT EXISTS questions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            generated_at TEXT NOT NULL,
            prompt       TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS model_outputs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            question_id INTEGER NOT NULL REFERENCES questions(id),
            model_name  TEXT NOT NULL,
            slot        TEXT NOT NULL,   -- randomised A/B/C/D shown to user
            option_a    TEXT NOT NULL,
            option_b    TEXT NOT NULL
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
    db.commit()

# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def load_model(model_name: str, model_path: str):
    print(f"  [{model_name}] Loading from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto",
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"  [{model_name}] Loaded on {device}")
    return tokenizer, model


def unload_model(model, tokenizer):
    """Free GPU memory before loading the next model."""
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_wyr(tokenizer, model) -> str:
    inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            pad_token_id=tokenizer.eos_token_id,
            **GEN_PARAMS,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()


def parse_options(text: str) -> tuple[str, str]:
    """Extract option A and B from generated WYR text."""
    # Format B: "Would you rather...\nA) foo\nB) bar"
    m = re.search(r"A\)\s*(.+?)\s*\nB\)\s*(.+)", text, re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    # Format A: "Would you rather X or Y?"
    m = re.search(r"[Ww]ould you rather\s+(.+?)\s+or\s+(.+?)[\?.]?$", text, re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    # Fallback: split on " or " anywhere
    parts = re.split(r"\s+or\s+", text, maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return None, None


# ---------------------------------------------------------------------------
# Status file — written after every run so you can verify the daemon is alive
# ---------------------------------------------------------------------------

def write_status(question_id: int, test_mode: bool, error: str = None):
    status = {
        "last_run":       datetime.utcnow().isoformat(),
        "question_id":    question_id,
        "test_mode":      test_mode,
        "error":          error,
        "pid":            os.getpid(),
    }
    with open(STATUS_PATH, "w") as f:
        json.dump(status, f, indent=2)
    print(f"✓ Status written to {STATUS_PATH}")


# ---------------------------------------------------------------------------
# Main job
# ---------------------------------------------------------------------------

def run(test_mode: bool = False):
    now_str = datetime.now().isoformat(timespec='seconds')
    print(f"\n{'='*60}")
    print(f"WYR Generation Job {'[TEST MODE]' if test_mode else ''} — {now_str}")
    print(f"{'='*60}")

    # Pull any votes the web server collected since last run and merge into local DB
    pull_and_merge_votes(DB_PATH)

    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    init_db(db)

    cur = db.execute(
        "INSERT INTO questions (generated_at, prompt) VALUES (?, ?)",
        (datetime.utcnow().isoformat(), PROMPT)
    )
    db.commit()
    question_id = cur.lastrowid
    print(f"Created question row id={question_id}")

    model_names = list(MODELS.keys())
    random.shuffle(model_names)
    slots = ["A", "B", "C", "D"]
    results = {}

    # In test mode, grab 4 distinct random questions without loading any model
    if test_mode:
        sample = random.sample(TEST_QUESTIONS, k=min(4, len(TEST_QUESTIONS)))

    for i, (slot, model_name) in enumerate(zip(slots, model_names)):
        print(f"\n--- Slot {slot}: {model_name} ---")

        if test_mode:
            opt_a, opt_b = sample[i]
            # Stamp the time so you can see freshness on the frontend
            ts = datetime.now().strftime("%H:%M:%S")
            opt_a = f"[{ts}] {opt_a}"
            print(f"  [TEST] Option A: {opt_a!r}")
            print(f"  [TEST] Option B: {opt_b!r}")
        else:
            while True:
                try:
                    tokenizer, model = load_model(model_name, MODELS[model_name])
                    raw = generate_wyr(tokenizer, model)
                    unload_model(model, tokenizer)
                    print(f"  Raw output: {raw!r}")
                    opt_a, opt_b = parse_options(raw)
                    print(f"  Parsed Option A: {opt_a!r}")
                    print(f"  Parsed Option B: {opt_b!r}")
                    if opt_a and opt_b:
                        break
                    else:
                        print("  Failed to parse options, retrying...")
                except Exception as e:
                    print(f"  ERROR: {e}")
                    opt_a, opt_b = "[ERR] option A", "[ERR] option B"
                    break

        db.execute(
            "INSERT INTO model_outputs (question_id, model_name, slot, option_a, option_b) VALUES (?,?,?,?,?)",
            (question_id, model_name, slot, opt_a, opt_b)
        )
        db.execute(
            "INSERT INTO votes (question_id, model_name, votes_a, votes_b) VALUES (?,?,0,0)",
            (question_id, model_name)
        )
        db.commit()

        results[slot] = {"slot": slot, "option_a": opt_a, "option_b": opt_b, "model_name": model_name}

    payload = {
        "question_id":  question_id,
        "generated_at": datetime.utcnow().isoformat(),
        "test_mode":    test_mode,
        # Public: what the frontend API returns (no model names)
        "slots": {
            slot: {"slot": d["slot"], "option_a": d["option_a"], "option_b": d["option_b"]}
            for slot, d in results.items()
        },
        # Private: used by web server to resolve votes — never forwarded to the browser
        "slot_model_map": {slot: d["model_name"] for slot, d in results.items()},
    }
    with open(JSON_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    write_status(question_id, test_mode)
    print(f"\n✓ Wrote {JSON_PATH}")
    push_question(JSON_PATH)
    print(f"✓ question_id={question_id} saved to {DB_PATH}")
    print(f"✓ Done at {datetime.now().isoformat(timespec='seconds')}\n")
    db.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WYR generation job")
    parser.add_argument("--test",     action="store_true",
                        help="Use dummy questions, skip model loading")
    parser.add_argument("--interval", type=int, default=120,
                        help="Seconds between runs in test mode (default: 120)")
    parser.add_argument("--once",     action="store_true",
                        help="Run exactly once then exit (useful for cron)")
    args = parser.parse_args()

    if args.test and not args.once:
        # Daemon loop for test mode
        print(f"TEST MODE — running every {args.interval}s. Ctrl+C to stop.")
        run_count = 0
        while True:
            run_count += 1
            print(f"\n[Run #{run_count}]")
            try:
                run(test_mode=True)
            except Exception as e:
                print(f"ERROR in run: {e}")
                write_status(None, True, str(e))
            print(f"Sleeping {args.interval}s ...")
            time.sleep(args.interval)
    else:
        # Single run (normal cron mode, or --test --once)
        run(test_mode=args.test)


# ---------------------------------------------------------------------------
# CRON SETUP
# ---------------------------------------------------------------------------
# Normal hourly cron (run once at :00 each hour):
#   0 * * * * /path/to/venv/bin/python /path/to/generate_job.py >> /path/to/wyr_job.log 2>&1
#
# Test mode via cron (every 2 minutes, single run):
#   */2 * * * * /path/to/venv/bin/python /path/to/generate_job.py --test --once >> /path/to/wyr_job.log 2>&1
#
# CHECK IF DAEMON IS RUNNING:
#   cat job_status.json          — see last run time, question_id, any errors
#   tail -f wyr_job.log          — live log output
#   ps aux | grep generate_job   — confirm process is alive
# ---------------------------------------------------------------------------