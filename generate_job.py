"""
generate_job.py — Hourly WYR generation job.

Loads each model one at a time (GPU-safe), generates a "Would You Rather"
question, writes results to current_question.json, and inserts a new row
into the SQLite DB with votes initialised at 0.

Run manually:
    python generate_job.py

Schedule hourly via cron (see bottom of file for instructions).
"""

import json
import sqlite3
import random
import re
import os
import gc
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Config — update paths to point at your finetuned merged model folders.
# Use a HuggingFace model ID (e.g. "Qwen/Qwen2.5-0.5B") to pull base weights.
# ---------------------------------------------------------------------------

MODELS = {
    "qwen":  "./model_outputs/qwen-wyr-merged",
    "llama": "./model_outputs/llama-wyr-merged",
    "gemma": "./model_outputs/gemma-wyr-merged",
    "smol":  "./model_outputs/smol-wyr-merged",
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
DB_PATH         = "wyr_votes.db"
JSON_PATH       = "current_question.json"

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
    return text.strip(), ""


# ---------------------------------------------------------------------------
# Main job
# ---------------------------------------------------------------------------

def run():
    print(f"\n{'='*60}")
    print(f"WYR Generation Job — {datetime.now().isoformat()}")
    print(f"{'='*60}")

    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    init_db(db)

    # Insert question row
    cur = db.execute(
        "INSERT INTO questions (generated_at, prompt) VALUES (?, ?)",
        (datetime.utcnow().isoformat(), PROMPT)
    )
    db.commit()
    question_id = cur.lastrowid
    print(f"Created question row id={question_id}")

    # Randomise slot assignment so the frontend doesn't reveal model identity
    model_names = list(MODELS.keys())
    random.shuffle(model_names)
    slots = ["A", "B", "C", "D"]

    results = {}  # slot -> {model_name, option_a, option_b, raw}

    for slot, model_name in zip(slots, model_names):
        model_path = MODELS[model_name]
        print(f"\n--- Slot {slot}: {model_name} ---")

        try:
            tokenizer, model = load_model(model_name, model_path)
            raw = generate_wyr(tokenizer, model)
            unload_model(model, tokenizer)
        except Exception as e:
            print(f"  ERROR loading/generating for {model_name}: {e}")
            raw = f"Would you rather option A or option B?"
            unload_model(None, None) if False else None  # skip unload on error

        print(f"  Raw output: {raw!r}")
        opt_a, opt_b = parse_options(raw)
        print(f"  Option A: {opt_a!r}")
        print(f"  Option B: {opt_b!r}")

        # Save to DB
        db.execute(
            """INSERT INTO model_outputs (question_id, model_name, slot, option_a, option_b)
               VALUES (?, ?, ?, ?, ?)""",
            (question_id, model_name, slot, opt_a, opt_b)
        )
        # Initialise vote row at 0
        db.execute(
            """INSERT INTO votes (question_id, model_name, votes_a, votes_b)
               VALUES (?, ?, 0, 0)""",
            (question_id, model_name)
        )
        db.commit()

        results[slot] = {
            "model_name": model_name,  # hidden from frontend JSON
            "slot":       slot,
            "option_a":   opt_a,
            "option_b":   opt_b,
        }

    # Write JSON for frontend — model_name intentionally excluded
    payload = {
        "question_id":   question_id,
        "generated_at":  datetime.utcnow().isoformat(),
        "slots": {
            slot: {
                "slot":     data["slot"],
                "option_a": data["option_a"],
                "option_b": data["option_b"],
            }
            for slot, data in results.items()
        }
    }

    with open(JSON_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n✓ Wrote {JSON_PATH}")
    print(f"✓ question_id={question_id} saved to {DB_PATH}")
    print(f"✓ Done at {datetime.now().isoformat()}\n")
    db.close()


if __name__ == "__main__":
    run()


# ---------------------------------------------------------------------------
# CRON SETUP
# ---------------------------------------------------------------------------
# Add this line to your crontab (`crontab -e`) to run every hour at :00:
#
#   0 * * * * /usr/bin/python3 /path/to/generate_job.py >> /path/to/wyr_job.log 2>&1
#
# Or use a virtual environment:
#   0 * * * * /path/to/venv/bin/python /path/to/generate_job.py >> /path/to/wyr_job.log 2>&1
#
# To run every hour at :30 instead:
#   30 * * * * /usr/bin/python3 /path/to/generate_job.py >> /path/to/wyr_job.log 2>&1
# ---------------------------------------------------------------------------
