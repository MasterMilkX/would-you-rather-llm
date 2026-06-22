# Would You Rather — LLM Fine-tuning & Voting Platform

A research platform for fine-tuning small language models to generate "Would You Rather" questions, collecting human preference votes, and analyzing model quality across question categories.

## Overview

Multiple small LLMs (TinyLlama, Qwen, Gemma, SmolLM, etc.) are fine-tuned on WYR questions and pitted against each other. Questions are served through a web app and a Discord bot where users vote A or B. Votes are synced back to a GPU server where models are periodically retrained on quality-rated data.

## Architecture

```
GPU Server                          Web Server
──────────────────────────────      ──────────────────────────────
generate_job.py (hourly cron)  ───► wyr_app_v2.py (Flask)
  - runs all models                   - serves current_question.json
  - writes current_question.json      - appends votes to votes_delta.jsonl
  - pushes JSON to web server         - handles thumbs up/down/flag →
  - pulls votes_delta.jsonl  ◄──────    model_mods.jsonl
  - merges votes into SQLite DB

discord_bot.py (always on)     ◄──► web server (pulls archived questions)
  - posts 1 question/hour             
  - users vote via buttons
  - votes go into wyr_votes.db
```

## Python Scripts

### Core Pipeline

**`generate_job.py`**
The main hourly generation job run on the GPU server. Loads each fine-tuned model one at a time (GPU-safe, sequential loading to avoid OOM), generates a WYR question per model, writes results to `current_question.json`, and inserts a new row into the SQLite database. After generating, it pushes the new question to the web server via SCP and pulls any pending votes from `votes_delta.jsonl` to merge into the local DB. Supports `--test` mode that skips model loading and uses dummy questions for pipeline testing.

**`wyr_app_v2.py`**
The lightweight Flask web server (runs on the web/VPS machine). Reads `current_question.json` pushed by the GPU server and serves it to users. Does not require its own SQLite DB — votes are appended to `votes_delta.jsonl` (an append-only log) and model feedback (upvotes, downvotes, flags) goes to `model_mods.jsonl`. The GPU server pulls both files on each generation cycle. Rotates to a new archived question automatically if the current one goes stale (after ~65 minutes).

**`wyr_server.py`**
A standalone Flask inference API for the v1 app. Loads a fine-tuned TinyLlama and a Qwen model and serves generation requests. Exposes `/generate` (single model) and `/compare` (both models side-by-side) endpoints with configurable temperature, top-p, and token count. Used by the older `wyr_app.py`.

**`wyr_app.py`**
The original v1 Flask app with a Y2K-themed frontend. Runs its own SQLite DB locally and calls `wyr_server.py` for generation. Supports 4-model comparison with randomized slot labels so voters don't know which model produced which question. Superseded by the two-machine `wyr_app_v2.py` setup.

**`sync.py`**
Shared sync utilities imported by both the GPU server and the web server. Contains all SCP/SSH helper functions, path constants, and the core logic for pushing questions, pulling/merging vote deltas, archiving unvoted questions, and syncing Discord vote deltas. Both machines import this module — the GPU server uses the push/pull functions while the web server uses the append functions.

### Sync & Data Migration

**`sync_votes.py`**
Lightweight on-demand vote sync — pulls pending votes from the web server and merges them into the local `wyr_votes.db` without running the full model generation pipeline. `generate_job.py` calls this automatically each hour, but you can run it manually any time to get votes into the DB between generation runs.

**`merge_server_votes.py`**
One-time migration helper for pulling the web server's entire `wyr_votes.db` and `votes_delta.jsonl` and merging them into the local GPU server DB. Safe to run multiple times (uses `INSERT OR IGNORE` + additive `UPDATE`). Used when switching machines or recovering from DB divergence.

**`export_archive.py`**
Bulk-exports unvoted questions from `wyr_votes.db` into the `questions_archive/` directory as individual JSON files, and optionally pushes them to the web server (`--push`). Run this once when setting up a new web server to pre-populate its rotation pool with existing unvoted questions. Supports `--dry-run` to preview what would be exported.

### Discord Bot

**`discord_bot.py`**
A Discord bot that posts one WYR question per hour to one or more configured channels. It works through the archive of unvoted questions, picking a single model's output per post. Users vote via A/B buttons and votes are written directly to `wyr_votes.db`. Tracks posted question/slot pairs in `discord_state.json` to avoid repeating. Falls back to random archived questions when the unvoted pool is exhausted. Uses a file lock to prevent duplicate instances.

### Data Processing

**`proc_moderate_data.py`**
Processes moderation feedback (thumbs up/down/flag votes from the web UI) to build a curated training dataset. Pulls `model_mods.jsonl` from the web server, separates upvotes, downvotes, and flags, and queries the DB for the actual question text corresponding to each rated output. Used to filter training data for retraining cycles — upvoted outputs become positive examples.

### Analysis (`analysis/`)

**`analysis/vote_ratio_analysis.py`**
Computes per-model vote balance scores — how close each model's questions come to a 50/50 A/B split (a proxy for question quality: polarizing questions are considered better). Produces two metrics: raw average balance and quality-weighted balance (upvotes +3, downvotes −1, flags −3). Outputs a bar chart PNG.

**`analysis/vote_ratio_heatmap.py`**
Extends the balance score analysis across question categories. Uses sentence embeddings (Qwen3-Embedding) to classify each question into a main and secondary category, then produces heatmap PNGs showing balance score per (model × category). Helps identify which models perform well or poorly on specific topic areas.

**`analysis/data_analysis_code.py`**
Core data analysis utilities used by the Jupyter notebooks. Handles question categorization via keyword-based matching and semantic embedding clustering (HDBSCAN), generates voting heatmaps by day-of-week and hour, and provides helpers for loading and querying the SQLite database for analysis.

### Fine-tuning (`finetuning-notebooks/`)

**`finetuning-notebooks/train_all_models.py`**
Batch training script that fine-tunes multiple LLMs sequentially on the WYR question dataset using LoRA/QLoRA via the `trl` SFTTrainer. Handles dataset formatting, model loading, and saving merged weights for each model.

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `transformers`, `peft`, `trl`, `flask`, `discord.py`, `sentence-transformers`, `matplotlib`, `seaborn`, `pandas`.

### Configuration

Edit `sync.py` to set `WEB_SERVER_USER`, `WEB_SERVER_HOST`, and `WEB_SERVER_DIR` to match your web server. SSH key auth (passwordless) is required between the two machines.

Edit `generate_job.py` to point the `MODELS` dict at your fine-tuned merged model folders (or HuggingFace model IDs for base weights).

### Running

**GPU server** (hourly cron):
```bash
python generate_job.py
```

**Web server**:
```bash
python wyr_app_v2.py
```

**Discord bot**:
```bash
export DISCORD_BOT_TOKEN=your-token
export DISCORD_CHANNEL_ID=your-channel-id
python discord_bot.py
```

See `finetuning-notebooks/cronjob.txt` for the cron setup used in production.

## Database

`wyr_votes.db` is a SQLite database with tables for questions, per-model outputs, votes, and moderation feedback. The web server does not hold a copy — it uses append-only JSONL files (`votes_delta.jsonl`, `model_mods.jsonl`) that the GPU server pulls and merges on each generation cycle.
