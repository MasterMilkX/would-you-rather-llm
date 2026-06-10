"""
discord_bot.py — WYR Discord Bot

Posts one "Would You Rather" question per hour from the recycled (unvoted)
question archive. Each post picks a single model's output (one slot) from one
archived question. Users vote A or B via buttons. Votes are saved directly to
wyr_votes.db so they appear in the existing vote tallies.

Setup:
    pip install "discord.py>=2.0"
    export DISCORD_BOT_TOKEN=your-token
    export DISCORD_CHANNEL_ID=your-channel-id
    python discord_bot.py

The bot tracks which question/slot pairs it has already posted in
discord_state.json so it never repeats. When all unvoted questions are
exhausted it falls back to random archived questions.
"""

import discord
from discord.ext import tasks
import sqlite3
import json
import random
import os
import sys
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_PATH     = "wyr_votes.db"
ARCHIVE_DIR = "questions_archive"
STATE_PATH  = "discord_state.json"

TOKEN      = os.environ.get("DISCORD_BOT_TOKEN", "")
CHANNEL_ID = int(os.environ.get("DISCORD_CHANNEL_ID", "0"))

if not TOKEN:
    sys.exit("ERROR: Set DISCORD_BOT_TOKEN env var")
if not CHANNEL_ID:
    sys.exit("ERROR: Set DISCORD_CHANNEL_ID env var")

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _db():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    return db


def get_unvoted_question_ids() -> list[int]:
    """Return IDs of archived questions with zero total votes in the DB."""
    archived_ids = _archived_ids()
    if not archived_ids:
        return []

    placeholders = ",".join("?" * len(archived_ids))
    db = _db()
    rows = db.execute(f"""
        SELECT q.id
        FROM questions q
        LEFT JOIN votes v ON v.question_id = q.id
        WHERE q.id IN ({placeholders})
        GROUP BY q.id
        HAVING COALESCE(SUM(v.votes_a + v.votes_b), 0) = 0
        ORDER BY q.id
    """, list(archived_ids)).fetchall()
    db.close()
    return [r["id"] for r in rows]


def record_vote(question_id: int, model_name: str, chosen_option: str) -> tuple[int, int]:
    """
    Increment votes_a or votes_b for question_id + model_name.
    Inserts the row if it doesn't exist yet (handles questions that were
    never put through generate_job's initial insert).
    Returns (votes_a, votes_b) after the update.
    """
    col = "votes_a" if chosen_option == "A" else "votes_b"
    db = _db()
    db.execute(
        "INSERT OR IGNORE INTO votes (question_id, model_name, votes_a, votes_b) VALUES (?,?,0,0)",
        (question_id, model_name)
    )
    db.execute(
        f"UPDATE votes SET {col} = {col} + 1 WHERE question_id = ? AND model_name = ?",
        (question_id, model_name)
    )
    db.commit()
    row = db.execute(
        "SELECT votes_a, votes_b FROM votes WHERE question_id = ? AND model_name = ?",
        (question_id, model_name)
    ).fetchone()
    db.close()
    return (row["votes_a"], row["votes_b"]) if row else (0, 0)


def get_vote_totals(question_id: int, model_name: str) -> tuple[int, int]:
    db = _db()
    row = db.execute(
        "SELECT votes_a, votes_b FROM votes WHERE question_id = ? AND model_name = ?",
        (question_id, model_name)
    ).fetchone()
    db.close()
    return (row["votes_a"], row["votes_b"]) if row else (0, 0)


# ---------------------------------------------------------------------------
# Archive helpers
# ---------------------------------------------------------------------------

def _archived_ids() -> list[int]:
    if not os.path.isdir(ARCHIVE_DIR):
        return []
    ids = []
    for fname in os.listdir(ARCHIVE_DIR):
        if fname.endswith(".json"):
            try:
                ids.append(int(fname[:-5]))
            except ValueError:
                pass
    return ids


def _load_archive(qid: int) -> dict | None:
    path = os.path.join(ARCHIVE_DIR, f"{qid}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def load_state() -> dict:
    if not os.path.exists(STATE_PATH):
        return {"posted_slots": [], "active": None}
    with open(STATE_PATH) as f:
        return json.load(f)


def save_state(state: dict):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


# ---------------------------------------------------------------------------
# Question picker
# ---------------------------------------------------------------------------

def pick_next_question() -> dict | None:
    """
    Pick an unvoted (question_id, slot) pair that hasn't been posted to Discord.
    Falls back to any archived question if the unvoted pool is empty.
    Returns a dict with: question_id, slot, model_name, option_a, option_b.
    """
    state = load_state()
    posted = {(p[0], p[1]) for p in state.get("posted_slots", [])}

    unvoted_ids = get_unvoted_question_ids()
    random.shuffle(unvoted_ids)

    for qid in unvoted_ids:
        q = _load_archive(qid)
        if not q:
            continue
        slots         = q.get("slots", {})
        slot_model    = q.get("slot_model_map", {})
        available     = [s for s in slots if (qid, s) not in posted]
        if not available:
            continue
        slot       = random.choice(available)
        slot_data  = slots[slot]
        model_name = slot_model.get(slot, "unknown")
        return {
            "question_id": qid,
            "slot":        slot,
            "model_name":  model_name,
            "option_a":    slot_data["option_a"],
            "option_b":    slot_data["option_b"],
        }

    # Fallback: all unvoted exhausted — pick any archived question/slot
    all_ids = _archived_ids()
    if not all_ids:
        return None
    qid = random.choice(all_ids)
    q   = _load_archive(qid)
    if not q:
        return None
    slots      = q.get("slots", {})
    slot_model = q.get("slot_model_map", {})
    if not slots:
        return None
    slot       = random.choice(list(slots.keys()))
    slot_data  = slots[slot]
    model_name = slot_model.get(slot, "unknown")
    return {
        "question_id": qid,
        "slot":        slot,
        "model_name":  model_name,
        "option_a":    slot_data["option_a"],
        "option_b":    slot_data["option_b"],
    }


# ---------------------------------------------------------------------------
# Discord UI
# ---------------------------------------------------------------------------

def _bar(n: int, total: int, width: int = 12) -> str:
    if total == 0:
        return "░" * width
    filled = round(n / total * width)
    return "█" * filled + "░" * (width - filled)


def build_embed(option_a: str, option_b: str, question_id: int,
                votes_a: int = 0, votes_b: int = 0) -> discord.Embed:
    total = votes_a + votes_b
    pct_a = int(votes_a / total * 100) if total else 0
    pct_b = 100 - pct_a if total else 0

    embed = discord.Embed(title="🤔  Would You Rather...", color=0x00b0f4)
    embed.add_field(
        name="🔵  Option A",
        value=f"{option_a}\n{_bar(votes_a, total)} {pct_a}% ({votes_a})",
        inline=False,
    )
    embed.add_field(
        name="🟡  Option B",
        value=f"{option_b}\n{_bar(votes_b, total)} {pct_b}% ({votes_b})",
        inline=False,
    )
    footer = f"#{question_id}"
    if total:
        footer += f" · {total} vote{'s' if total != 1 else ''}"
    embed.set_footer(text=footer)
    return embed


class VoteView(discord.ui.View):
    def __init__(self, question_id: int, slot: str, model_name: str,
                 option_a: str, option_b: str):
        super().__init__(timeout=None)
        self.question_id = question_id
        self.slot        = slot
        self.model_name  = model_name
        self.option_a    = option_a
        self.option_b    = option_b

        self.add_item(_VoteButton("A", question_id, slot, discord.ButtonStyle.primary))
        self.add_item(_VoteButton("B", question_id, slot, discord.ButtonStyle.secondary))


class _VoteButton(discord.ui.Button):
    def __init__(self, choice: str, question_id: int, slot: str,
                 style: discord.ButtonStyle):
        prefix = "🔵 " if choice == "A" else "🟡 "
        super().__init__(
            style=style,
            label=f"{prefix}Option {choice}",
            custom_id=f"wyr_{question_id}_{slot}_{choice}",
        )
        self.choice      = choice
        self.question_id = question_id
        self.slot        = slot

    async def callback(self, interaction: discord.Interaction):
        view: VoteView = self.view  # type: ignore[assignment]
        votes_a, votes_b = record_vote(self.question_id, view.model_name, self.choice)

        embed = build_embed(view.option_a, view.option_b,
                            self.question_id, votes_a, votes_b)
        total = votes_a + votes_b
        embed.set_footer(
            text=f"#{self.question_id} · "
                 f"You voted Option {self.choice} · "
                 f"{total} vote{'s' if total != 1 else ''}"
        )
        await interaction.response.edit_message(embed=embed, view=view)


# ---------------------------------------------------------------------------
# Bot
# ---------------------------------------------------------------------------

intents = discord.Intents.default()
bot     = discord.Client(intents=intents)


async def post_question():
    channel = bot.get_channel(CHANNEL_ID)
    if not channel:
        print(f"[bot] Channel {CHANNEL_ID} not found — check DISCORD_CHANNEL_ID")
        return

    q = pick_next_question()
    if not q:
        print("[bot] No questions available in archive")
        return

    votes_a, votes_b = get_vote_totals(q["question_id"], q["model_name"])
    embed = build_embed(q["option_a"], q["option_b"], q["question_id"], votes_a, votes_b)
    view  = VoteView(q["question_id"], q["slot"], q["model_name"],
                     q["option_a"], q["option_b"])
    msg   = await channel.send(embed=embed, view=view)

    state = load_state()
    posted = state.get("posted_slots", [])
    posted.append([q["question_id"], q["slot"]])
    state["posted_slots"] = posted
    state["active"] = {
        "message_id":  msg.id,
        "channel_id":  CHANNEL_ID,
        **q,
    }
    save_state(state)

    ts = datetime.now(timezone.utc).strftime("%H:%M UTC")
    print(f"[bot] {ts} — Posted q#{q['question_id']} slot {q['slot']} "
          f"({q['model_name']}): {q['option_a']!r} vs {q['option_b']!r}")


@tasks.loop(hours=1)
async def hourly_question():
    await post_question()


@bot.event
async def on_ready():
    print(f"[bot] Logged in as {bot.user} (discord.py {discord.__version__})")

    # Re-register persistent views so buttons on old messages keep working
    state  = load_state()
    active = state.get("active")
    if active:
        view = VoteView(
            active["question_id"], active["slot"], active["model_name"],
            active["option_a"], active["option_b"],
        )
        bot.add_view(view)
        print(f"[bot] Re-registered persistent view for q#{active['question_id']}")

    if not hourly_question.is_running():
        hourly_question.start()
    print("[bot] Hourly question loop started")


bot.run(TOKEN)
