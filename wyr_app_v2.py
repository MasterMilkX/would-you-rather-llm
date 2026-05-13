"""
wyr_app.py — Would You Rather LLM Frontend + Vote API

Reads current_question.json (written by generate_job.py) to serve the
active question. Votes are recorded in wyr_votes.db.

Usage:
    pip install flask
    python wyr_app.py
"""

import sqlite3
import json
import os
from flask import Flask, request, jsonify, g
from datetime import datetime

app   = Flask(__name__)
DB_PATH   = "wyr_votes.db"
JSON_PATH = "current_question.json"

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop("db", None)
    if db:
        db.close()

def init_db():
    with sqlite3.connect(DB_PATH) as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS questions (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                generated_at TEXT NOT NULL,
                prompt       TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS model_outputs (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                question_id  INTEGER NOT NULL REFERENCES questions(id),
                model_name   TEXT NOT NULL,
                slot         TEXT NOT NULL,
                option_a     TEXT NOT NULL,
                option_b     TEXT NOT NULL
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

init_db()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_current_question():
    """Read the JSON written by generate_job.py."""
    if not os.path.exists(JSON_PATH):
        return None
    with open(JSON_PATH) as f:
        return json.load(f)


def get_vote_totals(db, question_id):
    rows = db.execute(
        "SELECT model_name, votes_a, votes_b FROM votes WHERE question_id=?",
        (question_id,)
    ).fetchall()
    return {r["model_name"]: {"votes_a": r["votes_a"], "votes_b": r["votes_b"]} for r in rows}


def get_slot_to_model(db, question_id):
    rows = db.execute(
        "SELECT slot, model_name FROM model_outputs WHERE question_id=?",
        (question_id,)
    ).fetchall()
    return {r["slot"]: r["model_name"] for r in rows}

# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    q = load_current_question()
    return jsonify({
        "status": "ok",
        "current_question_id": q["question_id"] if q else None,
        "generated_at": q["generated_at"] if q else None,
    })


@app.get("/api/question")
def api_question():
    """Return the current question from the JSON file (no model names)."""
    q = load_current_question()
    if not q:
        return jsonify({"error": "No question available yet. Run generate_job.py first."}), 404
    return jsonify(q)


@app.post("/api/vote")
def api_vote():
    """
    Record a vote.

    Body:
        question_id   : int
        slot          : "A" | "B" | "C" | "D"   (the card the user voted on)
        chosen_option : "A" | "B"                (which WYR option they picked)
    """
    body = request.get_json(silent=True) or {}
    question_id    = body.get("question_id")
    slot           = body.get("slot", "").upper()
    chosen_option  = body.get("chosen_option", "").upper()

    if not question_id or slot not in ("A","B","C","D") or chosen_option not in ("A","B"):
        return jsonify({"error": "Required: question_id, slot (A-D), chosen_option (A/B)"}), 400

    db = get_db()

    # Resolve slot -> model_name
    slot_map = get_slot_to_model(db, question_id)
    model_name = slot_map.get(slot)
    if not model_name:
        return jsonify({"error": f"Slot {slot} not found for question {question_id}"}), 404

    # Increment the right vote column
    col = "votes_a" if chosen_option == "A" else "votes_b"
    db.execute(
        f"UPDATE votes SET {col} = {col} + 1 WHERE question_id=? AND model_name=?",
        (question_id, model_name)
    )
    db.commit()

    # Return updated totals + reveal model names
    totals = get_vote_totals(db, question_id)
    return jsonify({
        "question_id": question_id,
        "voted_slot": slot,
        "voted_option": chosen_option,
        "totals": totals,           # keyed by actual model name
        "slot_reveal": slot_map,    # reveals which slot = which model
    })


@app.get("/api/results/<int:question_id>")
def api_results(question_id):
    """Get full vote results for any question (reveals model names)."""
    db = get_db()
    totals   = get_vote_totals(db, question_id)
    slot_map = get_slot_to_model(db, question_id)
    if not totals:
        return jsonify({"error": "Question not found"}), 404
    return jsonify({
        "question_id": question_id,
        "totals": totals,
        "slot_reveal": slot_map,
    })

# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return HTML

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Would You Rather LLM</title>
<link href="https://fonts.googleapis.com/css2?family=VT323&family=Share+Tech+Mono&display=swap" rel="stylesheet">
<style>
  :root {
    --teal:      #a6fff8;
    --teal-dark: #5ce8df;
    --teal-dim:  #2a6e6a;
    --bg:        #0a1a1a;
    --win-bg:    #0d2b2b;
    --text:      #c8fffc;
    --text-dim:  #5ce8df;
    --pink:      #ff6ef7;
    --yellow:    #fffb6e;
    --scan: repeating-linear-gradient(
      0deg, transparent, transparent 2px,
      rgba(0,0,0,0.15) 2px, rgba(0,0,0,0.15) 4px
    );
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    background-image:
      radial-gradient(ellipse at 20% 20%, #0d3535 0%, transparent 50%),
      radial-gradient(ellipse at 80% 80%, #0a2828 0%, transparent 50%);
    font-family: 'Share Tech Mono', monospace;
    color: var(--text); min-height: 100vh; overflow-x: hidden; cursor: crosshair;
  }
  body::after {
    content: ''; position: fixed; inset: 0;
    background: var(--scan); pointer-events: none; z-index: 9999; opacity: 0.4;
  }
  .titlebar {
    display: flex; align-items: center; justify-content: space-between;
    background: linear-gradient(90deg, var(--teal-dim), var(--teal-dark), var(--teal-dim));
    padding: 6px 12px; border-bottom: 2px solid var(--teal);
    position: sticky; top: 0; z-index: 100;
  }
  .titlebar-btns { display: flex; gap: 6px; }
  .titlebar-btn {
    width:14px;height:14px;border:1px solid var(--bg);background:var(--teal);
    font-size:9px;display:flex;align-items:center;justify-content:center;
    color:var(--bg);cursor:pointer;font-family:'VT323',monospace;font-weight:bold;
  }
  .titlebar-title { font-family:'VT323',monospace;font-size:1.1rem;color:var(--bg);letter-spacing:2px; }
  header { text-align:center; padding:2.5rem 1rem 1rem; }
  .logo {
    font-family:'VT323',monospace;font-size:clamp(2.8rem,7vw,5rem);color:var(--teal);
    text-shadow:0 0 10px var(--teal),0 0 30px var(--teal-dark),2px 2px 0 var(--teal-dim);
    letter-spacing:4px;animation:glitch 4s infinite;
  }
  @keyframes glitch {
    0%,94%,100%{text-shadow:0 0 10px var(--teal),0 0 30px var(--teal-dark),2px 2px 0 var(--teal-dim);transform:none;}
    95%{transform:translate(-2px,0);text-shadow:2px 0 var(--pink),-2px 0 var(--teal);}
    97%{transform:translate(2px,0);text-shadow:-2px 0 var(--yellow),2px 0 var(--teal);}
    99%{transform:translate(0,1px);}
  }
  .subtitle{font-size:0.8rem;color:var(--text-dim);letter-spacing:3px;margin-top:6px;}
  .countdown-bar{text-align:center;padding:0.8rem;font-size:0.78rem;letter-spacing:2px;color:var(--text-dim);}
  .countdown-bar span{color:var(--teal);font-family:'VT323',monospace;font-size:1.1rem;}
  .gen-time{color:var(--text-dim);font-size:0.7rem;letter-spacing:2px;text-align:center;margin-bottom:1rem;}
  .status{text-align:center;font-size:0.78rem;color:var(--text-dim);letter-spacing:2px;min-height:1.2em;margin-bottom:1rem;padding:0 1rem;}
  .status.loading{color:var(--teal);animation:blink 0.8s infinite;}
  .status.error{color:var(--pink);}
  @keyframes blink{50%{opacity:0.3;}}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:20px;max-width:1100px;margin:0 auto;padding:0 20px 60px;}
  .win{
    border:2px solid var(--teal);background:var(--win-bg);
    box-shadow:0 0 0 1px var(--teal-dim),4px 4px 0 var(--teal-dim),0 0 20px rgba(166,255,248,0.08);
    opacity:0;transform:translateY(16px);transition:opacity 0.4s,transform 0.4s;
  }
  .win.visible{opacity:1;transform:none;}
  .win:nth-child(1){transition-delay:0.05s;}.win:nth-child(2){transition-delay:0.15s;}
  .win:nth-child(3){transition-delay:0.25s;}.win:nth-child(4){transition-delay:0.35s;}
  .win-title{
    background:linear-gradient(90deg,var(--teal-dim),var(--teal-dark));
    padding:5px 10px;display:flex;align-items:center;justify-content:space-between;
    border-bottom:1px solid var(--teal);
  }
  .win-title span{font-family:'VT323',monospace;font-size:1rem;color:var(--bg);letter-spacing:2px;}
  .win-controls{display:flex;gap:4px;}
  .win-ctrl{width:12px;height:12px;background:var(--bg);border:1px solid var(--bg);font-size:8px;display:flex;align-items:center;justify-content:center;color:var(--teal);font-family:'VT323';font-weight:bold;}
  .win-body{padding:18px;}
  .wyr-prompt{font-family:'VT323',monospace;font-size:1.3rem;color:var(--text-dim);letter-spacing:2px;margin-bottom:14px;text-align:center;}
  .option-btn{
    width:100%;padding:14px 12px;margin-bottom:10px;background:transparent;
    border:1px solid var(--teal-dim);color:var(--text);font-family:'Share Tech Mono',monospace;
    font-size:0.85rem;text-align:left;cursor:pointer;transition:all 0.15s;line-height:1.4;
  }
  .option-btn::before{content:'▶ ';color:var(--teal-dim);font-size:0.7rem;}
  .option-btn:hover:not(:disabled){border-color:var(--teal);background:rgba(166,255,248,0.07);color:var(--teal);}
  .option-btn:hover:not(:disabled)::before{color:var(--teal);}
  .option-btn.selected-a{border-color:var(--teal);background:rgba(166,255,248,0.12);color:var(--teal);}
  .option-btn.selected-b{border-color:var(--pink);background:rgba(255,110,247,0.08);color:var(--pink);}
  .option-btn:disabled{cursor:default;}
  .option-btn.placeholder{color:var(--teal-dim);font-style:italic;}
  .results{margin-top:14px;display:none;border-top:1px dashed var(--teal-dim);padding-top:12px;}
  .results.show{display:block;}
  .result-row{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;font-size:0.75rem;color:var(--text-dim);letter-spacing:1px;}
  .result-label{color:var(--teal);}
  .bar-wrap{display:flex;align-items:center;gap:6px;}
  .bar{height:8px;background:var(--teal-dim);display:inline-block;transition:width 0.5s;min-width:2px;}
  .bar.bar-b{background:var(--pink);}
  .reveal-tag{font-size:0.7rem;color:var(--pink);margin-top:10px;letter-spacing:1px;text-align:center;}
  .voted-badge{font-size:0.7rem;color:var(--yellow);letter-spacing:1px;text-align:center;margin-bottom:6px;}
  footer{text-align:center;padding:20px;font-size:0.7rem;color:var(--teal-dim);letter-spacing:2px;border-top:1px solid var(--teal-dim);}
  ::-webkit-scrollbar{width:6px;}::-webkit-scrollbar-track{background:var(--bg);}::-webkit-scrollbar-thumb{background:var(--teal-dim);}
  @media(max-width:640px){.grid{grid-template-columns:1fr;}.logo{font-size:2.2rem;}}
</style>
</head>
<body>

<div class="titlebar">
  <div class="titlebar-btns">
    <div class="titlebar-btn">■</div><div class="titlebar-btn">▼</div><div class="titlebar-btn">✕</div>
  </div>
  <div class="titlebar-title">WYR_LLM.EXE — RUNNING</div>
  <div style="font-size:0.65rem;color:var(--bg);letter-spacing:1px;" id="clock"></div>
</div>

<header>
  <div class="logo">WOULD YOU RATHER LLM</div>
  <div class="subtitle">:: FOUR MODELS :: ONE QUESTION :: YOU DECIDE ::</div>
</header>

<div class="countdown-bar">NEXT QUESTION IN: <span id="countdown">--:--:--</span></div>
<div class="gen-time" id="genTime"></div>
<div class="status" id="status">LOADING QUESTION_</div>

<div class="grid" id="grid">
  <div class="win" id="card-A"></div>
  <div class="win" id="card-B"></div>
  <div class="win" id="card-C"></div>
  <div class="win" id="card-D"></div>
</div>

<footer>[ WOULD YOU RATHER LLM // Y2K EDITION // ALL OUTPUTS AI GENERATED ]</footer>

<script>
const SLOTS = ['A','B','C','D'];
let state = { questionId: null, voted: false };

function tick() {
  document.getElementById('clock').textContent =
    new Date().toLocaleTimeString('en-US',{hour12:false});
}
tick(); setInterval(tick, 1000);

function updateCountdown() {
  const now = new Date();
  const next = new Date(now);
  next.setHours(next.getHours()+1,0,0,0);
  const diff = Math.max(0, next-now);
  const h = String(Math.floor(diff/3600000)).padStart(2,'0');
  const m = String(Math.floor((diff%3600000)/60000)).padStart(2,'0');
  const s = String(Math.floor((diff%60000)/1000)).padStart(2,'0');
  document.getElementById('countdown').textContent = `${h}:${m}:${s}`;
  if (diff < 1000 && state.questionId) setTimeout(()=>loadQuestion(), 5000);
}
setInterval(updateCountdown,1000); updateCountdown();

function buildCard(slot) {
  const el = document.getElementById(`card-${slot}`);
  el.classList.remove('visible');
  el.innerHTML = `
    <div class="win-title">
      <span>MODEL_${slot}.EXE</span>
      <div class="win-controls"><div class="win-ctrl">_</div><div class="win-ctrl">□</div><div class="win-ctrl">✕</div></div>
    </div>
    <div class="win-body">
      <div class="wyr-prompt">Would you rather...</div>
      <button class="option-btn placeholder" disabled>[ LOADING... ]</button>
      <button class="option-btn placeholder" disabled>[ LOADING... ]</button>
    </div>`;
  setTimeout(()=>el.classList.add('visible'),10);
}

function populateCard(slot, optA, optB) {
  document.getElementById(`card-${slot}`).querySelector('.win-body').innerHTML = `
    <div class="wyr-prompt">Would you rather...</div>
    <button class="option-btn" id="obtn-${slot}-A" onclick="pickOption('${slot}','A')">${esc(optA)}</button>
    <button class="option-btn" id="obtn-${slot}-B" onclick="pickOption('${slot}','B')">${esc(optB)}</button>
    <div class="results" id="results-${slot}"></div>`;
}

function esc(s){ return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

async function loadQuestion() {
  setStatus('loading','LOADING QUESTION_');
  SLOTS.forEach(buildCard);
  state.voted = false;
  try {
    const resp = await fetch('/api/question');
    if (!resp.ok){ const e=await resp.json(); setStatus('error',e.error||'FAILED TO LOAD'); return; }
    const data = await resp.json();
    state.questionId = data.question_id;
    const gt = new Date(data.generated_at+'Z');
    document.getElementById('genTime').textContent =
      `GENERATED AT: ${gt.toLocaleTimeString('en-US',{hour12:false})}`;
    SLOTS.forEach(slot=>{ const s=data.slots[slot]; if(s) populateCard(slot,s.option_a,s.option_b); });
    setStatus('','CLICK AN OPTION ON ANY CARD TO CAST YOUR VOTE_');
  } catch(e){ setStatus('error','CONNECTION ERROR: '+e.message); }
}

function pickOption(slot, option) {
  if (state.voted) return;
  const btnA = document.getElementById(`obtn-${slot}-A`);
  const btnB = document.getElementById(`obtn-${slot}-B`);
  if (!btnA) return;
  btnA.disabled=true; btnB.disabled=true;
  btnA.classList.toggle('selected-a', option==='A');
  btnB.classList.toggle('selected-b', option==='B');
  state.voted = true;
  submitVote(slot, option);
}

async function submitVote(slot, chosenOption) {
  setStatus('loading','RECORDING VOTE_');
  try {
    const resp = await fetch('/api/vote',{
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({question_id:state.questionId, slot, chosen_option:chosenOption})
    });
    const data = await resp.json();
    if (!resp.ok){ setStatus('error',data.error); return; }
    showResults(slot, data.totals, data.slot_reveal, chosenOption);
    setStatus('','VOTE RECORDED // MODELS REVEALED BELOW_');
  } catch(e){ setStatus('error','VOTE ERROR: '+e.message); }
}

function showResults(votedSlot, totals, slotReveal, chosenOption) {
  SLOTS.forEach(slot=>{
    const modelName = slotReveal[slot];
    if (!modelName||!totals[modelName]) return;
    const t = totals[modelName];
    const total = (t.votes_a+t.votes_b)||1;
    const pctA = Math.round(t.votes_a/total*100);
    const pctB = 100-pctA;
    const el = document.getElementById(`results-${slot}`);
    if (!el) return;
    el.innerHTML = `
      ${slot===votedSlot?`<div class="voted-badge">★ YOU VOTED ${chosenOption==='A'?'OPTION A':'OPTION B'} HERE ★</div>`:''}
      <div class="result-row">
        <span class="result-label">OPT A</span>
        <div class="bar-wrap"><span class="bar" style="width:${Math.round(pctA*.9)}px"></span><span>${t.votes_a}v / ${pctA}%</span></div>
      </div>
      <div class="result-row">
        <span class="result-label">OPT B</span>
        <div class="bar-wrap"><span class="bar bar-b" style="width:${Math.round(pctB*.9)}px"></span><span>${t.votes_b}v / ${pctB}%</span></div>
      </div>
      <div class="reveal-tag">★ MODEL: ${modelName.toUpperCase()} ★</div>`;
    el.classList.add('show');
  });
}

function setStatus(cls,msg){
  const el=document.getElementById('status');
  el.className='status'+(cls?' '+cls:'');
  el.textContent=msg;
}

loadQuestion();
</script>
</body>
</html>"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6969, debug=False)
