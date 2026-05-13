"""
Would You Rather — Dual Model Generation API Server
Loads a finetuned TinyLlama and a Qwen2.5 model and serves WYR question
generation, including a /compare endpoint that runs both side-by-side.

Usage:
    python wyr_server.py

Endpoints:
    GET  /health                — health check, reports both models + their devices
    GET  /generate              — generate one WYR question from TinyLlama (default)
    POST /generate              — generate from a specific model with custom params
    GET  /compare               — generate one question from each model simultaneously
    POST /compare               — compare with custom params

POST /generate body (all optional):
    {
        "model":       "tinyllama",          # "tinyllama" | "qwen" (default: tinyllama)
        "prompt":      "Would you rather",   # primer text
        "num":         3,                    # how many questions (1-10)
        "temperature": 0.9,
        "top_p":       0.95,
        "max_tokens":  80
    }

POST /compare body (all optional):
    {
        "prompt":      "Would you rather",
        "num":         1,
        "temperature": 0.9,
        "top_p":       0.95,
        "max_tokens":  80
    }
"""

import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Point these at your finetuned merged model folders.
# If you only finetuned TinyLlama so far, point QWEN_MODEL_DIR at the base
# Qwen model and it will still work — it just won't be domain-finetuned yet.
TINYLLAMA_MODEL_DIR = "./llama-wyr-merged"
QWEN_MODEL_DIR      = "./qwen-wyr-merged"   # or e.g. "Qwen/Qwen2.5-0.5B" for base

PORT = 6969

# Device strategy:
#   - If you have >=8 GB VRAM, both models fit on cuda:0.
#   - If tight on VRAM, set QWEN_DEVICE = "cpu" — Qwen2.5-0.5B is fast enough on CPU.
#   - "auto" lets HuggingFace decide per-model (recommended if unsure).
TINYLLAMA_DEVICE = "auto"
QWEN_DEVICE      = "auto"

# ---------------------------------------------------------------------------
# Load both models at startup
# ---------------------------------------------------------------------------

def load_model(model_dir, device):
    print(f"  Loading {model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"torch_dtype": "auto"}
    if device == "auto":
        kwargs["device_map"] = "auto"
    else:
        kwargs["device_map"] = {"": device}

    model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
    model.eval()
    print(f"  Loaded {model_dir} -> {next(model.parameters()).device}")
    return tokenizer, model


# print("Loading models...")
# tinyllama_tokenizer, tinyllama_model = load_model(TINYLLAMA_MODEL_DIR, TINYLLAMA_DEVICE)
# qwen_tokenizer,      qwen_model      = load_model(QWEN_MODEL_DIR,      QWEN_DEVICE)
# print("Both models ready.\n")




MODELS = {
    "tinyllama": (tinyllama_tokenizer, tinyllama_model),
    "qwen":      (qwen_tokenizer,      qwen_model),
}

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

DEFAULT_PROMPT      = "Would you rather"
DEFAULT_NUM         = 1
DEFAULT_TEMPERATURE = 0.9
DEFAULT_TOP_P       = 0.95
DEFAULT_MAX_TOKENS  = 80
REPETITION_PENALTY  = 1.2


def generate(tokenizer, model, prompt, num, temperature, top_p, max_tokens):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=num,
        )
    return [tokenizer.decode(o, skip_special_tokens=True).strip() for o in outputs]


def parse_params(body):
    """Parse and validate shared generation params from a request body dict."""
    num         = int(body.get("num", DEFAULT_NUM))
    temperature = float(body.get("temperature", DEFAULT_TEMPERATURE))
    top_p       = float(body.get("top_p", DEFAULT_TOP_P))
    max_tokens  = int(body.get("max_tokens", DEFAULT_MAX_TOKENS))
    prompt      = str(body.get("prompt", DEFAULT_PROMPT))

    if not (1 <= num <= 10):
        raise ValueError("'num' must be between 1 and 10")

    return prompt, num, temperature, top_p, max_tokens


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "models": {
            "tinyllama": {
                "dir":    TINYLLAMA_MODEL_DIR,
                "device": str(next(tinyllama_model.parameters()).device),
            },
            "qwen": {
                "dir":    QWEN_MODEL_DIR,
                "device": str(next(qwen_model.parameters()).device),
            },
        }
    })


@app.get("/generate")
def generate_get():
    questions = generate(
        tinyllama_tokenizer, tinyllama_model,
        DEFAULT_PROMPT, DEFAULT_NUM,
        DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_MAX_TOKENS,
    )
    return jsonify({"model": "tinyllama", "questions": questions})


@app.post("/generate")
def generate_post():
    body = request.get_json(silent=True) or {}

    model_name = body.get("model", "tinyllama").lower()
    if model_name not in MODELS:
        return jsonify({"error": f"Unknown model '{model_name}'. Choose: {list(MODELS)}"}), 400

    try:
        prompt, num, temperature, top_p, max_tokens = parse_params(body)
    except (ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400

    tokenizer, model = MODELS[model_name]
    questions = generate(tokenizer, model, prompt, num, temperature, top_p, max_tokens)

    return jsonify({
        "model":     model_name,
        "prompt":    prompt,
        "questions": questions,
        "params":    {"num": num, "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens},
    })


@app.get("/compare")
def compare_get():
    shared = (DEFAULT_PROMPT, DEFAULT_NUM, DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_MAX_TOKENS)
    return jsonify({
        "prompt":    DEFAULT_PROMPT,
        "tinyllama": generate(tinyllama_tokenizer, tinyllama_model, *shared),
        "qwen":      generate(qwen_tokenizer,      qwen_model,      *shared),
    })


@app.post("/compare")
def compare_post():
    body = request.get_json(silent=True) or {}

    try:
        prompt, num, temperature, top_p, max_tokens = parse_params(body)
    except (ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400

    args = (prompt, num, temperature, top_p, max_tokens)
    return jsonify({
        "prompt":    prompt,
        "params":    {"num": num, "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens},
        "tinyllama": generate(tinyllama_tokenizer, tinyllama_model, *args),
        "qwen":      generate(qwen_tokenizer,      qwen_model,      *args),
    })


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
