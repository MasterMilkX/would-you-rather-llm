#!/usr/bin/env python3
"""
WYR Judge Model
A supervised classifier that scores LLM-generated "Would You Rather" questions
from 0 (likely downvote/flag) to 1 (likely upvote).

Training data:
  Positive (label=1): formatted_polls.json + upvote_questions.json
  Negative (label=0): bad_questions.json (downvotes + flagged)
"""

import json
import re
import pickle
import argparse
import numpy as np
from pathlib import Path
from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin

# ─── Paths ───────────────────────────────────────────────────────────────────

BASE = Path(__file__).parent
FORMATTED_POLLS = BASE / "finetuning-notebooks" / "formatted_polls.json"
UPVOTE_QUESTIONS = BASE / "analysis" / "upvote_questions.json"
BAD_QUESTIONS = BASE / "analysis" / "bad_questions.json"
MODEL_PATH = BASE / "judge_model.pkl"

# ─── Data loading ─────────────────────────────────────────────────────────────

def load_datasets():
    with open(FORMATTED_POLLS) as f:
        polls = json.load(f)
    with open(UPVOTE_QUESTIONS) as f:
        upvotes = json.load(f)
    with open(BAD_QUESTIONS) as f:
        bad = json.load(f)

    positives = []
    for item in polls:
        positives.append({
            "optionA": item.get("optionA", ""),
            "optionB": item.get("optionB", ""),
            "title": item.get("title", ""),
        })
    for item in upvotes:
        positives.append({
            "optionA": item.get("optionA", ""),
            "optionB": item.get("optionB", ""),
            "title": item.get("title", ""),
        })

    negatives = []
    for item in bad.get("downvotes", []) + bad.get("flagged", []):
        negatives.append({
            "optionA": item.get("optionA", ""),
            "optionB": item.get("optionB", ""),
            "title": item.get("title", ""),
        })

    print(f"Loaded {len(positives)} positive examples, {len(negatives)} negative examples")
    return positives, negatives


# ─── Feature engineering ──────────────────────────────────────────────────────

def make_combined_text(records):
    """Concatenate optionA + optionB for TF-IDF."""
    return [
        (r["optionA"] + " [SEP] " + r["optionB"]).lower()
        for r in records
    ]


def _is_placeholder(text: str) -> bool:
    return text.strip().lower() in ("option a", "option b", "a", "b", "")


def _option_similarity(a: str, b: str) -> float:
    """Jaccard word overlap — high overlap means options aren't distinct."""
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return 1.0
    return len(wa & wb) / len(wa | wb)


def extract_handcrafted(records):
    """
    Returns a float matrix (n, 15).
      0  is_placeholder_A
      1  is_placeholder_B
      2  either_placeholder
      3  len_A (log)
      4  len_B (log)
      5  word_count_A
      6  word_count_B
      7  word_count_ratio (min/max → 0..1)
      8  option_similarity (Jaccard)
      9  has_wyr_prefix_in_options
      10 avg_word_length_A
      11 avg_word_length_B
      12 exclamation_count
      13 question_mark_count
      14 digit_ratio
    """
    X = []
    for r in records:
        a = r["optionA"]
        b = r["optionB"]

        is_ph_a = float(_is_placeholder(a))
        is_ph_b = float(_is_placeholder(b))

        len_a = np.log1p(len(a))
        len_b = np.log1p(len(b))

        words_a = a.split()
        words_b = b.split()
        wc_a = len(words_a)
        wc_b = len(words_b)

        wc_ratio = (min(wc_a, wc_b) / max(wc_a, wc_b)) if max(wc_a, wc_b) > 0 else 0.0

        sim = _option_similarity(a, b)
        has_wyr = float(bool(re.search(r"would you rather", a + b, re.IGNORECASE)))

        avg_wlen_a = float(np.mean([len(w) for w in words_a])) if words_a else 0.0
        avg_wlen_b = float(np.mean([len(w) for w in words_b])) if words_b else 0.0

        combined = a + b
        excl = combined.count("!")
        qmarks = combined.count("?")
        digits = sum(c.isdigit() for c in combined)
        digit_ratio = digits / max(len(combined), 1)

        X.append([
            is_ph_a, is_ph_b, float(is_ph_a or is_ph_b),
            len_a, len_b,
            wc_a, wc_b, wc_ratio,
            sim, has_wyr,
            avg_wlen_a, avg_wlen_b,
            excl, qmarks, digit_ratio,
        ])
    return np.array(X, dtype=np.float32)


# ─── Judge model class ────────────────────────────────────────────────────────

class WYRJudge(BaseEstimator):
    """
    Combines TF-IDF on text with handcrafted features, then logistic regression.
    Accepts list-of-dicts with keys: optionA, optionB, title.
    """

    def __init__(self, C=1.0, max_tfidf_features=5000):
        self.C = C
        self.max_tfidf_features = max_tfidf_features

    def _build_features(self, records, fit=False):
        texts = make_combined_text(records)
        if fit:
            tfidf_X = self.tfidf_.fit_transform(texts)
        else:
            tfidf_X = self.tfidf_.transform(texts)
        hand_X = extract_handcrafted(records)
        return hstack([tfidf_X, hand_X])

    def fit(self, X, y):
        self.tfidf_ = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=self.max_tfidf_features,
            sublinear_tf=True,
            min_df=1,
        )
        self.clf_ = LogisticRegression(
            C=self.C,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        )
        features = self._build_features(X, fit=True)
        self.clf_.fit(features, y)
        return self

    def predict_proba(self, X):
        features = self._build_features(X, fit=False)
        return self.clf_.predict_proba(features)

    def predict(self, X):
        return self.clf_.predict(self._build_features(X, fit=False))

    def score(self, X, y):
        return self.clf_.score(self._build_features(X, fit=False), y)


# ─── Training ────────────────────────────────────────────────────────────────

def train():
    positives, negatives = load_datasets()
    X = positives + negatives
    y = np.array([1] * len(positives) + [0] * len(negatives))

    model = WYRJudge(C=1.0)

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(f"\n5-fold CV Accuracy: {acc_scores.mean():.3f} ± {acc_scores.std():.3f}")

    # Manual ROC-AUC CV (predict_proba not directly supported by cross_val_score
    # without extra setup since we use a custom estimator)
    auc_scores = []
    for train_idx, val_idx in cv.split(X, y):
        X_tr = [X[i] for i in train_idx]
        X_val = [X[i] for i in val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        m = WYRJudge(C=1.0)
        m.fit(X_tr, y_tr)
        probs = m.predict_proba(X_val)[:, 1]
        auc_scores.append(roc_auc_score(y_val, probs))
    auc_scores = np.array(auc_scores)
    print(f"5-fold CV ROC-AUC:  {auc_scores.mean():.3f} ± {auc_scores.std():.3f}")

    # Fit on full dataset
    model.fit(X, y)

    # Training metrics
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    print(f"\nTraining ROC-AUC: {roc_auc_score(y, y_prob):.3f}")
    print("\nTraining classification report:")
    print(classification_report(y, y_pred, target_names=["negative", "positive"]))

    # Save
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {MODEL_PATH}")
    return model


# ─── Inference ───────────────────────────────────────────────────────────────

def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def score(optionA: str, optionB: str, title: str = "Would you rather", model=None) -> float:
    """
    Score a single WYR question.
    Returns a float in [0, 1] where 1 = predicted upvote.
    """
    if model is None:
        model = load_model()
    prob = model.predict_proba([{"optionA": optionA, "optionB": optionB, "title": title}])[0][1]
    return float(prob)


def score_batch(questions: list, model=None) -> list:
    """
    Score a batch of questions (list of dicts with optionA, optionB, optional title).
    Returns list of float scores in [0, 1].
    """
    if model is None:
        model = load_model()
    records = [
        {
            "optionA": q.get("optionA", ""),
            "optionB": q.get("optionB", ""),
            "title": q.get("title", "Would you rather"),
        }
        for q in questions
    ]
    return model.predict_proba(records)[:, 1].tolist()


# ─── Demo / inspection ───────────────────────────────────────────────────────

def demo(model):
    examples = [
        # Expected high scores
        {
            "optionA": "always have your room be the perfect temperature when going to sleep",
            "optionB": "be able to immediately find the perfect sleeping position",
            "label": "good (formatted_polls)",
        },
        {
            "optionA": "Be a knight but have to wear armor made of clay instead of steel",
            "optionB": "Have no hands",
            "label": "good (upvotes)",
        },
        {
            "optionA": "Have a photographic memory but only for things you find boring",
            "optionB": "Forget everything fun you experience within 24 hours",
            "label": "good (invented)",
        },
        {
            "optionA": "Always speak in rhyme but be incredibly persuasive",
            "optionB": "Speak normally but nobody ever believes anything you say",
            "label": "good (invented)",
        },
        # Expected low scores
        {
            "optionA": "option A",
            "optionB": "option B",
            "label": "bad (placeholder)",
        },
        {
            "optionA": "A meth head",
            "optionB": "A crackhead",
            "label": "bad (inappropriate)",
        },
        {
            "optionA": "Be the only black person in a white school",
            "optionB": "be the only female in all male highschool",
            "label": "bad (flagged)",
        },
        {
            "optionA": "leg hair.",
            "optionB": "Would you rather...\nA) A black and white dog with spots",
            "label": "bad (malformed)",
        },
    ]

    print("\n── Demo scoring ─────────────────────────────────────────────")
    print(f"{'Score':>6}  {'Bar':22}  Label")
    print("-" * 65)
    for ex in examples:
        s = score(ex["optionA"], ex["optionB"], model=model)
        bar = "█" * int(s * 20) + "░" * (20 - int(s * 20))
        print(f"  {s:.3f}  [{bar}]  {ex['label']}")

    # Top TF-IDF features by logistic regression coefficient
    print("\n── Top TF-IDF features ──────────────────────────────────────")
    vocab = np.array(model.tfidf_.get_feature_names_out())
    n_tfidf = len(vocab)
    coefs = model.clf_.coef_[0][:n_tfidf]
    top_pos = np.argsort(coefs)[-10:][::-1]
    top_neg = np.argsort(coefs)[:10]
    print("  → upvote:   ", list(vocab[top_pos]))
    print("  → downvote: ", list(vocab[top_neg]))


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="WYR judge model")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("train", help="Train and save the model")
    sub.add_parser("demo", help="Run demo examples (requires trained model)")

    p_score = sub.add_parser("score", help="Score a single question")
    p_score.add_argument("--optionA", required=True)
    p_score.add_argument("--optionB", required=True)
    p_score.add_argument("--title", default="Would you rather")

    p_file = sub.add_parser(
        "score-file",
        help="Score questions from a JSON file (list of {optionA, optionB, title?})",
    )
    p_file.add_argument("file")
    p_file.add_argument("--output", help="Write scored results to JSON file")

    args = parser.parse_args()

    if args.cmd == "train":
        m = train()
        demo(m)

    elif args.cmd == "demo":
        demo(load_model())

    elif args.cmd == "score":
        m = load_model()
        s = score(args.optionA, args.optionB, args.title, model=m)
        print(f"{s:.4f}")

    elif args.cmd == "score-file":
        with open(args.file) as f:
            questions = json.load(f)
        m = load_model()
        scores = score_batch(questions, model=m)
        results = []
        for q, s in zip(questions, scores):
            results.append({**q, "judge_score": round(s, 4)})
            print(f"{s:.4f}  {q.get('optionA','')[:55]} / {q.get('optionB','')[:55]}")
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nScored results written to {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
