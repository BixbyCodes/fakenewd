"""Evaluate the SAVED classical models without retraining them.

Produces results/saved_model_evaluation.json with:
- accuracy, precision, recall, F1, per-class metrics and confusion matrices
  for every saved ISOT and LIAR classifier on its own held-out test split;
- a majority-class baseline for each dataset;
- ISOT shortcut probes: how much the classifiers rely on the Reuters
  dateline, and how accuracy changes when only the title or the first N words
  are available (as happens when a user pastes a short claim).

Usage:  python -m src.evaluate_saved_models
Requires data/processed/hybrid/*_test.csv (and data/raw/isot for titles).
"""

import json
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.config import BASE_DIR, CLASSICAL_MODELS_DIR, RESULTS_DIR

SPLITS = BASE_DIR / "data" / "processed" / "hybrid"
RAW_ISOT = BASE_DIR / "data" / "raw" / "isot"
MODELS = ("logistic_regression", "linear_svm", "random_forest")
LABELS = {0: "REAL", 1: "FAKE"}


def binary_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    report = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=["REAL", "FAKE"],
        output_dict=True, zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_fake": float(precision),
        "recall_fake": float(recall),
        "f1_fake": float(f1),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "per_class": {
            name: {k: float(report[name][k]) for k in ("precision", "recall", "f1-score", "support")}
            for name in ("REAL", "FAKE")
        },
        "confusion_matrix": {
            "rows_true_cols_pred": ["REAL", "FAKE"],
            "matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        },
        "predicted_fake_share": float(np.mean(y_pred == 1)),
    }


def evaluate_dataset(dataset):
    test = pd.read_csv(SPLITS / "{}_test.csv".format(dataset)).fillna("")
    y = test.label.astype(int).to_numpy()
    vectorizer = joblib.load(CLASSICAL_MODELS_DIR / "{}_tfidf.joblib".format(dataset))
    X = vectorizer.transform(test.text.astype(str))
    majority = int(np.bincount(y).argmax())
    out = {
        "test_rows": int(len(test)),
        "label_counts": {LABELS[k]: int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "majority_baseline": {
            "predicts": LABELS[majority],
            "accuracy": float(np.mean(y == majority)),
        },
        "models": {},
    }
    loaded = {}
    for name in MODELS:
        model = joblib.load(CLASSICAL_MODELS_DIR / "{}_{}.joblib".format(dataset, name))
        loaded[name] = model
        out["models"][name] = binary_metrics(y, model.predict(X))
    return out, test, vectorizer, loaded


def isot_titles(test):
    """Recover ISOT titles (the processed text is 'title. body')."""
    frames = []
    for name in ("True.csv", "Fake.csv"):
        path = RAW_ISOT / name
        if path.exists():
            frames.append(pd.read_csv(path, usecols=["title", "text"]).fillna(""))
    if not frames:
        return None
    raw = pd.concat(frames, ignore_index=True)
    raw["joined"] = (raw["title"].astype(str) + ". " + raw["text"].astype(str)).str.strip()
    lookup = dict(zip(raw["joined"], raw["title"]))
    return test.text.map(lookup)


def isot_shortcut_probes(test, vectorizer, models):
    y = test.label.astype(int).to_numpy()
    texts = test.text.astype(str)
    probes = {}

    lr = models["logistic_regression"]
    names = np.array(vectorizer.get_feature_names_out())
    order = np.argsort(lr.coef_[0])
    probes["top_real_indicative_terms_lr"] = names[order[:20]].tolist()
    probes["top_fake_indicative_terms_lr"] = names[order[-20:]][::-1].tolist()

    probes["share_containing_reuters"] = {
        "REAL": float(texts[y == 0].str.contains("Reuters").mean()),
        "FAKE": float(texts[y == 1].str.contains("Reuters").mean()),
    }

    def strip_reuters(text):
        text = re.sub(r"^.*?\(Reuters\)\s*-\s*", "", text, count=1)
        return text.replace("Reuters", "")

    variants = {"reuters_removed": texts.map(strip_reuters)}

    titles = isot_titles(test)
    if titles is not None and titles.notna().mean() > 0.95:
        mask = titles.notna().to_numpy()
        variants["title_only"] = titles.fillna("")
        probes["title_match_rate"] = float(mask.mean())

    for n_words in (10, 25, 50, 100, 200):
        variants["first_{}_words".format(n_words)] = texts.map(
            lambda t, n=n_words: " ".join(t.split()[:n])
        )

    probes["accuracy_under_input_variants"] = {}
    for variant, items in variants.items():
        features = vectorizer.transform(items)
        probes["accuracy_under_input_variants"][variant] = {
            name: float(accuracy_score(y, model.predict(features)))
            for name, model in models.items()
        }
    return probes


def main():
    report = {
        "note": (
            "Saved models evaluated on their own held-out test splits. "
            "ISOT and LIAR are separate tasks; do not average them."
        )
    }
    isot, isot_test, isot_vec, isot_models = evaluate_dataset("isot")
    isot["shortcut_probes"] = isot_shortcut_probes(isot_test, isot_vec, isot_models)
    report["isot"] = isot
    liar, _, _, _ = evaluate_dataset("liar")
    report["liar"] = liar

    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / "saved_model_evaluation.json"
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("\nWritten:", path)


if __name__ == "__main__":
    main()
