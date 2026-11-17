"""Classical (TF-IDF) fake/real classifier inference.

Primary model: ISOT Linear SVM (calibrated with CalibratedClassifierCV), the
best of the three saved ISOT classifiers on the held-out ISOT test split.
Logistic Regression and Random Forest are reported alongside it as a
model-agreement check. Label mapping used in training (src/data.py):
0 = REAL, 1 = FAKE.

The probabilities are model scores learned from ISOT writing patterns. They
are not a verified probability that the claim is true or false.
"""

import argparse
import json
import re
import threading
from functools import lru_cache

import joblib

from src.config import CLASSICAL_MODELS_DIR, RESULTS_DIR

LABELS = {0: "REAL", 1: "FAKE"}
DATASET = "isot"
PRIMARY_MODEL = "linear_svm"
MODEL_NAMES = {
    "linear_svm": "Linear SVM (calibrated)",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
}
# Inputs shorter than this are far from the full-length ISOT training articles.
SHORT_INPUT_WORDS = 50
LOW_CONFIDENCE = 0.65

_lock = threading.Lock()


class ModelUnavailable(RuntimeError):
    """Raised when saved model files are missing or cannot be loaded."""


@lru_cache(maxsize=1)
def _load_vectorizer():
    path = CLASSICAL_MODELS_DIR / "{}_tfidf.joblib".format(DATASET)
    if not path.exists():
        raise ModelUnavailable("Saved TF-IDF vectorizer not found: {}".format(path.name))
    return joblib.load(path)


@lru_cache(maxsize=4)
def _load_model(name):
    path = CLASSICAL_MODELS_DIR / "{}_{}.joblib".format(DATASET, name)
    if not path.exists():
        raise ModelUnavailable("Saved model not found: {}".format(path.name))
    model = joblib.load(path)
    if list(getattr(model, "classes_", [])) != [0, 1]:
        raise ModelUnavailable("Unexpected class labels in {}".format(path.name))
    return model


def load_models():
    """Load (and cache) the vectorizer and all ISOT classifiers."""
    with _lock:
        vectorizer = _load_vectorizer()
        models = {name: _load_model(name) for name in MODEL_NAMES}
    return vectorizer, models


def models_available():
    try:
        load_models()
        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def recorded_metrics():
    """Held-out test metrics recorded for the saved ISOT classifiers."""
    path = RESULTS_DIR / "isot_classical.json"
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def reliability_notes(text):
    """Warnings about inputs that differ from the ISOT training distribution."""
    notes = []
    words = len(text.split())
    if words < SHORT_INPUT_WORDS:
        notes.append(
            "Short input ({} words). The classifier was trained on full ISOT "
            "articles; on ISOT headlines alone the Linear SVM was correct only "
            "about 69% of the time (results/saved_model_evaluation.json).".format(words)
        )
    if re.search(r"\(\s*reuters\s*\)", text, flags=re.IGNORECASE):
        notes.append(
            "The text contains a '(Reuters)' dateline. In ISOT almost every REAL "
            "article contains it and almost no FAKE article does, so the model "
            "treats it as strong evidence of REAL regardless of content."
        )
    return notes


def predict_news(text):
    """Classify text and return probabilities with provenance."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Article text cannot be empty.")

    vectorizer, models = load_models()
    features = vectorizer.transform([text])

    if features.nnz == 0:
        notes = ["None of the input words appear in the classifier vocabulary."]
    else:
        notes = []

    per_model = {}
    for name, model in models.items():
        proba = model.predict_proba(features)[0]
        label = int(proba.argmax())
        per_model[name] = {
            "model": MODEL_NAMES[name],
            "prediction": LABELS[label],
            "fake_probability": round(float(proba[1]) * 100, 2),
            "real_probability": round(float(proba[0]) * 100, 2),
        }

    primary = per_model[PRIMARY_MODEL]
    confidence = max(primary["fake_probability"], primary["real_probability"])
    agreeing = sum(1 for item in per_model.values() if item["prediction"] == primary["prediction"])
    metrics = recorded_metrics().get(PRIMARY_MODEL, {})

    return {
        "prediction": primary["prediction"],
        "confidence": confidence,
        "fake_probability": primary["fake_probability"],
        "real_probability": primary["real_probability"],
        "low_confidence": confidence < LOW_CONFIDENCE * 100,
        "model": MODEL_NAMES[PRIMARY_MODEL],
        "model_key": "{}_{}".format(DATASET, PRIMARY_MODEL),
        "vectorizer": "TF-IDF (1-2 grams, 150k features, English stop words)",
        "training_data": "ISOT Fake News Dataset (5,000 real + 5,000 fake sample, 70/15/15 split)",
        "label_mapping": {"0": "REAL", "1": "FAKE"},
        "recorded_test_accuracy": metrics.get("accuracy"),
        "model_agreement": "{}/{} ISOT classifiers predict {}".format(
            agreeing, len(per_model), primary["prediction"]
        ),
        "all_models": per_model,
        "input_words": len(text.split()),
        "vocabulary_hits": int(features.nnz),
        "reliability_notes": notes + reliability_notes(text),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EvidenceAware classifier")
    parser.add_argument("--text", required=True, help="News article text")
    args = parser.parse_args()

    result = predict_news(args.text)
    print("\n--- CLASSIFIER PREDICTION ---")
    print("Model:", result["model"], "trained on ISOT")
    print("Prediction:", result["prediction"])
    print("Fake probability:", "{}%".format(result["fake_probability"]))
    print("Real probability:", "{}%".format(result["real_probability"]))
    print("Agreement:", result["model_agreement"])
    for note in result["reliability_notes"]:
        print("Note:", note)
    print("\nThis is a model prediction, not independent verification.")
