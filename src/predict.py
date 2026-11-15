from pathlib import Path
import joblib
import argparse

BASE = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE / "models" / "classical"

vectorizer = joblib.load(MODEL_DIR / "isot_tfidf.joblib")
model = joblib.load(MODEL_DIR / "isot_logistic_regression.joblib")

def predict_news(text):
    if not text.strip():
        raise ValueError("Article text cannot be empty.")

    features = vectorizer.transform([text])
    probabilities = model.predict_proba(features)[0]

    label = int(probabilities.argmax())
    confidence = float(probabilities[label])

    return {
        "prediction": "FAKE" if label == 1 else "REAL",
        "confidence": round(confidence * 100, 2),
        "fake_probability": round(float(probabilities[1]) * 100, 2),
        "real_probability": round(float(probabilities[0]) * 100, 2)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fake News Detection")
    parser.add_argument("--text", required=True, help="News article text")
    args = parser.parse_args()

    result = predict_news(args.text)

    print("\n--- FAKE NEWS DETECTION ---")
    print("Prediction:", result["prediction"])
    print("Confidence:", f'{result["confidence"]}%')
    print("Fake probability:", f'{result["fake_probability"]}%')
    print("Real probability:", f'{result["real_probability"]}%')
    print("\nNote: This is a model prediction, not independent verification.")
