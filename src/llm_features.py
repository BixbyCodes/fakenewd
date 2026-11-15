import os
import json
import sqlite3
import hashlib
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
DB = Path("data/processed/llm_features.sqlite")

FIELDS = {
    "verdict", "plausibility", "verifiability",
    "source_credibility", "rationale"
}


def connect():
    DB.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB)
    c.execute("""
        CREATE TABLE IF NOT EXISTS features(
            text_hash TEXT PRIMARY KEY,
            text TEXT,
            verdict TEXT,
            plausibility REAL,
            verifiability REAL,
            source_credibility REAL,
            rationale TEXT,
            model TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    return c


def extract_one(text, client):
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        reasoning_effort="low",
        max_completion_tokens=1200,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Analyze the supplied news text as a cautious research "
                    "annotation assistant. Do not decide truth without evidence. "
                    "Use verdict real, fake, or uncertain. Scores must be "
                    "numbers between 0 and 1. Source credibility measures "
                    "whether a source is identified in the text, not whether "
                    "you personally trust it. Do not invent information. "
                    "Return one JSON object only, with exactly these keys: "
                    "verdict, plausibility, verifiability, "
                    "source_credibility, rationale. No markdown."
                )
            },
            {"role": "user", "content": str(text)[:3500]}
        ]
    )

    content = response.choices[0].message.content or ""
    start = content.find("{")
    if start < 0:
        raise ValueError("LLM response contains no JSON object")

    # Decode the first complete JSON object; ignore trailing characters.
    data, _ = json.JSONDecoder().raw_decode(content[start:])

    if not isinstance(data, dict) or set(data) != FIELDS:
        raise ValueError("LLM returned an invalid feature schema")

    if data["verdict"] not in {"real", "fake", "uncertain"}:
        raise ValueError("Invalid verdict")

    for key in ("plausibility", "verifiability", "source_credibility"):
        value = data[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Invalid numeric field: " + key)
        if not 0 <= value <= 1:
            raise ValueError("Score outside 0-1: " + key)

    if not isinstance(data["rationale"], str):
        raise ValueError("Invalid rationale")

    return data


def feature_rows(texts, limit=None, delay=0.15):
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY missing. Add it to .env.")

    client = Groq(api_key=key)
    c = connect()
    out = []
    successes = 0
    subset = list(texts if limit is None else texts[:limit])

    for i, text in enumerate(subset, 1):
        text = str(text)
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()

        row = c.execute(
            "SELECT verdict, plausibility, verifiability, "
            "source_credibility FROM features WHERE text_hash=? AND model=?",
            (h, MODEL)
        ).fetchone()

        success = 1.0

        if row is None:
            try:
                d = extract_one(text, client)
                row = (
                    d["verdict"],
                    float(d["plausibility"]),
                    float(d["verifiability"]),
                    float(d["source_credibility"])
                )
                c.execute(
                    "INSERT OR REPLACE INTO features"
                    "(text_hash,text,verdict,plausibility,verifiability,"
                    "source_credibility,rationale,model) "
                    "VALUES(?,?,?,?,?,?,?,?)",
                    (h, text, *row, d["rationale"], MODEL)
                )
                c.commit()
                time.sleep(delay)

            except Exception as e:
                print("LLM failed on row {}: {}".format(i, e))
                row = ("uncertain", 0.5, 0.0, 0.0)
                success = 0.0
        else:
            successes += 1

        if success:
            successes += 1

        out.append([
            {"real": 0.0, "fake": 1.0, "uncertain": 0.5}[row[0]],
            row[1],
            row[2],
            row[3],
            1.0 if row[0] == "uncertain" else 0.0,
            success
        ])

        if i % 10 == 0:
            print("LLM features: {}/{} | successful: {}".format(
                i, len(subset), successes
            ))

    c.close()
    print("LLM annotation success: {}/{}".format(successes, len(subset)))
    return out
