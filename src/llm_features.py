"""LLM-based structured assessment of a news text (Groq, gpt-oss-20b).

Used in two places:
- `assess_text` powers the "LLM assessment" panel of the application;
- `feature_rows` produced the LLM features for the XGBoost fusion experiment
  (src/fusion.py).

Both share one SQLite cache (data/processed/llm_features.sqlite) keyed by the
SHA-256 of the text and the model name. The output is an LLM assessment of
the text, not a verified fact check: the model sees only the submitted text.
"""

import hashlib
import logging
import time
from contextlib import closing

from src.config import LLM_FEATURE_CACHE, llm_model
from src.llm_client import (
    get_client,
    parse_json_object,
    sqlite_connect,
)

logger = logging.getLogger("eafnd.llm_features")

MODEL = llm_model()
DB = LLM_FEATURE_CACHE
TEXT_LIMIT = 3500  # characters sent to the LLM (unchanged from the experiments)

FIELDS = {
    "verdict", "plausibility", "verifiability",
    "source_credibility", "rationale"
}

SYSTEM_PROMPT = (
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


def connect():
    c = sqlite_connect(DB)
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


def text_hash(text):
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def validate_features(data):
    """Validate and clean the LLM feature object."""
    if not isinstance(data, dict) or not FIELDS.issubset(data):
        raise ValueError("LLM returned an invalid feature schema")

    verdict = str(data["verdict"]).strip().lower()
    if verdict not in {"real", "fake", "uncertain"}:
        raise ValueError("Invalid verdict")

    clean = {"verdict": verdict}
    for key in ("plausibility", "verifiability", "source_credibility"):
        value = data[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Invalid numeric field: " + key)
        if not 0 <= value <= 1:
            raise ValueError("Score outside 0-1: " + key)
        clean[key] = float(value)

    if not isinstance(data["rationale"], str):
        raise ValueError("Invalid rationale")
    clean["rationale"] = data["rationale"].strip()
    return clean


def extract_one(text, client):
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        reasoning_effort="low",
        max_completion_tokens=1200,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(text)[:TEXT_LIMIT]},
        ],
    )
    content = response.choices[0].message.content or ""
    return validate_features(parse_json_object(content))


def _cached(connection, h):
    row = connection.execute(
        "SELECT verdict, plausibility, verifiability, source_credibility, rationale "
        "FROM features WHERE text_hash=? AND model=?",
        (h, MODEL),
    ).fetchone()
    if row is None:
        return None
    return {
        "verdict": row[0],
        "plausibility": row[1],
        "verifiability": row[2],
        "source_credibility": row[3],
        "rationale": row[4] or "",
    }


def _store(connection, h, text, data):
    connection.execute(
        "INSERT OR REPLACE INTO features"
        "(text_hash,text,verdict,plausibility,verifiability,"
        "source_credibility,rationale,model) VALUES(?,?,?,?,?,?,?,?)",
        (h, text, data["verdict"], data["plausibility"], data["verifiability"],
         data["source_credibility"], data["rationale"], MODEL),
    )
    connection.commit()


def assess_text(text, use_cache=True):
    """Return the LLM assessment for one text (cached).

    Raises LLMUnavailable or Groq/validation errors; callers convert them
    to user-safe messages with src.llm_client.describe_error.
    """
    text = str(text)
    h = text_hash(text)

    if use_cache:
        try:
            with closing(connect()) as connection:
                hit = _cached(connection, h)
            if hit:
                hit.update(model=MODEL, cached=True)
                return hit
        except Exception as exc:  # cache problems must not block analysis
            logger.warning("LLM feature cache unavailable: %s", type(exc).__name__)

    data = extract_one(text, get_client())

    try:
        with closing(connect()) as connection:
            _store(connection, h, text, data)
    except Exception as exc:
        logger.warning("Could not cache LLM features: %s", type(exc).__name__)

    data.update(model=MODEL, cached=False)
    return data


def feature_rows(texts, limit=None, delay=0.15):
    """Numeric LLM features for the fusion experiment (src/fusion.py).

    Columns: verdict score (real=0, fake=1, uncertain=0.5), plausibility,
    verifiability, source_credibility, uncertain flag, success flag.
    Failed rows are filled with neutral values and success=0.
    """
    client = get_client()
    out = []
    successes = 0
    subset = list(texts if limit is None else texts[:limit])

    with closing(connect()) as c:
        for i, text in enumerate(subset, 1):
            text = str(text)
            h = text_hash(text)
            data = _cached(c, h)
            success = 1.0

            if data is None:
                try:
                    data = extract_one(text, client)
                    _store(c, h, text, data)
                    time.sleep(delay)
                except Exception as e:
                    print("LLM failed on row {}: {}".format(i, type(e).__name__))
                    data = {"verdict": "uncertain", "plausibility": 0.5,
                            "verifiability": 0.0, "source_credibility": 0.0}
                    success = 0.0

            if success:
                successes += 1

            out.append([
                {"real": 0.0, "fake": 1.0, "uncertain": 0.5}[data["verdict"]],
                data["plausibility"],
                data["verifiability"],
                data["source_credibility"],
                1.0 if data["verdict"] == "uncertain" else 0.0,
                success,
            ])

            if i % 10 == 0:
                print("LLM features: {}/{} | successful: {}".format(i, len(subset), successes))

    print("LLM annotation success: {}/{}".format(successes, len(subset)))
    return out
