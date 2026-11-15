"""LLM-based event and entity normalization for EAFND."""

import hashlib
import json
import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

MODEL = "openai/gpt-oss-20b"
CACHE_PATH = Path("data/processed/event_parse_cache.sqlite")


def get_text(record):
    """Accept a plain string, FEVER record, or ingested article record."""
    if isinstance(record, str):
        text = record.strip()
    elif isinstance(record, dict):
        if "claim" in record:
            text = str(record["claim"]).strip()
        else:
            headline = record.get("headline", "")
            body = record.get("body", "")
            text = "{}\n{}".format(headline, body).strip()
    else:
        raise TypeError("Input must be a string or dictionary.")

    if not text:
        raise ValueError("Input text cannot be empty.")

    return text


def validate_result(data):
    """Validate the model's event-normalization response."""
    required = {
        "event_summary",
        "entities",
        "time",
        "location",
        "attribution",
        "normalized_claim",
        "uncertainties",
    }

    if not isinstance(data, dict) or not required.issubset(data):
        raise ValueError("Model response is missing required JSON fields.")

    for field in ("event_summary", "normalized_claim"):
        if data[field] is not None and not isinstance(data[field], str):
            raise ValueError("{} must be a string or null.".format(field))

    for field in ("time", "location", "attribution"):
        if data[field] is not None and not isinstance(data[field], str):
            raise ValueError("{} must be a string or null.".format(field))

    if not isinstance(data["entities"], list):
        raise ValueError("entities must be a list.")

    for entity in data["entities"]:
        if not isinstance(entity, dict):
            raise ValueError("Each entity must be an object.")
        for field in ("name", "type", "role"):
            if not isinstance(entity.get(field), str):
                raise ValueError("Each entity needs string fields: name, type, role.")

    if not isinstance(data["uncertainties"], list):
        raise ValueError("uncertainties must be a list.")

    if not all(isinstance(item, str) for item in data["uncertainties"]):
        raise ValueError("Every uncertainty must be a string.")

    return data


def initialize_cache():
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(str(CACHE_PATH)) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS event_parses (
                cache_key TEXT PRIMARY KEY,
                result_json TEXT NOT NULL
            )
        """)


def parse_event(record, language="en", use_cache=True):
    """Extract event, entities, time, location and attribution."""
    load_dotenv()

    text = get_text(record)
    cache_key = hashlib.sha256(
        (MODEL + "\n" + language + "\n" + text).encode("utf-8")
    ).hexdigest()

    initialize_cache()

    if use_cache:
        with sqlite3.connect(str(CACHE_PATH)) as connection:
            row = connection.execute(
                "SELECT result_json FROM event_parses WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()

        if row:
            return json.loads(row[0])

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY was not found in .env.")

    client = Groq(api_key=api_key)

    prompt = """
You are an information extraction component in a fact-checking research system.

Extract the event information from the supplied text.

Rules:
1. Use only information explicitly stated in the text.
2. Never invent entities, dates, locations, sources or events.
3. Use null when a single value is not stated or cannot be determined.
4. Include people, organizations, places and other relevant named entities.
5. Give each entity a short type and its role in the described event.
6. Preserve the original meaning. Do not decide whether the claim is true.
7. Put ambiguity or missing context in uncertainties.
8. Return valid JSON only, with exactly these fields:

{
  "event_summary": "string or null",
  "entities": [
    {"name": "string", "type": "string", "role": "string"}
  ],
  "time": "string or null",
  "location": "string or null",
  "attribution": "string or null",
  "normalized_claim": "string",
  "uncertainties": ["string"]
}

Language: """ + language + """

Text:
""" + text

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You extract factual structure. Do not add unsupported details.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
        max_completion_tokens=1200,
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("The model returned an empty response.")

    result = validate_result(json.loads(content))
    result["model"] = MODEL
    result["source_text_hash"] = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

    with sqlite3.connect(str(CACHE_PATH)) as connection:
        connection.execute(
            "INSERT OR REPLACE INTO event_parses VALUES (?, ?)",
            (cache_key, json.dumps(result, ensure_ascii=False)),
        )

    return result


if __name__ == "__main__":
    sample = (
        "The Indian government announced a new education policy "
        "in New Delhi on 29 July 2020."
    )
    print(json.dumps(parse_event(sample), indent=2, ensure_ascii=False))
