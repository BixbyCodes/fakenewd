"""LLM-based event and entity extraction for EAFND.

The extracted structure (including `normalized_claim`) is shown to the user
and used to add entity-name queries for evidence retrieval. It never replaces
the submitted text: the original claim/article is always what is classified
and what retrieved evidence is assessed against.
"""

import hashlib
import json
import logging

from src.config import EVENT_PARSE_CACHE, LLM_TEXT_CHAR_LIMIT, llm_model
from src.llm_client import JsonCache, get_client, parse_json_object

logger = logging.getLogger("eafnd.event_parser")

MODEL = llm_model()
CACHE_PATH = EVENT_PARSE_CACHE
_cache = JsonCache(CACHE_PATH, "event_parses")

PROMPT_TEMPLATE = """
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

Language: """

REQUIRED = (
    "event_summary", "entities", "time", "location",
    "attribution", "normalized_claim", "uncertainties",
)


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


def _optional_string(value, field):
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if not isinstance(value, str):
        raise ValueError("{} must be a string or null.".format(field))
    value = value.strip()
    return value or None


def validate_result(data):
    """Validate and lightly clean the model's event-extraction response."""
    if not isinstance(data, dict) or not set(REQUIRED).issubset(data):
        raise ValueError("Model response is missing required JSON fields.")

    clean = {}
    for field in ("event_summary", "normalized_claim", "time", "location", "attribution"):
        clean[field] = _optional_string(data[field], field)

    if not isinstance(data["entities"], list):
        raise ValueError("entities must be a list.")

    entities = []
    for entity in data["entities"]:
        if not isinstance(entity, dict):
            raise ValueError("Each entity must be an object.")
        for field in ("name", "type", "role"):
            if not isinstance(entity.get(field), str):
                raise ValueError("Each entity needs string fields: name, type, role.")
        if entity["name"].strip():
            entities.append({k: entity[k].strip() for k in ("name", "type", "role")})
    clean["entities"] = entities

    if not isinstance(data["uncertainties"], list):
        raise ValueError("uncertainties must be a list.")
    if not all(isinstance(item, str) for item in data["uncertainties"]):
        raise ValueError("Every uncertainty must be a string.")
    clean["uncertainties"] = [item.strip() for item in data["uncertainties"] if item.strip()]

    return clean


def cache_key(text, language="en"):
    return hashlib.sha256(
        (MODEL + "\n" + language + "\n" + text).encode("utf-8")
    ).hexdigest()


def parse_event(record, language="en", use_cache=True):
    """Extract event, entities, time, location and attribution."""
    text = get_text(record)
    truncated = len(text) > LLM_TEXT_CHAR_LIMIT
    text = text[:LLM_TEXT_CHAR_LIMIT]
    key = cache_key(text, language)

    if use_cache:
        hit = _cache.get(key)
        if hit:
            hit["cached"] = True
            return hit

    client = get_client()
    prompt = PROMPT_TEMPLATE + language + "\n\nText:\n" + text

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
        max_completion_tokens=1500,
    )

    result = validate_result(parse_json_object(response.choices[0].message.content or ""))
    result["model"] = MODEL
    result["input_truncated"] = truncated
    result["source_text_hash"] = hashlib.sha256(text.encode("utf-8")).hexdigest()

    _cache.set(key, result)
    result["cached"] = False
    return result


if __name__ == "__main__":
    sample = (
        "The Indian government announced a new education policy "
        "in New Delhi on 29 July 2020."
    )
    print(json.dumps(parse_event(sample), indent=2, ensure_ascii=False))
