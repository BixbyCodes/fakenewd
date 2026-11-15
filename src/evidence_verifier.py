import os
import json
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from src.title_search import search_pages

load_dotenv()

DB_PATH = Path("data/processed/wiki_pages.sqlite")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


def retrieve_page_sentences(page_id, limit=5):
    """Retrieve actual sentences from an indexed Wikipedia page."""
    if not DB_PATH.exists():
        raise FileNotFoundError("Wikipedia SQLite index not found.")

    with sqlite3.connect(str(DB_PATH)) as connection:
        row = connection.execute(
            "SELECT shard_path, byte_offset FROM pages WHERE page_id=?",
            (page_id,)
        ).fetchone()

    if not row:
        return []

    shard_path, offset = row

    with open(shard_path, "rb") as handle:
        handle.seek(offset)
        page = json.loads(handle.readline())

    lines = page.get("lines", "")
    if not isinstance(lines, str):
        return []

    sentences = []

    for line in lines.splitlines():
        parts = line.split("\t", 2)

        if len(parts) < 2:
            continue

        try:
            sentence_id = int(parts[0])
        except ValueError:
            continue

        sentence = parts[1].strip()

        if sentence:
            sentences.append({
                "page_id": page_id,
                "sentence_id": sentence_id,
                "text": sentence
            })

        if len(sentences) >= limit:
            break

    return sentences


def verify_evidence(claim, evidence):
    """Compare retrieved evidence against a claim using Groq."""
    if not evidence:
        return []

    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY missing from .env")

    client = Groq(api_key=key)

    payload = [
        {
            "page_id": item["page_id"],
            "sentence_id": item["sentence_id"],
            "text": item["text"]
        }
        for item in evidence
    ]

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        reasoning_effort="low",
        max_completion_tokens=1800,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": """
You are a cautious evidence verification assistant.

Compare the supplied claim with each supplied evidence sentence.

For each sentence, assign exactly one label:
SUPPORTS: The sentence provides direct factual support for the claim.
REFUTES: The sentence directly contradicts the claim.
INSUFFICIENT: The sentence does not establish either conclusion.

Rules:
- Do not use outside knowledge.
- Do not assume missing information.
- Similar topics are not necessarily supporting evidence.
- Preserve the supplied page_id and sentence_id.
- Explain the comparison briefly.
- Return valid JSON with one key: results.
- results must be an array of objects with these keys:
  page_id, sentence_id, label, explanation.
"""
            },
            {
                "role": "user",
                "content": json.dumps({
                    "claim": claim,
                    "evidence": payload
                })
            }
        ]
    )

    content = response.choices[0].message.content or ""
    start = content.find("{")

    if start < 0:
        raise ValueError("Verifier returned no JSON.")

    data, _ = json.JSONDecoder().raw_decode(content[start:])

    if not isinstance(data.get("results"), list):
        raise ValueError("Invalid evidence verification response.")

    evidence_lookup = {
        (item["page_id"], item["sentence_id"]): item["text"]
        for item in evidence
    }

    verified = []

    for item in data["results"]:
        key = (item.get("page_id"), item.get("sentence_id"))
        label = item.get("label")

        if key not in evidence_lookup:
            continue

        if label not in {"SUPPORTS", "REFUTES", "INSUFFICIENT"}:
            continue

        verified.append({
            "page_id": key[0],
            "sentence_id": key[1],
            "sentence": evidence_lookup[key],
            "label": label,
            "explanation": str(item.get("explanation", ""))
        })

    return verified


def analyze_evidence(claim, normalized_claim=None, page_limit=5):
    """Retrieve candidate pages, extract sentences and verify them."""
    query = normalized_claim or claim

    candidates = search_pages(query, limit=page_limit)

    evidence = []
    for candidate in candidates:
        evidence.extend(
            retrieve_page_sentences(candidate["page_id"], limit=5)
        )

    verified = verify_evidence(claim, evidence)

    return {
        "query": query,
        "candidate_pages": candidates,
        "evidence_count": len(evidence),
        "verified_count": len(verified),
        "results": verified
    }
