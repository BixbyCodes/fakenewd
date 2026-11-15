"""Fast evidence retrieval using the SQLite Wikipedia page index."""

import json
import sqlite3
from pathlib import Path

from src.evidence_retriever import extract_sentence

INDEX_PATH = Path("data/processed/wiki_pages.sqlite")


def retrieve_indexed_evidence(page_id, sentence_id):
    if not INDEX_PATH.exists():
        raise FileNotFoundError("Wikipedia SQLite index not found.")

    connection = sqlite3.connect(str(INDEX_PATH))

    try:
        row = connection.execute(
            "SELECT shard_path, byte_offset FROM pages WHERE page_id = ?",
            (page_id,)
        ).fetchone()
    finally:
        connection.close()

    if row is None:
        return {"found": False, "error": "Page not found in index."}

    shard_path, byte_offset = row

    with open(shard_path, "rb") as handle:
        handle.seek(byte_offset)
        page = json.loads(handle.readline())

    sentence = extract_sentence(page, sentence_id)

    return {
        "found": sentence is not None,
        "page_id": page_id,
        "sentence_id": sentence_id,
        "sentence": sentence,
        "shard": Path(shard_path).name,
        "error": None if sentence else "Sentence not found."
    }
