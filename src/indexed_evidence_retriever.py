"""Fast evidence retrieval using the SQLite Wikipedia page index."""

import json
from contextlib import closing

from src.evidence_retriever import extract_sentence
from src.title_search import open_index, resolve_shard_path


def load_page(page_id):
    """Return the raw FEVER page record for page_id, or None if not indexed."""
    with closing(open_index()) as connection:
        row = connection.execute(
            "SELECT shard_path, byte_offset FROM pages WHERE page_id = ?",
            (page_id,)
        ).fetchone()

    if row is None:
        return None

    shard_path, byte_offset = row
    with open(resolve_shard_path(shard_path), "rb") as handle:
        handle.seek(byte_offset)
        return json.loads(handle.readline())


def retrieve_indexed_evidence(page_id, sentence_id):
    page = load_page(page_id)

    if page is None:
        return {"found": False, "error": "Page not found in index."}

    sentence = extract_sentence(page, sentence_id)

    return {
        "found": sentence is not None,
        "page_id": page_id,
        "sentence_id": sentence_id,
        "sentence": sentence,
        "error": None if sentence else "Sentence not found."
    }
