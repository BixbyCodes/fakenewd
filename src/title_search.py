"""Search Wikipedia page titles using SQLite FTS5 (BM25)."""

import os
import re
import sqlite3
from contextlib import closing
from pathlib import Path

from src.config import BASE_DIR, SQLITE_TIMEOUT_SECONDS, WIKI_INDEX_PATH

DB_PATH = WIKI_INDEX_PATH
MAX_QUERY_TERMS = 40

STOPWORDS = {
    "a", "an", "the", "is", "was", "were", "are",
    "be", "been", "being", "of", "in", "on", "at",
    "to", "for", "and", "or", "but", "by", "with",
    "from", "that", "this", "it", "as", "during",
    "released", "produced"
}


class EvidenceIndexUnavailable(RuntimeError):
    """The local FEVER Wikipedia index (or its shards) is not available."""


def open_index():
    """Open the Wikipedia index read-only. Use with contextlib.closing()."""
    path = Path(DB_PATH)
    if not path.exists():
        raise EvidenceIndexUnavailable(
            "The FEVER Wikipedia index was not found at {}.".format(path.name)
        )
    uri = path.resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True, timeout=SQLITE_TIMEOUT_SECONDS,
                           check_same_thread=False)


def resolve_shard_path(shard_path):
    """Shard paths are stored relative to the repository root."""
    path = Path(shard_path)
    candidates = [path] if path.is_absolute() else [BASE_DIR / path]
    override = os.getenv("EAFND_WIKI_PAGES_DIR")
    if override:
        candidates.append(Path(override) / path.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise EvidenceIndexUnavailable(
        "Wikipedia shard {} referenced by the index is missing.".format(path.name)
    )


def index_status():
    """Report whether evidence retrieval can run. Never raises."""
    try:
        with closing(open_index()) as connection:
            row = connection.execute(
                "SELECT shard_path FROM pages LIMIT 1"
            ).fetchone()
            connection.execute("SELECT 1 FROM page_search LIMIT 1").fetchone()
        if row is None:
            return {"available": False, "reason": "The Wikipedia index is empty."}
        resolve_shard_path(row[0])
        return {"available": True, "reason": None}
    except EvidenceIndexUnavailable as exc:
        return {"available": False, "reason": str(exc)}
    except sqlite3.Error:
        return {"available": False,
                "reason": "The Wikipedia index could not be opened or lacks the title search table."}


def query_terms(text):
    words = re.findall(r"[^\W_]+", text.lower(), flags=re.UNICODE)
    terms = [word for word in words if word not in STOPWORDS and len(word) > 1]
    return list(dict.fromkeys(terms))[:MAX_QUERY_TERMS]


def search_pages(claim, limit=10):
    if not claim or not claim.strip():
        return []

    if limit < 1:
        raise ValueError("limit must be at least 1")

    terms = query_terms(claim)
    if not terms:
        return []

    # OR allows a page to match important claim terms
    # without requiring every claim word in its title.
    query = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)

    with closing(open_index()) as connection:
        rows = connection.execute("""
            SELECT page_id, bm25(page_search) AS score
            FROM page_search
            WHERE page_search MATCH ?
            ORDER BY score
            LIMIT ?
        """, (query, limit)).fetchall()

    return [{"page_id": page_id, "score": score} for page_id, score in rows]


def page_exists(page_id):
    with closing(open_index()) as connection:
        row = connection.execute(
            "SELECT 1 FROM pages WHERE page_id = ?", (page_id,)
        ).fetchone()
    return row is not None


if __name__ == "__main__":
    claim = "Fox 2000 Pictures released the film Soul Food"

    for result in search_pages(claim):
        print(result)
