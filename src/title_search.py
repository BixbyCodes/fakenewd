"""Search Wikipedia page titles using SQLite FTS5."""

import re
import sqlite3
from pathlib import Path

DB_PATH = Path("data/processed/wiki_pages.sqlite")

STOPWORDS = {
    "a", "an", "the", "is", "was", "were", "are",
    "be", "been", "being", "of", "in", "on", "at",
    "to", "for", "and", "or", "but", "by", "with",
    "from", "that", "this", "it", "as", "during",
    "released", "produced"
}


def search_pages(claim, limit=10):
    if not claim or not claim.strip():
        return []

    if limit < 1:
        raise ValueError("limit must be at least 1")

    words = re.findall(r"[^\W_]+", claim.lower(), flags=re.UNICODE)

    terms = [
        word for word in words
        if word not in STOPWORDS and len(word) > 1
    ]

    if not terms:
        return []

    # OR allows a page to match important claim terms
    # without requiring every claim word in its title.
    query = " OR ".join(
        '"' + term.replace('"', '""') + '"'
        for term in dict.fromkeys(terms)
    )

    connection = sqlite3.connect(str(DB_PATH))

    try:
        rows = connection.execute("""
            SELECT page_id, bm25(page_search) AS score
            FROM page_search
            WHERE page_search MATCH ?
            ORDER BY score
            LIMIT ?
        """, (query, limit)).fetchall()

        return [
            {"page_id": page_id, "score": score}
            for page_id, score in rows
        ]
    finally:
        connection.close()


if __name__ == "__main__":
    claim = "Fox 2000 Pictures released the film Soul Food"

    for result in search_pages(claim):
        print(result)
