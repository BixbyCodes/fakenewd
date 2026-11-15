"""Build an FTS5 full-text index over Wikipedia page titles."""

import sqlite3
from pathlib import Path

DB_PATH = Path("data/processed/wiki_pages.sqlite")


def build_title_index():
    if not DB_PATH.exists():
        raise FileNotFoundError("Wikipedia page index not found.")

    connection = sqlite3.connect(str(DB_PATH))

    try:
        cursor = connection.cursor()

        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS page_search
            USING fts5(
                page_id UNINDEXED,
                title,
                tokenize='unicode61 remove_diacritics 2'
            )
        """)

        existing = cursor.execute(
            "SELECT COUNT(*) FROM page_search"
        ).fetchone()[0]

        if existing:
            print("Title index already contains", existing, "pages.")
            return

        total = cursor.execute(
            "SELECT COUNT(*) FROM pages"
        ).fetchone()[0]

        print("Building title index for", total, "pages...")

        cursor.execute("""
            INSERT INTO page_search(page_id, title)
            SELECT page_id, page_id FROM pages
        """)

        connection.commit()

        indexed = cursor.execute(
            "SELECT COUNT(*) FROM page_search"
        ).fetchone()[0]

        print("Title index complete.")
        print("Pages indexed:", indexed)

    finally:
        connection.close()


if __name__ == "__main__":
    build_title_index()
