"""Build a disk-efficient SQLite index for FEVER Wikipedia pages."""

import json
import sqlite3
from pathlib import Path


CORPUS_DIR = Path("data/raw/fever/wiki-pages")
INDEX_PATH = Path("data/processed/wiki_pages.sqlite")


def build_index():
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(str(INDEX_PATH))
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pages (
            page_id TEXT PRIMARY KEY,
            shard_path TEXT NOT NULL,
            byte_offset INTEGER NOT NULL
        )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_page_id ON pages(page_id)")
    connection.commit()

    shards = sorted(CORPUS_DIR.glob("wiki-*.jsonl"))

    if not shards:
        raise FileNotFoundError("No Wikipedia shards found.")

    total = 0

    for shard_number, shard in enumerate(shards, start=1):
        print("Indexing {}/{}: {}".format(
            shard_number, len(shards), shard.name
        ))

        batch = []

        with shard.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()

                if not line:
                    break

                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue

                page_id = record.get("id")

                if not isinstance(page_id, str) or not page_id:
                    continue

                batch.append((page_id, str(shard), offset))

                if len(batch) >= 10000:
                    cursor.executemany(
                        "INSERT OR IGNORE INTO pages VALUES (?, ?, ?)",
                        batch
                    )
                    connection.commit()
                    total += len(batch)
                    batch = []

        if batch:
            cursor.executemany(
                "INSERT OR IGNORE INTO pages VALUES (?, ?, ?)",
                batch
            )
            connection.commit()
            total += len(batch)

    count = cursor.execute("SELECT COUNT(*) FROM pages").fetchone()[0]

    connection.close()

    print("\nIndex complete.")
    print("Unique pages indexed:", count)
    print("Index location:", INDEX_PATH)


if __name__ == "__main__":
    build_index()
