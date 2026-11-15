"""Evaluate exact recovery of FEVER gold evidence from the Wikipedia index."""

import json
import sqlite3
from pathlib import Path

from src.evidence_retriever import extract_sentence

DATASET = Path("data/raw/fever/paper_dev.jsonl")
INDEX = Path("data/processed/wiki_pages.sqlite")


def evaluate(limit=1000):
    connection = sqlite3.connect(str(INDEX))
    cursor = connection.cursor()

    claims = 0
    references = 0
    recovered = 0
    missing_pages = 0
    missing_sentences = 0

    with DATASET.open(encoding="utf-8") as handle:
        for line in handle:
            if claims >= limit:
                break

            record = json.loads(line)

            if record.get("label") not in ("SUPPORTS", "REFUTES"):
                continue

            claims += 1

            for group in record.get("evidence") or []:
                for item in group:
                    if len(item) < 4:
                        continue

                    _, _, page_id, sentence_id = item

                    if page_id is None or sentence_id is None:
                        continue

                    references += 1

                    row = cursor.execute(
                        "SELECT shard_path, byte_offset FROM pages "
                        "WHERE page_id = ?",
                        (page_id,)
                    ).fetchone()

                    if row is None:
                        missing_pages += 1
                        continue

                    shard_path, offset = row

                    with open(shard_path, "rb") as shard:
                        shard.seek(offset)
                        page = json.loads(shard.readline())

                    sentence = extract_sentence(page, sentence_id)

                    if sentence:
                        recovered += 1
                    else:
                        missing_sentences += 1

    connection.close()

    rate = recovered / references * 100 if references else 0

    print("\nFEVER EVIDENCE RECOVERY")
    print("-----------------------")
    print("Claims evaluated:", claims)
    print("Evidence references:", references)
    print("Successfully recovered:", recovered)
    print("Missing pages:", missing_pages)
    print("Missing sentences:", missing_sentences)
    print("Recovery rate: {:.2f}%".format(rate))


if __name__ == "__main__":
    evaluate()
