"""Evaluate automatic Wikipedia title retrieval on FEVER."""

import json
from collections import defaultdict

from src.title_search import search_pages

DATASET = "data/raw/fever/paper_dev.jsonl"


def evaluate(limit=1000):
    total = 0
    hits = defaultdict(int)
    skipped = 0

    with open(DATASET, encoding="utf-8") as handle:
        for line in handle:
            if total >= limit:
                break

            record = json.loads(line)

            if record.get("label") not in ("SUPPORTS", "REFUTES"):
                continue

            gold_pages = set()

            for group in record.get("evidence") or []:
                for item in group:
                    if len(item) >= 4 and item[2] is not None:
                        gold_pages.add(item[2])

            if not gold_pages:
                skipped += 1
                continue

            results = search_pages(record["claim"], limit=10)
            retrieved = [item["page_id"] for item in results]

            total += 1

            for k in (1, 5, 10):
                if gold_pages.intersection(retrieved[:k]):
                    hits[k] += 1

            if total % 100 == 0:
                print("Evaluated:", total)

    print("\nFEVER TITLE RETRIEVAL EVALUATION")
    print("--------------------------------")
    print("Claims evaluated:", total)
    print("Claims skipped:", skipped)

    for k in (1, 5, 10):
        recall = hits[k] / total * 100 if total else 0
        print("Recall@{}: {:.2f}% ({}/{})".format(
            k, recall, hits[k], total
        ))


if __name__ == "__main__":
    evaluate()
