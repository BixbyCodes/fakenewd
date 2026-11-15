"""Measure whether gold pages enter the BM25 top-50 candidate pool."""

import json
from collections import Counter
from pathlib import Path

from src.title_search import search_pages

DATASET = "data/raw/fever/paper_dev.jsonl"
OUTPUT = Path("results/candidate_coverage.json")


def evaluate(limit=1000):
    total = 0
    coverage = Counter()
    examples = []

    with open(DATASET, encoding="utf-8") as handle:
        for line in handle:
            if total >= limit:
                break

            record = json.loads(line)

            if record.get("label") not in ("SUPPORTS", "REFUTES"):
                continue

            gold = {
                item[2]
                for group in (record.get("evidence") or [])
                for item in group
                if len(item) >= 4 and item[2] is not None
            }

            if not gold:
                continue

            candidates = search_pages(record["claim"], limit=50)
            ranked = [item["page_id"] for item in candidates]

            positions = {
                page: ranked.index(page) + 1
                for page in gold if page in ranked
            }

            total += 1

            if positions:
                best_rank = min(positions.values())
                coverage["in_top50"] += 1

                if best_rank <= 10:
                    coverage["in_top10"] += 1
                else:
                    coverage["only_11_to_50"] += 1
            else:
                coverage["missing_from_top50"] += 1
                examples.append({
                    "claim": record["claim"],
                    "gold_pages": sorted(gold),
                    "top_candidates": ranked[:10]
                })

            if total % 100 == 0:
                print("Analyzed:", total)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(examples, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\nBM25 CANDIDATE COVERAGE")
    print("----------------------")
    print("Claims:", total)

    for key in (
        "in_top10",
        "only_11_to_50",
        "missing_from_top50"
    ):
        count = coverage[key]
        print("{}: {} ({:.2f}%)".format(
            key, count, count / total * 100 if total else 0
        ))

    print("\nSaved missing examples:", OUTPUT)


if __name__ == "__main__":
    evaluate()
