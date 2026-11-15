"""Compare baseline and V3 retrieval errors on the same FEVER claims."""

import json
from pathlib import Path

from src.title_search import search_pages
from src.title_search_v3 import rerank_pages

DATASET = "data/raw/fever/paper_dev.jsonl"
OUTPUT = Path("results/reranking_errors_v3.json")


def evaluate(limit=1000):
    total = 0
    categories = {
        "both_success": [],
        "baseline_only": [],
        "v3_only": [],
        "both_fail": [],
    }

    with open(DATASET, encoding="utf-8") as handle:
        for line in handle:
            if total >= limit:
                break

            record = json.loads(line)

            if record.get("label") not in ("SUPPORTS", "REFUTES"):
                continue

            gold_pages = {
                item[2]
                for group in (record.get("evidence") or [])
                for item in group
                if len(item) >= 4 and item[2] is not None
            }

            if not gold_pages:
                continue

            claim = record["claim"]

            baseline = search_pages(claim, limit=10)
            v3 = rerank_pages(
                claim,
                candidate_limit=50,
                result_limit=10
            )

            baseline_pages = [item["page_id"] for item in baseline]
            v3_pages = [item["page_id"] for item in v3]

            baseline_hit = bool(gold_pages.intersection(baseline_pages))
            v3_hit = bool(gold_pages.intersection(v3_pages))

            if baseline_hit and v3_hit:
                category = "both_success"
            elif baseline_hit:
                category = "baseline_only"
            elif v3_hit:
                category = "v3_only"
            else:
                category = "both_fail"

            categories[category].append({
                "claim_id": record["id"],
                "claim": claim,
                "label": record["label"],
                "gold_pages": sorted(gold_pages),
                "baseline_top10": baseline_pages,
                "v3_top10": v3_pages,
            })

            total += 1

            if total % 100 == 0:
                print("Analyzed:", total)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT, "w", encoding="utf-8") as handle:
        json.dump(categories, handle, indent=2, ensure_ascii=False)

    print("\nV1 vs V3 ERROR ANALYSIS")
    print("-----------------------")
    print("Claims analyzed:", total)

    for name, records in categories.items():
        percentage = len(records) / total * 100 if total else 0
        print("{}: {} ({:.2f}%)".format(
            name, len(records), percentage
        ))

    print("\nSaved:", OUTPUT)

    for name in ("baseline_only", "v3_only", "both_fail"):
        print("\n--- {} (first 3) ---".format(name))

        for item in categories[name][:3]:
            print("\nClaim:", item["claim"])
            print("Gold:", item["gold_pages"])
            print("V1:", item["baseline_top10"][:5])
            print("V3:", item["v3_top10"][:5])


if __name__ == "__main__":
    evaluate()
