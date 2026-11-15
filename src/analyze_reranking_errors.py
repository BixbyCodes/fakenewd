"""Find claims harmed by reranking v2."""

import json

from src.title_search import search_pages
from src.title_search_v2 import rerank_pages

DATASET = "data/raw/fever/paper_dev.jsonl"


def get_gold_pages(record):
    return {
        item[2]
        for group in (record.get("evidence") or [])
        for item in group
        if len(item) >= 4 and item[2] is not None
    }


def evaluate_errors(limit=1000):
    analyzed = 0
    errors = []

    with open(DATASET, encoding="utf-8") as handle:
        for line in handle:
            if analyzed >= limit:
                break

            record = json.loads(line)

            if record.get("label") not in ("SUPPORTS", "REFUTES"):
                continue

            gold = get_gold_pages(record)

            if not gold:
                continue

            baseline = [
                x["page_id"]
                for x in search_pages(record["claim"], limit=10)
            ]

            improved = [
                x["page_id"]
                for x in rerank_pages(
                    record["claim"],
                    candidate_limit=50,
                    result_limit=10
                )
            ]

            analyzed += 1

            baseline_hit = bool(gold.intersection(baseline))
            v2_hit = bool(gold.intersection(improved))

            if baseline_hit and not v2_hit:
                errors.append({
                    "claim_id": record["id"],
                    "claim": record["claim"],
                    "gold_pages": sorted(gold),
                    "baseline_results": baseline,
                    "v2_results": improved
                })

    output = "results/reranking_errors.json"

    with open(output, "w", encoding="utf-8") as handle:
        json.dump(errors, handle, indent=2, ensure_ascii=False)

    print("Claims analyzed:", analyzed)
    print("Baseline-success / V2-failure cases:", len(errors))
    print("Saved:", output)

    for item in errors[:5]:
        print("\nClaim:", item["claim"])
        print("Gold pages:", item["gold_pages"])
        print("Baseline:", item["baseline_results"][:5])
        print("V2:", item["v2_results"][:5])


if __name__ == "__main__":
    evaluate_errors()
