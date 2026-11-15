
"""Compare original, normalized and combined FEVER retrieval."""

import json
from collections import defaultdict
from pathlib import Path

from src.event_parser import parse_event
from src.title_search_v3 import rerank_pages

DATASET = "data/raw/fever/paper_dev.jsonl"
OUTPUT = Path("results/normalized_retrieval.json")
K_VALUES = (1, 5, 10)


def get_gold_pages(record):
    return {
        item[2]
        for group in (record.get("evidence") or [])
        for item in group
        if len(item) >= 4 and item[2] is not None
    }


def build_queries(claim, parsed):
    entities = [
        entity["name"]
        for entity in parsed["entities"]
        if entity.get("name")
    ]

    return {
        "baseline": claim,
        "normalized": parsed["normalized_claim"] or claim,
        "combined": " ".join(
            part for part in (
                claim,
                parsed["event_summary"] or "",
                " ".join(entities),
            ) if part
        ),
    }


def evaluate(limit=10):
    totals = defaultdict(int)
    hits = defaultdict(lambda: defaultdict(int))
    skipped = 0
    details = []

    with open(DATASET, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)

            if record.get("label") not in ("SUPPORTS", "REFUTES"):
                continue

            gold_pages = get_gold_pages(record)

            if not gold_pages:
                skipped += 1
                continue

            claim = record["claim"]

            try:
                parsed = parse_event(claim)
                queries = build_queries(claim, parsed)
            except Exception as exc:
                print("Normalization failed:", record.get("id"), str(exc))
                continue

            claim_result = {
                "claim_id": record.get("id"),
                "claim": claim,
                "strategies": {},
            }

            for strategy, query in queries.items():
                results = rerank_pages(
                    query,
                    candidate_limit=50,
                    result_limit=10,
                )

                retrieved = [item["page_id"] for item in results]
                totals[strategy] += 1

                for k in K_VALUES:
                    if gold_pages.intersection(retrieved[:k]):
                        hits[strategy][k] += 1

                claim_result["strategies"][strategy] = {
                    "query": query,
                    "retrieved_pages": retrieved,
                    "gold_pages": sorted(gold_pages),
                }

            details.append(claim_result)
            print("Evaluated claims:", len(details))

            if len(details) >= limit:
                break

    summary = {}

    for strategy in ("baseline", "normalized", "combined"):
        summary[strategy] = {
            "claims": totals[strategy],
            "recall": {
                str(k): round(
                    hits[strategy][k] / totals[strategy] * 100, 2
                ) if totals[strategy] else 0
                for k in K_VALUES
            },
        }

    output = {
        "dataset": DATASET,
        "requested_claims": limit,
        "evaluated_claims": len(details),
        "skipped": skipped,
        "summary": summary,
        "details": details,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nNORMALIZED RETRIEVAL RESULTS")
    print(json.dumps(summary, indent=2))
    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    evaluate(limit=10)

