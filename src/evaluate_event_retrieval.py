
import json
from pathlib import Path
from collections import defaultdict

from src.fever_loader import iter_fever
from src.event_parser import parse_event
from src.title_search_v3 import rerank_pages

DATA_PATH = Path("data/raw/fever/paper_dev.jsonl")
OUTPUT_PATH = Path("results/event_retrieval_comparison.json")

SAMPLE_SIZE = 100
TOP_K = 10
CANDIDATE_LIMIT = 50
RRF_K = 60


def get_gold_pages(record):
    return {
        item["page"]
        for item in record["evidence"]
        if item.get("page")
    }


def retrieve_combined(original, normalized):
    scores = defaultdict(float)

    for query in (original, normalized):
        if not query:
            continue

        results = rerank_pages(
            query,
            candidate_limit=CANDIDATE_LIMIT,
            result_limit=TOP_K
        )

        for rank, item in enumerate(results, start=1):
            scores[item["page_id"]] += 1 / (RRF_K + rank)

    return [
        {"page_id": page, "score": score}
        for page, score in sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True
        )[:TOP_K]
    ]


def calculate_metrics(results, gold_sets):
    metrics = {}

    for k in (1, 5, 10):
        hits = sum(
            bool(
                gold.intersection(
                    item["page_id"] for item in results[i][:k]
                )
            )
            for i, gold in enumerate(gold_sets)
        )

        metrics[f"recall@{k}"] = hits / len(gold_sets)

    return metrics


def main():
    records = []

    for record in iter_fever(DATA_PATH):
        if (
            record["evidence"]
            and record["label"] in ("supports", "refutes")
        ):
            records.append(record)

        if len(records) >= SAMPLE_SIZE:
            break

    print(f"Claims selected: {len(records)}", flush=True)

    baseline_results = []
    event_results = []
    combined_results = []
    gold_sets = []
    details = []

    parse_failures = 0
    event_successes = 0

    for index, record in enumerate(records, start=1):
        claim = record["claim"]
        gold = get_gold_pages(record)

        baseline = rerank_pages(
            claim,
            candidate_limit=CANDIDATE_LIMIT,
            result_limit=TOP_K
        )

        normalized_claim = None

        try:
            parsed = parse_event(claim)
            normalized_claim = parsed.get("normalized_claim")

            if not isinstance(normalized_claim, str) or not normalized_claim.strip():
                raise ValueError("Empty normalized claim")

            event_successes += 1

        except Exception as exc:
            parse_failures += 1
            print(f"Parse failure {index}: {exc}", flush=True)

        event = (
            rerank_pages(
                normalized_claim,
                candidate_limit=CANDIDATE_LIMIT,
                result_limit=TOP_K
            )
            if normalized_claim else []
        )

        combined = retrieve_combined(claim, normalized_claim)

        baseline_results.append(baseline)
        event_results.append(event)
        combined_results.append(combined)
        gold_sets.append(gold)

        details.append({
            "claim_id": record["claim_id"],
            "claim": claim,
            "normalized_claim": normalized_claim,
            "gold_pages": sorted(gold),
            "baseline_top10": [x["page_id"] for x in baseline],
            "event_top10": [x["page_id"] for x in event],
            "combined_top10": [x["page_id"] for x in combined]
        })

        if index % 10 == 0:
            print(f"Evaluated: {index}/{len(records)}", flush=True)

    report = {
        "dataset": "FEVER dev",
        "claims_evaluated": len(records),
        "event_parse_successes": event_successes,
        "event_parse_failures": parse_failures,
        "metrics": {
            "baseline_original_claim": calculate_metrics(
                baseline_results, gold_sets
            ),
            "event_normalized_claim": calculate_metrics(
                event_results, gold_sets
            ),
            "combined_retrieval": calculate_metrics(
                combined_results, gold_sets
            )
        },
        "claim_details": details
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\nEVENT RETRIEVAL COMPARISON")
    print("-------------------------")
    print(f"Claims: {len(records)}")
    print(f"Parser successes: {event_successes}")
    print(f"Parser failures: {parse_failures}")

    for name, metrics in report["metrics"].items():
        print(f"\n{name}")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.2%}")

    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
