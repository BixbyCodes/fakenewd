"""Entity-aware Wikipedia title reranking with BM25 preservation."""

import math
import re
from collections import Counter

from src.title_search import search_pages

STOPWORDS = {
    "a", "an", "the", "is", "was", "were", "are",
    "be", "been", "being", "of", "in", "on", "at",
    "to", "for", "and", "or", "but", "by", "with",
    "from", "that", "this", "it", "as", "during",
    "released", "produced"
}


def tokenize(text):
    return [
        word for word in re.findall(r"[^\W_]+", text.lower(), flags=re.UNICODE)
        if word not in STOPWORDS and len(word) > 1
    ]


def rerank_pages(claim, candidate_limit=50, result_limit=10):
    if not claim or not claim.strip():
        return []

    if candidate_limit < 1 or result_limit < 1:
        raise ValueError("Limits must be positive.")

    claim_tokens = tokenize(claim)

    if not claim_tokens:
        return []

    candidates = search_pages(claim, limit=candidate_limit)

    if not candidates:
        return []

    title_tokens = [
        tokenize(item["page_id"].replace("_", " "))
        for item in candidates
    ]

    # Measure how distinctive each claim term is in this candidate pool.
    document_frequency = Counter()

    for tokens in title_tokens:
        document_frequency.update(set(tokens))

    total_candidates = len(candidates)

    weights = {
        token: math.log(
            (total_candidates + 1) /
            (document_frequency[token] + 1)
        ) + 1.0
        for token in set(claim_tokens)
    }

    total_weight = sum(weights.values())

    ranked = []

    for original_rank, (candidate, tokens) in enumerate(
        zip(candidates, title_tokens), start=1
    ):
        title_set = set(tokens)

        matched_weight = sum(
            weights[token]
            for token in set(claim_tokens)
            if token in title_set
        )

        weighted_coverage = (
            matched_weight / total_weight if total_weight else 0.0
        )

        # Reward exact contiguous phrases, without requiring them.
        phrase_score = 0.0

        for size in range(2, min(len(claim_tokens), 5) + 1):
            for start in range(len(claim_tokens) - size + 1):
                phrase = claim_tokens[start:start + size]

                for pos in range(len(tokens) - size + 1):
                    if tokens[pos:pos + size] == phrase:
                        phrase_score = max(
                            phrase_score, size / len(claim_tokens)
                        )

        # Keep the original BM25 rank influential.
        rank_score = 1.0 / original_rank

        final_score = (
            0.65 * rank_score
            + 0.25 * weighted_coverage
            + 0.10 * phrase_score
        )

        ranked.append({
            "page_id": candidate["page_id"],
            "score": round(final_score, 6),
            "weighted_coverage": round(weighted_coverage, 4),
            "phrase_score": round(phrase_score, 4),
            "baseline_rank": original_rank
        })

    ranked.sort(key=lambda item: item["score"], reverse=True)

    return ranked[:result_limit]


if __name__ == "__main__":
    claim = "Fox 2000 Pictures released the film Soul Food"

    print("Claim:", claim)
    print("\nV3 candidates:")

    for rank, result in enumerate(rerank_pages(claim), start=1):
        print(
            "{}. {} | score={} | BM25 rank={}".format(
                rank,
                result["page_id"],
                result["score"],
                result["baseline_rank"]
            )
        )
