"""Rerank Wikipedia title candidates using lexical and phrase matching."""

import re

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
        word for word in re.findall(r"[a-z0-9]+", text.lower())
        if word not in STOPWORDS and len(word) > 1
    ]


def normalize_title(title):
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def phrase_match(claim, title):
    """Check whether meaningful claim phrases occur in the title."""
    claim_tokens = tokenize(claim)
    normalized_title = normalize_title(title)

    phrases = [
        " ".join(claim_tokens[i:i + 2])
        for i in range(len(claim_tokens) - 1)
    ]

    if not phrases:
        return 0.0

    matches = sum(
        1 for phrase in phrases
        if phrase in normalized_title
    )

    return matches / len(phrases)


def rerank_pages(claim, candidate_limit=50, result_limit=10):
    if not claim or not claim.strip():
        return []

    if candidate_limit < 1 or result_limit < 1:
        raise ValueError("Limits must be positive.")

    claim_tokens = set(tokenize(claim))

    if not claim_tokens:
        return []

    candidates = search_pages(claim, limit=candidate_limit)
    ranked = []

    for original_rank, candidate in enumerate(candidates, start=1):
        page_id = candidate["page_id"]
        title_tokens = set(tokenize(page_id.replace("_", " ")))

        overlap = (
            len(claim_tokens & title_tokens) / len(claim_tokens)
        )

        phrase_score = phrase_match(claim, page_id.replace("_", " "))

        # Reciprocal rank score: higher is better.
        bm25_rank_score = 1.0 / original_rank

        final_score = (
            0.55 * overlap
            + 0.25 * phrase_score
            + 0.20 * bm25_rank_score
        )

        ranked.append({
            "page_id": page_id,
            "score": round(final_score, 6),
            "overlap": round(overlap, 4),
            "phrase_match": round(phrase_score, 4),
            "baseline_rank": original_rank
        })

    ranked.sort(key=lambda item: item["score"], reverse=True)

    return ranked[:result_limit]


if __name__ == "__main__":
    claim = "Fox 2000 Pictures released the film Soul Food"

    print("Claim:", claim)
    print("\nReranked candidates:")

    for rank, result in enumerate(rerank_pages(claim), start=1):
        print(
            "{}. {} | score={}".format(
                rank, result["page_id"], result["score"]
            )
        )
