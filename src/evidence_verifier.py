"""Wikipedia evidence retrieval and LLM support/refutation assessment.

Pipeline for one submitted claim/article:
1. Candidate pages: V3 title reranking (src/title_search_v3.py) on the
   ORIGINAL claim text, plus exact/near title matches for extracted entity
   names. The LLM-normalized claim is never used as a replacement query.
2. Sentences: for each page, the lead sentence plus the sentences sharing
   the most content words with the claim.
3. Assessment: an LLM labels each sentence SUPPORTS, REFUTES or INSUFFICIENT
   with respect to the submitted claim, using only that sentence.

The labels are LLM assessments of retrieved text, not ground truth. The
corpus is the 2017 FEVER Wikipedia snapshot, so recent events are usually
not covered.
"""

import hashlib
import json
import logging
import re
from urllib.parse import quote

from src.config import EVIDENCE_CACHE, llm_model
from src.indexed_evidence_retriever import load_page
from src.llm_client import (
    JsonCache,
    LLMUnavailable,
    describe_error,
    get_client,
    parse_json_object,
)
from src.title_search import (
    EvidenceIndexUnavailable,
    index_status,
    page_exists,
    query_terms,
)
from src.title_search_v3 import rerank_pages

logger = logging.getLogger("eafnd.evidence")

MODEL = llm_model()
LABELS = ("SUPPORTS", "REFUTES", "INSUFFICIENT")
_cache = JsonCache(EVIDENCE_CACHE, "evidence_assessments")

FEVER_TOKENS = {
    "-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]",
    "-LCB-": "{", "-RCB-": "}", "-COLON-": ":",
}

VERIFIER_PROMPT = """
You are a cautious evidence verification assistant.

Compare the supplied claim with each supplied evidence sentence.

For each sentence, assign exactly one label:
SUPPORTS: The sentence provides direct factual support for the claim.
REFUTES: The sentence directly contradicts the claim.
INSUFFICIENT: The sentence does not establish either conclusion.

Rules:
- Do not use outside knowledge.
- Do not assume missing information.
- Similar topics are not necessarily supporting evidence.
- A sentence about a different entity with a similar name is INSUFFICIENT.
- Preserve the supplied page_id and sentence_id.
- Explain the comparison briefly.
- Return valid JSON with one key: results.
- results must be an array of objects with these keys:
  page_id, sentence_id, label, explanation.
"""


def clean_fever_text(text):
    """Turn FEVER/PTB tokens (-LRB- etc.) back into readable characters."""
    for token, char in FEVER_TOKENS.items():
        text = text.replace(token, char)
    text = re.sub(r"\s+([,.;:%)\]])", r"\1", text)
    text = re.sub(r"\s+('s|'re|'ve|'ll|'d|n't)\b", r"\1", text)
    text = text.replace("`` ", '"').replace(" ''", '"')
    return re.sub(r"([(\[])\s+", r"\1", text)


def page_title(page_id):
    return clean_fever_text(page_id).replace("_", " ")


def wikipedia_url(page_id):
    title = clean_fever_text(page_id).replace(" ", "_")
    return "https://en.wikipedia.org/wiki/" + quote(title, safe="_(),'-.")


def page_sentences(page_id):
    """All non-empty sentences of an indexed page as dicts."""
    page = load_page(page_id)
    if page is None:
        return []
    lines = page.get("lines", "")
    if not isinstance(lines, str):
        return []

    sentences = []
    for line in lines.splitlines():
        parts = line.split("\t", 2)
        if len(parts) < 2:
            continue
        try:
            sentence_id = int(parts[0])
        except ValueError:
            continue
        sentence = parts[1].strip()
        if sentence:
            sentences.append({"page_id": page_id, "sentence_id": sentence_id, "text": sentence})
    return sentences


def retrieve_page_sentences(page_id, limit=5):
    """First `limit` sentences of an indexed page (kept for compatibility)."""
    return page_sentences(page_id)[:limit]


def select_sentences(page_id, claim_text, per_page=3):
    """Lead sentence plus the sentences with the highest claim-term overlap."""
    sentences = page_sentences(page_id)
    if not sentences:
        return []
    claim_tokens = set(query_terms(claim_text))

    scored = []
    for item in sentences:
        tokens = set(query_terms(clean_fever_text(item["text"])))
        overlap = len(claim_tokens & tokens)
        scored.append((overlap, -item["sentence_id"], item))

    chosen = {sentences[0]["sentence_id"]: sentences[0]}
    for overlap, _, item in sorted(scored, key=lambda x: (x[0], x[1]), reverse=True):
        if len(chosen) >= per_page:
            break
        if overlap > 0:
            chosen.setdefault(item["sentence_id"], item)

    result = []
    for item in sorted(chosen.values(), key=lambda x: x["sentence_id"]):
        tokens = set(query_terms(clean_fever_text(item["text"])))
        result.append(dict(item, term_overlap=len(claim_tokens & tokens)))
    return result


def retrieval_query(claim_text, max_words=40):
    """Original text for short claims; the opening of long articles."""
    words = claim_text.split()
    if len(words) <= max_words:
        return claim_text
    first_line = claim_text.strip().splitlines()[0]
    if 3 <= len(first_line.split()) <= max_words:
        return first_line
    return " ".join(words[:max_words])


def retrieve_candidates(claim_text, entities=None, page_limit=6):
    """Candidate Wikipedia pages from the claim text and entity names."""
    candidates = []
    seen = set()

    def add(page_id, source, score=None, query=None):
        if page_id not in seen and len(candidates) < page_limit:
            seen.add(page_id)
            candidates.append({
                "page_id": page_id,
                "title": page_title(page_id),
                "url": wikipedia_url(page_id),
                "source": source,
                "score": score,
                "query": query,
            })

    # Exact title matches for extracted entity names are high-precision.
    for entity in (entities or [])[:6]:
        name = entity.get("name", "").strip() if isinstance(entity, dict) else ""
        if len(name) < 2 or len(name) > 80:
            continue
        page_id = name.replace(" ", "_")
        if page_exists(page_id):
            add(page_id, "entity title match", query=name)

    query = retrieval_query(claim_text)
    reserved = max(page_limit - len(candidates), 2)
    for item in rerank_pages(query, candidate_limit=50, result_limit=reserved + len(seen)):
        add(item["page_id"], "claim-text title search", item["score"], query)
        if len(candidates) >= page_limit:
            break
    return candidates


def _assessment_key(claim, evidence):
    payload = json.dumps(
        {"model": MODEL, "claim": claim,
         "evidence": [[e["page_id"], e["sentence_id"]] for e in evidence]},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_evidence(claim, evidence, use_cache=True):
    """Ask the LLM to label each evidence sentence against the claim."""
    if not evidence:
        return []

    key = _assessment_key(claim, evidence)
    if use_cache:
        hit = _cache.get(key)
        if isinstance(hit, list):
            return hit

    client = get_client()
    payload = [
        {"page_id": item["page_id"], "sentence_id": item["sentence_id"],
         "text": clean_fever_text(item["text"])}
        for item in evidence
    ]

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        reasoning_effort="low",
        max_completion_tokens=2500,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": VERIFIER_PROMPT},
            {"role": "user", "content": json.dumps({"claim": claim, "evidence": payload})},
        ],
    )

    data = parse_json_object(response.choices[0].message.content or "")
    if not isinstance(data.get("results"), list):
        raise ValueError("Invalid evidence verification response.")

    lookup = {(item["page_id"], item["sentence_id"]): item for item in evidence}
    verified = []
    for item in data["results"]:
        if not isinstance(item, dict):
            continue
        try:
            key_pair = (item.get("page_id"), int(item.get("sentence_id")))
        except (TypeError, ValueError):
            continue
        label = str(item.get("label", "")).strip().upper()
        if key_pair not in lookup or label not in LABELS:
            continue
        source = lookup.pop(key_pair)  # pop: ignore duplicate labels
        verified.append({
            "page_id": key_pair[0],
            "sentence_id": key_pair[1],
            "sentence": clean_fever_text(source["text"]),
            "label": label,
            "explanation": str(item.get("explanation", "")).strip(),
        })

    _cache.set(key, verified)
    return verified


def summarize_labels(results):
    counts = {label: 0 for label in LABELS}
    for item in results:
        if item.get("label") in counts:
            counts[item["label"]] += 1
    if counts["SUPPORTS"] and counts["REFUTES"]:
        direction = "CONFLICTING"
    elif counts["REFUTES"]:
        direction = "REFUTED"
    elif counts["SUPPORTS"]:
        direction = "SUPPORTED"
    else:
        direction = "INSUFFICIENT"
    return counts, direction


def analyze_evidence(claim, entities=None, page_limit=6, per_page=3,
                     llm_available=True, normalized_claim=None):
    """Retrieve and assess evidence. Never raises; reports a status instead.

    `normalized_claim` is accepted for backward compatibility but is not used
    as the retrieval query (normalization can drop retrieval details).
    """
    report = {
        "status": "error",
        "message": None,
        "claim_assessed": claim,
        "query": retrieval_query(claim),
        "candidate_pages": [],
        "evidence": [],
        "results": [],
        "evidence_count": 0,
        "verified_count": 0,
        "counts": {label: 0 for label in LABELS},
        "direction": None,
        "corpus": "FEVER 2017 Wikipedia snapshot (local SQLite index)",
    }

    status = index_status()
    if not status["available"]:
        report.update(status="index_unavailable", message=status["reason"])
        return report

    try:
        candidates = retrieve_candidates(claim, entities, page_limit)
        report["candidate_pages"] = candidates
        if not candidates:
            report.update(status="no_candidates",
                          message="No Wikipedia page titles matched the claim terms.")
            return report

        evidence = []
        for candidate in candidates:
            evidence.extend(select_sentences(candidate["page_id"], claim, per_page))
        report["evidence"] = [
            dict(item, text=clean_fever_text(item["text"]), title=page_title(item["page_id"]),
                 url=wikipedia_url(item["page_id"]))
            for item in evidence
        ]
        report["evidence_count"] = len(evidence)
        if not evidence:
            report.update(status="no_sentences",
                          message="Candidate pages were found but contained no usable sentences.")
            return report
    except EvidenceIndexUnavailable as exc:
        report.update(status="index_unavailable", message=str(exc))
        return report
    except Exception:
        logger.exception("Evidence retrieval failed")
        report.update(status="error", message="Evidence retrieval failed unexpectedly.")
        return report

    if not llm_available:
        report.update(status="retrieved_not_assessed",
                      message="Evidence was retrieved but not assessed because the LLM is unavailable.")
        return report

    try:
        results = verify_evidence(claim, evidence)
    except Exception as exc:
        if not isinstance(exc, LLMUnavailable):
            logger.warning("Evidence assessment failed: %s", type(exc).__name__)
        report.update(status="retrieved_not_assessed", message=describe_error(exc))
        return report

    for item in results:
        item["title"] = page_title(item["page_id"])
        item["url"] = wikipedia_url(item["page_id"])
    counts, direction = summarize_labels(results)
    report.update(
        status="completed",
        results=results,
        verified_count=len(results),
        counts=counts,
        direction=direction,
    )
    if len(results) < len(evidence):
        report["message"] = "{} of {} retrieved sentences received a valid label.".format(
            len(results), len(evidence))
    return report
