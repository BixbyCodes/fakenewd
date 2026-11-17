"""Wikipedia evidence retrieval and LLM support/refutation assessment.

Two interchangeable retrieval backends feed one assessment step:

* ``local_index``    - the FEVER 2017 Wikipedia snapshot in a local SQLite
  index (data/processed/wiki_pages.sqlite). Used when the index is installed.
* ``wikipedia_api``  - the public MediaWiki API (src/wikipedia_api.py), the
  free fallback used on deployments without the multi-GB index.

``EAFND_EVIDENCE_BACKEND`` can force ``local``, ``wikipedia_api`` or ``off``
(default ``auto``: local index if present, otherwise the public API).

For one submitted claim:
1. Candidate pages and sentences are retrieved from the ORIGINAL claim text,
   enriched (never replaced) with extracted entity names.
2. A few relevant sentences are sent to the LLM, which labels each one
   SUPPORTS, REFUTES or INSUFFICIENT with respect to the claim and must quote
   the deciding phrase for SUPPORTS/REFUTES. Labels whose quote is not in the
   sentence are downgraded to INSUFFICIENT.

The labels are LLM assessments of retrieved text, not ground truth.
"""

import hashlib
import json
import logging
import os
import re
from urllib.parse import quote

from src import wikipedia_api
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
from src.wikipedia_api import WikipediaUnavailable

logger = logging.getLogger("eafnd.evidence")

MODEL = llm_model()
LABELS = ("SUPPORTS", "REFUTES", "INSUFFICIENT")
DECISIVE = ("SUPPORTS", "REFUTES")
PROMPT_VERSION = "v2"
_cache = JsonCache(EVIDENCE_CACHE, "evidence_assessments")

BACKEND_LOCAL = "local_index"
BACKEND_API = "wikipedia_api"
BACKEND_NAMES = {
    BACKEND_LOCAL: "FEVER 2017 Wikipedia snapshot (local SQLite index)",
    BACKEND_API: "English Wikipedia, current text (public MediaWiki API)",
}

FEVER_TOKENS = {
    "-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]",
    "-LCB-": "{", "-RCB-": "}", "-COLON-": ":",
}

VERIFIER_PROMPT = """
You are a cautious evidence verification assistant in a fact-checking research system.

You receive a CLAIM and a list of EVIDENCE sentences taken from Wikipedia. Each item
has page_title (the Wikipedia article the sentence comes from; pronouns such as "he",
"she", "it" or "the company" usually refer to that article's subject).

Judge every sentence on its own against the claim and assign exactly one label:

SUPPORTS: the sentence itself states, or directly entails, the key facts of the claim.
REFUTES: the sentence itself states something that cannot be true if the claim is true,
  about the same entity and the same attribute (for example a different number, ordinal,
  date, place, person or amount, or an explicit negation of what the claim asserts).
INSUFFICIENT: everything else, including a different entity or attribute, background or
  related information, a sentence that only shares names or topic with the claim, partial
  overlap that does not settle the claim, and sentences too vague to decide.

Rules:
- Use only the sentence and its page_title. Do not use outside knowledge and do not
  assume missing information.
- Related or irrelevant is NOT contradiction. If the sentence is about a different
  person, event or attribute than the claim, the label is INSUFFICIENT.
- Before choosing REFUTES or SUPPORTS, compare the specifics (numbers, ordinals, dates,
  names, places, quantities) of the claim with those of the sentence.
- If the claim has several parts, SUPPORTS requires the sentence to cover the essential
  parts. A sentence that explicitly contradicts an essential part is REFUTES; a sentence
  confirming only one part is INSUFFICIENT.
- For SUPPORTS or REFUTES, evidence_span must be an exact quotation (copied word for
  word, at most 200 characters) from the sentence that decides the label. For
  INSUFFICIENT use an empty string.
- explanation: one short sentence (at most 40 words) about what the sentence actually
  says relative to the claim. Do not invent facts, quotations or sources.
- Return one JSON object with one key, results: an array with one object per input
  item, each with exactly these keys: page_id, sentence_id, label, evidence_span,
  explanation. Preserve page_id and sentence_id exactly as given.
"""


def clean_fever_text(text):
    """Turn FEVER/PTB tokens (-LRB- etc.) back into readable characters."""
    for token, char in FEVER_TOKENS.items():
        text = text.replace(token, char)
    text = re.sub(r"\s+([,.;:%)\]])", r"\1", text)
    text = re.sub(r"\s+('s|'re|'ve|'ll|'d|n't)\b", r"\1", text)
    text = text.replace("`` ", '"').replace(" ''", '"')
    return re.sub(r"([(\[])\s+", r"\1", text)


def display_text(item):
    """Sentence text for display and assessment.

    Sentences from the public API are already clean (``raw``); FEVER
    sentences carry PTB tokenisation artefacts that are cleaned up.
    """
    text = item.get("text", "")
    return text if item.get("raw") else clean_fever_text(text)


def page_title(page_id):
    return clean_fever_text(page_id).replace("_", " ")


def wikipedia_url(page_id):
    title = clean_fever_text(page_id).replace(" ", "_")
    return "https://en.wikipedia.org/wiki/" + quote(title, safe="_(),'-.")


# ------------------------------------------------------------ local backend

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
    """Candidate Wikipedia pages (local index) from the claim text and entity names."""
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


def _retrieve_local(claim, entities, page_limit, per_page):
    candidates = retrieve_candidates(claim, entities, page_limit)
    evidence = []
    for candidate in candidates:
        evidence.extend(select_sentences(candidate["page_id"], claim, per_page))
    return {"candidates": candidates, "evidence": evidence, "queries": [retrieval_query(claim)]}


# -------------------------------------------------------- public API backend

def _retrieve_api(claim, entities, normalized_claim, page_limit):
    found = wikipedia_api.retrieve_evidence(
        claim, entities=entities, normalized_claim=normalized_claim, max_pages=page_limit)
    return {"candidates": found["pages"], "evidence": found["sentences"],
            "queries": found["queries"]}


# --------------------------------------------------------- backend selection

def evidence_backend_mode():
    return os.getenv("EAFND_EVIDENCE_BACKEND", "auto").strip().lower()


def backend_status():
    """Which backend would be used. Never raises and makes no network request."""
    mode = evidence_backend_mode()
    local = index_status()
    if mode in ("auto", "local") and local["available"]:
        return {"backend": BACKEND_LOCAL, "available": True, "reason": None,
                "label": "local FEVER index"}
    if mode in ("auto", "wikipedia_api", "api", "wikipedia"):
        return {"backend": BACKEND_API, "available": True,
                "reason": None if mode != "auto" else local["reason"],
                "label": "live Wikipedia API"}
    if mode == "off":
        return {"backend": None, "available": False,
                "reason": "Evidence retrieval is disabled by configuration.", "label": "disabled"}
    return {"backend": None, "available": False, "reason": local["reason"], "label": "unavailable"}


# --------------------------------------------------------------- assessment

def _normalize_quote(text):
    return re.sub(r"\s+", " ", re.sub(r"[‘’]", "'",
                  re.sub(r"[“”]", '"', text or ""))).strip().lower()


def _assessment_key(claim, payload):
    body = json.dumps(
        {"model": MODEL, "prompt": PROMPT_VERSION, "claim": claim,
         "evidence": [[p["page_id"], p["sentence_id"],
                       hashlib.sha256(p["text"].encode("utf-8")).hexdigest()[:16]]
                      for p in payload]},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def verify_evidence(claim, evidence, use_cache=True):
    """Ask the LLM to label each evidence sentence against the claim."""
    if not evidence:
        return []

    payload = [
        {"page_id": item["page_id"], "page_title": item.get("title") or page_title(item["page_id"]),
         "sentence_id": item["sentence_id"], "text": display_text(item)}
        for item in evidence
    ]

    key = _assessment_key(claim, payload)
    if use_cache:
        hit = _cache.get(key)
        if isinstance(hit, list):
            return hit

    client = get_client()
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        reasoning_effort="low",
        max_completion_tokens=3500,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": VERIFIER_PROMPT},
            {"role": "user", "content": json.dumps({"claim": claim, "evidence": payload},
                                                   ensure_ascii=False)},
        ],
    )

    data = parse_json_object(response.choices[0].message.content or "")
    if not isinstance(data.get("results"), list):
        raise ValueError("Invalid evidence verification response.")

    lookup = {(p["page_id"], p["sentence_id"]): p for p in payload}
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
        explanation = str(item.get("explanation") or "").strip()
        span = str(item.get("evidence_span") or "").strip()
        downgraded = False

        if label in DECISIVE:
            # A decisive label must be grounded in a quotation that really is
            # part of the retrieved sentence; otherwise do not trust it.
            if not span or _normalize_quote(span) not in _normalize_quote(source["text"]):
                label, downgraded = "INSUFFICIENT", True
                explanation = ("The model's decisive label was not backed by a quotation from "
                               "the sentence, so it is treated as insufficient. " + explanation).strip()
                span = ""
        else:
            span = ""

        verified.append({
            "page_id": key_pair[0],
            "sentence_id": key_pair[1],
            "sentence": source["text"],
            "label": label,
            "evidence_span": span,
            "explanation": explanation,
            "downgraded": downgraded,
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


def _empty_report(claim):
    return {
        "status": "error",
        "message": None,
        "claim_assessed": claim,
        "query": retrieval_query(claim),
        "queries": [],
        "backend": None,
        "candidate_pages": [],
        "evidence": [],
        "results": [],
        "evidence_count": 0,
        "verified_count": 0,
        "counts": {label: 0 for label in LABELS},
        "direction": None,
        "corpus": None,
    }


def _retrieve(claim, entities, normalized_claim, page_limit, per_page):
    """Run the selected backend. Returns (backend, retrieved dict, note).

    Raises EvidenceIndexUnavailable / WikipediaUnavailable when no backend
    can be used.
    """
    choice = backend_status()
    if not choice["available"]:
        raise EvidenceIndexUnavailable(choice["reason"] or "No evidence backend is available.")

    note = None
    if choice["backend"] == BACKEND_LOCAL:
        try:
            return BACKEND_LOCAL, _retrieve_local(claim, entities, page_limit, per_page), None
        except EvidenceIndexUnavailable as exc:
            if evidence_backend_mode() != "auto":
                raise
            note = "The local index failed ({}); the public Wikipedia API was used.".format(exc)
            logger.warning("Local evidence index failed, falling back to the API")

    return BACKEND_API, _retrieve_api(claim, entities, normalized_claim, page_limit), note


def analyze_evidence(claim, entities=None, page_limit=6, per_page=3,
                     llm_available=True, normalized_claim=None):
    """Retrieve and assess evidence. Never raises; reports a status instead.

    The original `claim` is always what evidence is assessed against.
    `normalized_claim` and `entities` only enrich the Wikipedia search queries.
    """
    report = _empty_report(claim)

    try:
        backend, retrieved, note = _retrieve(claim, entities, normalized_claim, page_limit, per_page)
    except EvidenceIndexUnavailable as exc:
        report.update(status="index_unavailable", message=str(exc))
        return report
    except WikipediaUnavailable as exc:
        local_reason = index_status()["reason"]
        message = str(exc)
        if local_reason and evidence_backend_mode() == "auto":
            message += " (The local FEVER index is also unavailable: {}.)".format(local_reason.rstrip("."))
        report.update(status="retrieval_unavailable", backend=BACKEND_API,
                      corpus=BACKEND_NAMES[BACKEND_API], message=message)
        return report
    except Exception:
        logger.exception("Evidence retrieval failed")
        report.update(status="error", message="Evidence retrieval failed unexpectedly.")
        return report

    candidates, evidence = retrieved["candidates"], retrieved["evidence"]
    report.update(backend=backend, corpus=BACKEND_NAMES[backend], queries=retrieved["queries"],
                  candidate_pages=candidates, evidence_count=len(evidence))
    if note:
        report["message"] = note

    evidence = [dict(item, text=display_text(item), raw=True,
                     title=item.get("title") or page_title(item["page_id"]),
                     url=item.get("url") or wikipedia_url(item["page_id"]))
                for item in evidence]
    report["evidence"] = evidence

    if not candidates:
        report.update(status="no_candidates",
                      message="No Wikipedia page matched the claim terms.")
        return report
    if not evidence:
        report.update(status="no_sentences",
                      message="Pages were found, but no sentence was relevant enough to the "
                              "claim to be assessed.")
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

    by_key = {(item["page_id"], item["sentence_id"]): item for item in evidence}
    for item in results:
        source = by_key.get((item["page_id"], item["sentence_id"]), {})
        item["title"] = source.get("title") or page_title(item["page_id"])
        item["url"] = source.get("url") or wikipedia_url(item["page_id"])
    counts, direction = summarize_labels(results)
    report.update(
        status="completed",
        results=results,
        verified_count=len(results),
        counts=counts,
        direction=direction,
    )
    messages = [note] if note else []
    if len(results) < len(evidence):
        messages.append("{} of {} retrieved sentences received a valid label.".format(
            len(results), len(evidence)))
    if any(item.get("downgraded") for item in results):
        messages.append("Some decisive labels were downgraded to INSUFFICIENT because the model "
                        "did not quote the deciding text.")
    report["message"] = " ".join(messages) or None
    return report
