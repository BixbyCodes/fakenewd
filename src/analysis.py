"""End-to-end EvidenceAware analysis pipeline.

    input validation -> classifier -> (LLM assessment || event extraction)
    -> evidence retrieval + assessment -> rule-based interpretation

Each stage reports its own status. A failing optional stage (LLM, evidence
index) never stops the classifier result from being returned, and no stage
pretends to have run when it did not. The interpretation reports agreement
or disagreement between independent signals; it does not average them into a
single score and the LLM never overrides the trained classifier.
"""

import hashlib
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor

from src.config import (
    MAX_HEADLINE_CHARS,
    MAX_INPUT_CHARS,
    MIN_INPUT_CHARS,
    MIN_INPUT_WORDS,
    llm_model,
)
from src.ingestion import normalize_text
from src.llm_client import api_key_available, describe_error

logger = logging.getLogger("eafnd.analysis")

EVIDENCE_CLAIM_MAX_WORDS = 60


class InputError(ValueError):
    """Invalid user input. The message is safe to display."""


def validate_input(body, headline=""):
    """Normalize and validate the submitted text. Returns (text, headline)."""
    if not isinstance(body, str) or not isinstance(headline or "", str):
        raise InputError("Input must be text.")
    headline = normalize_text(headline or "")
    body = normalize_text(body)

    if len(headline) > MAX_HEADLINE_CHARS:
        raise InputError("The headline is longer than {} characters.".format(MAX_HEADLINE_CHARS))
    if not body and not headline:
        raise InputError("Please paste a claim or article to analyze.")
    if len(body) > MAX_INPUT_CHARS:
        raise InputError(
            "The article is {:,} characters long; the limit is {:,}. "
            "Please submit a shorter excerpt.".format(len(body), MAX_INPUT_CHARS)
        )

    if headline and body and not body.lower().startswith(headline.lower()):
        text = headline + "\n\n" + body
    else:
        text = body or headline

    if len(text) < MIN_INPUT_CHARS or len(text.split()) < MIN_INPUT_WORDS:
        raise InputError(
            "Please enter at least {} words ({} characters) so there is a claim to analyze.".format(
                MIN_INPUT_WORDS, MIN_INPUT_CHARS)
        )
    if not re.search(r"[^\W\d_]{2,}", text):
        raise InputError("The input does not appear to contain readable words.")
    return text, headline


def claim_for_evidence(text, headline=""):
    """The part of the submission that evidence is assessed against.

    Short submissions are used verbatim. For long articles the headline (or
    the opening sentences) is used, because sentence-level verification needs
    a specific claim. This is shown to the user.
    """
    words = text.split()
    if len(words) <= EVIDENCE_CLAIM_MAX_WORDS:
        return text
    if headline:
        return headline
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    claim = ""
    for sentence in sentences:
        candidate = (claim + " " + sentence).strip()
        if len(candidate.split()) > EVIDENCE_CLAIM_MAX_WORDS and claim:
            break
        claim = candidate
    return " ".join(claim.split()[:EVIDENCE_CLAIM_MAX_WORDS])


def component(status, message=None, seconds=None):
    return {"status": status, "message": message, "seconds": seconds}


def _timed(fn, *args, **kwargs):
    start = time.perf_counter()
    value = fn(*args, **kwargs)
    return value, round(time.perf_counter() - start, 2)


def run_classifier(text):
    from src.predict import predict_news
    try:
        result, seconds = _timed(predict_news, text)
        return result, component("ok", seconds=seconds)
    except Exception:
        logger.exception("Classifier failed")
        return None, component("error", "The trained classifier could not be loaded or run.")


def run_llm_assessment(text):
    from src.llm_features import assess_text
    try:
        result, seconds = _timed(assess_text, text)
        return result, component("ok", seconds=seconds)
    except Exception as exc:
        logger.warning("LLM assessment failed: %s", type(exc).__name__)
        return None, component("unavailable", describe_error(exc))


def run_event_extraction(text):
    from src.event_parser import parse_event
    try:
        result, seconds = _timed(parse_event, text)
        return result, component("ok", seconds=seconds)
    except Exception as exc:
        logger.warning("Event extraction failed: %s", type(exc).__name__)
        return None, component("unavailable", describe_error(exc))


def run_evidence(claim, entities, llm_available):
    from src.evidence_verifier import analyze_evidence
    report, seconds = _timed(analyze_evidence, claim, entities=entities,
                             llm_available=llm_available)
    status_map = {
        "completed": "ok",
        "index_unavailable": "unavailable",
        "retrieved_not_assessed": "partial",
        "no_candidates": "ok",
        "no_sentences": "ok",
    }
    return report, component(status_map.get(report["status"], "error"),
                             report.get("message"), seconds)


def capability_status():
    """Which components can run in this environment."""
    from src.predict import models_available
    from src.title_search import index_status

    evidence = index_status()
    return {
        "classifier": models_available(),
        "llm": api_key_available(),
        "llm_model": llm_model(),
        "evidence_index": evidence["available"],
        "evidence_reason": evidence["reason"],
    }


def run_analysis(body, headline="", progress=None):
    """Run the full pipeline. Raises InputError only for invalid input."""
    text, headline = validate_input(body, headline)

    def step(message):
        if progress:
            progress(message)

    llm_ready = api_key_available()
    report = {
        "input": {
            "text": text,
            "headline": headline or None,
            "characters": len(text),
            "words": len(text.split()),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        },
        "components": {},
    }

    step("Running the trained classifier")
    report["classifier"], report["components"]["classifier"] = run_classifier(text)

    if llm_ready:
        step("Requesting LLM assessment and event extraction")
        with ThreadPoolExecutor(max_workers=2) as pool:
            llm_future = pool.submit(run_llm_assessment, text)
            event_future = pool.submit(run_event_extraction, text)
            report["llm"], report["components"]["llm"] = llm_future.result()
            report["event"], report["components"]["event"] = event_future.result()
    else:
        missing = component("unavailable", describe_error(_missing_key()))
        report["llm"], report["components"]["llm"] = None, missing
        report["event"], report["components"]["event"] = None, dict(missing)

    step("Retrieving and assessing Wikipedia evidence")
    entities = (report["event"] or {}).get("entities") or []
    claim = claim_for_evidence(text, headline)
    # If both LLM calls just failed (bad key, outage), skip a third call.
    llm_working = llm_ready and (
        report["components"]["llm"]["status"] == "ok"
        or report["components"]["event"]["status"] == "ok"
    )
    report["evidence"], report["components"]["evidence"] = run_evidence(
        claim, entities, llm_working
    )

    report["interpretation"] = interpret(report["classifier"], report["llm"], report["evidence"])
    return report


def _missing_key():
    from src.llm_client import LLMUnavailable
    return LLMUnavailable(
        "GROQ_API_KEY is not configured, so LLM-based components are disabled."
    )


def interpret(classifier, llm, evidence):
    """Combine independent signals into a cautious, rule-based summary."""
    points = []
    reasons = []
    tone = "neutral"

    prediction = classifier["prediction"] if classifier else None
    direction = evidence.get("direction") if evidence and evidence.get("status") == "completed" else None
    counts = (evidence or {}).get("counts") or {}

    if classifier:
        points.append(
            "Classifier: the ISOT-trained {} predicts {} ({:.1f}% model probability for that "
            "class). This reflects writing patterns learned from ISOT, not a fact check.".format(
                classifier["model"], prediction, classifier["confidence"])
        )
        if classifier["low_confidence"]:
            reasons.append("The classifier's probability is close to 50%.")
        if not classifier["model_agreement"].startswith("3/3"):
            reasons.append("The three ISOT classifiers do not all agree ({}).".format(
                classifier["model_agreement"]))
        for note in classifier["reliability_notes"]:
            if note.startswith("Short input"):
                reasons.append("The input is much shorter than the articles the classifier was trained on.")
            elif "Reuters" in note:
                reasons.append("The '(Reuters)' dateline strongly biases this classifier towards REAL.")
    else:
        reasons.append("The trained classifier did not produce a result.")

    status = (evidence or {}).get("status")
    if direction == "REFUTED":
        points.append("Evidence: {} retrieved Wikipedia sentence(s) were assessed by the LLM as "
                      "contradicting the claim.".format(counts.get("REFUTES", 0)))
    elif direction == "SUPPORTED":
        points.append("Evidence: {} retrieved Wikipedia sentence(s) were assessed by the LLM as "
                      "supporting the claim.".format(counts.get("SUPPORTS", 0)))
    elif direction == "CONFLICTING":
        points.append("Evidence: retrieved sentences were assessed as both supporting ({}) and "
                      "contradicting ({}) the claim.".format(counts.get("SUPPORTS", 0),
                                                            counts.get("REFUTES", 0)))
        reasons.append("Retrieved evidence is conflicting.")
    elif direction == "INSUFFICIENT":
        points.append("Evidence: none of the retrieved sentences was assessed as directly "
                      "supporting or contradicting the claim.")
        reasons.append("Retrieved evidence is insufficient to support or contradict the claim.")
    elif status in ("no_candidates", "no_sentences"):
        points.append("Evidence: no relevant Wikipedia evidence was retrieved.")
        reasons.append("No evidence was retrieved.")
    elif status == "retrieved_not_assessed":
        points.append("Evidence: sentences were retrieved but not assessed ({})".format(
            evidence.get("message") or "LLM unavailable"))
        reasons.append("Evidence was not assessed.")
    else:
        points.append("Evidence: evidence verification did not run in this environment.")
        reasons.append("Evidence verification was unavailable.")

    if llm:
        verdict = llm["verdict"].upper()
        if verdict == "UNCERTAIN":
            relation = "is uncertain"
            reasons.append("The LLM assessment is uncertain.")
        elif prediction and verdict == prediction:
            relation = "agrees with the classifier"
        elif prediction:
            relation = "differs from the classifier"
            reasons.append("The LLM assessment differs from the classifier prediction.")
        else:
            relation = "was produced without a classifier result"
        points.append("LLM assessment: {} rates the text as {} and {}. The LLM sees only the "
                      "submitted text and does not override the classifier.".format(
                          llm.get("model", "The LLM"), verdict, relation))

    if prediction == "FAKE" and direction == "REFUTED":
        headline = "Classifier predicts FAKE and retrieved evidence contradicts the claim"
        tone = "refutes"
    elif prediction == "REAL" and direction == "SUPPORTED":
        headline = "Classifier predicts REAL and retrieved evidence supports the claim"
        tone = "supports"
    elif prediction == "REAL" and direction == "REFUTED":
        headline = "Disagreement: classifier predicts REAL but retrieved evidence contradicts the claim"
        tone = "conflict"
        reasons.insert(0, "The classifier and the retrieved evidence point in opposite directions.")
    elif prediction == "FAKE" and direction == "SUPPORTED":
        headline = "Disagreement: classifier predicts FAKE but retrieved evidence supports the claim"
        tone = "conflict"
        reasons.insert(0, "The classifier and the retrieved evidence point in opposite directions.")
    elif direction == "CONFLICTING":
        headline = "Retrieved evidence is conflicting; classifier predicts {}".format(prediction or "n/a")
        tone = "conflict"
    elif prediction:
        if direction == "INSUFFICIENT" or status in ("no_candidates", "no_sentences"):
            headline = "Classifier predicts {}; retrieved evidence is insufficient".format(prediction)
        else:
            headline = "Classifier predicts {}; evidence was not checked".format(prediction)
        tone = "uncertain"
    else:
        headline = "No classifier result is available"
        tone = "uncertain"

    seen = set()
    reasons = [r for r in reasons if not (r in seen or seen.add(r))]
    return {
        "headline": headline,
        "tone": tone,
        "points": points,
        "human_review": bool(reasons),
        "review_level": "strong" if tone == "conflict" else ("recommended" if reasons else "routine"),
        "review_reasons": reasons,
    }
