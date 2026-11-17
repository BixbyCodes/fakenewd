"""Live evidence retrieval from English Wikipedia via the public MediaWiki API.

Used when the local FEVER SQLite index is not installed (for example on
Streamlit Community Cloud). Only the Python standard library is used.

Steps for one claim (see `retrieve_evidence`):
1. Build a few search queries from the ORIGINAL claim, extracted entity names
   and (only to enrich the search) the LLM-normalized claim.
2. Search Wikipedia (generator=search) and get page intro extracts in the same
   request; skip disambiguation pages; fuse the rankings of all queries.
3. For the top pages fetch the plain-text article (truncated locally).
4. Split the text into sentences, drop headings and near-duplicates.
5. Rank sentences by IDF-weighted overlap with the original claim and keep a
   small number, so that the LLM sees a few relevant sentences and never a
   whole article.

Politeness: descriptive User-Agent, timeouts, one bounded retry on 429/503,
at most a handful of requests per claim, and an in-process TTL cache.
The text returned is Wikipedia's *current* content, not the 2017 FEVER
snapshot, and is not independently verified.
"""

import json
import logging
import math
import os
import re
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict

logger = logging.getLogger("eafnd.wikipedia")

DEFAULT_API_URL = "https://en.wikipedia.org/w/api.php"
DEFAULT_USER_AGENT = (
    "EvidenceAware/1.0 (research prototype; https://github.com/BixbyCodes/fakenewd)"
)
REQUEST_TIMEOUT_SECONDS = 8
MAX_RESPONSE_BYTES = 3 * 1024 * 1024
MAX_RETRY_WAIT_SECONDS = 3.0
CACHE_TTL_SECONDS = 3600
CACHE_MAX_ENTRIES = 256

MAX_QUERY_CHARS = 300
MAX_QUERIES = 4
SEARCH_RESULTS_CLAIM = 5
SEARCH_RESULTS_ENTITY = 3
MAX_ARTICLE_CHARS = 20000  # of the plain-text article kept per deep page
MIN_SENTENCE_CHARS = 25
MAX_SENTENCE_CHARS = 700
MIN_RELEVANCE = 0.15

SKIP_ENTITY_TYPES = {
    "date", "time", "number", "quantity", "percent", "percentage", "money",
    "currency", "duration", "ordinal", "year",
}

STOPWORDS = {
    "a", "an", "the", "is", "was", "were", "are", "be", "been", "being", "of", "in",
    "on", "at", "to", "for", "and", "or", "but", "by", "with", "from", "that", "this",
    "it", "as", "during", "has", "have", "had", "his", "her", "its", "their", "he",
    "she", "they", "him", "them", "who", "which", "than", "then", "also", "not", "no",
    "there", "into", "about", "over", "after", "before", "will", "would", "can",
    "could", "may", "might", "said", "says", "new", "one",
}


class WikipediaUnavailable(RuntimeError):
    """The Wikipedia API could not be used. The message is safe to display."""


# --------------------------------------------------------------------- cache

class _TTLCache:
    def __init__(self, maxsize, ttl):
        self.maxsize = maxsize
        self.ttl = ttl
        self._data = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            stamp, value = item
            if time.monotonic() - stamp > self.ttl:
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return value

    def set(self, key, value):
        with self._lock:
            self._data[key] = (time.monotonic(), value)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def clear(self):
        with self._lock:
            self._data.clear()


_CACHE = _TTLCache(CACHE_MAX_ENTRIES, CACHE_TTL_SECONDS)


def clear_cache():
    _CACHE.clear()


# ---------------------------------------------------------------------- HTTP

def _retry_after(error):
    try:
        wait = float(error.headers.get("Retry-After", "1"))
    except (TypeError, ValueError, AttributeError):
        wait = 1.0
    return max(0.2, min(wait, MAX_RETRY_WAIT_SECONDS))


def _http_get(params):
    """One GET against the MediaWiki API; raises WikipediaUnavailable."""
    query = dict(params, format="json", formatversion=2)
    url = os.getenv("WIKIPEDIA_API_URL", DEFAULT_API_URL) + "?" + urllib.parse.urlencode(query)
    request = urllib.request.Request(url, headers={
        "User-Agent": os.getenv("WIKIPEDIA_USER_AGENT", DEFAULT_USER_AGENT),
        "Accept": "application/json",
    })

    for attempt in range(2):
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                payload = response.read(MAX_RESPONSE_BYTES + 1)
            if len(payload) > MAX_RESPONSE_BYTES:
                raise WikipediaUnavailable("The Wikipedia API response was unexpectedly large.")
            data = json.loads(payload.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 503) and attempt == 0:
                time.sleep(_retry_after(exc))
                continue
            if exc.code == 429:
                raise WikipediaUnavailable(
                    "The Wikipedia API rate limit was reached. Please try again shortly.") from exc
            raise WikipediaUnavailable(
                "The Wikipedia API returned an error (HTTP {}).".format(exc.code)) from exc
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, OSError) as exc:
            raise WikipediaUnavailable("Could not reach the Wikipedia API.") from exc
        except ValueError as exc:
            raise WikipediaUnavailable("The Wikipedia API returned an unreadable response.") from exc

        if not isinstance(data, dict):
            raise WikipediaUnavailable("The Wikipedia API returned an unexpected response.")
        if "error" in data:
            raise WikipediaUnavailable("The Wikipedia API reported an error for this request.")
        return data

    raise WikipediaUnavailable("The Wikipedia API did not respond.")  # pragma: no cover


def _query(params):
    key = json.dumps(params, sort_keys=True)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    data = _http_get(params)
    _CACHE.set(key, data)
    return data


# -------------------------------------------------------------- API wrappers

def normalize_space(text):
    return re.sub(r"\s+", " ", text or "").strip()


def search_pages(query, limit=5):
    """Search pages and return [{title, page_id, rank, url, extract}] by rank.

    The intro extract of each hit comes back in the same request. Disambiguation
    and missing pages are skipped.
    """
    query = normalize_space(query)[:MAX_QUERY_CHARS]
    if not query:
        return []
    data = _query({
        "action": "query", "generator": "search", "gsrsearch": query,
        "gsrnamespace": 0, "gsrlimit": max(1, min(int(limit), 10)),
        "prop": "extracts|pageprops|info", "exintro": 1, "explaintext": 1,
        "exlimit": "max", "ppprop": "disambiguation", "inprop": "url", "redirects": 1,
    })
    pages = (data.get("query") or {}).get("pages") or []
    hits = []
    for page in pages:
        if not isinstance(page, dict) or page.get("missing") or not page.get("title"):
            continue
        if "disambiguation" in (page.get("pageprops") or {}):
            continue
        title = page["title"]
        hits.append({
            "title": title,
            "page_id": page.get("pageid"),
            "rank": page.get("index") if isinstance(page.get("index"), int) else 999,
            "url": page.get("fullurl") or title_url(title),
            "extract": page.get("extract") or "",
        })
    hits.sort(key=lambda hit: hit["rank"])
    return hits


def get_page_text(title, max_chars=MAX_ARTICLE_CHARS):
    """Plain-text article for one page (truncated locally), or None."""
    data = _query({
        "action": "query", "titles": title, "prop": "extracts|info",
        "explaintext": 1, "inprop": "url", "redirects": 1,
    })
    pages = (data.get("query") or {}).get("pages") or []
    for page in pages:
        if isinstance(page, dict) and not page.get("missing") and page.get("extract"):
            return {
                "title": page.get("title", title),
                "url": page.get("fullurl") or title_url(page.get("title", title)),
                "text": page["extract"][:max_chars],
            }
    return None


def title_url(title):
    return "https://en.wikipedia.org/wiki/" + urllib.parse.quote(
        title.replace(" ", "_"), safe="_(),'-.")


# ------------------------------------------------------------ text handling

_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "vs", "no", "inc", "ltd",
    "co", "corp", "gen", "col", "lt", "sgt", "capt", "rev", "hon", "gov", "sen", "rep",
    "fig", "vol", "etc", "dept", "est", "approx", "ca", "cf", "ft", "u.s", "u.k",
}
_SENTENCE_END = re.compile(r'([.!?]["”’)\]]*)\s+(?=["“‘(\[]?[A-Z0-9])')
_HEADING = re.compile(r"^=+\s*.*?\s*=+$")
_SENTENCE_PUNCT = re.compile(r'[.!?]["”’)\]]*$')


def split_sentences(text):
    """Split plain text into sentences, skipping headings and fragments."""
    sentences = []
    for line in re.split(r"\n+", text or ""):
        paragraph = normalize_space(line)
        if not paragraph or _HEADING.match(paragraph):
            continue
        start = 0
        pieces = []
        for match in _SENTENCE_END.finditer(paragraph):
            before = paragraph[start:match.start(1)]
            token_match = re.search(r"(\S+)$", before)
            token = token_match.group(1).strip("\"'“‘([") if token_match else ""
            if match.group(1).startswith("."):
                if (token.lower() in _ABBREVIATIONS
                        or re.fullmatch(r"[A-Z]", token)
                        or re.fullmatch(r"(?:[A-Za-z]\.)+[A-Za-z]", token)):
                    continue
            pieces.append(paragraph[start:match.end(1)])
            start = match.end()
        pieces.append(paragraph[start:])
        for piece in pieces:
            piece = piece.strip()
            if not _SENTENCE_PUNCT.search(piece):
                continue  # headings, list items and text cut off by truncation
            if not MIN_SENTENCE_CHARS <= len(piece) <= MAX_SENTENCE_CHARS:
                continue
            if len(piece.split()) < 4 or "\\displaystyle" in piece or "{" in piece:
                continue
            sentences.append(piece)
    return sentences


def _norm_key(text):
    return re.sub(r"[^\w]+", " ", text.lower()).strip()


def tokenize(text):
    return [t for t in re.findall(r"[^\W_]+", (text or "").lower())
            if t not in STOPWORDS and (len(t) > 1 or t.isdigit())]


def dedupe_sentences(items, threshold=0.92):
    """Drop exact and near-duplicate sentences (items are dicts with 'text')."""
    kept, keys, token_sets = [], set(), []
    for item in items:
        key = _norm_key(item["text"])
        if not key or key in keys:
            continue
        tokens = set(key.split())
        duplicate = False
        for other in token_sets:
            union = len(tokens | other)
            if union and len(tokens & other) / union >= threshold:
                duplicate = True
                break
        if duplicate:
            continue
        keys.add(key)
        token_sets.append(tokens)
        kept.append(item)
    return kept


def rank_sentences(claim, sentences, entity_terms=(), limit=12, per_page=4,
                   min_relevance=MIN_RELEVANCE):
    """Rank sentence dicts (text, title, sentence_id, page_rank) against a claim.

    relevance = IDF-weighted share of the claim's content words found in the
    sentence (page-title words count half, since Wikipedia sentences often use
    pronouns for the page subject). Sentences below `min_relevance` are
    dropped, so topically unrelated sentences are never sent for assessment.
    """
    claim_tokens = set(tokenize(claim))
    if not claim_tokens or not sentences:
        return []

    sentence_tokens = [set(tokenize(s["text"])) for s in sentences]
    title_tokens = [set(tokenize(s.get("title", ""))) for s in sentences]
    pool = len(sentences)
    df = {t: sum(1 for toks in sentence_tokens if t in toks) for t in claim_tokens}

    def idf(token):
        return math.log((pool + 1) / (df[token] + 0.5)) + 1.0

    present = [t for t in claim_tokens
               if df[t] > 0 or any(t in title for title in title_tokens)]
    total = sum(idf(t) for t in present)
    if total <= 0:
        return []

    # Entity phrases (as token sets). A sentence is about an entity when every
    # word of the phrase appears in the sentence or in its page title.
    phrases = [set(tokenize(t)) for t in entity_terms if t]
    phrases = [p for p in phrases if p]
    ranked = []
    for item, tokens, title in zip(sentences, sentence_tokens, title_tokens):
        matched = sum(idf(t) for t in present if t in tokens)
        matched += 0.5 * sum(idf(t) for t in present if t not in tokens and t in title)
        relevance = matched / total
        if phrases:
            if not any(p <= (tokens | title) for p in phrases):
                relevance *= 0.5  # shares common words only, not the claim's entities
            elif any(p <= tokens for p in phrases):
                relevance += 0.05
        if relevance < min_relevance:
            continue
        score = relevance
        if item.get("sentence_id") == 0:
            score += 0.05
        score += 0.08 / (1 + item.get("page_rank", 0))
        ranked.append(dict(item, relevance=round(relevance, 4), score=round(score, 4)))

    ranked.sort(key=lambda s: s["score"], reverse=True)
    selected, per_page_count = [], {}
    for item in ranked:
        if len(selected) >= limit:
            break
        count = per_page_count.get(item["title"], 0)
        if count >= per_page:
            continue
        per_page_count[item["title"]] = count + 1
        selected.append(item)
    return selected


# ------------------------------------------------------------------ queries

_CAPITALIZED = re.compile(
    r"\b[A-Z][\w'’.-]*(?:\s+(?:(?:of|the|de|von|van|and|for)\s+)*[A-Z][\w'’.-]*)*")


def capitalized_phrases(text, max_phrases=3):
    """Fallback entity guesses: runs of capitalized words in the claim."""
    phrases = []
    for match in _CAPITALIZED.finditer(text or ""):
        phrase = match.group(0).strip(" .,'")
        words = phrase.split()
        if len(words) == 1 and (match.start() == 0 or phrase.lower() in STOPWORDS):
            continue
        if len(phrase) >= 3 and phrase.lower() not in {p.lower() for p in phrases}:
            phrases.append(phrase)
    return phrases[:max_phrases]


def _trim(text, limit=MAX_QUERY_CHARS):
    text = normalize_space(text)
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0]


def build_queries(claim, entities=None, normalized_claim=None, max_queries=MAX_QUERIES):
    """Search queries as (kind, text). The original claim always comes first.

    Entity names and the normalized claim only add queries; they never replace
    the claim.
    """
    queries = [("claim", _trim(claim))]
    names = []
    for entity in entities or []:
        if not isinstance(entity, dict):
            continue
        name = normalize_space(str(entity.get("name", "")))
        if str(entity.get("type", "")).strip().lower() in SKIP_ENTITY_TYPES:
            continue
        if 2 <= len(name) <= 80 and name.lower() not in {n.lower() for n in names}:
            names.append(name)
    if not names:
        names = capitalized_phrases(claim)
    for name in names[:2]:
        queries.append(("entity", name))
    if normalized_claim and _norm_key(normalized_claim) != _norm_key(claim):
        queries.append(("normalized", _trim(normalized_claim)))

    unique, seen = [], set()
    for kind, text in queries:
        if text and text.lower() not in seen:
            seen.add(text.lower())
            unique.append((kind, text))
    return unique[:max_queries]


# ---------------------------------------------------------------- retrieval

def retrieve_evidence(claim, entities=None, normalized_claim=None, max_pages=6,
                      deep_pages=2, max_sentences=12, per_page=4):
    """Retrieve candidate pages and ranked evidence sentences for a claim.

    Raises WikipediaUnavailable if no search request succeeded. Returns
    {"queries", "pages", "sentences"}; "sentences" may be empty.
    """
    queries = build_queries(claim, entities, normalized_claim)
    fused = {}
    succeeded, last_error = 0, None

    for kind, text in queries:
        try:
            hits = search_pages(
                text, SEARCH_RESULTS_CLAIM if kind == "claim" else SEARCH_RESULTS_ENTITY)
        except WikipediaUnavailable as exc:
            last_error = exc
            logger.warning("Wikipedia search failed: %s", exc)
            continue
        succeeded += 1
        for position, hit in enumerate(hits):
            entry = fused.setdefault(hit["title"], dict(hit, score=0.0, queries=[], kinds=set()))
            entry["score"] += 1.0 / (5 + position)
            if kind == "entity" and hit["title"].lower() == text.lower():
                entry["score"] += 0.5
            entry["queries"].append(text)
            entry["kinds"].add(kind)

    if not succeeded:
        raise last_error or WikipediaUnavailable("Wikipedia search did not return a response.")

    pages = sorted(fused.values(), key=lambda p: p["score"], reverse=True)[:max_pages]

    pool = []
    for page_rank, page in enumerate(pages):
        text, url = page["extract"], page["url"]
        if page_rank < deep_pages:
            try:
                article = get_page_text(page["title"])
            except WikipediaUnavailable as exc:
                logger.warning("Wikipedia article fetch failed: %s", exc)
                article = None
            if article:
                text, url = article["text"], article["url"]
        page["url"] = url
        for sentence_id, sentence in enumerate(split_sentences(text)):
            pool.append({
                "page_id": page["title"].replace(" ", "_"),
                "title": page["title"],
                "url": url,
                "sentence_id": sentence_id,
                "text": sentence,
                "page_rank": page_rank,
                "raw": True,
            })
    pool = dedupe_sentences(pool)

    entity_terms = [text for kind, text in queries if kind == "entity"]
    sentences = rank_sentences(claim, pool, entity_terms, max_sentences, per_page)

    candidates = [{
        "page_id": page["title"].replace(" ", "_"),
        "title": page["title"],
        "url": page["url"],
        "source": "Wikipedia search ({})".format(
            "entity" if "entity" in page["kinds"] and "claim" not in page["kinds"] else "claim text"),
        "score": round(page["score"], 4),
        "query": page["queries"][0],
    } for page in pages]

    return {"queries": [text for _, text in queries], "pages": candidates,
            "sentences": sentences, "sentences_considered": len(pool)}
