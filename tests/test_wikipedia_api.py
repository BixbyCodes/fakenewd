"""Tests for public-Wikipedia evidence retrieval and backend selection.

No test touches the network or the real Groq API: urllib and the LLM client
are replaced with fakes, and caches are redirected to temporary directories.
"""

import email.message
import json
import os
import socket
import sqlite3
import tempfile
import unittest
import urllib.error
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from src import evidence_verifier as ev
from src import title_search, wikipedia_api as wiki
from src.llm_client import JsonCache
from src.wikipedia_api import WikipediaUnavailable

CLAIM = "The first inauguration of Bill Clinton made him the 50th President of the United States."


def api_page(title, index, extract, disambiguation=False, missing=False):
    page = {"pageid": index * 100, "ns": 0, "title": title, "index": index,
            "extract": extract, "fullurl": "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")}
    if disambiguation:
        page["pageprops"] = {"disambiguation": ""}
    if missing:
        page["missing"] = True
    return page


def api_response(*pages):
    return {"batchcomplete": True, "query": {"pages": list(pages)}}


class NoNetwork(unittest.TestCase):
    """Fail loudly if a test would reach the real network."""

    def setUp(self):
        wiki.clear_cache()
        patcher = mock.patch("urllib.request.urlopen",
                             side_effect=AssertionError("unexpected network access"))
        patcher.start()
        self.addCleanup(patcher.stop)


class SearchParsingTests(NoNetwork):
    def test_parses_ranks_and_skips_disambiguation_and_missing(self):
        response = api_response(
            api_page("Hillary Clinton", 2, "Hillary Diane Rodham Clinton is an American politician."),
            api_page("Bill Clinton", 1, "William Jefferson Clinton was the 42nd president."),
            api_page("Clinton (disambiguation)", 3, "Clinton may refer to:", disambiguation=True),
            api_page("Gone", 4, "", missing=True),
        )
        with mock.patch.object(wiki, "_http_get", return_value=response):
            hits = wiki.search_pages("Clinton")
        self.assertEqual([h["title"] for h in hits], ["Bill Clinton", "Hillary Clinton"])
        self.assertEqual(hits[0]["url"], "https://en.wikipedia.org/wiki/Bill_Clinton")
        self.assertIn("42nd", hits[0]["extract"])

    def test_missing_url_falls_back_to_title_url(self):
        page = api_page("Bill Clinton", 1, "Text.")
        del page["fullurl"]
        with mock.patch.object(wiki, "_http_get", return_value=api_response(page)):
            self.assertEqual(wiki.search_pages("x")[0]["url"], "https://en.wikipedia.org/wiki/Bill_Clinton")

    def test_empty_results(self):
        with mock.patch.object(wiki, "_http_get", return_value={"batchcomplete": True}):
            self.assertEqual(wiki.search_pages("zzzz qqqq"), [])
            result = wiki.retrieve_evidence("zzzz qqqq xxxx yyyy")
        self.assertEqual(result["pages"], [])
        self.assertEqual(result["sentences"], [])

    def test_blank_query_makes_no_request(self):
        with mock.patch.object(wiki, "_http_get") as http:
            self.assertEqual(wiki.search_pages("   "), [])
        http.assert_not_called()

    def test_results_are_cached(self):
        with mock.patch.object(wiki, "_http_get", return_value=api_response(api_page("A", 1, "Text."))) as http:
            wiki.search_pages("same query")
            wiki.search_pages("same query")
            wiki.search_pages("other query")
        self.assertEqual(http.call_count, 2)


class PageTextTests(NoNetwork):
    def test_returns_truncated_text_and_url(self):
        response = api_response({"title": "Bill Clinton", "extract": "A" * 500,
                                 "fullurl": "https://en.wikipedia.org/wiki/Bill_Clinton"})
        with mock.patch.object(wiki, "_http_get", return_value=response):
            page = wiki.get_page_text("Bill Clinton", max_chars=100)
        self.assertEqual(len(page["text"]), 100)
        self.assertEqual(page["title"], "Bill Clinton")

    def test_missing_page(self):
        with mock.patch.object(wiki, "_http_get", return_value=api_response({"title": "X", "missing": True})):
            self.assertIsNone(wiki.get_page_text("X"))


class SentenceExtractionTests(unittest.TestCase):
    def test_headings_fragments_and_abbreviations(self):
        text = (
            "Dr. Smith met U.S. officials in St. Louis on Monday. The talks lasted two hours.\n"
            "== Early life ==\n"
            "Early life\n"
            "Too short.\n"
            "He was born in a small town and later moved to the capital, where he studied law and\n"
            "This sentence is cut off because the text was truncated and has no ending"
        )
        sentences = wiki.split_sentences(text)
        self.assertEqual(sentences, [
            "Dr. Smith met U.S. officials in St. Louis on Monday.",
            "The talks lasted two hours.",
        ])

    def test_whitespace_normalized_and_quotes_kept(self):
        sentences = wiki.split_sentences(
            'The mayor said  "we  will  win  this  election."   Then   he   left the building quickly.')
        self.assertEqual(sentences[0], 'The mayor said "we will win this election."')
        self.assertEqual(sentences[1], "Then he left the building quickly.")

    def test_empty_input(self):
        self.assertEqual(wiki.split_sentences(""), [])
        self.assertEqual(wiki.split_sentences(None), [])


class DedupeTests(unittest.TestCase):
    def test_exact_and_near_duplicates_removed(self):
        items = [
            {"text": "Bill Clinton was the 42nd president of the United States."},
            {"text": "Bill Clinton was the 42nd president of the United States."},
            {"text": "bill clinton was the 42nd President of the United States"},
            {"text": "Hillary Clinton served as United States secretary of state."},
        ]
        kept = wiki.dedupe_sentences(items)
        self.assertEqual(len(kept), 2)
        self.assertTrue(kept[0]["text"].startswith("Bill Clinton"))


class RankingTests(unittest.TestCase):
    @staticmethod
    def sentence(title, sentence_id, text, page_rank=0):
        return {"title": title, "page_id": title.replace(" ", "_"), "sentence_id": sentence_id,
                "text": text, "page_rank": page_rank}

    def test_relevant_sentences_first_and_unrelated_dropped(self):
        pool = [
            self.sentence("Giant panda", 0, "The giant panda is a bear species endemic to China.", 2),
            self.sentence("Bill Clinton", 0,
                          "William Jefferson Clinton is an American politician who served as the 42nd "
                          "president of the United States from 1993 to 2001."),
            self.sentence("Bill Clinton", 5, "He married Hillary Rodham in 1975 after law school."),
            self.sentence("Hillary Clinton", 3, "She worked as a lawyer and an advocate for children.", 1),
        ]
        ranked = wiki.rank_sentences(CLAIM, pool, entity_terms=["Bill Clinton"])
        titles = [(s["title"], s["sentence_id"]) for s in ranked]
        self.assertEqual(titles[0], ("Bill Clinton", 0))
        self.assertNotIn(("Giant panda", 0), titles)
        self.assertNotIn(("Hillary Clinton", 3), titles)
        self.assertTrue(all("relevance" in s and "score" in s for s in ranked))

    def test_limits(self):
        pool = [self.sentence("Bill Clinton", i, "Bill Clinton was the 42nd president, number {}.".format(i))
                for i in range(10)]
        self.assertEqual(len(wiki.rank_sentences(CLAIM, pool, limit=3, per_page=10)), 3)
        self.assertEqual(len(wiki.rank_sentences(CLAIM, pool, limit=10, per_page=2)), 2)

    def test_empty_inputs(self):
        self.assertEqual(wiki.rank_sentences("", [self.sentence("A", 0, "Some text here.")]), [])
        self.assertEqual(wiki.rank_sentences(CLAIM, []), [])


class QueryTests(unittest.TestCase):
    def test_original_claim_first_and_entities_enrich(self):
        entities = [{"name": "Bill Clinton", "type": "person", "role": "subject"},
                    {"name": "1993", "type": "date", "role": "time"},
                    {"name": "United States", "type": "country", "role": "location"},
                    {"name": "Extra Person", "type": "person", "role": "other"}]
        queries = wiki.build_queries(CLAIM, entities, "Bill Clinton became the 50th US President.")
        self.assertEqual(queries[0], ("claim", CLAIM))
        texts = [q[1] for q in queries]
        self.assertIn("Bill Clinton", texts)
        self.assertNotIn("1993", texts)           # dates are not entities to search for
        self.assertNotIn("Extra Person", texts)   # at most two entity queries
        self.assertEqual(queries[-1][0], "normalized")
        self.assertLessEqual(len(queries), wiki.MAX_QUERIES)

    def test_identical_normalized_claim_not_repeated(self):
        queries = wiki.build_queries("Water boils at 100 degrees.", [], "water boils at 100 degrees.")
        self.assertEqual([k for k, _ in queries].count("normalized"), 0)

    def test_capitalized_phrase_fallback(self):
        queries = wiki.build_queries(CLAIM, None, None)
        self.assertIn(("entity", "Bill Clinton"), queries)

    def test_long_claim_is_trimmed(self):
        queries = wiki.build_queries("word " * 200)
        self.assertLessEqual(len(queries[0][1]), wiki.MAX_QUERY_CHARS)


def fake_response(payload):
    body = json.dumps(payload).encode("utf-8") if not isinstance(payload, bytes) else payload
    response = mock.MagicMock()
    response.read.return_value = body
    response.__enter__.return_value = response
    return response


def http_error(code):
    headers = email.message.Message()
    headers["Retry-After"] = "0"
    return urllib.error.HTTPError("https://example.invalid", code, "error", headers, None)


class HttpTests(unittest.TestCase):
    def setUp(self):
        wiki.clear_cache()
        for patcher in (mock.patch.object(wiki.time, "sleep"),):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_success_sends_descriptive_user_agent_and_timeout(self):
        with mock.patch("urllib.request.urlopen", return_value=fake_response({"query": {"pages": []}})) as opened:
            data = wiki._http_get({"action": "query", "titles": "X"})
        self.assertEqual(data, {"query": {"pages": []}})
        request = opened.call_args[0][0]
        self.assertIn("EvidenceAware", request.get_header("User-agent"))
        self.assertEqual(opened.call_args[1]["timeout"], wiki.REQUEST_TIMEOUT_SECONDS)
        self.assertIn("formatversion=2", request.full_url)

    def test_rate_limit_retries_once_then_fails_safely(self):
        with mock.patch("urllib.request.urlopen", side_effect=[http_error(429), http_error(429)]) as opened:
            with self.assertRaises(WikipediaUnavailable) as ctx:
                wiki._http_get({"action": "query"})
        self.assertEqual(opened.call_count, 2)
        self.assertIn("rate limit", str(ctx.exception))
        self.assertNotIn("http", str(ctx.exception).lower().replace("(http", ""))

    def test_rate_limit_then_success(self):
        with mock.patch("urllib.request.urlopen",
                        side_effect=[http_error(429), fake_response({"ok": True})]):
            self.assertEqual(wiki._http_get({"action": "query"}), {"ok": True})

    def test_server_error_not_retried(self):
        with mock.patch("urllib.request.urlopen", side_effect=http_error(500)) as opened:
            with self.assertRaises(WikipediaUnavailable) as ctx:
                wiki._http_get({"action": "query"})
        self.assertEqual(opened.call_count, 1)
        self.assertIn("500", str(ctx.exception))

    def test_network_failures(self):
        for failure in (urllib.error.URLError("dns failure with secret-host"), socket.timeout("slow"),
                        ConnectionResetError("reset")):
            with mock.patch("urllib.request.urlopen", side_effect=failure):
                with self.assertRaises(WikipediaUnavailable) as ctx:
                    wiki._http_get({"action": "query"})
            self.assertIn("Could not reach", str(ctx.exception))
            self.assertNotIn("secret-host", str(ctx.exception))

    def test_bad_json_and_api_error(self):
        with mock.patch("urllib.request.urlopen", return_value=fake_response(b"<html>not json</html>")):
            with self.assertRaises(WikipediaUnavailable):
                wiki._http_get({"action": "query"})
        with mock.patch("urllib.request.urlopen",
                        return_value=fake_response({"error": {"code": "maxlag", "info": "internal detail"}})):
            with self.assertRaises(WikipediaUnavailable) as ctx:
                wiki._http_get({"action": "query"})
        self.assertNotIn("internal detail", str(ctx.exception))

    def test_oversized_response_rejected(self):
        with mock.patch.object(wiki, "MAX_RESPONSE_BYTES", 10), \
                mock.patch("urllib.request.urlopen", return_value=fake_response(b"x" * 50)):
            with self.assertRaises(WikipediaUnavailable):
                wiki._http_get({"action": "query"})


class RetrieveEvidenceTests(NoNetwork):
    CLINTON = "William Jefferson Clinton is an American politician who served as the 42nd president of the United States."

    def http(self, params):
        if params.get("generator") == "search":
            return api_response(api_page("Bill Clinton", 1, self.CLINTON),
                                api_page("Hillary Clinton", 2, "Hillary Clinton is an American politician and lawyer."))
        return api_response({"title": "Bill Clinton", "fullurl": "https://en.wikipedia.org/wiki/Bill_Clinton",
                             "extract": self.CLINTON + "\n== Early life ==\nHe was born in Hope, Arkansas, in 1946."})

    def test_end_to_end_with_mocked_api(self):
        with mock.patch.object(wiki, "_http_get", side_effect=self.http):
            result = wiki.retrieve_evidence(CLAIM, entities=[{"name": "Bill Clinton", "type": "person"}])
        self.assertEqual(result["queries"][0], CLAIM)
        self.assertEqual(result["pages"][0]["title"], "Bill Clinton")
        top = result["sentences"][0]
        self.assertIn("42nd president", top["text"])
        self.assertTrue(top["raw"])
        self.assertEqual(top["url"], "https://en.wikipedia.org/wiki/Bill_Clinton")
        # intro sentence appears once although it is in both the extract and the article
        self.assertEqual(sum(1 for s in result["sentences"] if "42nd president" in s["text"]), 1)

    def test_all_searches_failing_raises(self):
        with mock.patch.object(wiki, "_http_get", side_effect=WikipediaUnavailable("Could not reach the Wikipedia API.")):
            with self.assertRaises(WikipediaUnavailable):
                wiki.retrieve_evidence(CLAIM)

    def test_partial_failure_is_tolerated(self):
        calls = {"n": 0}

        def flaky(params):
            calls["n"] += 1
            if calls["n"] == 2:
                raise WikipediaUnavailable("Could not reach the Wikipedia API.")
            return self.http(params)

        with mock.patch.object(wiki, "_http_get", side_effect=flaky):
            result = wiki.retrieve_evidence(CLAIM, entities=[{"name": "Bill Clinton", "type": "person"}])
        self.assertTrue(result["pages"])

    def test_article_fetch_failure_falls_back_to_intro(self):
        def http(params):
            if params.get("generator") == "search":
                return api_response(api_page("Bill Clinton", 1, self.CLINTON))
            raise WikipediaUnavailable("Could not reach the Wikipedia API.")

        with mock.patch.object(wiki, "_http_get", side_effect=http):
            result = wiki.retrieve_evidence(CLAIM)
        self.assertIn("42nd president", result["sentences"][0]["text"])


class FakeGroq:
    """Records calls; answers evidence-verification requests from a callable."""

    def __init__(self, answer=None, error=None):
        self.answer = answer
        self.error = error
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        if self.error:
            raise self.error
        user = json.loads(kwargs["messages"][1]["content"])
        self.requests.append(user)
        content = json.dumps({"results": self.answer(user)})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def label_all(label="INSUFFICIENT", span_for=None):
    def answer(user):
        out = []
        for item in user["evidence"]:
            use = label if (span_for is None or span_for in item["text"]) else "INSUFFICIENT"
            out.append({"page_id": item["page_id"], "sentence_id": item["sentence_id"], "label": use,
                        "evidence_span": span_for if use != "INSUFFICIENT" else "",
                        "explanation": "Compared with the claim."})
        return out
    return answer


def retrieved(text="Bill Clinton served as the 42nd president of the United States.", sentence_id=0):
    return {"queries": [CLAIM],
            "pages": [{"page_id": "Bill_Clinton", "title": "Bill Clinton",
                       "url": "https://en.wikipedia.org/wiki/Bill_Clinton",
                       "source": "Wikipedia search (claim text)", "score": 1.0, "query": CLAIM}],
            "sentences": [{"page_id": "Bill_Clinton", "title": "Bill Clinton", "sentence_id": sentence_id,
                           "url": "https://en.wikipedia.org/wiki/Bill_Clinton", "text": text, "raw": True}],
            "sentences_considered": 1}


class VerifierCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        wiki.clear_cache()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.object(
            ev, "_cache", JsonCache(Path(self.tmp.name) / "evidence.sqlite", "evidence_assessments")))
        self.stack.enter_context(mock.patch.dict(os.environ, {"EAFND_EVIDENCE_BACKEND": "auto"}))
        self.stack.enter_context(mock.patch.object(
            title_search, "DB_PATH", Path(self.tmp.name) / "missing.sqlite"))


class LLMClassificationTests(VerifierCase):
    def evidence(self, text="Bill Clinton served as the 42nd president of the United States."):
        return [{"page_id": "Bill_Clinton", "title": "Bill Clinton", "sentence_id": 0,
                 "text": text, "raw": True}]

    def test_decisive_label_kept_with_valid_quote(self):
        client = FakeGroq(label_all("REFUTES", span_for="42nd president"))
        with mock.patch.object(ev, "get_client", return_value=client):
            results = ev.verify_evidence(CLAIM, self.evidence())
        self.assertEqual(results[0]["label"], "REFUTES")
        self.assertEqual(results[0]["evidence_span"], "42nd president")
        self.assertFalse(results[0]["downgraded"])

    def test_decisive_label_without_real_quote_is_downgraded(self):
        def answer(user):
            item = user["evidence"][0]
            return [{"page_id": item["page_id"], "sentence_id": item["sentence_id"], "label": "REFUTES",
                     "evidence_span": "was the 50th president", "explanation": "Contradicts."}]

        with mock.patch.object(ev, "get_client", return_value=FakeGroq(answer)):
            results = ev.verify_evidence(CLAIM, self.evidence())
        self.assertEqual(results[0]["label"], "INSUFFICIENT")
        self.assertTrue(results[0]["downgraded"])

    def test_missing_quote_and_invalid_items_are_handled(self):
        def answer(user):
            item = user["evidence"][0]
            return [
                {"page_id": item["page_id"], "sentence_id": item["sentence_id"], "label": "SUPPORTS",
                 "explanation": "x"},                                    # no quote -> downgraded
                {"page_id": "Invented", "sentence_id": 0, "label": "REFUTES", "evidence_span": "x"},
                {"page_id": item["page_id"], "sentence_id": "abc", "label": "REFUTES"},
                "not an object",
                {"page_id": item["page_id"], "sentence_id": item["sentence_id"], "label": "BOGUS"},
            ]

        with mock.patch.object(ev, "get_client", return_value=FakeGroq(answer)):
            results = ev.verify_evidence(CLAIM, self.evidence())
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["label"], "INSUFFICIENT")

    def test_page_title_is_sent_and_cache_depends_on_text(self):
        client = FakeGroq(label_all())
        with mock.patch.object(ev, "get_client", return_value=client):
            ev.verify_evidence(CLAIM, self.evidence())
            ev.verify_evidence(CLAIM, self.evidence())           # cached
            ev.verify_evidence(CLAIM, self.evidence("A different sentence about the same page."))
        self.assertEqual(len(client.requests), 2)
        self.assertEqual(client.requests[0]["evidence"][0]["page_title"], "Bill Clinton")

    def test_prompt_demands_direct_contradiction(self):
        self.assertIn("Related or irrelevant is NOT contradiction", ev.VERIFIER_PROMPT)
        self.assertIn("evidence_span", ev.VERIFIER_PROMPT)


class BackendSelectionTests(VerifierCase):
    def build_local_index(self):
        shard = Path(self.tmp.name) / "wiki-001.jsonl"
        page = {"id": "Telemundo", "lines": "0\tTelemundo is an American Spanish-language television network .\n"}
        shard.write_text(json.dumps(page) + "\n", encoding="utf-8")
        db = Path(self.tmp.name) / "local.sqlite"
        connection = sqlite3.connect(str(db))
        connection.execute("CREATE TABLE pages (page_id TEXT PRIMARY KEY, shard_path TEXT, byte_offset INTEGER)")
        connection.execute("CREATE VIRTUAL TABLE page_search USING fts5(page_id UNINDEXED, title, "
                           "tokenize='unicode61 remove_diacritics 2')")
        connection.execute("INSERT INTO pages VALUES (?,?,0)", ("Telemundo", str(shard)))
        connection.execute("INSERT INTO page_search VALUES (?,?)", ("Telemundo", "Telemundo"))
        connection.commit()
        connection.close()
        self.stack.enter_context(mock.patch.object(title_search, "DB_PATH", db))

    def test_local_index_preferred_when_present(self):
        self.build_local_index()
        api = self.stack.enter_context(mock.patch.object(
            wiki, "retrieve_evidence", side_effect=AssertionError("API must not be used")))
        self.assertEqual(ev.backend_status()["backend"], "local_index")
        with mock.patch.object(ev, "get_client", return_value=FakeGroq(label_all())):
            report = ev.analyze_evidence("Telemundo is a television network.",
                                         entities=[{"name": "Telemundo", "type": "organization"}])
        self.assertEqual(report["backend"], "local_index")
        self.assertEqual(report["status"], "completed")
        api.assert_not_called()

    def test_api_used_when_local_index_missing(self):
        self.stack.enter_context(mock.patch.object(wiki, "retrieve_evidence", return_value=retrieved()))
        status = ev.backend_status()
        self.assertEqual(status["backend"], "wikipedia_api")
        self.assertIn("not found", status["reason"])
        client = FakeGroq(label_all("REFUTES", span_for="42nd president"))
        with mock.patch.object(ev, "get_client", return_value=client):
            report = ev.analyze_evidence(CLAIM, entities=[{"name": "Bill Clinton", "type": "person"}],
                                         normalized_claim="Bill Clinton became the 50th US President.")
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["backend"], "wikipedia_api")
        self.assertEqual(report["counts"]["REFUTES"], 1)
        self.assertEqual(report["direction"], "REFUTED")
        result = report["results"][0]
        self.assertEqual(result["url"], "https://en.wikipedia.org/wiki/Bill_Clinton")
        self.assertEqual(result["title"], "Bill Clinton")
        # The ORIGINAL claim is what the LLM assesses, never the normalized one.
        self.assertEqual(client.requests[0]["claim"], CLAIM)
        self.assertIn("public MediaWiki API", report["corpus"])

    def test_normalized_claim_and_entities_only_enrich_queries(self):
        retrieve = self.stack.enter_context(mock.patch.object(wiki, "retrieve_evidence", return_value=retrieved()))
        with mock.patch.object(ev, "get_client", return_value=FakeGroq(label_all())):
            report = ev.analyze_evidence(CLAIM, entities=[{"name": "Bill Clinton", "type": "person"}],
                                         normalized_claim="Normalized wording")
        self.assertEqual(retrieve.call_args[0][0], CLAIM)
        self.assertEqual(retrieve.call_args[1]["normalized_claim"], "Normalized wording")
        self.assertEqual(report["claim_assessed"], CLAIM)

    def test_forced_api_mode_ignores_local_index(self):
        self.build_local_index()
        self.stack.enter_context(mock.patch.dict(os.environ, {"EAFND_EVIDENCE_BACKEND": "wikipedia_api"}))
        self.assertEqual(ev.backend_status()["backend"], "wikipedia_api")

    def test_forced_local_mode_without_index_does_not_use_api(self):
        self.stack.enter_context(mock.patch.dict(os.environ, {"EAFND_EVIDENCE_BACKEND": "local"}))
        api = self.stack.enter_context(mock.patch.object(wiki, "retrieve_evidence"))
        report = ev.analyze_evidence(CLAIM)
        self.assertEqual(report["status"], "index_unavailable")
        api.assert_not_called()

    def test_disabled_mode(self):
        self.stack.enter_context(mock.patch.dict(os.environ, {"EAFND_EVIDENCE_BACKEND": "off"}))
        self.assertFalse(ev.backend_status()["available"])
        self.assertEqual(ev.analyze_evidence(CLAIM)["status"], "index_unavailable")

    def test_both_backends_failing_reports_clear_error(self):
        self.stack.enter_context(mock.patch.object(
            wiki, "retrieve_evidence", side_effect=WikipediaUnavailable("Could not reach the Wikipedia API.")))
        report = ev.analyze_evidence(CLAIM)
        self.assertEqual(report["status"], "retrieval_unavailable")
        self.assertIn("Could not reach the Wikipedia API", report["message"])
        self.assertIn("local FEVER index is also unavailable", report["message"])
        self.assertEqual(report["results"], [])

    def test_runtime_local_failure_falls_back_to_api(self):
        self.build_local_index()
        self.stack.enter_context(mock.patch.object(
            ev, "_retrieve_local", side_effect=ev.EvidenceIndexUnavailable("shard missing")))
        self.stack.enter_context(mock.patch.object(wiki, "retrieve_evidence", return_value=retrieved()))
        with mock.patch.object(ev, "get_client", return_value=FakeGroq(label_all())):
            report = ev.analyze_evidence(CLAIM)
        self.assertEqual(report["backend"], "wikipedia_api")
        self.assertIn("local index failed", report["message"])

    def test_empty_search_results_report_no_candidates(self):
        self.stack.enter_context(mock.patch.object(
            wiki, "retrieve_evidence",
            return_value={"queries": [CLAIM], "pages": [], "sentences": [], "sentences_considered": 0}))
        report = ev.analyze_evidence(CLAIM)
        self.assertEqual(report["status"], "no_candidates")
        self.assertEqual(report["evidence_count"], 0)

    def test_pages_without_relevant_sentences(self):
        data = retrieved()
        data["sentences"] = []
        self.stack.enter_context(mock.patch.object(wiki, "retrieve_evidence", return_value=data))
        report = ev.analyze_evidence(CLAIM)
        self.assertEqual(report["status"], "no_sentences")

    def test_llm_unavailable_still_returns_retrieved_evidence(self):
        self.stack.enter_context(mock.patch.object(wiki, "retrieve_evidence", return_value=retrieved()))
        report = ev.analyze_evidence(CLAIM, llm_available=False)
        self.assertEqual(report["status"], "retrieved_not_assessed")
        self.assertEqual(report["evidence_count"], 1)
        self.assertEqual(report["results"], [])


class SmartFakeGroq:
    """Answers all three LLM tasks (assessment, event extraction, evidence)."""

    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        system = kwargs["messages"][0]["content"]
        if "annotation assistant" in system:
            data = {"verdict": "uncertain", "plausibility": 0.4, "verifiability": 0.9,
                    "source_credibility": 0.0, "rationale": "No source is identified."}
        elif "extract factual structure" in system:
            data = {"event_summary": "Claim about a presidency.", "time": None, "location": None,
                    "attribution": None, "normalized_claim": "Bill Clinton became the 50th President.",
                    "entities": [{"name": "Bill Clinton", "type": "person", "role": "subject"}],
                    "uncertainties": []}
        else:
            user = json.loads(kwargs["messages"][1]["content"])
            data = {"results": label_all("REFUTES", span_for="42nd president")(user)}
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(data)))])


class AppWithPublicApiTests(VerifierCase):
    def setUp(self):
        super().setUp()
        from src import event_parser, llm_features
        tmp = Path(self.tmp.name)
        self.stack.enter_context(mock.patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}))
        self.stack.enter_context(mock.patch.object(llm_features, "DB", tmp / "llm.sqlite"))
        self.stack.enter_context(mock.patch.object(
            event_parser, "_cache", JsonCache(tmp / "event.sqlite", "event_parses")))
        fake = SmartFakeGroq()
        for module in (llm_features, event_parser, ev):
            self.stack.enter_context(mock.patch.object(module, "get_client", return_value=fake))

    def run_app(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=120)
        app.run()
        app.text_area(key="body_input").set_value(CLAIM)
        app.button(key="FormSubmitter:analyze_form-Analyze").click().run()
        return app

    def test_app_shows_retrieved_sentence_assessment_and_source_link(self):
        retrieve = self.stack.enter_context(mock.patch.object(wiki, "retrieve_evidence", return_value=retrieved()))
        app = self.run_app()
        self.assertEqual(len(app.exception), 0)
        retrieve.assert_called_once()
        self.assertEqual(retrieve.call_args[0][0], CLAIM)
        rendered = " ".join(m.value for m in app.markdown)
        for expected in ("Retrieved Wikipedia sentence", "LLM assessment of this sentence",
                         "Open on Wikipedia", "https://en.wikipedia.org/wiki/Bill_Clinton",
                         "42nd president", "Deciding phrase quoted by the LLM"):
            self.assertIn(expected, rendered)
        self.assertTrue(any(e.label.startswith("REFUTES") and "Bill Clinton" in e.label for e in app.expander))
        self.assertNotIn("Evidence retrieval did not run", rendered)

    def test_app_explains_failure_without_pretending_verification(self):
        self.stack.enter_context(mock.patch.object(
            wiki, "retrieve_evidence", side_effect=WikipediaUnavailable("Could not reach the Wikipedia API.")))
        app = self.run_app()
        self.assertEqual(len(app.exception), 0)
        rendered = " ".join(m.value for m in app.markdown)
        self.assertIn("Evidence retrieval did not run", rendered)
        self.assertIn("Could not reach the Wikipedia API", rendered)
        self.assertIn("No evidence check was performed", rendered)
        self.assertNotIn("Retrieved Wikipedia sentence", rendered)


if __name__ == "__main__":
    unittest.main()
