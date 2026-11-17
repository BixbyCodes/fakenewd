"""Tests for the end-to-end EvidenceAware pipeline.

No test calls the real Groq API: LLM clients are replaced with fakes, and
caches are redirected to temporary directories.
"""

import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from src import analysis, event_parser, evidence_verifier, llm_features, predict, title_search
from src.config import CLASSICAL_MODELS_DIR, MAX_INPUT_CHARS
from src.llm_client import JsonCache, LLMUnavailable, describe_error, get_client, parse_json_object

MODELS_PRESENT = (CLASSICAL_MODELS_DIR / "isot_tfidf.joblib").exists()


class FakeClient:
    """Mimics groq.Groq().chat.completions.create."""

    def __init__(self, responses=None, error=None):
        self.responses = list(responses or [])
        self.error = error
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        content = self.responses.pop(0) if self.responses else ""
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def build_tiny_index(directory):
    """A miniature FEVER-style shard + SQLite index with FTS5 titles."""
    shard = Path(directory) / "wiki-001.jsonl"
    pages = [
        {"id": "Telemundo", "lines": "0\tTelemundo is an American Spanish-language television network .\n"
                                     "1\tIt is owned by NBCUniversal .\n"},
        {"id": "English_language", "lines": "0\tEnglish is a West Germanic language .\n"},
    ]
    offsets = []
    with shard.open("wb") as handle:
        for page in pages:
            offsets.append(handle.tell())
            handle.write((json.dumps(page) + "\n").encode("utf-8"))
    db = Path(directory) / "wiki_pages.sqlite"
    connection = sqlite3.connect(str(db))
    connection.execute("CREATE TABLE pages (page_id TEXT PRIMARY KEY, shard_path TEXT, byte_offset INTEGER)")
    connection.execute("CREATE VIRTUAL TABLE page_search USING fts5(page_id UNINDEXED, title, "
                       "tokenize='unicode61 remove_diacritics 2')")
    for page, offset in zip(pages, offsets):
        connection.execute("INSERT INTO pages VALUES (?,?,?)", (page["id"], str(shard), offset))
        connection.execute("INSERT INTO page_search VALUES (?,?)", (page["id"], page["id"]))
    connection.commit()
    connection.close()
    return db


class IsolatedCaches(unittest.TestCase):
    """Redirect every SQLite cache into a temporary directory."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        tmp = Path(self.tmp.name)
        self.stack = ExitStack()
        # Local-only evidence backend: these tests must never reach Wikipedia.
        self.stack.enter_context(mock.patch.dict(os.environ, {"EAFND_EVIDENCE_BACKEND": "local"}))
        self.stack.enter_context(mock.patch.object(llm_features, "DB", tmp / "llm.sqlite"))
        self.stack.enter_context(mock.patch.object(
            event_parser, "_cache", JsonCache(tmp / "event.sqlite", "event_parses")))
        self.stack.enter_context(mock.patch.object(
            evidence_verifier, "_cache", JsonCache(tmp / "evidence.sqlite", "evidence_assessments")))

    def tearDown(self):
        self.stack.close()
        self.tmp.cleanup()


class InputValidationTests(unittest.TestCase):
    def test_empty_input_rejected(self):
        for value in ("", "   ", "\n\n"):
            with self.assertRaises(analysis.InputError):
                analysis.validate_input(value)

    def test_too_short_input_rejected(self):
        with self.assertRaises(analysis.InputError):
            analysis.validate_input("Fake news")

    def test_non_text_rejected(self):
        with self.assertRaises(analysis.InputError):
            analysis.validate_input("12345 67890 11121 31415 92653")
        with self.assertRaises(analysis.InputError):
            analysis.validate_input(None)

    def test_long_input_rejected(self):
        with self.assertRaises(analysis.InputError) as ctx:
            analysis.validate_input("word " * (MAX_INPUT_CHARS // 4))
        self.assertIn("limit", str(ctx.exception))

    def test_headline_combined_and_whitespace_normalized(self):
        text, headline = analysis.validate_input("Body   text says the minister resigned.", "  Minister resigns ")
        self.assertEqual(headline, "Minister resigns")
        self.assertTrue(text.startswith("Minister resigns\n\nBody text says"))

    def test_headline_not_duplicated(self):
        text, _ = analysis.validate_input("Minister resigns. The minister resigned today.", "Minister resigns")
        self.assertEqual(text.count("Minister resigns"), 1)

    def test_claim_for_evidence(self):
        short = "Telemundo is an English-language television network."
        self.assertEqual(analysis.claim_for_evidence(short), short)
        article = "First sentence is the lead claim here. " + "More words follow here. " * 40
        self.assertEqual(analysis.claim_for_evidence(article, "Headline claim"), "Headline claim")
        self.assertLessEqual(len(analysis.claim_for_evidence(article).split()), 60)


@unittest.skipUnless(MODELS_PRESENT, "saved ISOT models not present")
class ClassifierTests(unittest.TestCase):
    def test_models_load_with_expected_classes(self):
        vectorizer, models = predict.load_models()
        self.assertEqual(set(models), {"linear_svm", "logistic_regression", "random_forest"})
        for model in models.values():
            self.assertEqual(list(model.classes_), [0, 1])
        self.assertGreater(len(vectorizer.vocabulary_), 1000)

    def test_prediction_schema(self):
        result = predict.predict_news("The government announced a new budget on Tuesday, officials said.")
        for key in ("prediction", "confidence", "fake_probability", "real_probability", "model",
                    "all_models", "reliability_notes", "model_agreement", "label_mapping"):
            self.assertIn(key, result)
        self.assertIn(result["prediction"], {"FAKE", "REAL"})
        self.assertAlmostEqual(result["fake_probability"] + result["real_probability"], 100, delta=0.05)
        self.assertEqual(result["confidence"], max(result["fake_probability"], result["real_probability"]))

    def test_label_mapping_index_1_is_fake(self):
        vectorizer, models = predict.load_models()
        fake_models = {}
        for name, model in models.items():
            stub = mock.Mock(classes_=np.array([0, 1]))
            stub.predict_proba.return_value = np.array([[0.1, 0.9]])
            fake_models[name] = stub
        with mock.patch.object(predict, "load_models", return_value=(vectorizer, fake_models)):
            result = predict.predict_news("Some text about the election results announced today.")
        self.assertEqual(result["prediction"], "FAKE")
        self.assertEqual(result["fake_probability"], 90.0)
        self.assertEqual(result["real_probability"], 10.0)

    def test_short_input_and_reuters_flags(self):
        notes = predict.reliability_notes("WASHINGTON (Reuters) - Senators met.")
        self.assertTrue(any("Short input" in n for n in notes))
        self.assertTrue(any("Reuters" in n for n in notes))

    def test_empty_text_raises(self):
        with self.assertRaises(ValueError):
            predict.predict_news("  ")


class LLMClientTests(unittest.TestCase):
    def test_missing_api_key(self):
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": ""}):
            with self.assertRaises(LLMUnavailable):
                get_client()

    def test_error_messages_do_not_leak_details(self):
        secret = "gsk_SECRETVALUE"

        class AuthenticationError(Exception):
            pass

        message = describe_error(AuthenticationError("Invalid key " + secret))
        self.assertNotIn(secret, message)
        self.assertNotIn(secret, describe_error(RuntimeError("boom " + secret)))

    def test_parse_json_object(self):
        self.assertEqual(parse_json_object('text {"a": 1} trailing'), {"a": 1})
        for bad in ("", "   ", "no json", "[1, 2]"):
            with self.assertRaises(ValueError):
                parse_json_object(bad)


class LLMAssessmentTests(IsolatedCaches):
    GOOD = json.dumps({"verdict": "uncertain", "plausibility": 0.5, "verifiability": 0.7,
                       "source_credibility": 0.2, "rationale": "No source given."})

    def test_assessment_and_cache(self):
        client = FakeClient([self.GOOD])
        with mock.patch.object(llm_features, "get_client", return_value=client):
            first = llm_features.assess_text("A claim about a minister resigning today.")
            second = llm_features.assess_text("A claim about a minister resigning today.")
        self.assertEqual(first["verdict"], "uncertain")
        self.assertFalse(first["cached"])
        self.assertTrue(second["cached"])
        self.assertEqual(client.calls, 1)

    def test_invalid_schema_rejected(self):
        bad = json.dumps({"verdict": "maybe", "plausibility": 2})
        with mock.patch.object(llm_features, "get_client", return_value=FakeClient([bad])):
            with self.assertRaises(ValueError):
                llm_features.assess_text("Another claim to assess for the test.")


class EventParserTests(IsolatedCaches):
    VALID = {
        "event_summary": "Minister resigned.",
        "entities": [{"name": "Jane Doe", "type": "person", "role": "subject"}],
        "time": "Tuesday", "location": None, "attribution": "Reuters",
        "normalized_claim": "Jane Doe resigned.", "uncertainties": ["No date given."],
    }

    def test_validate_result_accepts_valid(self):
        result = event_parser.validate_result(dict(self.VALID))
        self.assertEqual(result["entities"][0]["name"], "Jane Doe")

    def test_validate_result_rejects_invalid(self):
        for broken in (
            {k: v for k, v in self.VALID.items() if k != "entities"},
            dict(self.VALID, entities="Jane"),
            dict(self.VALID, entities=[{"name": "Jane"}]),
            dict(self.VALID, uncertainties=[1]),
            dict(self.VALID, time=["Tuesday"]),
        ):
            with self.assertRaises(ValueError):
                event_parser.validate_result(broken)

    def test_parse_event_with_fake_client(self):
        client = FakeClient([json.dumps(self.VALID)])
        with mock.patch.object(event_parser, "get_client", return_value=client):
            result = event_parser.parse_event("Jane Doe, the minister, resigned on Tuesday.")
        self.assertEqual(result["normalized_claim"], "Jane Doe resigned.")
        self.assertIn("source_text_hash", result)

    def test_empty_response_raises(self):
        with mock.patch.object(event_parser, "get_client", return_value=FakeClient([""])):
            with self.assertRaises(ValueError):
                event_parser.parse_event("Some claim that gets an empty response.")


class EvidenceTests(IsolatedCaches):
    CLAIM = "Telemundo is an English-language television network."

    def setUp(self):
        super().setUp()
        self.db = build_tiny_index(self.tmp.name)
        self.stack.enter_context(mock.patch.object(title_search, "DB_PATH", self.db))

    def verifier_response(self, label="REFUTES"):
        return json.dumps({"results": [
            {"page_id": "Telemundo", "sentence_id": 0, "label": label, "evidence_span": "Spanish-language",
             "explanation": "Spanish-language."},
            {"page_id": "Telemundo", "sentence_id": 1, "label": "INSUFFICIENT", "evidence_span": "",
             "explanation": "Ownership."},
            {"page_id": "Invented_Page", "sentence_id": 0, "label": "SUPPORTS", "evidence_span": "x",
             "explanation": "made up"},
        ]})

    def test_index_status_and_search(self):
        self.assertTrue(title_search.index_status()["available"])
        pages = [p["page_id"] for p in title_search.search_pages(self.CLAIM)]
        self.assertIn("Telemundo", pages)

    def test_sentences_cleaned(self):
        sentences = evidence_verifier.select_sentences("Telemundo", self.CLAIM)
        self.assertEqual(sentences[0]["sentence_id"], 0)
        self.assertEqual(evidence_verifier.clean_fever_text("Film -LRB-1997-RRB- .") , "Film (1997).")

    def test_full_evidence_analysis(self):
        client = FakeClient([self.verifier_response()])
        with mock.patch.object(evidence_verifier, "get_client", return_value=client):
            report = evidence_verifier.analyze_evidence(
                self.CLAIM, entities=[{"name": "Telemundo", "type": "org", "role": "subject"}])
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["candidate_pages"][0]["source"], "entity title match")
        self.assertEqual(report["counts"]["REFUTES"], 1)
        self.assertEqual(report["direction"], "REFUTED")
        # Labels for sentences that were never retrieved must be discarded.
        self.assertNotIn("Invented_Page", [r["page_id"] for r in report["results"]])

    def test_missing_index(self):
        with mock.patch.object(title_search, "DB_PATH", Path(self.tmp.name) / "missing.sqlite"):
            report = evidence_verifier.analyze_evidence(self.CLAIM)
        self.assertEqual(report["status"], "index_unavailable")
        self.assertEqual(report["results"], [])

    def test_missing_shard(self):
        os.remove(Path(self.tmp.name) / "wiki-001.jsonl")
        report = evidence_verifier.analyze_evidence(self.CLAIM)
        self.assertEqual(report["status"], "index_unavailable")

    def test_llm_failure_keeps_retrieved_evidence(self):
        class APITimeoutError(Exception):
            pass

        with mock.patch.object(evidence_verifier, "get_client",
                               return_value=FakeClient(error=APITimeoutError("slow"))):
            report = evidence_verifier.analyze_evidence(self.CLAIM)
        self.assertEqual(report["status"], "retrieved_not_assessed")
        self.assertGreater(report["evidence_count"], 0)
        self.assertEqual(report["results"], [])
        self.assertIn("did not respond", report["message"])

    def test_no_candidates(self):
        report = evidence_verifier.analyze_evidence("Zzyzx quorbleflax snarkwidget plumbus.")
        self.assertEqual(report["status"], "no_candidates")


class InterpretationTests(unittest.TestCase):
    def classifier(self, prediction, confidence=90.0, notes=None):
        return {"prediction": prediction, "confidence": confidence, "model": "Linear SVM",
                "low_confidence": confidence < 65, "model_agreement": "3/3 ISOT classifiers predict X",
                "reliability_notes": notes or []}

    def evidence(self, direction, supports=0, refutes=0):
        return {"status": "completed", "direction": direction,
                "counts": {"SUPPORTS": supports, "REFUTES": refutes, "INSUFFICIENT": 1}}

    def test_agreement_fake_refuted(self):
        out = analysis.interpret(self.classifier("FAKE"), None, self.evidence("REFUTED", refutes=2))
        self.assertEqual(out["tone"], "refutes")

    def test_agreement_real_supported(self):
        out = analysis.interpret(self.classifier("REAL"), None, self.evidence("SUPPORTED", supports=1))
        self.assertEqual(out["tone"], "supports")

    def test_disagreement_reported(self):
        out = analysis.interpret(self.classifier("REAL"), None, self.evidence("REFUTED", refutes=1))
        self.assertEqual(out["tone"], "conflict")
        self.assertTrue(out["headline"].startswith("Disagreement"))
        self.assertEqual(out["review_level"], "strong")

    def test_insufficient_evidence(self):
        out = analysis.interpret(self.classifier("FAKE"), None, self.evidence("INSUFFICIENT"))
        self.assertIn("insufficient", out["headline"])
        self.assertTrue(out["human_review"])

    def test_evidence_unavailable_is_not_presented_as_checked(self):
        out = analysis.interpret(self.classifier("REAL"), None, {"status": "index_unavailable"})
        self.assertIn("not checked", out["headline"])

    def test_llm_does_not_override_classifier(self):
        llm = {"verdict": "fake", "model": "m"}
        out = analysis.interpret(self.classifier("REAL"), llm, {"status": "index_unavailable"})
        self.assertIn("REAL", out["headline"])
        self.assertIn("The LLM assessment differs from the classifier prediction.", out["review_reasons"])


@unittest.skipUnless(MODELS_PRESENT, "saved ISOT models not present")
class PipelineIntegrationTests(IsolatedCaches):
    def test_runs_without_api_key_or_index(self):
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": ""}), \
                mock.patch.object(title_search, "DB_PATH", Path(self.tmp.name) / "missing.sqlite"):
            report = analysis.run_analysis("The parliament passed the new budget law on Monday after a long debate.")
        self.assertEqual(report["components"]["classifier"]["status"], "ok")
        self.assertIn(report["classifier"]["prediction"], {"FAKE", "REAL"})
        self.assertEqual(report["components"]["llm"]["status"], "unavailable")
        self.assertIsNone(report["llm"])
        self.assertEqual(report["evidence"]["status"], "index_unavailable")
        self.assertIn("not checked", report["interpretation"]["headline"])

    def test_runs_with_failing_llm(self):
        class RateLimitError(Exception):
            pass

        failing = FakeClient(error=RateLimitError("429"))
        db = build_tiny_index(self.tmp.name)
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}), \
                mock.patch.object(title_search, "DB_PATH", db), \
                mock.patch.object(llm_features, "get_client", return_value=failing), \
                mock.patch.object(event_parser, "get_client", return_value=failing), \
                mock.patch.object(evidence_verifier, "get_client", return_value=failing):
            report = analysis.run_analysis("Telemundo is an English-language television network.")
        self.assertEqual(report["components"]["llm"]["status"], "unavailable")
        self.assertIn("rate limit", report["components"]["llm"]["message"])
        self.assertEqual(report["evidence"]["status"], "retrieved_not_assessed")
        self.assertEqual(failing.calls, 2)  # evidence assessment skipped after two failures

    def test_streamlit_app_renders_without_llm_or_index(self):
        from streamlit.testing.v1 import AppTest

        with mock.patch.dict(os.environ, {"GROQ_API_KEY": ""}), \
                mock.patch.object(title_search, "DB_PATH", Path(self.tmp.name) / "missing.sqlite"):
            app = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"),
                                    default_timeout=120)
            app.run()
            self.assertEqual(len(app.exception), 0)
            app.text_area(key="body_input").set_value(
                "The parliament passed the new budget law on Monday after a long debate.")
            app.button(key="FormSubmitter:analyze_form-Analyze").click().run()
            self.assertEqual(len(app.exception), 0)
            rendered = " ".join(m.value for m in app.markdown)
            self.assertIn("Classifier prediction", rendered)
            self.assertIn("Evidence retrieval did not run", rendered)
            self.assertIn("LLM assessment unavailable", rendered)

            app.text_area(key="body_input").set_value("")
            app.button(key="FormSubmitter:analyze_form-Analyze").click().run()
            self.assertTrue(any("Please" in w.value for w in app.warning))


if __name__ == "__main__":
    unittest.main()
