import unittest

from src.ingestion import create_article_record, normalize_text, validate_source_url


class IngestionTests(unittest.TestCase):
    def test_normalizes_whitespace_and_unicode(self):
        self.assertEqual(normalize_text("  A   headline\n\n  next  "), "A headline\n\nnext")

    def test_creates_record_and_stable_id(self):
        first = create_article_record("Headline", "Body text", "https://example.org/a")
        second = create_article_record("Headline", "Body text", "https://example.org/a")
        self.assertEqual(first.record_id, second.record_id)
        self.assertEqual(first.language, "en")
        self.assertTrue(first.ingested_at)

    def test_rejects_empty_headline(self):
        with self.assertRaises(ValueError):
            create_article_record(" ", "Body")

    def test_rejects_empty_body(self):
        with self.assertRaises(ValueError):
            create_article_record("Headline", " ")

    def test_rejects_invalid_url(self):
        with self.assertRaises(ValueError):
            validate_source_url("ftp://example.org/file")

    def test_allows_missing_url(self):
        self.assertIsNone(validate_source_url(None))


if __name__ == "__main__":
    unittest.main()
