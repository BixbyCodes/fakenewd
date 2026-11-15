import json
import tempfile
import unittest
from pathlib import Path

from src.evidence_retriever import extract_sentence, retrieve_evidence


class EvidenceRetrieverTests(unittest.TestCase):

    def test_extract_sentence(self):
        page = {
            "lines": "0\tFirst sentence.\tEntity\n1\tSecond sentence.\n"
        }
        self.assertEqual(
            extract_sentence(page, 1),
            "Second sentence."
        )

    def test_missing_sentence(self):
        page = {"lines": "0\tFirst sentence.\n"}
        self.assertIsNone(extract_sentence(page, 5))

    def test_retrieve_exact_page_and_sentence(self):
        with tempfile.TemporaryDirectory() as temp:
            shard = Path(temp) / "wiki-001.jsonl"
            page = {
                "id": "Soul_Food_-LRB-film-RRB-",
                "text": "Example page text.",
                "lines": "0\tFox 2000 Pictures released Soul Food.\n"
            }

            with shard.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(page) + "\n")

            result = retrieve_evidence(
                "Soul_Food_-LRB-film-RRB-", 0, temp
            )

            self.assertTrue(result["found"])
            self.assertEqual(
                result["sentence"],
                "Fox 2000 Pictures released Soul Food."
            )

    def test_missing_page(self):
        with tempfile.TemporaryDirectory() as temp:
            shard = Path(temp) / "wiki-001.jsonl"
            shard.write_text("", encoding="utf-8")

            result = retrieve_evidence("Missing_Page", 0, temp)
            self.assertFalse(result["found"])


if __name__ == "__main__":
    unittest.main()
