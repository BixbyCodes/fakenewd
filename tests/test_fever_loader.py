import json
import tempfile
import unittest
from pathlib import Path

from src.fever_loader import iter_fever, normalize_fever_record, inspect_fever


class FeverLoaderTests(unittest.TestCase):
    def test_label_mapping_and_evidence(self):
        raw = {
            "id": 123,
            "claim": "A test claim.",
            "label": "SUPPORTS",
            "evidence": [[[1, 2, "Example_Page", 3]]],
        }
        row = normalize_fever_record(raw)
        self.assertEqual(row["label"], "supports")
        self.assertEqual(row["evidence_count"], 1)
        self.assertEqual(row["evidence"][0]["page"], "Example_Page")

    def test_unlabelled_test_row(self):
        row = normalize_fever_record({
            "id": 456, "claim": "Another claim", "label": None, "evidence": []
        })
        self.assertIsNone(row["label"])
        self.assertEqual(row["evidence_count"], 0)

    def test_streaming_jsonl_and_inspection(self):
        rows = [
            {"id": 1, "claim": "Claim one", "label": "SUPPORTS", "evidence": [[[0, 1, "P", 2]]]},
            {"id": 2, "claim": "Claim two", "label": "REFUTES", "evidence": []},
            {"id": 3, "claim": "Claim three", "label": "NOT ENOUGH INFO", "evidence": []},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tiny.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            loaded = list(iter_fever(path))
            self.assertEqual(len(loaded), 3)
            summary = inspect_fever(path, sample_size=10)
            self.assertEqual(summary["sampled_records"], 3)
            self.assertEqual(summary["label_counts"]["supports"], 1)
            self.assertEqual(summary["records_with_evidence"], 1)

    def test_rejects_missing_claim(self):
        with self.assertRaises(ValueError):
            normalize_fever_record({"id": 1, "label": "SUPPORTS"})


if __name__ == "__main__":
    unittest.main()
