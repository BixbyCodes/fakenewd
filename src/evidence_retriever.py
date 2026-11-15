"""Memory-efficient exact evidence retrieval from FEVER Wikipedia shards."""

import glob
import json
from pathlib import Path
from typing import Dict, Optional


def extract_sentence(page: Dict, sentence_id: int) -> Optional[str]:
    """Extract a sentence using its original FEVER sentence ID."""
    lines = page.get("lines", "")

    if not isinstance(lines, str):
        return None

    for line in lines.splitlines():
        parts = line.split("\t", 2)

        if len(parts) < 2:
            continue

        try:
            current_id = int(parts[0])
        except ValueError:
            continue

        if current_id == sentence_id:
            sentence = parts[1].strip()
            return sentence if sentence else None

    return None


def retrieve_evidence(
    page_id: str,
    sentence_id: int,
    corpus_dir: str = "data/raw/fever/wiki-pages",
) -> Optional[Dict]:
    """Search Wikipedia shards and retrieve one exact evidence sentence."""

    shard_paths = sorted(
        glob.glob(str(Path(corpus_dir) / "wiki-*.jsonl"))
    )

    if not shard_paths:
        raise FileNotFoundError(
            "No Wikipedia shards found in: {}".format(corpus_dir)
        )

    for shard_path in shard_paths:
        with open(shard_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue

                try:
                    page = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if page.get("id") != page_id:
                    continue

                sentence = extract_sentence(page, sentence_id)

                if sentence is None:
                    return {
                        "page_id": page_id,
                        "sentence_id": sentence_id,
                        "sentence": None,
                        "found": True,
                        "error": "Page found, but sentence was empty or missing.",
                        "shard": Path(shard_path).name,
                    }

                return {
                    "page_id": page_id,
                    "sentence_id": sentence_id,
                    "sentence": sentence,
                    "found": True,
                    "error": None,
                    "shard": Path(shard_path).name,
                }

    return {
        "page_id": page_id,
        "sentence_id": sentence_id,
        "sentence": None,
        "found": False,
        "error": "Wikipedia page not found.",
        "shard": None,
    }
