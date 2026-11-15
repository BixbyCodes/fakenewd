"""Streaming FEVER JSONL loader and label normalization.

Python 3.9 compatible. Does not load the full dataset into RAM.
"""

import gzip
import json
from pathlib import Path
from typing import Dict, Iterator, Optional, Union

LABEL_MAP = {
    "SUPPORTS": "supports",
    "REFUTES": "refutes",
    "NOT ENOUGH INFO": "not_enough_info",
}

PathLike = Union[str, Path]


def open_jsonl(path: PathLike):
    """Open plain or gzip-compressed JSONL as UTF-8 text."""
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(str(path), "rt", encoding="utf-8")
    return open(str(path), "r", encoding="utf-8")


def normalize_fever_record(raw: Dict) -> Dict:
    """Convert one raw FEVER row into the project's stable schema."""
    if not isinstance(raw, dict):
        raise ValueError("Each FEVER line must contain a JSON object.")

    claim = raw.get("claim")
    if not isinstance(claim, str) or not claim.strip():
        raise ValueError("FEVER record is missing a valid 'claim'.")

    raw_label = raw.get("label")
    label = LABEL_MAP.get(raw_label) if raw_label is not None else None

    evidence_sets = raw.get("evidence") or []
    evidence = []

    for group in evidence_sets:
        for item in group:
            if not isinstance(item, (list, tuple)) or len(item) < 4:
                continue

            # Ignore placeholder annotations without actual evidence.
            if item[2] is None or item[3] is None:
                continue

            evidence.append({
                "annotation_id": item[0],
                "evidence_id": item[1],
                "page": item[2],
                "sentence_id": item[3],
            })

    return {
        "claim_id": raw.get("id"),
        "claim": claim.strip(),
        "label": label,
        "evidence": evidence,
        "evidence_count": len(evidence),
        "source_dataset": "FEVER",
    }


def iter_fever(path: PathLike, limit: Optional[int] = None) -> Iterator[Dict]:
    """Yield normalized records one at a time."""
    count = 0
    with open_jsonl(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                yield normalize_fever_record(raw)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    "Invalid FEVER row at line {}: {}".format(line_number, exc)
                )
            count += 1
            if limit is not None and count >= limit:
                break


def inspect_fever(path: PathLike, sample_size: int = 1000) -> Dict:
    """Count labels and evidence availability over a bounded sample."""
    counts = {
        "supports": 0,
        "refutes": 0,
        "not_enough_info": 0,
        "unlabelled": 0,
    }
    with_evidence = 0
    total = 0

    for row in iter_fever(path, limit=sample_size):
        total += 1
        label = row["label"] or "unlabelled"
        counts[label] = counts.get(label, 0) + 1
        if row["evidence_count"] > 0:
            with_evidence += 1

    return {
        "sampled_records": total,
        "label_counts": counts,
        "records_with_evidence": with_evidence,
    }
