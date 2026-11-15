# Evidence-Aware Fake News Detection (EAFND)

A lightweight, local-first research prototype. The first milestone implements input validation,
text normalization, and a stable article/claim record. It does **not** yet make a fake/real
prediction or claim to verify evidence.

## Dataset decision

Start with **FEVER** as the first evidence-verification benchmark. FEVER provides claims
labelled `SUPPORTS`, `REFUTES`, or `NOT ENOUGH INFO`, with annotated evidence for many
examples. It is suitable for testing the evidence-verification part of this project.

Important limitation: FEVER is a fact-verification dataset, not a news-article corpus.
After the evidence pipeline works, add a news dataset such as FakeNewsNet (GossipCop/PolitiFact)
for a separate news-classification evaluation. Do not merge scores across these datasets as
if they were one benchmark.

## Requirements

- Python 3.10+
- No API key or large model required for Milestone 1

## Setup (macOS)

```bash
cd evidence_aware_fnd
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## First module: article/claim ingestion

Run the demo:

```bash
python -m src.main
```

Run tests:

```bash
python -m unittest discover -s tests -v
```

The module:
1. accepts a headline, body, optional source URL, language, and metadata;
2. normalizes whitespace and Unicode text without destroying punctuation;
3. validates required fields and URL format;
4. creates a reproducible JSON record with a SHA-256 content identifier.

## Planned milestones

1. **Input and normalization** — current.
2. FEVER dataset adapter and label mapping.
3. Claim/event extraction.
4. Evidence retrieval and source provenance.
5. Evidence support/refutation assessment.
6. A baseline classifier and selective/uncertain output.
7. Evaluation: macro-F1, per-class precision/recall, evidence coverage, and ablations.

## Research integrity

The system currently has no trained classifier, retrieval index, or evidence verifier.
Do not describe Milestone 1 as a fake-news detector. It is the data-ingestion foundation.
