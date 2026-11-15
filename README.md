# 🔎 EvidenceAware — Evidence-Aware Fake News Detection

**EvidenceAware (EAFND)** is an AI-assisted fake news detection and evidence analysis research prototype. It combines machine-learning classification, event and entity extraction, Wikipedia evidence retrieval, and LLM-assisted evidence assessment.

The goal is to move beyond simple fake/real classification by providing supporting context and traceable evidence for human review.

## Features

- **News classification:** Predicts FAKE or REAL using a trained ISOT text classifier.
- **Confidence scores:** Displays fake and real probabilities.
- **Event extraction:** Extracts events, entities, dates, locations, and attribution using an LLM.
- **Evidence retrieval:** Searches an indexed Wikipedia corpus for relevant pages and sentences.
- **Evidence assessment:** Uses an LLM to classify retrieved sentences as SUPPORTS, REFUTES, or INSUFFICIENT.
- **Interactive interface:** Presents predictions, extracted entities, retrieved evidence, and technical details in a Streamlit application.

## Architecture

The current prototype follows this workflow:

1. **Input:** Accept a headline or news article.
2. **Classification:** Generate a preliminary fake/real prediction.
3. **Normalization:** Extract event information and named entities.
4. **Retrieval:** Search Wikipedia for potentially relevant evidence.
5. **Verification:** Assess retrieved sentences against the submitted claim.
6. **Output:** Display classification probabilities, extracted information, evidence assessments, and limitations.

## Tech Stack

- Python
- Streamlit
- Scikit-learn
- XGBoost
- SQLite / FTS5
- Groq API — `openai/gpt-oss-20b`
- TF-IDF and classical machine-learning models
- Wikipedia / FEVER evidence corpus

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/BixbyCodes/fakenewd.git
cd fakenewd
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the API

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b
```

Obtain an API key from [Groq Console](https://console.groq.com/).

Never commit your API key.

### 5. Prepare the datasets

The datasets and large Wikipedia index are not included in this GitHub repository.

You will need the relevant local datasets and the processed Wikipedia SQLite index to run the complete evidence retrieval workflow.

The expected Wikipedia index location is:

```text
data/processed/wiki_pages.sqlite
```

### 6. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal.

## Project Structure

```text
evidence_aware_fnd/
│
├── app.py
├── src/
│   ├── classical.py
│   ├── data.py
│   ├── event_parser.py
│   ├── evidence_retriever.py
│   ├── evidence_verifier.py
│   ├── indexed_evidence_retriever.py
│   ├── ingestion.py
│   ├── llm_features.py
│   ├── pipeline.py
│   ├── predict.py
│   └── title_search.py
│
├── models/
│   ├── classical/
│   └── fusion/
│
├── results/
├── tests/
├── requirements.txt
└── README.md
```

## Datasets and Evaluation

The project uses different datasets for different research tasks:

| Dataset | Purpose |
|---|---|
| ISOT | Fake news classification |
| LIAR | Short-claim classification |
| FEVER | Evidence retrieval and fact verification |

These datasets represent different tasks and should not be treated as one unified benchmark.

### Selected experimental results

- ISOT Linear SVM: approximately 99.04% test accuracy.
- LIAR Linear SVM: approximately 64.54% test accuracy.
- FEVER Wikipedia title retrieval: Recall@10 of 83.8% on 1,000 evaluated claims.
- Gold-reference evidence recovery: 99.37% in the evaluated sample.

**Important:** Gold-reference recovery is not end-to-end retrieval accuracy. The classification results are dataset-specific and may reflect dataset artifacts. The hybrid experiments used limited LLM annotations and should not be interpreted as a fully evaluated hybrid system.

## Limitations

- Classification predictions are not definitive fact-checking verdicts.
- Wikipedia retrieval may miss relevant evidence.
- LLM-generated explanations and evidence labels can be incorrect.
- The current evidence corpus is based on Wikipedia and FEVER, not comprehensive live news sources.
- Performance on ISOT and LIAR does not establish real-world generalization.
- Human review is recommended for uncertain or high-impact claims.

## Research Objective

EvidenceAware explores how machine-learning classification, event normalization, evidence retrieval, and LLM-assisted reasoning can work together in a unified fake news detection framework.

Further work includes improving retrieval, evaluating evidence assessments, integrating graph-based context, and conducting systematic ablation studies.

## Author

**Utsav Tripathi**

GitHub: [@BixbyCodes](https://github.com/BixbyCodes)

Repository: [BixbyCodes/fakenewd](https://github.com/BixbyCodes/fakenewd)

---

*EvidenceAware is a research prototype. Its predictions and evidence assessments should be treated as decision-support signals, not authoritative truth judgments.*
