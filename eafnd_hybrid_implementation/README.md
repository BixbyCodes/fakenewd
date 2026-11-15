# Evidence-Aware Fake News Detection (EAFND)

This package implements a reproducible hybrid research pipeline using:
1. TF-IDF + Logistic Regression, Linear SVM, Random Forest
2. RoBERTa-base binary classifier
3. Groq LLM-derived structured features (zero-shot)
4. XGBoost late-fusion meta-classifier
5. Robustness, ablation, cross-dataset evaluation and SHAP

## Important methodological notes
- ISOT is sampled to 5,000 real + 5,000 fake articles with seed 42.
- LIAR preserves the official train/valid/test files and maps `true`, `mostly-true`
  to real (0), and the remaining four labels to fake (1). This binary mapping is
  an operational choice and must be reported as such.
- ISOT has no official split. The pipeline uses stratified 70/15/15 splits.
- RoBERTa training is optional on an 8 GB Mac. Use `--skip-roberta` for the first run.
- Groq features are cached in SQLite. API calls are made only for rows missing from cache.
- Cross-dataset results are directional transfer tests, not directly comparable to
  in-domain test scores.
- This implementation does not claim exact reproduction of a Qwen model if Groq
  GPT-OSS is used; record the actual model in results/metadata.json.
- The supplied ISOT/LIAR sources have known limitations (publisher/topic leakage,
  short political statements vs long news articles). Interpret results accordingly.

## Setup
From the repository root:
```bash
source .venv/bin/activate
pip install -r eafnd_hybrid_implementation/requirements.txt
cp .env.example .env
```
Keep `.env` private and ignored by git.

## Run
Copy the package's `src` and `scripts` folders into your repository, or run it in
this folder after placing `data/raw/isot/{True.csv,Fake.csv}` and
`data/raw/liar/{train.tsv,valid.tsv,test.tsv}` under `data/raw`.

First run the fast pipeline:
```bash
python -m src.pipeline --skip-roberta --llm-limit 300
```
Then increase `--llm-limit` to process the full split. For the full research run,
train RoBERTa separately:
```bash
python -m src.train_roberta --epochs 2
python -m src.pipeline --use-roberta --llm-limit 1000
```
The pipeline writes metrics and artifacts under `results/` and `models/`.

Do not present a small `--llm-limit` run as full-dataset results. The pipeline reports
sample sizes and model IDs in metadata.
