# nlp-sentiment-api

A small, offline-trainable sentiment classifier: TF-IDF features plus
logistic regression, with a CLI for training and prediction.

## What it does

- Ships with a small built-in synthetic corpus (32 labeled sentences) so
  training works with no downloads and no external data.
- `run.py train` fits the model and pickles it; `run.py predict` loads the
  pickle and classifies new text with class probabilities.
- The model is wrapped in a `SentimentModel` class so it can be embedded in
  a service (a FastAPI/Flask wrapper can call `predict`/`predict_proba`
  directly).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python run.py train --out models/sentiment.pkl
python run.py predict "this product is great"
```

## Tests

```bash
pytest -q
```

## Honest limits

- The training corpus is tiny and synthetic. The model separates obvious
  positive/negative phrasing but is **not** a production sentiment system
  and has not been evaluated on a real benchmark.
- No API server is included; `SentimentModel` is the integration point.
