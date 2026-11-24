"""Logistic-regression sentiment classifier over TF-IDF features."""
from __future__ import annotations

import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .data import load_corpus


class SentimentModel:
    def __init__(self, vectorizer: TfidfVectorizer, classifier: LogisticRegression):
        self.vectorizer = vectorizer
        self.classifier = classifier

    def predict(self, text: str) -> str:
        label = int(self.classifier.predict(self.vectorizer.transform([text]))[0])
        return "positive" if label == 1 else "negative"

    def predict_proba(self, text: str) -> dict:
        probs = self.classifier.predict_proba(self.vectorizer.transform([text]))[0]
        return {"negative": round(float(probs[0]), 4), "positive": round(float(probs[1]), 4)}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump({"vectorizer": self.vectorizer, "classifier": self.classifier}, fh)

    @classmethod
    def load(cls, path: str | Path) -> "SentimentModel":
        with Path(path).open("rb") as fh:
            bundle = pickle.load(fh)
        return cls(bundle["vectorizer"], bundle["classifier"])


def train_default_model() -> SentimentModel:
    texts, labels = load_corpus()
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
    features = vectorizer.fit_transform(texts)
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(features, labels)
    return SentimentModel(vectorizer, classifier)
