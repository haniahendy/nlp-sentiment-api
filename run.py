"""CLI: train the baseline model and classify text.

Usage:
    python run.py train --out models/sentiment.pkl
    python run.py predict "this product is great" [--model models/sentiment.pkl]
"""
from __future__ import annotations

import argparse

from sentiment import SentimentModel, train_default_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline sentiment classifier")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="train on the built-in corpus")
    train_p.add_argument("--out", default="models/sentiment.pkl")

    pred_p = sub.add_parser("predict", help="classify a piece of text")
    pred_p.add_argument("text")
    pred_p.add_argument("--model", default="models/sentiment.pkl")

    args = parser.parse_args()

    if args.command == "train":
        model = train_default_model()
        model.save(args.out)
        print(f"saved model to {args.out}")
    else:
        model = SentimentModel.load(args.model)
        print(model.predict(args.text), model.predict_proba(args.text))


if __name__ == "__main__":
    main()
