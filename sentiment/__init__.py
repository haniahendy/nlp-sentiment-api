"""Tiny sentiment analysis package: train, persist, and predict."""
from .model import SentimentModel, train_default_model

__all__ = ["SentimentModel", "train_default_model"]
