"""Small built-in training corpus.

The corpus is synthetic and ships with the repo so training works offline
with no downloads. It is intentionally small; the model is a baseline, not
a production sentiment system.
"""

POSITIVE = [
    "I loved this product, it works perfectly",
    "absolutely fantastic experience, highly recommend",
    "great quality and fast delivery",
    "the support team was wonderful and helpful",
    "best purchase I have made this year",
    "excellent value for the price",
    "super easy to set up and use",
    "works exactly as described, very happy",
    "brilliant design and solid build",
    "five stars, would buy again",
    "the results exceeded my expectations",
    "smooth, reliable, and well documented",
    "outstanding service from start to finish",
    "really impressed with the performance",
    "simple, clean, and effective",
    "a joy to use every day",
]

NEGATIVE = [
    "terrible quality, broke after one day",
    "worst purchase ever, do not buy",
    "completely useless and a waste of money",
    "the support team never responded",
    "arrived damaged and missing parts",
    "horrible experience from start to finish",
    "does not work as described at all",
    "very disappointed, returning it immediately",
    "poor build quality and cheap materials",
    "constant crashes and bugs everywhere",
    "slow, confusing, and frustrating",
    "not worth the price at all",
    "the worst service I have ever received",
    "regret buying this, total disappointment",
    "misleading description, nothing works",
    "avoid this product, it is awful",
]


def load_corpus():
    """Return (texts, labels) with label 1 = positive, 0 = negative."""
    texts = POSITIVE + NEGATIVE
    labels = [1] * len(POSITIVE) + [0] * len(NEGATIVE)
    return texts, labels
