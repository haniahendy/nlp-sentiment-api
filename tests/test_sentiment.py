from sentiment import SentimentModel, train_default_model
from sentiment.data import load_corpus


def test_corpus_is_balanced():
    texts, labels = load_corpus()
    assert len(texts) == len(labels)
    assert labels.count(1) == labels.count(0)


def test_model_fits_training_corpus():
    model = train_default_model()
    texts, labels = load_corpus()
    expected = ["positive" if y == 1 else "negative" for y in labels]
    assert [model.predict(t) for t in texts] == expected


def test_predict_proba_sums_to_one():
    model = train_default_model()
    probs = model.predict_proba("great product")
    assert abs(probs["positive"] + probs["negative"] - 1.0) < 1e-6


def test_obvious_sentiment_direction():
    model = train_default_model()
    assert model.predict("absolutely fantastic and wonderful") == "positive"
    assert model.predict("terrible, worst experience ever") == "negative"


def test_save_and_load_roundtrip(tmp_path):
    model = train_default_model()
    out = tmp_path / "m.pkl"
    model.save(out)
    loaded = SentimentModel.load(out)
    assert loaded.predict("great quality and fast delivery") == model.predict("great quality and fast delivery")
