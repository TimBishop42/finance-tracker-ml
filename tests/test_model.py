"""Tests for the production TransactionCategorizer in src/ml/model.py.

These tests replaced the previous dead-code tests that exercised LightGBMModel,
a class used nowhere in production.
"""
import pytest
import numpy as np

from src.ml.model import TransactionCategorizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_transaction(business_name: str, comment: str = "") -> dict:
    return {
        "transaction_id": 1,
        "date": "2024-03-20T10:00:00Z",
        "amount": 10.00,
        "business_name": business_name,
        "comment": comment,
    }


def _make_transactions(names):
    return [_make_transaction(n) for n in names]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def categorizer(tmp_path, monkeypatch):
    """Return a fresh TransactionCategorizer backed by a temp model directory."""
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    # Re-import settings so it picks up the patched env var
    import importlib
    import src.config as cfg_mod
    importlib.reload(cfg_mod)
    import src.ml.model as model_mod
    importlib.reload(model_mod)
    return model_mod.TransactionCategorizer()


@pytest.fixture
def sample_transactions():
    names = ["STARBUCKS", "UBER", "WOOLWORTHS", "COLES", "BWS"]
    return _make_transactions(names)


@pytest.fixture
def sample_categories():
    return ["Coffee", "Transport", "Groceries", "Groceries", "Alcohol"]


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def test_model_initialization(categorizer):
    """A freshly created categorizer must have a model, vectorizer, and categories."""
    assert categorizer.model is not None
    assert categorizer.vectorizer is not None
    assert len(categorizer.categories) > 0


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def test_predict_returns_correct_count(categorizer, sample_transactions):
    results = categorizer.predict(sample_transactions)
    assert len(results) == len(sample_transactions)


def test_predict_result_shape(categorizer, sample_transactions):
    results = categorizer.predict(sample_transactions)
    for r in results:
        assert "predicted_category" in r
        assert "confidence_score" in r
        assert 0.0 <= r["confidence_score"] <= 1.0
        assert r["predicted_category"] in categorizer.categories


def test_predict_preserves_transaction_fields(categorizer, sample_transactions):
    results = categorizer.predict(sample_transactions)
    for original, result in zip(sample_transactions, results):
        assert result["business_name"] == original["business_name"]
        assert result["amount"] == original["amount"]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def test_train_without_corrections(categorizer, sample_transactions, sample_categories):
    """Training with user_corrections=None must not raise."""
    categorizer.train(
        transactions=sample_transactions,
        categories=sample_categories,
        confidence_scores=[1.0] * len(sample_transactions),
        user_corrections=None,
    )
    info = categorizer.get_model_info()
    assert info["is_loaded"] is True


def test_train_with_corrections(categorizer, sample_transactions, sample_categories):
    """Training with valid user corrections must succeed and apply them."""
    corrections = {0: "Eating Out"}  # Override index 0 from Coffee -> Eating Out
    categorizer.train(
        transactions=sample_transactions,
        categories=sample_categories,
        confidence_scores=[1.0] * len(sample_transactions),
        user_corrections=corrections,
    )
    info = categorizer.get_model_info()
    assert info["is_loaded"] is True


def test_train_corrections_oob_index_skipped(categorizer, sample_transactions, sample_categories):
    """Out-of-bounds correction indices must be skipped without raising."""
    corrections = {999: "Coffee"}  # way out of bounds
    # Should not raise
    categorizer.train(
        transactions=sample_transactions,
        categories=sample_categories,
        confidence_scores=[1.0] * len(sample_transactions),
        user_corrections=corrections,
    )


def test_train_unknown_correction_category_skipped(categorizer, sample_transactions, sample_categories):
    """Correction with a category not in self.categories must be skipped."""
    corrections = {0: "UNKNOWN_CATEGORY_XYZ"}
    # Should not raise
    categorizer.train(
        transactions=sample_transactions,
        categories=sample_categories,
        confidence_scores=[1.0] * len(sample_transactions),
        user_corrections=corrections,
    )


def test_train_unknown_ground_truth_category_skipped(categorizer, sample_transactions):
    """Transactions with unknown ground-truth categories must be skipped gracefully."""
    mixed_categories = ["Coffee", "TOTALLY_UNKNOWN", "Groceries", "Groceries", "Alcohol"]
    # Should not raise — the unknown row is simply skipped
    categorizer.train(
        transactions=sample_transactions,
        categories=mixed_categories,
        confidence_scores=[1.0] * len(sample_transactions),
        user_corrections=None,
    )


# ---------------------------------------------------------------------------
# Accuracy metrics
# ---------------------------------------------------------------------------

def test_train_reports_accuracy_metrics(categorizer, sample_transactions, sample_categories):
    """Performance metrics must include train_accuracy after training."""
    categorizer.train(
        transactions=sample_transactions,
        categories=sample_categories,
        confidence_scores=[1.0] * len(sample_transactions),
        user_corrections=None,
    )
    metrics = categorizer.get_model_info()["metadata"]["performance_metrics"]
    assert "train_accuracy" in metrics
    assert "val_accuracy" in metrics  # present (may be None for small datasets)


# ---------------------------------------------------------------------------
# Model info
# ---------------------------------------------------------------------------

def test_get_model_info(categorizer):
    info = categorizer.get_model_info()
    assert "version" in info
    assert "categories" in info
    assert info["is_loaded"] is True
