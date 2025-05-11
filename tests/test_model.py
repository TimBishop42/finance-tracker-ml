import pytest
from sklearn.metrics import accuracy_score

from src.ml.model.lightgbm_model import LightGBMModel


@pytest.fixture
def model():
    return LightGBMModel()


@pytest.fixture
def sample_data():
    business_names = [
        "STARBUCKS",
        "UBER",
        "NETFLIX",
        "AMAZON",
        "WALMART",
    ]
    amounts = [25.50, 15.00, 12.99, 49.99, 100.00]
    categories = ["FOOD", "TRANSPORT", "ENTERTAINMENT", "SHOPPING", "SHOPPING"]
    return business_names, amounts, categories


def test_model_initialization(model):
    assert model.model is None
    assert model.categories is None


def test_model_training(model, sample_data):
    business_names, amounts, categories = sample_data
    confidence_scores = [1.0] * len(categories)
    
    model.train(
        business_names=business_names,
        amounts=amounts,
        categories=categories,
        confidence_scores=confidence_scores,
    )
    
    assert model.model is not None
    assert model.categories is not None
    assert len(model.categories) == len(set(categories))


def test_model_prediction(model, sample_data):
    business_names, amounts, categories = sample_data
    confidence_scores = [1.0] * len(categories)
    
    # Train the model
    model.train(
        business_names=business_names,
        amounts=amounts,
        categories=categories,
        confidence_scores=confidence_scores,
    )
    
    # Test prediction
    predicted_categories, confidence_scores = model.predict(business_names, amounts)
    
    assert len(predicted_categories) == len(business_names)
    assert len(confidence_scores) == len(business_names)
    assert all(0 <= score <= 1 for score in confidence_scores)


def test_model_save_load(model, sample_data, tmp_path):
    business_names, amounts, categories = sample_data
    confidence_scores = [1.0] * len(categories)
    
    # Train the model
    model.train(
        business_names=business_names,
        amounts=amounts,
        categories=categories,
        confidence_scores=confidence_scores,
    )
    
    # Save model
    model_path = tmp_path / "model.joblib"
    model.save(str(model_path))
    
    # Create new model and load
    new_model = LightGBMModel()
    new_model.load(str(model_path))
    
    # Compare predictions
    orig_preds, _ = model.predict(business_names, amounts)
    new_preds, _ = new_model.predict(business_names, amounts)
    
    assert orig_preds == new_preds


def test_user_corrections(model, sample_data):
    business_names, amounts, categories = sample_data
    confidence_scores = [1.0] * len(categories)
    
    # Train initial model
    model.train(
        business_names=business_names,
        amounts=amounts,
        categories=categories,
        confidence_scores=confidence_scores,
    )
    
    # Get initial predictions
    initial_preds, _ = model.predict(business_names, amounts)
    
    # Train with corrections
    corrections = {0: "ENTERTAINMENT"}  # Change first prediction
    model.train(
        business_names=business_names,
        amounts=amounts,
        categories=categories,
        confidence_scores=confidence_scores,
        user_corrections=corrections,
    )
    
    # Get new predictions
    new_preds, _ = model.predict(business_names, amounts)
    
    # Verify predictions changed
    assert new_preds != initial_preds 