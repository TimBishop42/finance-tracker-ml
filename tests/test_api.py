from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_transaction():
    return {
        "transaction_id": 12345,
        "date": "2024-03-20T10:00:00Z",
        "amount": 25.50,
        "business_name": "STARBUCKS",
        "comment": "Coffee purchase",
    }


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_batch(client, sample_transaction):
    request_data = {
        "transactions": [sample_transaction],
        "categories": ["FOOD", "TRANSPORT", "ENTERTAINMENT"],
    }
    
    response = client.post("/api/v1/predict/batch", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "categorized_transactions" in data
    assert len(data["categorized_transactions"]) == 1
    
    transaction = data["categorized_transactions"][0]
    assert transaction["transaction_id"] == sample_transaction["transaction_id"]
    assert transaction["business_name"] == sample_transaction["business_name"]
    assert "predicted_category" in transaction
    assert "confidence_score" in transaction
    assert 0 <= transaction["confidence_score"] <= 1


def test_predict_batch_validation(client):
    # Test empty transactions
    response = client.post("/api/v1/predict/batch", json={"transactions": []})
    assert response.status_code == 422
    
    # Test too many transactions
    response = client.post(
        "/api/v1/predict/batch",
        json={"transactions": [{"transaction_id": i, "date": "2024-03-20T10:00:00Z", "amount": 25.50, "business_name": "TEST"} for i in range(201)]},
    )
    assert response.status_code == 422


def test_train_endpoint(client, sample_transaction):
    request_data = {
        "transactions": [sample_transaction],
        "categories": ["FOOD"],
        "confidence_scores": [0.95],
        "user_corrections": {"0": "ENTERTAINMENT"},
    }
    
    response = client.post("/api/v1/train", json=request_data)
    assert response.status_code == 204


def test_train_endpoint_validation(client, sample_transaction):
    # Test missing required fields
    response = client.post("/api/v1/train", json={"transactions": [sample_transaction]})
    assert response.status_code == 422
    
    # Test mismatched lengths
    response = client.post(
        "/api/v1/train",
        json={
            "transactions": [sample_transaction],
            "categories": ["FOOD", "TRANSPORT"],  # Mismatched length
            "confidence_scores": [0.95],
        },
    )
    assert response.status_code == 422 