# Finance Transaction Categorization ML Service

A Python-based ML service for categorizing financial transactions using LightGBM. This service exposes a REST API that can be called by the main Java backend to categorize transactions in batch.

## Architecture

### Core Components
- **FastAPI**: High-performance async API framework
- **LightGBM**: Primary ML model for transaction categorization
- **Poetry**: Dependency management
- **Docker**: Containerization

### Project Structure
```
finance-tracker-ml/
├── src/
│   ├── api/          # API routes and request/response models
│   ├── ml/           # ML model implementation and training
│   └── config.py     # Configuration management
└── tests/            # Test suite
```

## API Documentation

### Batch Prediction Endpoint

`POST /api/v1/predict/batch`

Predicts categories for a batch of transactions.

#### Request
```json
{
  "transactions": [
    {
      "transaction_id": 12345,
      "date": "2024-03-20T10:00:00Z",
      "amount": 25.50,
      "business_name": "STARBUCKS",
      "comment": "Coffee purchase"
    }
  ],
  "categories": ["FOOD", "TRANSPORT", "ENTERTAINMENT"] // Optional, if not provided will use default categories
}
```

#### Response
```json
{
  "categorized_transactions": [
    {
      "transaction_id": 12345,
      "date": "2024-03-20T10:00:00Z",
      "amount": 25.50,
      "business_name": "STARBUCKS",
      "comment": "Coffee purchase",
      "predicted_category": "FOOD",
      "confidence_score": 0.95
    }
  ]
}
```

### Training Endpoint

`POST /api/v1/train`

Updates the model with new training data from user corrections.

#### Request
```json
{
  "transactions": [
    {
      "transaction_id": 12345,
      "date": "2024-03-20T10:00:00Z",
      "amount": 25.50,
      "business_name": "STARBUCKS",
      "comment": "Coffee purchase"
    }
  ],
  "categories": ["FOOD"],
  "confidence_scores": [0.95],
  "user_corrections": {
    "0": "ENTERTAINMENT"  // Index of transaction -> corrected category
  }
}
```

#### Response
- 204 No Content on success
- 400 Bad Request if training data is invalid
- 500 Internal Server Error if training fails

## Design Choices

1. **ML Model**: LightGBM
   - Fast training and prediction
   - Good handling of categorical features
   - Extensible architecture to support other models in future

2. **Batch Processing**:
   - All prediction requests are batch-based
   - Optimized for 1-200 transactions per request
   - Async processing for better performance

3. **Model Management**:
   - Model versioning support
   - Training data persistence
   - Performance monitoring

## Development

### Prerequisites
- Python 3.9+
- Poetry
- Docker (optional)

### Setup
```bash
# Install dependencies
poetry install

# Run tests
poetry run pytest

# Start service
poetry run uvicorn src.api.main:app --reload
```

### Docker
```bash
# Build image
docker build -t finance-tracker-ml .

# Run container
docker run -p 8000:8000 finance-tracker-ml
```
