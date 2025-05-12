# Finance Transaction Categorization ML Service

A Python-based ML service for categorizing financial transactions using LightGBM. This service exposes a REST API that can be called by the main Java backend to categorize transactions in batch.

## Architecture

### Core Components
- **FastAPI**: High-performance async API framework
- **LightGBM**: Primary ML model for transaction categorization
- **Poetry**: Dependency management
- **Docker**: Containerization
- **PostgreSQL**: Database for category management and model metadata

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

4. **Database Integration**:
   - PostgreSQL for category management
   - Model metadata storage
   - Training history tracking

## Development

### Prerequisites
- Python 3.9+
- Poetry
- Docker
- Docker Compose

### Local Development

1. **Standalone Service**:
```bash
# Install dependencies
poetry install

# Run tests
poetry run pytest

# Start service
poetry run uvicorn src.api.main:app --reload
```

2. **Using Docker**:
```bash
# Build and run with docker-compose
docker-compose up --build

# Or use just commands
just docker-build-push
```

3. **Using Master Docker Compose**:
```bash
# From the main finance-tracker repository
docker-compose up -d finance-tracker-ml
```

### Available Commands

The project includes a `justfile` with common development commands:

```bash
just install          # Install dependencies
just run             # Run service locally
just test            # Run tests
just test-cov        # Run tests with coverage
just format          # Format code
just type-check      # Run type checking
just docker-build    # Build Docker image
just docker-push     # Push Docker image
just docker-build-push # Build and push Docker image
just docker-run      # Run Docker container
just clean           # Clean up cache files
```

### Configuration

The service can be configured using environment variables or a `.env` file:

```env
# API Settings
API_V1_STR=/api/v1
PROJECT_NAME="Finance Transaction Categorization ML Service"

# Model Settings
MODEL_DIR=models
MODEL_FILENAME=model.joblib

# ML Settings
TFIDF_MAX_FEATURES=1000
TFIDF_NGRAM_RANGE=(1,2)
LGBM_N_ESTIMATORS=100
LGBM_LEARNING_RATE=0.1
LGBM_NUM_LEAVES=31

# Database Settings
DB_HOST=postgres
DB_PORT=5432
DB_NAME=finance
DB_USER=finance-user
DB_PASSWORD=change_me

# CORS Settings
BACKEND_CORS_ORIGINS=["http://localhost:3000","http://localhost:8080"]
```

### Integration with Main Project

This service is designed to be integrated with the main finance-tracker project:

1. **Network**: Uses the `finance_network` Docker network
2. **Database**: Connects to the shared PostgreSQL instance
3. **API**: Exposes endpoints for the Java backend to call
4. **Volume**: Uses persistent storage for model files

### Health Checks

The service includes health checks for:
- API availability
- Model loading
- Database connectivity
- Resource usage

### Monitoring

The service exposes metrics for:
- Prediction latency
- Model accuracy
- Training performance
- Resource utilization
