from typing import List

from fastapi import APIRouter, HTTPException

from ..models.schemas import CategorizedTransaction, PredictRequest, PredictResponse
from ...ml.model.lightgbm_model import LightGBMModel
from src.ml.model import TransactionCategorizer
from src.api.main import PREDICTION_COUNTER, PREDICTION_LATENCY
import time

router = APIRouter()
model = LightGBMModel()  # In production, this would be loaded from disk
categorizer = TransactionCategorizer()


@router.post("/batch", response_model=PredictResponse)
async def predict_batch(request: PredictRequest) -> PredictResponse:
    """Predict categories for a batch of transactions."""
    try:
        start_time = time.time()
        categorized_transactions = categorizer.predict(request.transactions)
        PREDICTION_LATENCY.labels(endpoint="batch").observe(time.time() - start_time)
        PREDICTION_COUNTER.labels(endpoint="batch").inc()
        return PredictResponse(categorized_transactions=categorized_transactions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model-info")
async def get_model_info():
    """Get model metadata and performance metrics."""
    try:
        return categorizer.get_model_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 