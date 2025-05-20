from typing import List

from fastapi import APIRouter, HTTPException

from ..models.schemas import CategorizedTransaction, PredictRequest, PredictResponse
from src.ml.model import TransactionCategorizer
from src.api.metrics import PREDICTION_COUNTER, PREDICTION_LATENCY
from src.api.logging import get_logger
import time

router = APIRouter()
categorizer = TransactionCategorizer()
logger = get_logger(__name__)

@router.post("/batch", response_model=PredictResponse)
async def predict_batch(request: PredictRequest) -> PredictResponse:
    """Predict categories for a batch of transactions."""
    try:
        logger.info(f"Processing batch prediction request with {len(request.transactions)} transactions")
        start_time = time.time()
        
        categorized_transactions = categorizer.predict(request.transactions)
        
        latency = time.time() - start_time
        PREDICTION_LATENCY.labels(endpoint="batch").observe(latency)
        PREDICTION_COUNTER.labels(endpoint="batch").inc()
        
        logger.info(f"Batch prediction completed in {latency:.2f}s for {len(categorized_transactions)} transactions")
        return PredictResponse(categorized_transactions=categorized_transactions)
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model-info")
async def get_model_info():
    """Get model metadata and performance metrics."""
    try:
        logger.debug("Model info requested")
        return categorizer.get_model_info()
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 