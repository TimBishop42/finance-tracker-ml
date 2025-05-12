from fastapi import APIRouter, HTTPException
from src.api.models.schemas import TrainRequest
from src.ml.model import TransactionCategorizer
from src.api.metrics import TRAINING_COUNTER
from src.api.logging import get_logger

router = APIRouter()
categorizer = TransactionCategorizer()
logger = get_logger(__name__)

@router.post("")
async def train(request: TrainRequest):
    """Train the model with new data including user corrections."""
    try:
        logger.info(f"Starting model training with {len(request.transactions)} transactions")
        logger.info(f"User corrections: {request.user_corrections}")
        
        categorizer.train(
            request.transactions,
            request.categories,
            request.confidence_scores,
            request.user_corrections
        )
        
        TRAINING_COUNTER.labels(status="success").inc()
        logger.info("Model training completed successfully")
        return {"status": "success"}
    except Exception as e:
        TRAINING_COUNTER.labels(status="error").inc()
        logger.error(f"Error in model training: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 