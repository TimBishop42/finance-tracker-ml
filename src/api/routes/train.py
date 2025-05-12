from fastapi import APIRouter, HTTPException
from src.api.models.schemas import TrainRequest
from src.ml.model import TransactionCategorizer
from src.api.main import TRAINING_COUNTER

router = APIRouter()
categorizer = TransactionCategorizer()

@router.post("")
async def train(request: TrainRequest):
    """Train the model with new data including user corrections."""
    try:
        categorizer.train(
            request.transactions,
            request.categories,
            request.confidence_scores,
            request.user_corrections
        )
        TRAINING_COUNTER.labels(status="success").inc()
        return {"status": "success"}
    except Exception as e:
        TRAINING_COUNTER.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=str(e)) 