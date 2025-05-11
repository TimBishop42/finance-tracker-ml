from fastapi import APIRouter, HTTPException

from ..models.schemas import TrainRequest
from ...ml.model.lightgbm_model import LightGBMModel

router = APIRouter()
model = LightGBMModel()  # In production, this would be loaded from disk


@router.post("", status_code=204)
async def train(request: TrainRequest) -> None:
    try:
        # Extract features
        business_names = [t.business_name for t in request.transactions]
        amounts = [t.amount for t in request.transactions]
        
        # Train model
        model.train(
            business_names=business_names,
            amounts=amounts,
            categories=request.categories,
            confidence_scores=request.confidence_scores,
            user_corrections=request.user_corrections,
        )
        
        # Save model (in production, this would be to a persistent storage)
        model.save("model.joblib")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 