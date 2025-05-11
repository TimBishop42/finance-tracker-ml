from typing import List

from fastapi import APIRouter, HTTPException

from ..models.schemas import CategorizedTransaction, PredictRequest, PredictResponse
from ...ml.model.lightgbm_model import LightGBMModel

router = APIRouter()
model = LightGBMModel()  # In production, this would be loaded from disk


@router.post("/batch", response_model=PredictResponse)
async def predict_batch(request: PredictRequest) -> PredictResponse:
    try:
        # Extract features
        business_names = [t.business_name for t in request.transactions]
        amounts = [t.amount for t in request.transactions]
        
        # Get predictions
        predicted_categories, confidence_scores = model.predict(business_names, amounts)
        
        # Create response
        categorized_transactions = [
            CategorizedTransaction(
                **transaction.dict(),
                predicted_category=category,
                confidence_score=score,
            )
            for transaction, category, score in zip(
                request.transactions, predicted_categories, confidence_scores
            )
        ]
        
        return PredictResponse(categorized_transactions=categorized_transactions)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 