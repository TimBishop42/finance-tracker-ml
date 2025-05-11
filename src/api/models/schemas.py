from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class Transaction(BaseModel):
    transaction_id: int
    date: datetime
    amount: float
    business_name: str
    comment: Optional[str] = None


class PredictRequest(BaseModel):
    transactions: List[Transaction] = Field(..., min_items=1, max_items=200)
    categories: Optional[List[str]] = None


class CategorizedTransaction(Transaction):
    predicted_category: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    categorized_transactions: List[CategorizedTransaction]


class TrainRequest(BaseModel):
    transactions: List[Transaction] = Field(..., min_items=1)
    categories: List[str] = Field(..., min_items=1)
    confidence_scores: List[float] = Field(..., min_items=1)
    user_corrections: Optional[Dict[int, str]] = None 