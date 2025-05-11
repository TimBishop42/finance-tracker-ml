from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np


class BaseModel(ABC):
    """Base class for all transaction categorization models."""

    @abstractmethod
    def predict(self, business_names: List[str], amounts: List[float]) -> Tuple[List[str], List[float]]:
        """
        Predict categories and confidence scores for a batch of transactions.

        Args:
            business_names: List of business names
            amounts: List of transaction amounts

        Returns:
            Tuple of (predicted_categories, confidence_scores)
        """
        pass

    @abstractmethod
    def train(
        self,
        business_names: List[str],
        amounts: List[float],
        categories: List[str],
        confidence_scores: List[float],
        user_corrections: dict[int, str] | None = None,
    ) -> None:
        """
        Train the model on new data.

        Args:
            business_names: List of business names
            amounts: List of transaction amounts
            categories: List of true categories
            confidence_scores: List of confidence scores from previous predictions
            user_corrections: Optional dict mapping transaction indices to corrected categories
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the model to disk."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the model from disk."""
        pass 