from typing import List, Tuple

import lightgbm as lgb
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from .base import BaseModel


class LightGBMModel(BaseModel):
    def __init__(self):
        self.model = None
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words="english",
        )
        self.scaler = StandardScaler()
        self.categories = None

    def _prepare_features_train(
        self, business_names: List[str], amounts: List[float]
    ) -> np.ndarray:
        """Prepare features for model training - fits transformers."""
        # Convert business names to TF-IDF features
        text_features = self.vectorizer.fit_transform(business_names).toarray()
        
        # Scale amount features
        amount_features = self.scaler.fit_transform(np.array(amounts).reshape(-1, 1))
        
        # Combine features
        return np.hstack([text_features, amount_features])

    def _prepare_features_predict(
        self, business_names: List[str], amounts: List[float]
    ) -> np.ndarray:
        """Prepare features for prediction - uses fitted transformers."""
        # Convert business names to TF-IDF features
        text_features = self.vectorizer.transform(business_names).toarray()
        
        # Scale amount features
        amount_features = self.scaler.transform(np.array(amounts).reshape(-1, 1))
        
        # Combine features
        return np.hstack([text_features, amount_features])

    def predict(self, business_names: List[str], amounts: List[float]) -> Tuple[List[str], List[float]]:
        if self.model is None:
            raise RuntimeError("Model not trained or loaded")

        features = self._prepare_features_predict(business_names, amounts)
        probabilities = self.model.predict_proba(features)
        
        # Get predicted categories and confidence scores
        predicted_indices = np.argmax(probabilities, axis=1)
        confidence_scores = np.max(probabilities, axis=1)
        
        predicted_categories = [self.categories[idx] for idx in predicted_indices]
        
        return predicted_categories, confidence_scores.tolist()

    def train(
        self,
        business_names: List[str],
        amounts: List[float],
        categories: List[str],
        confidence_scores: List[float],
        user_corrections: dict[int, str] | None = None,
    ) -> None:
        # Update categories if needed
        if self.categories is None:
            self.categories = sorted(set(categories))
        
        # Apply user corrections if provided
        if user_corrections:
            for idx, corrected_category in user_corrections.items():
                categories[idx] = corrected_category
        
        # Prepare features using training method
        features = self._prepare_features_train(business_names, amounts)
        
        # Convert categories to indices
        category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        labels = np.array([category_to_idx[cat] for cat in categories])
        
        # Initialize or update model with better parameters to prevent overfitting
        self.model = lgb.LGBMClassifier(
            n_estimators=50,  # Reduced to prevent overfitting
            learning_rate=0.05,  # Reduced learning rate
            num_leaves=15,  # Reduced complexity
            min_child_samples=5,  # Prevent overfitting on small datasets
            min_split_gain=0.1,  # Require minimum gain for splits
            subsample=0.8,  # Use bagging
            colsample_bytree=0.8,  # Feature bagging
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=0.1,  # L2 regularization
            random_state=42,
        )
        
        # Train model
        self.model.fit(
            features,
            labels,
            sample_weight=confidence_scores,  # Weight by confidence
        )

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("No model to save")
        
        import joblib
        
        model_data = {
            "model": self.model,
            "vectorizer": self.vectorizer,
            "scaler": self.scaler,
            "categories": self.categories,
        }
        joblib.dump(model_data, path)

    def load(self, path: str) -> None:
        import joblib
        
        model_data = joblib.load(path)
        self.model = model_data["model"]
        self.vectorizer = model_data["vectorizer"]
        self.scaler = model_data["scaler"]
        self.categories = model_data["categories"] 