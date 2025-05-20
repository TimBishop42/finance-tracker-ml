import joblib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb
from src.config import settings

class TransactionCategorizer:
    def __init__(self):
        self.model: Optional[lgb.LGBMClassifier] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.categories: List[str] = []
        self.model_version: str = "1.0.0"
        self.model_metadata: Dict = {
            "version": self.model_version,
            "created_at": datetime.utcnow().isoformat(),
            "last_trained": None,
            "performance_metrics": {}
        }
        self._load_model()

    def _load_model(self) -> None:
        """Load the model and vectorizer from disk."""
        model_path = Path(settings.MODEL_DIR) / settings.MODEL_FILENAME
        if model_path.exists():
            model_data = joblib.load(model_path)
            self.model = model_data["model"]
            self.vectorizer = model_data["vectorizer"]
            self.categories = model_data["categories"]
            self.model_metadata = model_data.get("metadata", self.model_metadata)
        else:
            self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize a new model with default settings."""
        self.vectorizer = TfidfVectorizer(
            max_features=settings.TFIDF_MAX_FEATURES,
            ngram_range=settings.TFIDF_NGRAM_RANGE
        )
        
        # Initialize with some common business names and their categories
        business_category_map = {
            "STARBUCKS": "Coffee",
            "COSTA": "Coffee",
            "CAFE_NERO": "Coffee",
            "UBER": "Transport",
            "LYFT": "Transport",
            "TAXI": "Transport",
            "SHELL": "Fuel",
            "BP": "Fuel",
            "TESCO": "Groceries",
            "SAINSBURYS": "Groceries",
            "ASDA": "Groceries",
            "VET4PETS": "Vet",
            "VETS": "Vet",
            "PETSATHOME": "Pet Food",
            "PETSHOP": "Pet Food",
            "AMAZON": "House",
            "IKEA": "House",
            "WATER_COMPANY": "Bills",
            "ELECTRIC_COMPANY": "Bills",
            "GAS_COMPANY": "Bills",
            "WINE_SHOP": "Alcohol",
            "BEER_SHOP": "Alcohol",
            "CHOCOLATE_SHOP": "Chocolate",
            "SWEET_SHOP": "Chocolate",
            "BABY_SHOP": "Baby",
            "MOTHERCARE": "Baby",
            "RESTAURANT": "Eating Out",
            "CAFE": "Eating Out",
            "MISCELLANEOUS_SHOP": "Miscellaneous"
        }
        
        # Create training texts by combining business names with their categories
        initial_texts = [f"{business} {category}" for business, category in business_category_map.items()]
        self.vectorizer.fit(initial_texts)
        
        self.model = lgb.LGBMClassifier(
            n_estimators=settings.LGBM_N_ESTIMATORS,
            learning_rate=settings.LGBM_LEARNING_RATE,
            num_leaves=settings.LGBM_NUM_LEAVES
        )
        self.categories = settings.DEFAULT_CATEGORIES
        
        # Train initial model with dummy data
        X = self.vectorizer.transform(initial_texts)
        y = np.zeros(len(initial_texts))  # All transactions in first category
        self.model.fit(X, y)
        
        # Save the initialized model
        self._save_model()

    def _save_model(self) -> None:
        """Save the model, vectorizer, and metadata to disk."""
        model_path = Path(settings.MODEL_DIR)
        model_path.mkdir(exist_ok=True)
        
        # Update metadata
        self.model_metadata["last_trained"] = datetime.utcnow().isoformat()
        
        # Save model data
        model_data = {
            "model": self.model,
            "vectorizer": self.vectorizer,
            "categories": self.categories,
            "metadata": self.model_metadata
        }
        joblib.dump(model_data, model_path / settings.MODEL_FILENAME)

    def predict(self, transactions: List[Dict]) -> List[Dict]:
        """Predict categories for a list of transactions."""
        if not self.model or not self.vectorizer:
            raise RuntimeError("Model not loaded or initialized")

        # Extract features
        texts = [f"{t.business_name} {t.comment or ''}" for t in transactions]
        features = self.vectorizer.transform(texts)

        # Get predictions
        probabilities = self.model.predict_proba(features)
        predictions = self.model.predict(features)

        # Format results
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            transaction = transactions[i]
            pred_idx = int(pred)  # Convert to int once and reuse
            results.append({
                "transaction_id": transaction.transaction_id,
                "date": transaction.date,
                "amount": transaction.amount,
                "business_name": transaction.business_name,
                "comment": transaction.comment,
                "predicted_category": self.categories[pred_idx],
                "confidence_score": float(probs[pred_idx])
            })

        return results

    def train(self, transactions: List[Dict], categories: List[str], 
             confidence_scores: List[float], user_corrections: Dict[int, str]) -> None:
        """Train the model with new data including user corrections."""
        if not self.model or not self.vectorizer:
            raise RuntimeError("Model not loaded or initialized")

        # Prepare training data
        texts = [f"{t['business_name']} {t.get('comment', '')}" for t in transactions]
        X = self.vectorizer.fit_transform(texts)
        
        # If categories is empty, use user_corrections as the source of truth
        if not categories and user_corrections:
            # Create a list of categories from user corrections
            categories = [""] * len(transactions)  # Initialize with empty strings
            for idx, cat in user_corrections.items():
                categories[idx] = cat
        
        # Update categories if needed
        new_categories = set(categories)
        if new_categories != set(self.categories):
            self.categories = list(new_categories)
            self.model = lgb.LGBMClassifier(
                n_estimators=settings.LGBM_N_ESTIMATORS,
                learning_rate=settings.LGBM_LEARNING_RATE,
                num_leaves=settings.LGBM_NUM_LEAVES
            )

        # Prepare labels
        y = np.array([self.categories.index(cat) for cat in categories])
        
        # Apply user corrections (this will override any existing categories)
        for idx, corrected_cat in user_corrections.items():
            y[idx] = self.categories.index(corrected_cat)

        # Train model
        self.model.fit(X, y)

        # Update performance metrics
        train_accuracy = self.model.score(X, y)
        self.model_metadata["performance_metrics"] = {
            "train_accuracy": train_accuracy,
            "n_samples": len(transactions),
            "n_categories": len(self.categories)
        }

        # Save updated model
        self._save_model()

    def get_model_info(self) -> Dict:
        """Get model metadata and performance metrics."""
        return {
            "version": self.model_version,
            "metadata": self.model_metadata,
            "categories": self.categories,
            "is_loaded": self.model is not None
        } 