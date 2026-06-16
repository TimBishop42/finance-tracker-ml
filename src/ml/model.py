import joblib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from src.config import settings
import logging

logger = logging.getLogger(__name__)

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
            max_features=2000,  # Increased for better feature capture
            ngram_range=(1, 4),  # Increased to capture more patterns
            stop_words='english',
            min_df=1,  # Allow rare terms
            max_df=0.95,
            analyzer='char_wb'
        )
        
        # Initialize with a more diverse set of business names and their categories
        business_category_map = {
            # Coffee shops
            "STARBUCKS": "Coffee",
            "COSTA": "Coffee",
            "CAFE_NERO": "Coffee",
            "GLORIA_JEANS": "Coffee",
            "COFFEE_CLUB": "Coffee",
            "ZAMBRERO": "Eating Out",
            "ROLLD": "Eating Out",
            "MCDONALDS": "Eating Out",
            "KFC": "Eating Out",
            "HUNGRY_JACKS": "Eating Out",
            "SUBWAY": "Eating Out",
            "GRILLD": "Eating Out",
            
            # Transport
            "UBER": "Transport",
            "LYFT": "Transport",
            "TAXI": "Transport",
            "SYDNEY_TRAINS": "Transport",
            "BUS": "Transport",
            "TRAM": "Transport",
            "FERRY": "Transport",
            
            # Fuel
            "SHELL": "Fuel",
            "BP": "Fuel",
            "CALTEX": "Fuel",
            "7-ELEVEN": "Fuel",
            "METRO": "Fuel",
            
            # Groceries
            "WOOLWORTHS": "Groceries",
            "COLES": "Groceries",
            "ALDI": "Groceries",
            "IGA": "Groceries",
            "FRESH_MARKET": "Groceries",
            "FRUIT_SHOP": "Groceries",
            
            # Pet related
            "VET4PETS": "Vet",
            "VETS": "Vet",
            "PETSTOCK": "Pet Food",
            "PETSATHOME": "Pet Food",
            "PETSHOP": "Pet Food",
            
            # House and home
            "AMAZON": "House",
            "IKEA": "House",
            "BUNNINGS": "House",
            "HARVEY_NORMAN": "House",
            "JB_HIFI": "House",
            "OFFICEWORKS": "House",
            
            # Bills
            "WATER_CORP": "Bills",
            "ELECTRICITY": "Bills",
            "GAS": "Bills",
            "INTERNET": "Bills",
            "PHONE": "Bills",
            "INSURANCE": "Bills",
            
            # Alcohol
            "BWS": "Alcohol",
            "DAN_MURPHYS": "Alcohol",
            "LIQUORLAND": "Alcohol",
            "BOTTLE_SHOP": "Alcohol",
            
            # Chocolate and sweets
            "CADBURY": "Chocolate",
            "NESTLE": "Chocolate",
            "SWEET_SHOP": "Chocolate",
            "CANDY_STORE": "Chocolate",
            
            # Baby
            "BABY_BUNTING": "Baby",
            "MOTHERCARE": "Baby",
            "BABY_SHOP": "Baby",
            "TOYS_R_US": "Baby",
            
            # Miscellaneous
            "POST_OFFICE": "Miscellaneous",
            "NEWSAGENT": "Miscellaneous",
            "PHARMACY": "Miscellaneous",
            "CHEMIST": "Miscellaneous"
        }
        
        # Create training texts and labels
        initial_texts = []
        initial_labels = []
        
        # Add each business name multiple times with variations
        for business, category in business_category_map.items():
            # Add the business name as is
            initial_texts.append(business)
            initial_labels.append(category)
            
            # Add with common suffixes
            for suffix in [" PTY LTD", " STORE", " SHOP", " AUSTRALIA", " SYDNEY", " MELBOURNE", " DIRECT DEBIT", " RECEIPT", " TRANSFER", " PAYMENT"]:
                initial_texts.append(f"{business}{suffix}")
                initial_labels.append(category)
            
            # Add with category name
            initial_texts.append(f"{business} {category}")
            initial_labels.append(category)
            
            # Add with common variations
            if " " in business:
                parts = business.split()
                initial_texts.append("".join(parts))  # Remove spaces
                initial_labels.append(category)
                initial_texts.append("_".join(parts))  # Use underscores
                initial_labels.append(category)
        
        # Fit vectorizer and transform texts
        X = self.vectorizer.fit_transform(initial_texts)
        
        # Initialize categories
        self.categories = settings.DEFAULT_CATEGORIES
        
        # Create label indices
        y = np.array([self.categories.index(cat) for cat in initial_labels])
        
        # Initialize and train model with better parameters
        self.model = lgb.LGBMClassifier(
            objective='multiclass',
            n_estimators=settings.LGBM_N_ESTIMATORS,
            learning_rate=settings.LGBM_LEARNING_RATE,
            num_leaves=settings.LGBM_NUM_LEAVES,
            num_class=len(self.categories),
            random_state=42,
            class_weight='balanced',
            min_child_samples=5,
            min_child_weight=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            importance_type='gain',
            min_data_in_leaf=5,
            min_gain_to_split=0.1,
            max_depth=4
        )
        
        # Train initial model
        self.model.fit(X, y)
        
        # Log feature importance
        feature_names = self.vectorizer.get_feature_names_out()
        importance = self.model.feature_importances_
        top_features = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)[:10]
        logger.info(f"Top 10 important features: {top_features}")
        
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
        logger.info(f"Model saved with {len(self.categories)} categories: {self.categories}")

    def _clear_model(self) -> None:
        """Clear corrupted model file and reinitialize."""
        model_path = Path(settings.MODEL_DIR) / settings.MODEL_FILENAME
        if model_path.exists():
            logger.warning("Removing corrupted model file")
            model_path.unlink()
        logger.info("Reinitializing model from scratch")
        self._initialize_model()

    def predict(self, transactions: List[Dict]) -> List[Dict]:
        """Predict categories for a list of transactions."""
        if not self.model or not self.vectorizer:
            raise RuntimeError("Model not loaded or initialized")

        logger.info(f"Starting prediction for {len(transactions)} transactions")
        logger.info(f"Available categories: {self.categories}")

        # Extract features
        texts = []
        for t in transactions:
            business_name = t.business_name if hasattr(t, 'business_name') else t.get('business_name', '')
            comment = t.comment if hasattr(t, 'comment') else t.get('comment', '')
            # Clean and normalize business name
            business_name = business_name.strip().upper()
            texts.append(f"{business_name} {comment or ''}")
        
        logger.info(f"Extracted texts for prediction: {texts}")
        
        # Transform texts to features
        features = self.vectorizer.transform(texts)
        logger.info(f"Feature matrix shape: {features.shape}")
        logger.info(f"Feature names: {self.vectorizer.get_feature_names_out()}")

        # Get predictions
        probabilities = self.model.predict_proba(features)
        predictions = self.model.predict(features)
        logger.info(f"Raw predictions: {predictions}")
        logger.info(f"Prediction probabilities: {probabilities}")
        logger.info(f"Probabilities shape: {probabilities.shape}")
        logger.info(f"Current categories count: {len(self.categories)}")
        logger.info(f"Model classes: {getattr(self.model, 'classes_', 'N/A')}")

        # NOTE (ML-C3 fix): do NOT reset the model here. A fitted classifier's
        # predict_proba width equals the number of categories actually seen during
        # training, which is normally fewer than len(self.categories) (the full
        # category list). The previous code treated that as "corruption" and called
        # _clear_model() mid-request, discarding the freshly-trained model and
        # serving every prediction from the synthetic baseline. predict() returns a
        # valid category index into self.categories, and predict_proba columns map
        # back to category indices via self.model.classes_ (handled below).
        model_classes = list(getattr(self.model, "classes_", range(len(self.categories))))

        # Format results
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            transaction = transactions[i]
            pred_idx = int(pred)  # Convert to int once and reuse
            
            # Final safety check
            if pred_idx >= len(self.categories):
                logger.warning(f"Prediction index {pred_idx} still out of bounds, using fallback")
                pred_idx = 0  # Fallback to first category
            
            # Handle both Pydantic models and dicts
            transaction_id = transaction.transaction_id if hasattr(transaction, 'transaction_id') else transaction.get('transaction_id')
            date = transaction.date if hasattr(transaction, 'date') else transaction.get('date')
            amount = transaction.amount if hasattr(transaction, 'amount') else transaction.get('amount')
            business_name = transaction.business_name if hasattr(transaction, 'business_name') else transaction.get('business_name')
            comment = transaction.comment if hasattr(transaction, 'comment') else transaction.get('comment')
            
            # Get top 3 predictions. probs is indexed by the model's class columns,
            # so map each column position back to its category index via model_classes.
            top_cols = np.argsort(probs)[-3:][::-1]
            top_categories = []
            top_probs = []
            for col in top_cols:
                cat_idx = int(model_classes[col])
                if 0 <= cat_idx < len(self.categories):
                    top_categories.append(self.categories[cat_idx])
                    top_probs.append(float(probs[col]))
            
            result = {
                "transaction_id": transaction_id,
                "date": date,
                "amount": amount,
                "business_name": business_name,
                "comment": comment,
                "predicted_category": self.categories[pred_idx],
                "confidence_score": float(np.max(probs)),
                "top_predictions": list(zip(top_categories, top_probs))
            }
            logger.info(f"Formatted result for transaction {i}: {result}")
            results.append(result)

        return results

    def train(self, transactions: List[Dict], categories: List[str],
             confidence_scores: List[float], user_corrections: Optional[Dict[int, str]] = None) -> None:
        """Train the model with new data including user corrections."""
        if not self.model or not self.vectorizer:
            raise RuntimeError("Model not loaded or initialized")

        logger.info(f"Starting model training with {len(transactions)} transactions")
        logger.info(f"Current categories: {self.categories}")
        logger.info(f"Training categories (ground truth): {categories}")
        logger.info(f"User corrections: {user_corrections}")

        # If this is a large training set (>100 transactions), treat as full retrain
        is_full_retrain = len(transactions) > 100
        if is_full_retrain:
            logger.info(f"Large training set detected ({len(transactions)} transactions), performing full retrain")
            # Clear and reinitialize model and vectorizer for full retrain
            self._clear_model()

        # Prepare training data
        texts = []
        for t in transactions:
            business_name = t.business_name if hasattr(t, 'business_name') else t.get('business_name', '')
            comment = t.comment if hasattr(t, 'comment') else t.get('comment', '')
            # Clean and normalize business name
            business_name = business_name.strip().upper()
            texts.append(f"{business_name} {comment or ''}")

        logger.info(f"Training texts: {texts}")

        # Update categories if needed - add any new categories from training data.
        # ML-H7: guard user_corrections before calling .values()
        all_categories = set(self.categories)
        all_categories.update(categories)
        if user_corrections:
            all_categories.update(user_corrections.values())

        if all_categories != set(self.categories):
            logger.info(f"Updating categories from {self.categories} to {sorted(all_categories)}")
            self.categories = sorted(all_categories)

            # Reinitialize model with new number of categories (ML-M10: use settings)
            self.model = lgb.LGBMClassifier(
                objective='multiclass',
                n_estimators=settings.LGBM_N_ESTIMATORS,
                learning_rate=settings.LGBM_LEARNING_RATE,
                num_leaves=settings.LGBM_NUM_LEAVES,
                num_class=len(self.categories),
                random_state=42,
                class_weight='balanced',
                min_child_samples=5,
                min_child_weight=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                importance_type='gain',
                min_data_in_leaf=5,
                min_gain_to_split=0.1,
                max_depth=4
            )
        else:
            # Keep existing model but update parameters to prevent overfitting if needed
            if not hasattr(self.model, 'n_estimators') or self.model.n_estimators > settings.LGBM_N_ESTIMATORS:
                logger.info("Updating model parameters to prevent overfitting")
                self.model = lgb.LGBMClassifier(
                    objective='multiclass',
                    n_estimators=settings.LGBM_N_ESTIMATORS,
                    learning_rate=settings.LGBM_LEARNING_RATE,
                    num_leaves=settings.LGBM_NUM_LEAVES,
                    num_class=len(self.categories),
                    random_state=42,
                    class_weight='balanced',
                    min_child_samples=5,
                    min_child_weight=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    importance_type='gain',
                    min_data_in_leaf=5,
                    min_gain_to_split=0.1,
                    max_depth=4
                )

        # ML-M5: for incremental training freeze the vocabulary; only use fit_transform
        # on a full retrain so IDF weights are not distorted by synthetic tokens.
        if is_full_retrain:
            logger.info("Full retrain: refitting vectorizer on all training data")
            X_train = self.vectorizer.fit_transform(texts)
        else:
            logger.info("Incremental training: freezing vectorizer vocabulary (transform only)")
            X_train = self.vectorizer.transform(texts)

        # ML-M9: build label array, skipping transactions whose category is not in
        # self.categories (unknown categories are logged as warnings).
        valid_indices = []
        y_list = []
        for i, cat in enumerate(categories):
            if cat not in self.categories:
                logger.warning(
                    f"Unknown category '{cat}' for transaction index {i}; skipping."
                )
                continue
            valid_indices.append(i)
            y_list.append(self.categories.index(cat))

        if not valid_indices:
            raise ValueError("No valid training samples after filtering unknown categories.")

        X_train = X_train[valid_indices]
        y = np.array(y_list)

        # ML-H7 + ML-H8: apply user corrections with None guard and bounds check
        if user_corrections:
            for idx, corrected_cat in user_corrections.items():
                # ML-H8: bounds check against the (possibly filtered) label array
                if idx >= len(y):
                    logger.warning(
                        f"User correction index {idx} is out of bounds (y length {len(y)}); skipping."
                    )
                    continue
                if corrected_cat in self.categories:
                    logger.info(f"Applying user correction: transaction {idx} -> {corrected_cat}")
                    y[idx] = self.categories.index(corrected_cat)
                else:
                    logger.warning(f"Unknown category '{corrected_cat}' in user corrections; skipping.")

        logger.info(f"Final training labels: {y}")
        logger.info(f"Label distribution: {np.bincount(y)}")
        logger.info(f"Training data shape: {X_train.shape}")

        unique_labels = len(np.unique(y))
        if unique_labels < 2:
            logger.warning(f"Only {unique_labels} unique labels found. Model may not learn effectively.")

        # ML-M6: 80/20 train/validation split for honest accuracy reporting.
        # Skip split when there are fewer than 20 samples.
        n_samples = len(y)
        if n_samples >= 20:
            X_fit, X_val, y_fit, y_val = train_test_split(
                X_train, y, test_size=0.2, random_state=42, stratify=y if unique_labels > 1 else None
            )
            self.model.fit(X_fit, y_fit)
            train_accuracy = float(self.model.score(X_fit, y_fit))
            val_accuracy = float(self.model.score(X_val, y_val))
            logger.info(f"Train accuracy: {train_accuracy:.4f}, Val accuracy: {val_accuracy:.4f}")
        else:
            logger.info(f"Fewer than 20 samples ({n_samples}); skipping train/val split.")
            self.model.fit(X_train, y)
            train_accuracy = float(self.model.score(X_train, y))
            val_accuracy = None

        # Update performance metrics
        self.model_metadata["performance_metrics"] = {
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "n_samples": n_samples,
            "n_categories": len(self.categories),
            "unique_labels": unique_labels
        }

        logger.info(f"Model training completed. Train accuracy: {train_accuracy}, Val accuracy: {val_accuracy}")

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