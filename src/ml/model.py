import joblib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
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
            n_estimators=50,  # Reduced to prevent overfitting
            learning_rate=0.05,  # Reduced learning rate  
            num_leaves=15,  # Reduced complexity
            num_class=len(self.categories),
            random_state=42,
            class_weight='balanced',
            min_child_samples=5,  # Prevent overfitting
            min_child_weight=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            importance_type='gain',
            min_data_in_leaf=5,  # Increased from 1
            min_gain_to_split=0.1,  # Increased from 0.0
            max_depth=4  # Reduced complexity
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

        # CRITICAL FIX: Validate that model predictions match current categories
        max_pred_idx = np.max(predictions) if len(predictions) > 0 else -1
        if max_pred_idx >= len(self.categories):
            logger.error(f"Model prediction index {max_pred_idx} exceeds categories bounds {len(self.categories)-1}.")
            logger.error(f"This indicates model/categories mismatch. Clearing and reinitializing model...")
            # Clear corrupted model and reinitialize
            self._clear_model()
            # Retry prediction
            probabilities = self.model.predict_proba(features)
            predictions = self.model.predict(features)
            logger.info(f"Retried predictions: {predictions}")

        # Additional validation for probabilities shape
        if probabilities.shape[1] != len(self.categories):
            logger.error(f"Probabilities shape {probabilities.shape} doesn't match categories {len(self.categories)}")
            logger.error("Clearing and reinitializing model to fix shape mismatch...")
            self._clear_model()
            probabilities = self.model.predict_proba(features)
            predictions = self.model.predict(features)

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
            
            # Get top 3 predictions with bounds checking
            top_indices = np.argsort(probs)[-3:][::-1]
            top_categories = []
            top_probs = []
            for idx in top_indices:
                if idx < len(self.categories):
                    top_categories.append(self.categories[idx])
                    top_probs.append(float(probs[idx]))
            
            result = {
                "transaction_id": transaction_id,
                "date": date,
                "amount": amount,
                "business_name": business_name,
                "comment": comment,
                "predicted_category": self.categories[pred_idx],
                "confidence_score": float(probs[pred_idx]),
                "top_predictions": list(zip(top_categories, top_probs))
            }
            logger.info(f"Formatted result for transaction {i}: {result}")
            results.append(result)

        return results

    def train(self, transactions: List[Dict], categories: List[str], 
             confidence_scores: List[float], user_corrections: Dict[int, str]) -> None:
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
        
        # Update categories if needed - add any new categories from training data
        all_categories = set(self.categories)
        all_categories.update(categories)
        all_categories.update(user_corrections.values())
        
        if all_categories != set(self.categories):
            logger.info(f"Updating categories from {self.categories} to {sorted(all_categories)}")
            self.categories = sorted(all_categories)
            
            # Reinitialize model with new number of categories
            self.model = lgb.LGBMClassifier(
                objective='multiclass',
                n_estimators=50,  # Reduced to prevent overfitting
                learning_rate=0.05,  # Reduced learning rate
                num_leaves=15,  # Reduced complexity
                num_class=len(self.categories),
                random_state=42,
                class_weight='balanced',
                min_child_samples=5,  # Prevent overfitting
                min_child_weight=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                importance_type='gain',
                min_data_in_leaf=5,  # Increased from 1
                min_gain_to_split=0.1,  # Increased from 0.0
                max_depth=4  # Reduced complexity
            )
        else:
            # Keep existing model but update parameters to prevent overfitting if needed
            if not hasattr(self.model, 'n_estimators') or self.model.n_estimators > 100:
                logger.info("Updating model parameters to prevent overfitting")
                self.model = lgb.LGBMClassifier(
                    objective='multiclass',
                    n_estimators=50,
                    learning_rate=0.05,
                    num_leaves=15,
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
        
        # Handle vectorizer differently for full retrain vs incremental training
        if is_full_retrain:
            # For full retrain, refit vectorizer on all training data
            logger.info("Full retrain: refitting vectorizer on all training data")
            X_train = self.vectorizer.fit_transform(texts)
        else:
            # For incremental training, preserve existing vectorizer vocabulary
            logger.info("Incremental training: using existing vectorizer")
            old_vocabulary = getattr(self.vectorizer, 'vocabulary_', {})
            old_feature_count = len(old_vocabulary) if old_vocabulary else 0
            
            # Create expanded training corpus that includes both new data and representative old patterns
            expanded_texts = texts.copy()
            
            # Add some representative patterns from existing vocabulary to preserve knowledge
            if old_vocabulary:
                # Add top vocabulary terms as synthetic examples
                vocab_items = sorted(old_vocabulary.items(), key=lambda x: x[1])[:100]  # Top 100 terms
                for term, _ in vocab_items:
                    if len(term) > 3:  # Only meaningful terms
                        expanded_texts.append(term)
            
            logger.info(f"Expanded training corpus size: {len(expanded_texts)} (original: {len(texts)})")
            
            # Refit vectorizer on expanded corpus
            X = self.vectorizer.fit_transform(expanded_texts)
            new_feature_count = X.shape[1]
            
            # If feature dimensions changed, we need to reinitialize the model
            if new_feature_count != old_feature_count:
                logger.info(f"Feature dimensions changed from {old_feature_count} to {new_feature_count}, reinitializing model")
                self.model = lgb.LGBMClassifier(
                    objective='multiclass',
                    n_estimators=50,  # Reduced to prevent overfitting
                    learning_rate=0.05,  # Reduced learning rate
                    num_leaves=15,  # Reduced complexity
                    num_class=len(self.categories),
                    random_state=42,
                    class_weight='balanced',
                    min_child_samples=5,  # Prevent overfitting
                    min_child_weight=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    importance_type='gain',
                    min_data_in_leaf=5,  # Increased from 1
                    min_gain_to_split=0.1,  # Increased from 0.0
                    max_depth=4  # Reduced complexity
                )
            
            # Transform only the actual training texts
            X_train = self.vectorizer.transform(texts)
        
        # Prepare labels from the GROUND TRUTH categories, not predictions
        y = np.array([self.categories.index(cat) for cat in categories])
        
        # Apply user corrections (these override the ground truth)
        for idx, corrected_cat in user_corrections.items():
            if corrected_cat in self.categories:
                logger.info(f"Applying user correction: transaction {idx} -> {corrected_cat}")
                y[idx] = self.categories.index(corrected_cat)
            else:
                logger.warning(f"Unknown category {corrected_cat} in user corrections")

        logger.info(f"Final training labels: {y}")
        logger.info(f"Label distribution: {np.bincount(y)}")
        logger.info(f"Training data shape: {X_train.shape}")

        # Check if we have enough variety in the data
        unique_labels = len(np.unique(y))
        if unique_labels < 2:
            logger.warning(f"Only {unique_labels} unique labels found. Model may not learn effectively.")
        
        # Train model
        self.model.fit(X_train, y)

        # Update performance metrics
        train_accuracy = self.model.score(X_train, y)
        self.model_metadata["performance_metrics"] = {
            "train_accuracy": train_accuracy,
            "n_samples": len(transactions),
            "n_categories": len(self.categories),
            "unique_labels": unique_labels
        }

        logger.info(f"Model training completed with accuracy: {train_accuracy}")

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