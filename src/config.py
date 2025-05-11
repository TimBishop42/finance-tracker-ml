from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Finance Transaction Categorization ML Service"
    
    # Model Settings
    MODEL_DIR: Path = Path("models")
    MODEL_FILENAME: str = "model.joblib"
    DEFAULT_CATEGORIES: list[str] = [
        "FOOD",
        "TRANSPORT",
        "ENTERTAINMENT",
        "SHOPPING",
        "BILLS",
        "INCOME",
        "OTHER",
    ]
    
    # ML Settings
    TFIDF_MAX_FEATURES: int = 1000
    TFIDF_NGRAM_RANGE: tuple[int, int] = (1, 2)
    LGBM_N_ESTIMATORS: int = 100
    LGBM_LEARNING_RATE: float = 0.1
    LGBM_NUM_LEAVES: int = 31
    
    # CORS Settings
    BACKEND_CORS_ORIGINS: list[str] = ["*"]
    
    class Config:
        case_sensitive = True
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings() 