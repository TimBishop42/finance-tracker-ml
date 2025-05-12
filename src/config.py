from functools import lru_cache
from pathlib import Path
from typing import Optional, List, Union
from pydantic import AnyHttpUrl, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Finance Transaction Categorization ML Service"
    
    # Model Settings
    MODEL_DIR: str = "models"
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
    TFIDF_NGRAM_RANGE: tuple = (1, 2)
    LGBM_N_ESTIMATORS: int = 100
    LGBM_LEARNING_RATE: float = 0.1
    LGBM_NUM_LEAVES: int = 31
    
    # Database settings
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "finance"
    DB_USER: str = "finance-user"
    DB_PASSWORD: str = "change_me"
    
    # CORS Settings
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []

    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    class Config:
        case_sensitive = True
        env_file = ".env"

# Create settings instance
settings = Settings()

# Export settings
__all__ = ["settings"] 