from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    database_url: str = "postgresql://support:support_pass@localhost:5432/support_intelligence"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "ticket_resolutions"
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "support-intelligence"
    model_dir: str = "./models"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    transformer_model: str = "distilbert-base-uncased"
    data_dir: str = "./data"
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    retrieval_top_k: int = 10
    rerank_top_k: int = 5
    anomaly_zscore_threshold: float = 2.0
    anomaly_window_days: int = 30

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache
def get_settings() -> Settings:
    return Settings()
