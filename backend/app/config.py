"""Application configuration using Pydantic Settings."""

from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # OpenAI
    openai_api_key: str
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    max_tokens: int = 1024
    temperature: float = 0.7

    # Database
    database_url: str

    # Qdrant
    qdrant_url: str
    qdrant_api_key: str
    collection_name: str = "humanoid_robotics_docs"

    # Application
    app_env: str = "development"
    cors_origins: str = "http://localhost:3000"

    # Retrieval
    retrieval_top_k: int = 5
    similarity_threshold: float = 0.7

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins from comma-separated string."""
        return [origin.strip() for origin in self.cors_origins.split(",")]

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.app_env == "production"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
