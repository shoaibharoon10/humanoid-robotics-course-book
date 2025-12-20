"""OpenAI embedding service."""

from typing import List
from openai import AsyncOpenAI

from app.config import get_settings


class EmbeddingService:
    """Service for generating embeddings using OpenAI."""

    def __init__(self):
        settings = get_settings()
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.embedding_model

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        response = await self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding

    async def embed_texts(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Generate embeddings for multiple texts with batching."""
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = await self.client.embeddings.create(
                model=self.model,
                input=batch
            )
            embeddings.extend([e.embedding for e in response.data])

        return embeddings


# Singleton instance
_embedding_service = None


def get_embedding_service() -> EmbeddingService:
    """Get the embedding service singleton."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
