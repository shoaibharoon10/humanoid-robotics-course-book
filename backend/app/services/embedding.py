"""Google Gemini embedding service."""

from typing import List
import google.generativeai as genai

from app.config import get_settings


class EmbeddingService:
    """Service for generating embeddings using Google Gemini."""

    def __init__(self):
        settings = get_settings()
        genai.configure(api_key=settings.google_api_key)
        self.model = settings.embedding_model

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        result = genai.embed_content(
            model=self.model,
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']

    def embed_texts(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Generate embeddings for multiple texts with batching."""
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            for text in batch:
                result = genai.embed_content(
                    model=self.model,
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])

        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a search query."""
        result = genai.embed_content(
            model=self.model,
            content=query,
            task_type="retrieval_query"
        )
        return result['embedding']


# Singleton instance
_embedding_service = None


def get_embedding_service() -> EmbeddingService:
    """Get the embedding service singleton."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
