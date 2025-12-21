"""Qdrant vector database service."""

from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    VectorParams,
    Distance,
    Filter,
    FieldCondition,
    MatchValue,
)
import uuid

from app.config import get_settings


class QdrantService:
    """Service for Qdrant vector operations."""

    # Default collection name fallback
    DEFAULT_COLLECTION_NAME = "humanoid_robotics_docs"

    def __init__(self):
        settings = get_settings()
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
        # Use collection_name from settings with fallback
        self.collection_name = getattr(settings, 'collection_name', None) or self.DEFAULT_COLLECTION_NAME
        self.vector_size = 768  # Google text-embedding-004 dimension

    def ensure_collection(self):
        """Create collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
            print(f"Created collection: {self.collection_name}")
        return exists

    def upsert_points(self, points: List[Dict[str, Any]]):
        """Upsert points to the collection."""
        qdrant_points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=point["vector"],
                payload=point["payload"]
            )
            for point in points
        ]

        self.client.upsert(
            collection_name=self.collection_name,
            points=qdrant_points
        )

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter_module: Optional[str] = None,
        score_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        # Build filter if module specified
        search_filter = None
        if filter_module:
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="module",
                        match=MatchValue(value=filter_module)
                    )
                ]
            )

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=search_filter,
            score_threshold=score_threshold
        )

        return [
            {
                "id": str(result.id),
                "score": result.score,
                "payload": result.payload
            }
            for result in results
        ]

    def delete_collection(self):
        """Delete the entire collection."""
        self.client.delete_collection(collection_name=self.collection_name)

    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status.value
            }
        except Exception as e:
            return {"error": str(e)}


# Singleton instance
_qdrant_service = None


def get_qdrant_service() -> QdrantService:
    """Get the Qdrant service singleton."""
    global _qdrant_service
    if _qdrant_service is None:
        _qdrant_service = QdrantService()
    return _qdrant_service
