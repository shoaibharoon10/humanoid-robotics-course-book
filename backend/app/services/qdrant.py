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
        """Search for similar documents using qdrant-client 1.12+ API."""
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

        # Use query_points (qdrant-client 1.12+) instead of deprecated search()
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=search_filter,
            score_threshold=score_threshold
        )

        # query_points returns QueryResponse with .points attribute
        return [
            {
                "id": str(point.id),
                "score": point.score,
                "payload": point.payload
            }
            for point in results.points
        ]

    def delete_collection(self):
        """Delete the entire collection."""
        self.client.delete_collection(collection_name=self.collection_name)

    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information."""
        try:
            info = self.client.get_collection(self.collection_name)
            # Handle different qdrant-client versions
            points_count = getattr(info, 'points_count', None) or 0
            vectors_count = getattr(info, 'vectors_count', None) or points_count
            status = getattr(info, 'status', None)
            status_value = status.value if status else "unknown"
            return {
                "name": self.collection_name,
                "vectors_count": vectors_count,
                "points_count": points_count,
                "status": status_value
            }
        except Exception as e:
            return {"error": str(e)}

    def check_health(self) -> Dict[str, Any]:
        """
        Check Qdrant health with detailed status.
        Returns healthy if collection exists and has vectors.
        """
        try:
            # First check if we can connect to Qdrant
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]

            # Check if our collection exists
            if self.collection_name not in collection_names:
                return {
                    "healthy": False,
                    "status": "collection_missing",
                    "detail": f"Collection '{self.collection_name}' not found. Available: {collection_names}",
                    "collection_name": self.collection_name
                }

            # Get collection details - handle different qdrant-client versions
            info = self.client.get_collection(self.collection_name)

            # Try different attribute names for vector/point count
            points_count = getattr(info, 'points_count', None) or 0
            vectors_count = getattr(info, 'vectors_count', None) or points_count

            return {
                "healthy": True,
                "status": "connected",
                "detail": f"Collection '{self.collection_name}' has {points_count} points",
                "collection_name": self.collection_name,
                "points_count": points_count
            }

        except Exception as e:
            return {
                "healthy": False,
                "status": "connection_error",
                "detail": str(e),
                "collection_name": self.collection_name
            }


# Singleton instance
_qdrant_service = None


def get_qdrant_service() -> QdrantService:
    """Get the Qdrant service singleton."""
    global _qdrant_service
    if _qdrant_service is None:
        _qdrant_service = QdrantService()
    return _qdrant_service
