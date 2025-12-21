"""RAG retrieval service."""

from typing import List, Optional
from app.services.embedding import get_embedding_service
from app.services.qdrant import get_qdrant_service
from app.models.schemas import Source, SearchResult


class RetrievalService:
    """Service for RAG document retrieval."""

    def __init__(self):
        self.embedding_service = get_embedding_service()
        self.qdrant_service = get_qdrant_service()

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_module: Optional[str] = None,
        score_threshold: float = 0.7
    ) -> List[Source]:
        """Retrieve relevant documents for a query."""
        # Generate query embedding (using query-optimized embedding)
        query_embedding = self.embedding_service.embed_query(query)

        # Search in Qdrant
        results = self.qdrant_service.search(
            query_vector=query_embedding,
            top_k=top_k,
            filter_module=filter_module,
            score_threshold=score_threshold
        )

        # Convert to Source objects
        sources = []
        for result in results:
            payload = result["payload"]
            sources.append(Source(
                title=payload.get("title", "Untitled"),
                url_path=payload.get("url_path", ""),
                section=payload.get("heading", None),
                relevance_score=result["score"],
                snippet=self._truncate_text(payload.get("text", ""), 300)
            ))

        return sources

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filter_module: Optional[str] = None
    ) -> List[SearchResult]:
        """Search for documents without chat context."""
        # Generate query embedding (using query-optimized embedding)
        query_embedding = self.embedding_service.embed_query(query)

        # Search in Qdrant
        results = self.qdrant_service.search(
            query_vector=query_embedding,
            top_k=top_k,
            filter_module=filter_module,
            score_threshold=0.5  # Lower threshold for search
        )

        # Convert to SearchResult objects
        search_results = []
        for result in results:
            payload = result["payload"]
            search_results.append(SearchResult(
                doc_id=payload.get("doc_id", ""),
                title=payload.get("title", "Untitled"),
                section=payload.get("heading", ""),
                url_path=payload.get("url_path", ""),
                snippet=self._truncate_text(payload.get("text", ""), 200),
                score=result["score"]
            ))

        return search_results

    def build_context(self, sources: List[Source], max_tokens: int = 3000) -> str:
        """Build context string from retrieved sources."""
        context_parts = []
        current_length = 0

        for i, source in enumerate(sources, 1):
            source_text = f"[Source {i}: {source.title}]\n{source.snippet}\n"
            source_length = len(source_text)

            if current_length + source_length > max_tokens * 4:  # Rough char estimate
                break

            context_parts.append(source_text)
            current_length += source_length

        return "\n".join(context_parts)

    @staticmethod
    def _truncate_text(text: str, max_length: int) -> str:
        """Truncate text to max length."""
        if len(text) <= max_length:
            return text
        return text[:max_length - 3] + "..."


# Singleton instance
_retrieval_service = None


def get_retrieval_service() -> RetrievalService:
    """Get the retrieval service singleton."""
    global _retrieval_service
    if _retrieval_service is None:
        _retrieval_service = RetrievalService()
    return _retrieval_service
