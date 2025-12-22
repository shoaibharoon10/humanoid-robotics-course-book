"""Pydantic schemas for API request/response validation."""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from uuid import UUID


class Source(BaseModel):
    """Source citation from retrieved documents."""
    title: str
    url_path: str
    section: Optional[str] = None
    relevance_score: float
    snippet: str


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""
    message: str = Field(..., min_length=1, max_length=4000)
    conversation_id: Optional[UUID] = None
    session_id: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    """Response body for non-streaming chat."""
    message_id: UUID
    conversation_id: UUID
    content: str
    sources: List[Source]
    model: str
    tokens_used: int
    created_at: datetime


class SearchRequest(BaseModel):
    """Request body for semantic search."""
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    filter_module: Optional[str] = None


class SearchResult(BaseModel):
    """Individual search result."""
    doc_id: str
    title: str
    section: str
    url_path: str
    snippet: str
    score: float


class SearchResults(BaseModel):
    """Response body for search endpoint."""
    results: List[SearchResult]
    query: str
    total_found: int


class MessageResponse(BaseModel):
    """Message in conversation history."""
    id: UUID
    role: str
    content: str
    sources: Optional[List[Source]] = None
    created_at: datetime

    class Config:
        from_attributes = True


class ConversationResponse(BaseModel):
    """Conversation with messages."""
    id: UUID
    title: Optional[str]
    created_at: datetime
    updated_at: datetime
    messages: List[MessageResponse]

    class Config:
        from_attributes = True


class ConversationListItem(BaseModel):
    """Conversation summary for list view."""
    id: UUID
    title: Optional[str]
    created_at: datetime
    message_count: int

    class Config:
        from_attributes = True


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    database: str
    qdrant: str
    qdrant_detail: Optional[str] = None
    llm: str
