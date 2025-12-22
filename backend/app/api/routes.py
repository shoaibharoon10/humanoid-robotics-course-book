"""API routes for the RAG chatbot."""

import json
import uuid
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.models.database import get_db, User, Conversation, Message
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    SearchRequest,
    SearchResults,
    ConversationResponse,
    ConversationListItem,
    MessageResponse,
    HealthResponse,
    Source,
)
from app.services.retrieval import get_retrieval_service
from app.services.chat import get_chat_service
from app.services.qdrant import get_qdrant_service
from app.config import get_settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint - returns 200 quickly for Railway healthchecks."""
    settings = get_settings()

    # Check Qdrant with detailed status
    qdrant_status = "unknown"
    qdrant_detail = None
    try:
        qdrant = get_qdrant_service()
        health_info = qdrant.check_health()
        qdrant_status = "healthy" if health_info.get("healthy") else "unhealthy"
        qdrant_detail = health_info.get("detail")
    except Exception as e:
        qdrant_status = "error"
        qdrant_detail = str(e)

    return HealthResponse(
        status="healthy",
        database="available",  # DB initializes async, assume available
        qdrant=qdrant_status,
        qdrant_detail=qdrant_detail,
        llm="configured" if settings.google_api_key else "missing"
    )


@router.post("/chat")
async def chat_streaming(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db)
):
    """Chat endpoint with streaming response (SSE)."""
    try:
        retrieval_service = get_retrieval_service()
        chat_service = get_chat_service()

        # Get or create user
        user = await _get_or_create_user(db, request.session_id)

        # Get or create conversation
        conversation = await _get_or_create_conversation(
            db, user.id, request.conversation_id
        )

        # Retrieve relevant context
        sources = await retrieval_service.retrieve(
            query=request.message,
            top_k=get_settings().retrieval_top_k
        )
        context = retrieval_service.build_context(sources)

        # Get conversation history
        history = await _get_conversation_history(db, conversation.id)

        # Save user message
        user_message = Message(
            conversation_id=conversation.id,
            role="user",
            content=request.message
        )
        db.add(user_message)
        await db.commit()

        # Generate streaming response
        async def generate():
            full_response = ""
            try:
                # Send sources first
                sources_data = [s.model_dump() for s in sources]
                yield f"data: {json.dumps({'type': 'sources', 'data': sources_data})}\n\n"

                # Stream tokens
                async for token in chat_service.generate_stream(
                    user_message=request.message,
                    context=context,
                    conversation_history=history
                ):
                    full_response += token
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

                # Save assistant message
                assistant_message = Message(
                    conversation_id=conversation.id,
                    role="assistant",
                    content=full_response,
                    sources=[s.model_dump() for s in sources],
                    model=get_settings().chat_model,
                    tokens_used=await chat_service.get_usage(request.message, context, full_response)
                )
                db.add(assistant_message)
                await db.commit()

                # Send completion
                yield f"data: {json.dumps({'type': 'done', 'message_id': str(assistant_message.id), 'conversation_id': str(conversation.id)})}\n\n"
            except Exception as stream_error:
                # Send error event to client
                error_msg = str(stream_error)
                print(f"Streaming error: {error_msg}")
                yield f"data: {json.dumps({'type': 'error', 'error': error_msg})}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    except Exception as e:
        # Log the error for debugging
        error_msg = str(e)
        print(f"Chat endpoint error: {error_msg}")
        # Return a proper JSON error response (not streaming)
        raise HTTPException(
            status_code=500,
            detail=f"Chat service error: {error_msg}"
        )


@router.post("/chat/sync", response_model=ChatResponse)
async def chat_sync(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db)
):
    """Non-streaming chat endpoint for testing."""
    retrieval_service = get_retrieval_service()
    chat_service = get_chat_service()

    # Get or create user
    user = await _get_or_create_user(db, request.session_id)

    # Get or create conversation
    conversation = await _get_or_create_conversation(
        db, user.id, request.conversation_id
    )

    # Retrieve relevant context
    sources = await retrieval_service.retrieve(
        query=request.message,
        top_k=get_settings().retrieval_top_k
    )
    context = retrieval_service.build_context(sources)

    # Get conversation history
    history = await _get_conversation_history(db, conversation.id)

    # Save user message
    user_msg = Message(
        conversation_id=conversation.id,
        role="user",
        content=request.message
    )
    db.add(user_msg)

    # Generate response
    response_content = await chat_service.generate_response(
        user_message=request.message,
        context=context,
        conversation_history=history
    )

    # Save assistant message
    tokens_used = await chat_service.get_usage(request.message, context, response_content)
    assistant_msg = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=response_content,
        sources=[s.model_dump() for s in sources],
        model=get_settings().chat_model,
        tokens_used=tokens_used
    )
    db.add(assistant_msg)
    await db.commit()

    return ChatResponse(
        message_id=assistant_msg.id,
        conversation_id=conversation.id,
        content=response_content,
        sources=sources,
        model=get_settings().chat_model,
        tokens_used=tokens_used,
        created_at=assistant_msg.created_at
    )


@router.post("/search", response_model=SearchResults)
async def search(request: SearchRequest):
    """Semantic search endpoint."""
    retrieval_service = get_retrieval_service()

    results = await retrieval_service.search(
        query=request.query,
        top_k=request.top_k,
        filter_module=request.filter_module
    )

    return SearchResults(
        results=results,
        query=request.query,
        total_found=len(results)
    )


@router.get("/conversations", response_model=list[ConversationListItem])
async def list_conversations(
    session_id: str,
    db: AsyncSession = Depends(get_db)
):
    """List conversations for a user."""
    # Get user
    result = await db.execute(
        select(User).where(User.session_id == session_id)
    )
    user = result.scalar_one_or_none()

    if not user:
        return []

    # Get conversations with message count
    result = await db.execute(
        select(
            Conversation,
            func.count(Message.id).label("message_count")
        )
        .outerjoin(Message)
        .where(Conversation.user_id == user.id)
        .group_by(Conversation.id)
        .order_by(Conversation.updated_at.desc())
    )

    conversations = []
    for conv, count in result.all():
        conversations.append(ConversationListItem(
            id=conv.id,
            title=conv.title,
            created_at=conv.created_at,
            message_count=count
        ))

    return conversations


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get a conversation with all messages."""
    try:
        conv_uuid = uuid.UUID(conversation_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid conversation ID")

    result = await db.execute(
        select(Conversation).where(Conversation.id == conv_uuid)
    )
    conversation = result.scalar_one_or_none()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Get messages
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conv_uuid)
        .order_by(Message.created_at)
    )
    messages = result.scalars().all()

    return ConversationResponse(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=[
            MessageResponse(
                id=msg.id,
                role=msg.role,
                content=msg.content,
                sources=[Source(**s) for s in (msg.sources or [])],
                created_at=msg.created_at
            )
            for msg in messages
        ]
    )


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Delete a conversation."""
    try:
        conv_uuid = uuid.UUID(conversation_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid conversation ID")

    result = await db.execute(
        select(Conversation).where(Conversation.id == conv_uuid)
    )
    conversation = result.scalar_one_or_none()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    await db.delete(conversation)
    await db.commit()

    return {"status": "deleted"}


# Helper functions

async def _get_or_create_user(db: AsyncSession, session_id: str) -> User:
    """Get or create a user by session ID."""
    result = await db.execute(
        select(User).where(User.session_id == session_id)
    )
    user = result.scalar_one_or_none()

    if not user:
        user = User(session_id=session_id)
        db.add(user)
        await db.commit()
        await db.refresh(user)

    return user


async def _get_or_create_conversation(
    db: AsyncSession,
    user_id: uuid.UUID,
    conversation_id: Optional[uuid.UUID]
) -> Conversation:
    """Get or create a conversation."""
    if conversation_id:
        result = await db.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )
        conversation = result.scalar_one_or_none()
        if conversation:
            return conversation

    # Create new conversation
    conversation = Conversation(user_id=user_id)
    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)
    return conversation


async def _get_conversation_history(
    db: AsyncSession,
    conversation_id: uuid.UUID,
    limit: int = 10
) -> list[dict]:
    """Get recent conversation history."""
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
    )
    messages = result.scalars().all()

    # Return in chronological order
    return [
        {"role": msg.role, "content": msg.content}
        for msg in reversed(messages)
    ]
