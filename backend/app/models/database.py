"""SQLAlchemy database models for Neon Postgres."""

import uuid
from datetime import datetime
from typing import AsyncGenerator

from sqlalchemy import Column, String, Text, Integer, Float, ForeignKey, DateTime, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.sql import func

from app.config import get_settings


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""
    pass


class User(Base):
    """User model for authentication and session management."""
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # Authentication fields
    email = Column(String(255), unique=True, nullable=True, index=True)
    username = Column(String(100), unique=True, nullable=True, index=True)
    phone_number = Column(String(20), nullable=True)
    hashed_password = Column(String(255), nullable=True)
    is_active = Column(Integer, default=1)  # 1 = active, 0 = inactive
    # Legacy session field for backward compatibility
    session_id = Column(String(255), unique=True, nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_active = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    metadata_ = Column("metadata", JSON, default={})

    # Relationships
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")


class Conversation(Base):
    """Conversation model for chat threads."""
    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title = Column(String(500))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")


class Message(Base):
    """Message model for chat history."""
    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(20), nullable=False)  # 'user', 'assistant', 'system'
    content = Column(Text, nullable=False)
    tokens_used = Column(Integer)
    model = Column(String(100))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    sources = Column(JSON, default=[])  # Retrieved document references
    retrieval_score = Column(Float)

    # Relationships
    conversation = relationship("Conversation", back_populates="messages")


# Database engine and session
_engine = None
_async_session_maker = None


def get_engine():
    """Get or create the database engine."""
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.database_url,
            echo=not settings.is_production,
            pool_pre_ping=True,
        )
    return _engine


def get_session_maker():
    """Get or create the async session maker."""
    global _async_session_maker
    if _async_session_maker is None:
        _async_session_maker = async_sessionmaker(
            get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _async_session_maker


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting database sessions."""
    async_session = get_session_maker()
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    """Initialize the database by creating all tables and running migrations."""
    try:
        engine = get_engine()
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        # Run schema migrations for auth columns
        await run_auth_migrations(engine)
    except Exception as e:
        print(f"Warning: Database initialization failed: {e}")
        print("Application will continue but database features may be unavailable")


async def run_auth_migrations(engine):
    """Add authentication columns to users table if they don't exist."""
    from sqlalchemy import text

    migrations = [
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS email VARCHAR(255) UNIQUE",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS username VARCHAR(100) UNIQUE",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS phone_number VARCHAR(20)",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS hashed_password VARCHAR(255)",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS is_active INTEGER DEFAULT 1",
    ]

    async with engine.begin() as conn:
        for sql in migrations:
            try:
                await conn.execute(text(sql))
                print(f"Migration OK: {sql[:40]}...")
            except Exception as e:
                # Column likely already exists
                if "already exists" not in str(e).lower() and "duplicate" not in str(e).lower():
                    print(f"Migration skip: {sql[:40]}... ({type(e).__name__})")

        # Try to make session_id nullable (for auth users who don't need session_id)
        try:
            await conn.execute(text("ALTER TABLE users ALTER COLUMN session_id DROP NOT NULL"))
            print("Migration OK: session_id now nullable")
        except Exception:
            pass  # Already nullable or doesn't support ALTER

        # Create indexes
        try:
            await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_users_email ON users(email)"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_users_username ON users(username)"))
            print("Migration OK: indexes created")
        except Exception:
            pass

    print("Auth migrations complete")
