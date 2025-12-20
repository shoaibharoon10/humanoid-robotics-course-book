"""FastAPI application entry point for RAG Chatbot."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.config import get_settings
from app.api.routes import router
from app.models.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown."""
    # Startup
    settings = get_settings()
    print(f"Starting RAG Chatbot API in {settings.app_env} mode")

    # Initialize database
    await init_db()
    print("Database initialized")

    yield

    # Shutdown
    print("Shutting down RAG Chatbot API")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Humanoid Robotics RAG Chatbot",
        description="RAG-powered chatbot for the Physical AI & Humanoid Robotics textbook",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    app.include_router(router, prefix="/api/v1")

    return app


app = create_app()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Humanoid Robotics RAG Chatbot API",
        "docs": "/docs",
        "health": "/api/v1/health"
    }
