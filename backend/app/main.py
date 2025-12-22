"""FastAPI application entry point for RAG Chatbot."""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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

    # Initialize database (non-blocking - errors are caught in init_db)
    try:
        await init_db()
        print("Database initialized")
    except Exception as e:
        print(f"Warning: Database initialization error: {e}")
        print("Application will continue - database may initialize later")

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

    # Configure CORS - support wildcard or explicit origins
    cors_origins = settings.cors_origins.strip()
    if cors_origins == "*":
        # Allow all origins
        allowed_origins = ["*"]
    else:
        # Use configured origins + explicit Vercel URL
        allowed_origins = settings.cors_origins_list.copy()
        vercel_url = "https://humanoid-robotics-course-book.vercel.app"
        if vercel_url not in allowed_origins:
            allowed_origins.append(vercel_url)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=cors_origins != "*",  # credentials not allowed with wildcard
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    app.include_router(router, prefix="/api/v1")

    # Add global exception handler to ensure CORS headers on errors
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Handle all unhandled exceptions with proper CORS headers."""
        print(f"Global exception handler caught: {type(exc).__name__}: {exc}")

        # Build CORS headers for error response
        origin = request.headers.get("origin", "*")
        cors_headers = {
            "Access-Control-Allow-Origin": origin if cors_origins != "*" else "*",
            "Access-Control-Allow-Credentials": "true" if cors_origins != "*" else "false",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        }

        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal server error: {str(exc)}"},
            headers=cors_headers
        )

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
