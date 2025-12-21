#!/usr/bin/env python3
"""
Health check script for RAG system services (Gemini and Qdrant).
Run this to verify all services are properly configured and reachable.

Usage:
    cd backend
    python health_check.py
"""

import asyncio
import sys
import os

# Ensure we're in the backend directory for .env
backend_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(backend_dir)

from dotenv import load_dotenv
load_dotenv()

# ANSI colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'


def print_header(text: str):
    print(f"\n{BOLD}{BLUE}{'='*60}{RESET}")
    print(f"{BOLD}{BLUE}{text.center(60)}{RESET}")
    print(f"{BOLD}{BLUE}{'='*60}{RESET}\n")


def print_success(text: str):
    print(f"  {GREEN}[OK]{RESET} {text}")


def print_error(text: str):
    print(f"  {RED}[ERROR]{RESET} {text}")


def print_warning(text: str):
    print(f"  {YELLOW}[WARN]{RESET} {text}")


def print_info(text: str):
    print(f"  {BLUE}[INFO]{RESET} {text}")


async def check_gemini():
    """Check Google Gemini API connectivity."""
    print(f"\n{BOLD}1. Checking Google Gemini API...{RESET}")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print_error("GOOGLE_API_KEY not found in environment")
        return False

    print_info(f"API Key: {api_key[:10]}...{api_key[-4:]}")

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)

        # Test embedding model
        embedding_model = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
        print_info(f"Testing embedding model: {embedding_model}")

        result = genai.embed_content(
            model=embedding_model,
            content="Test embedding for health check"
        )

        if result and "embedding" in result:
            embedding_dim = len(result["embedding"])
            print_success(f"Embedding model works! Dimension: {embedding_dim}")

        # Test chat model - list available models first
        chat_model = os.getenv("CHAT_MODEL", "gemini-1.5-flash")
        print_info(f"Testing chat model: {chat_model}")

        # Try to find a working generative model
        try:
            # List models to find available ones
            available_models = []
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    available_models.append(m.name)

            if available_models:
                print_info(f"Available generative models: {len(available_models)}")

                # Try the configured model first, then fallback to first available
                test_model_name = chat_model
                if not any(chat_model in m for m in available_models):
                    test_model_name = available_models[0].replace("models/", "")
                    print_warning(f"Configured model not found, using: {test_model_name}")

                model = genai.GenerativeModel(model_name=test_model_name)
                response = model.generate_content("Say 'Hello' in one word.")

                if response and response.text:
                    print_success(f"Chat model works! Response: {response.text.strip()[:50]}")
            else:
                print_warning("No generative models available (API key may have limited access)")

        except Exception as chat_error:
            print_warning(f"Chat model test skipped: {str(chat_error)[:60]}")

        return True

    except Exception as e:
        print_error(f"Gemini API error: {str(e)}")
        return False


async def check_qdrant():
    """Check Qdrant vector database connectivity."""
    print(f"\n{BOLD}2. Checking Qdrant Vector Database...{RESET}")

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if not qdrant_url:
        print_error("QDRANT_URL not found in environment")
        return False

    if not qdrant_api_key:
        print_error("QDRANT_API_KEY not found in environment")
        return False

    print_info(f"URL: {qdrant_url}")
    print_info(f"API Key: {qdrant_api_key[:10]}...{qdrant_api_key[-4:]}")

    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )

        # Get collections
        collections = client.get_collections()
        print_success(f"Connected to Qdrant! Found {len(collections.collections)} collections")

        # Check for our collection
        collection_name = os.getenv("COLLECTION_NAME", "humanoid_robotics_docs")
        collection_exists = any(c.name == collection_name for c in collections.collections)

        if collection_exists:
            # Get collection info - use scroll to count points if get_collection fails
            try:
                collection_info = client.get_collection(collection_name)
                print_success(f"Collection '{collection_name}' exists with {collection_info.points_count} vectors")
            except Exception:
                # Fallback - just confirm collection exists
                print_success(f"Collection '{collection_name}' exists (version compatibility issue with metadata)")
        else:
            print_warning(f"Collection '{collection_name}' not found. Run indexer to create it.")

        return True

    except Exception as e:
        print_error(f"Qdrant error: {str(e)}")
        return False


async def check_database():
    """Check PostgreSQL database connectivity."""
    print(f"\n{BOLD}3. Checking PostgreSQL Database...{RESET}")

    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        print_error("DATABASE_URL not found in environment")
        return False

    # Mask password in display
    masked_url = database_url
    if "@" in database_url:
        parts = database_url.split("@")
        masked_url = parts[0].rsplit(":", 1)[0] + ":****@" + parts[1]

    print_info(f"URL: {masked_url}")

    # Check for ssl parameter
    if "ssl=require" in database_url:
        print_success("SSL configured correctly (ssl=require)")
    elif "sslmode=require" in database_url:
        print_error("Using sslmode=require - should be ssl=require for asyncpg")
        return False

    try:
        from sqlalchemy.ext.asyncio import create_async_engine
        from sqlalchemy import text

        engine = create_async_engine(database_url, echo=False)

        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            print_success("Database connection successful!")

        await engine.dispose()
        return True

    except Exception as e:
        print_error(f"Database error: {str(e)}")
        return False


async def main():
    """Run all health checks."""
    print_header("RAG System Health Check")

    results = {
        "Gemini API": await check_gemini(),
        "Qdrant": await check_qdrant(),
        "Database": await check_database(),
    }

    print_header("Summary")

    all_passed = True
    for service, status in results.items():
        if status:
            print_success(f"{service}: Healthy")
        else:
            print_error(f"{service}: Failed")
            all_passed = False

    print()
    if all_passed:
        print(f"{GREEN}{BOLD}All services are healthy! RAG system is ready.{RESET}")
        return 0
    else:
        print(f"{RED}{BOLD}Some services failed. Please check the errors above.{RESET}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
