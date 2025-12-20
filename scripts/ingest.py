#!/usr/bin/env python3
"""
Ingestion script for RAG chatbot.

Usage:
    python scripts/ingest.py --docs-path ./docusaurus/docs

Environment Variables Required:
    OPENAI_API_KEY
    QDRANT_URL
    QDRANT_API_KEY
"""

import asyncio
import argparse
import os
import sys
from pathlib import Path
from typing import List
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from dotenv import load_dotenv
from openai import AsyncOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import uuid

from chunk_documents import MarkdownChunker, Chunk, chunk_to_payload


class DocumentIngester:
    """Ingestion pipeline for markdown documents."""

    def __init__(
        self,
        docs_path: str,
        openai_api_key: str,
        qdrant_url: str,
        qdrant_api_key: str,
        collection_name: str = "humanoid_robotics_docs"
    ):
        self.docs_path = Path(docs_path)
        self.chunker = MarkdownChunker(max_tokens=1500, overlap_tokens=200)
        self.openai = AsyncOpenAI(api_key=openai_api_key)
        self.qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.collection_name = collection_name
        self.embedding_model = "text-embedding-3-small"
        self.vector_size = 1536

    def ensure_collection(self, recreate: bool = False):
        """Create collection if it doesn't exist."""
        collections = self.qdrant.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if exists and recreate:
            print(f"Deleting existing collection: {self.collection_name}")
            self.qdrant.delete_collection(self.collection_name)
            exists = False

        if not exists:
            print(f"Creating collection: {self.collection_name}")
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
        else:
            print(f"Using existing collection: {self.collection_name}")

    async def ingest_all(self, force_reindex: bool = False):
        """Ingest all markdown files."""
        # Ensure collection exists
        self.ensure_collection(recreate=force_reindex)

        # Discover markdown files
        md_files = list(self.docs_path.glob("**/*.md"))
        print(f"Found {len(md_files)} markdown files")

        # Chunk all documents
        all_chunks: List[Chunk] = []
        for file_path in md_files:
            try:
                chunks = self.chunker.chunk_file(file_path)
                all_chunks.extend(chunks)
                print(f"  {file_path.name}: {len(chunks)} chunks")
            except Exception as e:
                print(f"  Error processing {file_path}: {e}")

        print(f"\nTotal chunks: {len(all_chunks)}")

        # Generate embeddings
        print("\nGenerating embeddings...")
        embeddings = await self._generate_embeddings([c.text for c in all_chunks])
        print(f"Generated {len(embeddings)} embeddings")

        # Upsert to Qdrant
        print("\nUpserting to Qdrant...")
        self._upsert_chunks(all_chunks, embeddings)
        print(f"Successfully indexed {len(all_chunks)} chunks")

        # Print summary
        self._print_summary(all_chunks)

    async def _generate_embeddings(
        self,
        texts: List[str],
        batch_size: int = 100
    ) -> List[List[float]]:
        """Generate embeddings using OpenAI API with batching."""
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = await self.openai.embeddings.create(
                model=self.embedding_model,
                input=batch
            )
            embeddings.extend([e.embedding for e in response.data])
            print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)} chunks")

        return embeddings

    def _upsert_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]):
        """Upsert chunks with embeddings to Qdrant."""
        points = []
        for chunk, embedding in zip(chunks, embeddings):
            payload = chunk_to_payload(chunk)
            payload["indexed_at"] = datetime.utcnow().isoformat()

            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload=payload
            )
            points.append(point)

        # Upsert in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=batch
            )
            print(f"  Upserted {min(i + batch_size, len(points))}/{len(points)} points")

    def _print_summary(self, chunks: List[Chunk]):
        """Print ingestion summary."""
        modules = {}
        total_tokens = 0

        for chunk in chunks:
            module = chunk.module
            if module not in modules:
                modules[module] = {"count": 0, "tokens": 0}
            modules[module]["count"] += 1
            modules[module]["tokens"] += chunk.token_count
            total_tokens += chunk.token_count

        print("\n" + "=" * 50)
        print("INGESTION SUMMARY")
        print("=" * 50)

        for module, stats in sorted(modules.items()):
            print(f"  {module}: {stats['count']} chunks, {stats['tokens']:,} tokens")

        print("-" * 50)
        print(f"  TOTAL: {len(chunks)} chunks, {total_tokens:,} tokens")

        # Estimate embedding cost
        cost_per_1m = 0.02  # text-embedding-3-small
        estimated_cost = (total_tokens / 1_000_000) * cost_per_1m
        print(f"  Estimated embedding cost: ${estimated_cost:.4f}")
        print("=" * 50)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Ingest markdown documents for RAG")
    parser.add_argument(
        "--docs-path",
        type=str,
        default="./docusaurus/docs",
        help="Path to docs folder"
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Delete and recreate collection"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="humanoid_robotics_docs",
        help="Qdrant collection name"
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Validate environment
    required_vars = ["OPENAI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"]
    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        print(f"Error: Missing environment variables: {', '.join(missing)}")
        print("Please set them in .env file or environment")
        sys.exit(1)

    # Create ingester
    ingester = DocumentIngester(
        docs_path=args.docs_path,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        qdrant_url=os.getenv("QDRANT_URL"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY"),
        collection_name=args.collection
    )

    # Run ingestion
    await ingester.ingest_all(force_reindex=args.force_reindex)


if __name__ == "__main__":
    asyncio.run(main())
