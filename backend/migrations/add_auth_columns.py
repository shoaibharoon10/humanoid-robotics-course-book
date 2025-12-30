"""
Migration script to add authentication columns to users table.
Run this once to update the database schema.
"""

import asyncio
import os
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

async def migrate():
    """Add authentication columns to users table."""
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL environment variable not set")
        return

    engine = create_async_engine(database_url)

    async with engine.begin() as conn:
        # Add new columns if they don't exist
        migrations = [
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS email VARCHAR(255) UNIQUE",
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS username VARCHAR(100) UNIQUE",
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS phone_number VARCHAR(20)",
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS hashed_password VARCHAR(255)",
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS is_active INTEGER DEFAULT 1",
            "ALTER TABLE users ALTER COLUMN session_id DROP NOT NULL",
            "CREATE INDEX IF NOT EXISTS ix_users_email ON users(email)",
            "CREATE INDEX IF NOT EXISTS ix_users_username ON users(username)",
        ]

        for sql in migrations:
            try:
                await conn.execute(text(sql))
                print(f"OK: {sql[:50]}...")
            except Exception as e:
                print(f"SKIP: {sql[:50]}... ({e})")

    print("\nMigration complete!")

if __name__ == "__main__":
    asyncio.run(migrate())
