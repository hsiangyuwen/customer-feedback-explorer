# app/database.py
import asyncpg
from pgvector.asyncpg import register_vector
import os
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, Text, text
from pgvector.sqlalchemy import Vector
from collections.abc import AsyncGenerator
import re  # For stripping the driver part

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
print(f"DEBUG: DATABASE_URL in database.py = {DATABASE_URL}")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set")

engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False
)

Base = declarative_base()


class FeedbackDB(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    original_id = Column(Text, nullable=True, index=True)
    text = Column(Text, nullable=False)
    embedding = Column(Vector(768))


async def get_db_connection():
    """
    Establishes a direct asyncpg connection.
    Strips '+asyncpg' if present in the global DATABASE_URL for asyncpg compatibility.
    """
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL not configured for get_db_connection")

    # Prepare DSN for asyncpg: remove the "+driver" part if present
    # Example: "postgresql+asyncpg://user:pass@host/db" -> "postgresql://user:pass@host/db"
    asyncpg_dsn = re.sub(r"\+[^:/]+", "", DATABASE_URL)
    print(f"DEBUG: asyncpg_dsn for direct connection = {asyncpg_dsn}")

    conn_pg = await asyncpg.connect(dsn=asyncpg_dsn)  # Use the modified DSN
    await register_vector(conn_pg)  # Register pgvector types for this connection
    return conn_pg


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(
            lambda sync_conn: sync_conn.execute(
                text("CREATE EXTENSION IF NOT EXISTS vector;")
            )
        )
        await conn.run_sync(Base.metadata.create_all)
        print(
            "Database extension 'vector' checked/created and tables created via SQLAlchemy."
        )

    # Create HNSW index using a direct asyncpg connection
    conn_pg_direct = None
    try:
        print("Attempting to connect directly via asyncpg for HNSW index operations...")
        conn_pg_direct = await get_db_connection()  # This now uses the corrected DSN
        print("Successfully connected via asyncpg for HNSW index operations.")

        index_exists = await conn_pg_direct.fetchval(f"""
            SELECT EXISTS (
                SELECT 1
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE c.relname = 'ix_feedback_embedding_hnsw' AND n.nspname = 'public'
            );
        """)

        if not index_exists:
            print("HNSW index not found, attempting to create...")
            await conn_pg_direct.execute(f"""
                CREATE INDEX IF NOT EXISTS ix_feedback_embedding_hnsw
                ON feedback
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
            """)
            print("HNSW index created on feedback.embedding column.")
        else:
            print("HNSW index already exists on feedback.embedding column.")

    except Exception as e:
        print(f"Error during HNSW index creation or check: {e}")
        # Log the full traceback for detailed debugging if needed
        import traceback

        traceback.print_exc()
    finally:
        if conn_pg_direct:
            await conn_pg_direct.close()
            print("Direct asyncpg connection for index check/creation closed.")


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async_session = AsyncSessionLocal()
    try:
        yield async_session
    finally:
        await async_session.close()
