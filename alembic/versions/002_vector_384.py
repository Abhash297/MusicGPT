"""Switch embedding column from VECTOR(1536) to VECTOR(384)

OpenAI text-embedding-3-small (1536 dims) → sentence-transformers all-MiniLM-L6-v2 (384 dims)

SWAP: to go back to OpenAI, run the downgrade() or create a new migration
that ALTERs the column back to VECTOR(1536) and updates the HNSW index.

Revision ID: 002
Revises: 001
Create Date: 2026-04-23
"""

from alembic import op

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Drop the existing HNSW index — cannot ALTER a column with a vector index attached
    op.execute("DROP INDEX IF EXISTS ix_songs_embedding_hnsw")

    # Change dimension: 1536 → 384
    # USING cast forces Postgres to rewrite every row; NULL embeddings stay NULL.
    op.execute(
        "ALTER TABLE songs ALTER COLUMN embedding TYPE vector(384) USING embedding::vector(384)"
    )

    # Recreate HNSW index for the new dimension
    op.execute(
        """
        CREATE INDEX ix_songs_embedding_hnsw
        ON songs
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        """
    )


def downgrade() -> None:
    # SWAP: run this to go back to OpenAI 1536-dim embeddings
    op.execute("DROP INDEX IF EXISTS ix_songs_embedding_hnsw")
    op.execute(
        "ALTER TABLE songs ALTER COLUMN embedding TYPE vector(1536) USING embedding::vector(1536)"
    )
    op.execute(
        """
        CREATE INDEX ix_songs_embedding_hnsw
        ON songs
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        """
    )
