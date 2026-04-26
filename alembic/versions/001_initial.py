"""Initial schema: generations + songs with pgvector and FTS

Revision ID: 001
Revises:
Create Date: 2026-04-23
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR, UUID
from pgvector.sqlalchemy import Vector

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # pgvector extension — required before any vector column
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "generations",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("title", sa.Text, nullable=True),
        sa.Column("prompt", sa.Text, nullable=True),
        sa.Column("num_outputs", sa.SmallInteger, nullable=True),
        sa.Column("raw_metadata", JSONB, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_generations_created_at", "generations", ["created_at"])

    op.create_table(
        "songs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "generation_id",
            UUID(as_uuid=True),
            sa.ForeignKey("generations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("path_number", sa.SmallInteger, nullable=False),
        sa.Column("url", sa.Text, nullable=True),
        sa.Column("title", sa.Text, nullable=True),
        sa.Column("sounds_desc", sa.Text, nullable=True),
        sa.Column("lyrics", sa.Text, nullable=True),
        sa.Column("acoustic_prompt", sa.Text, nullable=True),
        sa.Column("acoustic_prompt_descriptive", sa.Text, nullable=True),
        sa.Column("all_tags", ARRAY(sa.Text), nullable=True),
        sa.Column("primary_genre", sa.String(100), nullable=True),
        sa.Column("primary_mood", sa.String(100), nullable=True),
        sa.Column("bpm", sa.Integer, nullable=True),
        sa.Column("key", sa.String(50), nullable=True),
        sa.Column("vocal_gender", ARRAY(sa.Text), nullable=True),
        sa.Column("embedding", Vector(1536), nullable=True),
        sa.Column("search_vector", TSVECTOR, nullable=True),
        sa.Column("clicks", sa.BigInteger, nullable=False, server_default="0"),
        sa.Column("impressions", sa.BigInteger, nullable=False, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    # HNSW index for vector ANN — faster queries than IVFFlat at this dataset size,
    # no need to pre-select nlist. m=16, ef_construction=64 are safe defaults.
    op.execute(
        """
        CREATE INDEX ix_songs_embedding_hnsw
        ON songs
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        """
    )

    op.create_index(
        "ix_songs_search_vector_gin",
        "songs",
        ["search_vector"],
        postgresql_using="gin",
    )
    op.create_index("ix_songs_generation_id", "songs", ["generation_id"])
    op.create_index("ix_songs_created_at", "songs", ["created_at"])
    op.create_index("ix_songs_primary_genre", "songs", ["primary_genre"])
    op.create_index("ix_songs_primary_mood", "songs", ["primary_mood"])

    # Trigger to keep search_vector in sync on INSERT/UPDATE.
    # Weights: title=A, acoustic_prompt=B, all_tags=C, acoustic_prompt_descriptive=D.
    # title in A because an exact title match should beat any tag match.
    # acoustic_prompt in B because it contains verbatim technical terms ("128 BPM", "C major").
    # all_tags in C for broad categorical matching.
    # acoustic_prompt_descriptive in D — richest prose but lowest precision signal.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION songs_search_vector_update() RETURNS trigger AS $$
        BEGIN
            NEW.search_vector :=
                setweight(to_tsvector('english', coalesce(NEW.title, '')), 'A') ||
                setweight(to_tsvector('english', coalesce(NEW.acoustic_prompt, '')), 'B') ||
                setweight(to_tsvector('simple',  array_to_string(coalesce(NEW.all_tags, '{}'), ' ')), 'C') ||
                setweight(to_tsvector('english', coalesce(NEW.acoustic_prompt_descriptive, '')), 'D');
            RETURN NEW;
        END
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        """
        CREATE TRIGGER songs_search_vector_trigger
        BEFORE INSERT OR UPDATE OF title, acoustic_prompt, all_tags, acoustic_prompt_descriptive
        ON songs
        FOR EACH ROW EXECUTE FUNCTION songs_search_vector_update();
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS songs_search_vector_trigger ON songs")
    op.execute("DROP FUNCTION IF EXISTS songs_search_vector_update()")
    op.drop_table("songs")
    op.drop_table("generations")
    op.execute("DROP EXTENSION IF EXISTS vector")
