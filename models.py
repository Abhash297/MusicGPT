"""
SQLAlchemy async models for MusicGPT hybrid search.

Schema decisions:
- Two tables: `generations` (one per source record) + `songs` (one per audio path).
  conversion_path_1 and conversion_path_2 are sibling rows in `songs`, linked via
  generation_id. This makes diversity enforcement trivial: check generation_id.

- First-class columns: title, primary_genre, primary_mood, bpm, key, vocal_gender,
  all_tags, acoustic_prompt, acoustic_prompt_descriptive. These are the fields
  queried in search (FTS, filters, embedding source). Everything else goes to
  raw_metadata JSONB so we don't lose data and can promote fields later.

- embedding (vector(1536)): nullable. Records missing acoustic_prompt_descriptive
  fall back to acoustic_prompt, then a constructed fallback from structured fields.
  If nothing is embeddable, embedding is NULL and those records only participate
  in FTS, never in vector search.

- search_vector (tsvector): stored generated column updated via trigger.
  Weights: title=A, acoustic_prompt=B (has verbatim "128 BPM", "C major"),
  all_tags=C, acoustic_prompt_descriptive=D.

- Indexes:
  - GIN on search_vector (FTS)
  - HNSW on embedding (vector ANN)
  - btree on generation_id (diversity lookup, O(1))
  - btree on created_at (recency sort)
  - btree on primary_genre, primary_mood (filter queries)
  - NOT indexed: raw_metadata, lyrics, url — not queried directly

- clicks/impressions: on the songs table, updated by feedback.py via batched UPSERT.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    __allow_unmapped__ = True


class Generation(Base):
    """
    One row per source record (one generation event).
    Holds the top-level metadata that applies to both audio variants.
    """

    __tablename__ = "generations"

    id: Column = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Column = Column(Text, nullable=True)
    prompt: Column = Column(Text, nullable=True)
    num_outputs: Column = Column(SmallInteger, nullable=True)
    raw_metadata: Column = Column(JSONB, nullable=True)  # full search_metadata blob
    created_at: Column = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    songs: list[Song] = relationship("Song", back_populates="generation", lazy="select")

    __table_args__ = (
        Index("ix_generations_created_at", "created_at"),
    )


class Song(Base):
    """
    One row per audio path (conversion_path_1 or conversion_path_2).
    Each song is an independently searchable unit with its own embedding
    and engagement counters. Lineage diversity is enforced via generation_id.
    """

    __tablename__ = "songs"

    id: Column = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    generation_id: Column = Column(
        UUID(as_uuid=True),
        ForeignKey("generations.id", ondelete="CASCADE"),
        nullable=False,
    )
    path_number: Column = Column(SmallInteger, nullable=False)  # 1 or 2
    url: Column = Column(Text, nullable=True)

    # --- Searchable text fields ---
    title: Column = Column(Text, nullable=True)
    sounds_desc: Column = Column(Text, nullable=True)  # sounds_1 or sounds_2
    lyrics: Column = Column(Text, nullable=True)

    # --- Promoted first-class columns (avoid JSONB extraction at query time) ---
    acoustic_prompt: Column = Column(Text, nullable=True)
    acoustic_prompt_descriptive: Column = Column(Text, nullable=True)
    all_tags: Column = Column(ARRAY(Text), nullable=True)
    primary_genre: Column = Column(String(100), nullable=True)
    primary_mood: Column = Column(String(100), nullable=True)
    bpm: Column = Column(Integer, nullable=True)
    key: Column = Column(String(50), nullable=True)
    vocal_gender: Column = Column(ARRAY(Text), nullable=True)

    # --- Vector embedding ---
    # Source priority: acoustic_prompt_descriptive > acoustic_prompt > constructed fallback.
    # NULL only when no text at all is available. Such records participate in FTS only.
    #
    # 384 dims  — sentence-transformers all-MiniLM-L6-v2 (current, free, local)
    # SWAP: change to Vector(1536) if switching back to OpenAI text-embedding-3-small
    embedding: Column = Column(Vector(384), nullable=True)

    # --- Full-text search ---
    # Updated by seed/ingest code; a trigger keeps it in sync on UPDATE.
    # Weights: title=A, acoustic_prompt=B, all_tags=C, acoustic_prompt_descriptive=D
    search_vector: Column = Column(TSVECTOR, nullable=True)

    # --- Engagement counters (updated asynchronously by feedback.py) ---
    clicks: Column = Column(BigInteger, nullable=False, default=0)
    impressions: Column = Column(BigInteger, nullable=False, default=0)

    created_at: Column = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    generation: Generation = relationship("Generation", back_populates="songs")

    __table_args__ = (
        # Vector ANN search
        Index(
            "ix_songs_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        # Full-text search
        Index("ix_songs_search_vector_gin", "search_vector", postgresql_using="gin"),
        # Lineage diversity lookup
        Index("ix_songs_generation_id", "generation_id"),
        # Filter / sort
        Index("ix_songs_created_at", "created_at"),
        Index("ix_songs_primary_genre", "primary_genre"),
        Index("ix_songs_primary_mood", "primary_mood"),
        # Deliberately NOT indexing: url, lyrics, sounds_desc, raw_metadata.
        # url is a display field; lyrics is too large for a useful btree;
        # raw_metadata is accessed via the promoted columns above.
    )
