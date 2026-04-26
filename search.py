"""
Hybrid search: vector similarity (pgvector) + full-text search (tsvector),
fused via Reciprocal Rank Fusion (RRF).

Part 2 — Bug Analysis
=====================

The broken query below is taken directly from production. Two distinct bugs
cause keyword queries ("C major", "128 BPM", "female vocal") to almost never
surface the right songs, while vibe queries ("sad rainy piano") seem fine.

Bug 1 — FTS: `to_tsquery` used with raw user input
---------------------------------------------------
`to_tsquery('english', $2)` requires pre-formatted tsquery syntax with explicit
`&` / `|` / `<->` operators between tokens. Passing a raw multi-word string
like "C major" or "female vocal" throws a Postgres syntax error at runtime.
The error causes fts_ranked to return zero rows, so the FTS branch of the join
produces nothing. Single-word queries like "piano" happen to be valid
single-token tsqueries, which is why vibe queries appear to work.

Impact: Any query referencing technical terms with spaces (key signatures,
vocal descriptors, tempo ranges) silently loses its entire FTS signal. Those
songs can only surface via vector similarity, but vector embeddings represent
semantic proximity — "C major" sits near "D minor" and "G major" in embedding
space, not specifically at rank 1. The lexical boost that should have rescued
exact-match songs never fires.

Fix: Replace `to_tsquery` with `websearch_to_tsquery`, which accepts natural
language as users type it — handles phrases, stop words, and multi-word input
without requiring operator syntax.

Bug 2 — Fusion: LIMIT without ORDER BY means candidates are not top-ranked
---------------------------------------------------------------------------
Both CTEs use LIMIT 100 but have no ORDER BY on the outer query:

    SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> $1) AS rank
    FROM songs
    LIMIT 100          ← no ORDER BY here

PostgreSQL evaluates window functions BEFORE LIMIT, but LIMIT without ORDER BY
returns rows in heap storage order (insertion order on disk), not by vector
distance. The ROW_NUMBER() values are computed correctly across all rows, but
LIMIT then cuts to 100 arbitrary rows from disk — the actual nearest neighbors
may never be fetched. The same applies to fts_ranked: it returns 100 arbitrary
FTS-matching rows, not the top 100 by ts_rank.

Impact: vector_ranked does not contain the true top-100 nearest neighbors.
It contains 100 random songs labeled with their correct rank positions. A song
with rank=1 (the nearest neighbor) might not be in the CTE at all because it
happens to live on a later heap page. The RRF scores are computed over a random
subset, making the fusion meaningless for precision retrieval.

Fix: Add ORDER BY to both CTEs before LIMIT so Postgres fetches the correct
candidate set before truncating.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import asyncpg

DATABASE_URL = "postgresql://musicgpt:musicgpt@localhost:5432/musicgpt"

RRF_K = 60
VECTOR_LIMIT = 100
FTS_LIMIT = 100


# ---------------------------------------------------------------------------
# BROKEN VERSION (exact production query — DO NOT USE)
# ---------------------------------------------------------------------------

BROKEN_QUERY = """
WITH vector_ranked AS (
  SELECT id,
         ROW_NUMBER() OVER (ORDER BY embedding <=> $1) AS rank
  FROM songs
  LIMIT 100
),
fts_ranked AS (
  SELECT id,
         ROW_NUMBER() OVER (ORDER BY ts_rank(search_vector, query) DESC) AS rank
  FROM songs, to_tsquery('english', $2) query
  WHERE search_vector @@ query
  LIMIT 100
),
fused AS (
  SELECT
    COALESCE(v.id, f.id) AS id,
    (1.0 / (60 + COALESCE(v.rank, 100))) + (1.0 / (60 + COALESCE(f.rank, 100))) AS score
  FROM vector_ranked v
  FULL OUTER JOIN fts_ranked f ON v.id = f.id
)
SELECT s.*, fused.score
FROM fused JOIN songs s ON s.id = fused.id
ORDER BY fused.score DESC
LIMIT $3;
"""
# Bug 1: to_tsquery('english', $2) — crashes on any multi-word input;
#         fts_ranked silently returns zero rows.
# Bug 2: LIMIT 100 without ORDER BY — returns 100 heap-order rows, not the
#         top-100 by vector distance or ts_rank. Nearest neighbors may be absent.


# ---------------------------------------------------------------------------
# FIXED VERSION
# ---------------------------------------------------------------------------

HYBRID_SEARCH_QUERY = """
WITH vector_ranked AS (
    SELECT
        id,
        ROW_NUMBER() OVER (ORDER BY embedding <=> $1) AS rank
    FROM songs
    WHERE embedding IS NOT NULL
    ORDER BY embedding <=> $1
    LIMIT $3
),
fts_ranked AS (
    SELECT
        id,
        ROW_NUMBER() OVER (
            ORDER BY ts_rank_cd(search_vector, websearch_to_tsquery('english', $2)) DESC
        ) AS rank
    FROM songs
    WHERE search_vector @@ websearch_to_tsquery('english', $2)
    ORDER BY ts_rank_cd(search_vector, websearch_to_tsquery('english', $2)) DESC
    LIMIT $4
),
rrf AS (
    SELECT
        COALESCE(v.id, f.id) AS id,
        COALESCE(1.0 / ($5 + v.rank), 0.0) +
        COALESCE(1.0 / ($5 + f.rank), 0.0) AS hybrid_score,
        v.rank AS vector_rank,
        f.rank AS fts_rank
    FROM vector_ranked v
    FULL OUTER JOIN fts_ranked f ON v.id = f.id
)
SELECT
    s.id,
    s.generation_id,
    s.title,
    s.primary_genre,
    s.primary_mood,
    s.bpm,
    s.key,
    s.all_tags,
    s.clicks,
    s.impressions,
    s.created_at,
    s.url,
    r.hybrid_score,
    r.vector_rank,
    r.fts_rank
FROM rrf r
JOIN songs s ON s.id = r.id
ORDER BY r.hybrid_score DESC
LIMIT 10;
"""
# Fix 1: websearch_to_tsquery accepts natural language; handles phrases and
#         multi-word input without requiring tsquery operator syntax.
# Fix 2: ORDER BY added before LIMIT in both CTEs so Postgres fetches the
#         actual top-N candidates, not an arbitrary heap-order slice.
# Fix 3: WHERE embedding IS NOT NULL added to vector_ranked to exclude songs
#         with no embeddable text; those participate in FTS only.
# Note:   ts_rank_cd (cover density) normalises by document length, preventing
#         long acoustic_prompt_descriptive blobs from inflating ts_rank scores.


async def hybrid_search(
    query_text: str,
    query_embedding: list[float],
    *,
    vector_limit: int = VECTOR_LIMIT,
    fts_limit: int = FTS_LIMIT,
    rrf_k: int = RRF_K,
) -> list[dict[str, Any]]:
    """
    Run hybrid search and return raw result rows (before re-ranking).

    Args:
        query_text: The raw user query string.
        query_embedding: Pre-computed embedding of the query (384 dims).
        vector_limit: How many candidates to pull from the vector stage.
        fts_limit: How many candidates to pull from the FTS stage.
        rrf_k: RRF damping constant (60 per Cormack et al. 2009).

    Returns:
        List of dicts with song fields + hybrid_score, vector_rank, fts_rank.
    """
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        rows = await conn.fetch(
            HYBRID_SEARCH_QUERY,
            query_embedding,    # $1
            query_text,         # $2
            vector_limit,       # $3
            fts_limit,          # $4
            float(rrf_k),       # $5
        )
        return [dict(r) for r in rows]
    finally:
        await conn.close()
