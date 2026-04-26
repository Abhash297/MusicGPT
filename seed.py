"""
seed.py — Fetch the dataset, parse DynamoDB-style records, generate embeddings,
and insert at least 30 generations (60 songs) into Postgres.

Requirements:
    pip install sentence-transformers asyncpg psycopg2-binary python-dotenv httpx

Embedding backend (swap by changing EMBED_BACKEND below):
    "local"  — sentence-transformers all-MiniLM-L6-v2 (free, runs on your machine, 384 dims)
    "openai" — OpenAI text-embedding-3-small (paid API, 1536 dims)

Environment:
    DATABASE_URL   (optional, defaults to local Postgres)
    OPENAI_API_KEY (only needed when EMBED_BACKEND = "openai")
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from typing import Any

import httpx
import asyncpg
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://musicgpt:musicgpt@localhost:5432/musicgpt",
)
DATASET_URL = (
    "https://lalals.s3.us-east-1.amazonaws.com/"
    "ai_backend_assets/technical_assessment_datasets/song_metadata.json"
)

# ---------------------------------------------------------------------------
# EMBEDDING BACKEND — switch here
# ---------------------------------------------------------------------------
EMBED_BACKEND = "local"   # "local" | "openai"

# --- local: sentence-transformers all-MiniLM-L6-v2 (free, 384 dims) ---
LOCAL_EMBED_MODEL = "all-MiniLM-L6-v2"
LOCAL_EMBED_DIMS  = 384
LOCAL_EMBED_BATCH = 32   # sentence-transformers handles batches natively

# --- openai: text-embedding-3-small (paid, 1536 dims) ---
# SWAP: to go back to OpenAI set EMBED_BACKEND = "openai" above
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# OPENAI_EMBED_MODEL = "text-embedding-3-small"
# OPENAI_EMBED_DIMS  = 1536
# OPENAI_EMBED_BATCH = 20


# ---------------------------------------------------------------------------
# DynamoDB unwrapper
# ---------------------------------------------------------------------------

def unwrap(value: Any) -> Any:
    """
    Recursively unwrap DynamoDB marshalled JSON.

    {"S": "x"}       → "x"
    {"N": "128"}     → 128 (int) or 128.0 (float)
    {"L": [...]}     → [unwrap(item) for item in ...]
    {"M": {...}}     → {k: unwrap(v) for k, v in ...}
    {"NULL": True}   → None
    {"BOOL": True}   → True
    plain value      → returned as-is (shouldn't happen in well-formed data)
    """
    if not isinstance(value, dict):
        return value

    if "S" in value:
        return value["S"]
    if "N" in value:
        n = value["N"]
        try:
            return int(n)
        except ValueError:
            return float(n)
    if "L" in value:
        return [unwrap(item) for item in value["L"]]
    if "M" in value:
        return {k: unwrap(v) for k, v in value["M"].items()}
    if "NULL" in value:
        return None
    if "BOOL" in value:
        return value["BOOL"]
    # Unknown wrapper type — return the dict as-is with a warning
    logger.warning("Unknown DynamoDB wrapper: %s", list(value.keys()))
    return value


# ---------------------------------------------------------------------------
# Field extraction
# ---------------------------------------------------------------------------

def extract_fields(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Extract and normalize all fields from a raw source record.
    Returns a flat dict ready for DB insertion.
    Handles missing keys, None values, and type mismatches defensively.
    """
    sm_raw = raw.get("search_metadata", {})
    # search_metadata is a plain dict whose values are DynamoDB-wrapped;
    # unwrap each value individually rather than the top-level dict.
    sm = {k: unwrap(v) for k, v in sm_raw.items()} if sm_raw else {}

    core = sm.get("core_attributes") or {}
    genre_block  = core.get("genre") or {}
    mood_block   = core.get("mood") or {}
    tech_block   = sm.get("technical") or {}
    vocals_block = core.get("vocals") or {}

    # all_tags: L of S → list of strings after unwrap
    all_tags_raw = sm.get("all_tags") or []
    all_tags = [t for t in all_tags_raw if isinstance(t, str) and t]

    # algo_extra_tags: also worth including for FTS but store separately
    extra_tags = sm.get("algo_extra_tags") or []
    extra_tags = [t for t in extra_tags if isinstance(t, str) and t]

    # Merge all tags for the array column
    all_tags_merged = list(dict.fromkeys(all_tags + extra_tags))  # dedup, order-preserving

    bpm_raw = tech_block.get("bpm")
    try:
        bpm = int(bpm_raw) if bpm_raw is not None else None
    except (TypeError, ValueError):
        bpm = None

    vocal_gender_raw = vocals_block.get("vocal_gender") or []
    vocal_gender = [g for g in vocal_gender_raw if isinstance(g, str) and g]

    # acoustic_prompt_descriptive: richest embedding source
    apd = sm.get("acoustic_prompt_descriptive")
    if not apd or not apd.strip():
        apd = None

    # Fallback chain for embedding text:
    # 1. acoustic_prompt_descriptive (richest prose)
    # 2. acoustic_prompt (technical summary)
    # 3. constructed: title + genre + mood + key + BPM
    acoustic_prompt = sm.get("acoustic_prompt") or ""
    if not acoustic_prompt.strip():
        acoustic_prompt = None

    embed_text: str | None = apd or acoustic_prompt
    if not embed_text:
        parts = [
            raw.get("title") or "",
            genre_block.get("primary_genre") or "",
            mood_block.get("primary_mood") or "",
            tech_block.get("key") or "",
        ]
        if bpm:
            parts.append(f"{bpm} BPM")
        constructed = ", ".join(p for p in parts if p)
        embed_text = constructed if constructed else None

    return {
        # Generation-level fields
        "id": raw.get("id") or str(uuid.uuid4()),
        "title": (raw.get("title") or "").strip() or None,
        "prompt": (raw.get("prompt") or "").strip() or None,
        "num_outputs": _safe_int(raw.get("num_outputs")),
        "raw_metadata": sm,  # store the unwrapped (clean) metadata as JSONB
        # Per-path fields (indexed by path_number 1/2)
        "conversion_path_1": raw.get("conversion_path_1") or None,
        "conversion_path_2": raw.get("conversion_path_2") or None,
        "sounds_1": (raw.get("sounds_1") or "").strip() or None,
        "sounds_2": (raw.get("sounds_2") or "").strip() or None,
        "lyrics_1": (raw.get("lyrics_1") or raw.get("lyrics") or "").strip() or None,
        "lyrics_2": (raw.get("lyrics_2") or raw.get("lyrics") or "").strip() or None,
        # Promoted columns (shared between both paths)
        "primary_genre": genre_block.get("primary_genre") or None,
        "primary_mood": mood_block.get("primary_mood") or None,
        "bpm": bpm,
        "key": tech_block.get("key") or None,
        "vocal_gender": vocal_gender or None,
        "all_tags": all_tags_merged or None,
        "acoustic_prompt": acoustic_prompt,
        "acoustic_prompt_descriptive": apd,
        # Embedding source text (not stored in DB — used to call OpenAI)
        "_embed_text": embed_text,
    }


def _safe_int(val: Any) -> int | None:
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------------

async def get_embeddings(texts: list[str]) -> list[list[float] | None]:
    """
    Embed texts using the configured backend.

    EMBED_BACKEND = "local"  → sentence-transformers all-MiniLM-L6-v2 (free, 384 dims)
    EMBED_BACKEND = "openai" → OpenAI text-embedding-3-small (paid, 1536 dims)
    """
    if EMBED_BACKEND == "local":
        return _embed_local(texts)
    elif EMBED_BACKEND == "openai":
        return await _embed_openai(texts)
    else:
        raise ValueError(f"Unknown EMBED_BACKEND: {EMBED_BACKEND!r}")


def _embed_local(texts: list[str]) -> list[list[float] | None]:
    """
    Local embedding via sentence-transformers all-MiniLM-L6-v2.
    Free, no API key, runs on CPU. Produces 384-dim vectors.
    Model is downloaded once (~90 MB) and cached in ~/.cache/huggingface.
    """
    from sentence_transformers import SentenceTransformer  # lazy import

    logger.info("Loading local model %s ...", LOCAL_EMBED_MODEL)
    model = SentenceTransformer(LOCAL_EMBED_MODEL)

    # Replace empty strings — model handles them but returns near-zero vectors
    safe_texts = [t if t and t.strip() else "music" for t in texts]

    logger.info("Encoding %d texts locally (batch=%d)...", len(safe_texts), LOCAL_EMBED_BATCH)
    vectors = model.encode(
        safe_texts,
        batch_size=LOCAL_EMBED_BATCH,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return [v.tolist() for v in vectors]


# SWAP: uncomment this block and set EMBED_BACKEND = "openai" to use OpenAI instead
# async def _embed_openai(texts: list[str]) -> list[list[float] | None]:
#     """
#     OpenAI text-embedding-3-small — paid API, 1536 dims.
#     Requires OPENAI_API_KEY env var and schema VECTOR(1536).
#     """
#     from openai import AsyncOpenAI
#     client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
#     results: list[list[float] | None] = []
#     for i in range(0, len(texts), OPENAI_EMBED_BATCH):
#         batch = texts[i : i + OPENAI_EMBED_BATCH]
#         safe_batch = [t if t else "music" for t in batch]
#         response = await client.embeddings.create(model=OPENAI_EMBED_MODEL, input=safe_batch)
#         results.extend(item.embedding for item in sorted(response.data, key=lambda x: x.index))
#     return results

async def _embed_openai(texts: list[str]) -> list[list[float] | None]:
    raise NotImplementedError("Set EMBED_BACKEND='openai' and uncomment the block above")


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

async def fetch_dataset() -> list[dict[str, Any]]:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(DATASET_URL)
        resp.raise_for_status()
        data = resp.json()
    # Dataset may be a list or {"Items": [...]}
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "Items" in data:
        return data["Items"]
    raise ValueError(f"Unexpected dataset shape: {type(data)}")


async def seed(limit: int = 48) -> None:
    logger.info("Fetching dataset from %s", DATASET_URL)
    raw_records = await fetch_dataset()
    records = raw_records[:limit]
    logger.info("Processing %d records", len(records))

    extracted = [extract_fields(r) for r in records]

    # Collect embedding texts
    embed_texts = [e["_embed_text"] or "music" for e in extracted]
    logger.info("Generating embeddings (%d texts)...", len(embed_texts))
    embeddings = await get_embeddings(embed_texts)

    conn = await asyncpg.connect(DATABASE_URL)
    try:
        async with conn.transaction():
            for i, (fields, embedding) in enumerate(zip(extracted, embeddings)):
                # asyncpg has no native vector codec — pass as pgvector text literal "[x,y,...]"
                embedding_pg = f"[{','.join(str(v) for v in embedding)}]" if embedding else None
                gen_id = uuid.UUID(fields["id"]) if len(fields["id"]) == 36 else uuid.uuid4()

                # Insert generation row
                await conn.execute(
                    """
                    INSERT INTO generations (id, title, prompt, num_outputs, raw_metadata)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    gen_id,
                    fields["title"],
                    fields["prompt"],
                    fields["num_outputs"],
                    json.dumps(fields["raw_metadata"]) if fields["raw_metadata"] else None,
                )

                # Insert both song paths
                for path_num in (1, 2):
                    url = fields[f"conversion_path_{path_num}"]
                    sounds = fields[f"sounds_{path_num}"]
                    lyrics = fields[f"lyrics_{path_num}"]

                    # The trigger sets search_vector on INSERT automatically
                    await conn.execute(
                        """
                        INSERT INTO songs (
                            id, generation_id, path_number, url,
                            title, sounds_desc, lyrics,
                            acoustic_prompt, acoustic_prompt_descriptive,
                            all_tags, primary_genre, primary_mood,
                            bpm, key, vocal_gender,
                            embedding,
                            clicks, impressions
                        ) VALUES (
                            $1, $2, $3, $4,
                            $5, $6, $7,
                            $8, $9,
                            $10, $11, $12,
                            $13, $14, $15,
                            $16,
                            0, 0
                        )
                        ON CONFLICT (generation_id, path_number) DO NOTHING
                        """,
                        uuid.uuid4(),
                        gen_id,
                        path_num,
                        url,
                        fields["title"],
                        sounds,
                        lyrics,
                        fields["acoustic_prompt"],
                        fields["acoustic_prompt_descriptive"],
                        fields["all_tags"],
                        fields["primary_genre"],
                        fields["primary_mood"],
                        fields["bpm"],
                        fields["key"],
                        fields["vocal_gender"],
                        embedding_pg,
                    )

                if (i + 1) % 10 == 0:
                    logger.info("  %d/%d generations inserted", i + 1, len(extracted))

    finally:
        await conn.close()

    logger.info(
        "Seed complete: %d generations, %d songs",
        len(extracted),
        len(extracted) * 2,
    )


if __name__ == "__main__":
    asyncio.run(seed())
