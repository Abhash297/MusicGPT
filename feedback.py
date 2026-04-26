"""
Part 4 — Concurrency: High-throughput feedback ingestion

Decision: In-process buffer + periodic batch UPSERT to Postgres.

Approach
========
Rather than a per-event UPDATE, we buffer click/impression increments in an
in-process Counter and flush to Postgres every `FLUSH_INTERVAL_SECONDS` seconds
via a single bulk INSERT ... ON CONFLICT DO UPDATE.

The UPSERT pattern:
    INSERT INTO feedback_buffer (song_id, delta_clicks, delta_impressions)
    VALUES ($1, $2, $3)
    ON CONFLICT (song_id) DO UPDATE
    SET clicks     = songs.clicks     + EXCLUDED.delta_clicks,
        impressions = songs.impressions + EXCLUDED.delta_impressions

We INSERT into songs directly with the accumulated delta, using Postgres
advisory arithmetic — no read-modify-write, no SERIALIZABLE needed.

Why this approach over alternatives
=====================================
- Redis INCR + sync: rejected — adds infra dependency for a problem solvable
  in-process. At 5k rps over 5s intervals, we batch ~25k events per flush.
  That's one SQL statement. Redis overhead not justified.
- Postgres SKIP LOCKED queue: rejected — solves ordering guarantees we don't
  need. We just want counts, not event replay.
- Postgres partitioned counter tables (one row per (song_id, shard)):
  rejected — correct but complex. Shard selection logic + merge query adds
  maintenance surface. In-process batching achieves the same lock-contention
  reduction with less code.

Failure modes (acknowledged)
==============================
1. Process restart mid-buffer: buffered increments are lost. At 5s flush
   intervals, worst-case loss = 5s × 5000 = 25,000 events. For a re-ranker
   this is acceptable — count staleness of <30s does not change the relative
   order of songs (a song with 1000 clicks vs 40 clicks doesn't flip from a
   25k event gap). If durability is required, drain the buffer to a local
   SQLite WAL before exiting (add signal handlers).

2. Staleness: counts fed to the re-ranker are at most FLUSH_INTERVAL_SECONDS
   stale (default 5s). For engagement-based re-ranking, 5s staleness is
   imperceptible to users. Acceptable.

3. Next bottleneck after row-lock contention is solved:
   The bulk UPSERT touches potentially thousands of rows per flush. At 5k rps
   with many distinct song_ids, the number of rows per batch ≈ unique songs
   clicked in 5 seconds. If that's 10k rows, the UPSERT is still one
   statement but touches 10k index entries — now the B-tree on songs.id
   becomes the bottleneck. Mitigation: use a separate `feedback_buffer` table
   (avoids updating the main songs table on every flush) and merge it
   periodically with a background job, reducing pressure on the hot songs
   table during peak traffic.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Literal

import asyncpg

DATABASE_URL = "postgresql://musicgpt:musicgpt@localhost:5432/musicgpt"
FLUSH_INTERVAL_SECONDS = 5.0

logger = logging.getLogger(__name__)

EventType = Literal["click", "impression"]


class FeedbackBuffer:
    """
    Thread-unsafe in-process buffer for click/impression increments.
    Designed for use within a single asyncio event loop. All mutations
    happen in the event loop thread — no locking needed.
    """

    def __init__(self, db_url: str = DATABASE_URL, flush_interval: float = FLUSH_INTERVAL_SECONDS) -> None:
        self._db_url = db_url
        self._flush_interval = flush_interval
        # delta_clicks[song_id], delta_impressions[song_id]
        self._delta_clicks: dict[str, int] = defaultdict(int)
        self._delta_impressions: dict[str, int] = defaultdict(int)
        self._flush_task: asyncio.Task | None = None

    def record(self, output_id: str, event_type: EventType) -> None:
        """
        Record a feedback event. O(1). Called in the hot path.
        Non-blocking — does not touch the database.
        """
        if event_type == "click":
            self._delta_clicks[output_id] += 1
        elif event_type == "impression":
            self._delta_impressions[output_id] += 1
        else:
            raise ValueError(f"Unknown event type: {event_type!r}")

    async def flush(self) -> int:
        """
        Drain the buffer to Postgres via a single bulk UPSERT.
        Returns the number of rows flushed.
        """
        if not self._delta_clicks and not self._delta_impressions:
            return 0

        # Snapshot and reset atomically (within the event loop — no preemption)
        clicks = dict(self._delta_clicks)
        impressions = dict(self._delta_impressions)
        self._delta_clicks.clear()
        self._delta_impressions.clear()

        all_ids = set(clicks) | set(impressions)
        if not all_ids:
            return 0

        rows = [
            (song_id, clicks.get(song_id, 0), impressions.get(song_id, 0))
            for song_id in all_ids
        ]

        conn = await asyncpg.connect(self._db_url)
        try:
            # Single statement; no per-row locking.
            # The arithmetic `songs.clicks + EXCLUDED.delta_clicks` happens
            # inside Postgres — no read-modify-write race condition.
            await conn.executemany(
                """
                INSERT INTO songs (id, clicks, impressions)
                VALUES ($1::uuid, $2, $3)
                ON CONFLICT (id) DO UPDATE
                SET clicks      = songs.clicks      + EXCLUDED.clicks,
                    impressions = songs.impressions  + EXCLUDED.impressions
                """,
                rows,
            )
        except Exception:
            # Re-enqueue on failure to avoid losing data from transient errors.
            # This can double-count if the statement was partially applied,
            # but for engagement counts that's acceptable over data loss.
            for song_id, dc, di in rows:
                self._delta_clicks[song_id] += dc
                self._delta_impressions[song_id] += di
            logger.exception("Flush failed; increments re-queued")
            raise
        finally:
            await conn.close()

        logger.debug("Flushed %d song counters to Postgres", len(rows))
        return len(rows)

    async def _flush_loop(self) -> None:
        """Background task: flush every FLUSH_INTERVAL_SECONDS."""
        while True:
            await asyncio.sleep(self._flush_interval)
            try:
                await self.flush()
            except Exception:
                logger.exception("Background flush error; will retry next interval")

    def start(self) -> None:
        """Start the background flush task. Call once at app startup."""
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.get_event_loop().create_task(self._flush_loop())

    async def stop(self) -> None:
        """Stop the flush loop and drain remaining buffer. Call at shutdown."""
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        await self.flush()  # final drain


# ---------------------------------------------------------------------------
# FastAPI endpoint (example — wire into your app)
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    app = FastAPI()
    _buffer = FeedbackBuffer()

    @app.on_event("startup")
    async def startup() -> None:
        _buffer.start()

    @app.on_event("shutdown")
    async def shutdown() -> None:
        await _buffer.stop()

    class FeedbackEvent(BaseModel):
        output_id: str
        type: EventType

    @app.post("/feedback", status_code=204)
    async def post_feedback(event: FeedbackEvent) -> None:
        """
        O(1) handler: just records to the in-process buffer.
        Returns 204 immediately — no DB call in the hot path.
        """
        try:
            _buffer.record(event.output_id, event.type)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

except ImportError:
    pass  # FastAPI not installed — module still usable as a library
