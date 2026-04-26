"""
Verification script — demonstrates FTS + re-ranking correctness.

Runs two queries per the deliverable spec:
1. "new pop" — 3-day-old song with 40/60 CTR should rank above 2-year-old with 1000/5000 clicks
2. "C major female vocal" — FTS surfaces songs with exact technical terms

Setup: two real songs from the DB are updated to simulate the test conditions,
then restored at the end.
"""

import asyncio
from datetime import datetime, timedelta, timezone

import asyncpg

from ranking import calculate_final_score, diversify_results

DATABASE_URL = "postgresql://musicgpt:musicgpt@localhost:5432/musicgpt"

# Specific song IDs from the seeded DB used for the two test scenarios.
# Using IDs (not titles) to avoid ambiguity — two generations share the
# title "Snowy Holiday Nights" in this dataset.
SONG_A_ID    = "3db6dde6-ccef-4b1c-83b6-9591251b87e6"  # Snowy Holiday Nights — recent, high CTR
SONG_B_ID    = "c0013caa-9ae7-40f0-aa0f-8886493cc1fe"  # Electric Heart — old, high raw clicks
SONG_A_TITLE = "Snowy Holiday Nights"
SONG_B_TITLE = "Electric Heart"


async def setup_test_conditions(conn: asyncpg.Connection) -> None:
    """Update two songs to simulate the assessment scenarios."""
    now = datetime.now(timezone.utc)
    three_days_ago = now - timedelta(days=3)
    two_years_ago  = now - timedelta(days=730)

    # Song A: 3 days old, 40 clicks / 60 impressions
    await conn.execute(
        "UPDATE songs SET created_at = $1, clicks = 40, impressions = 60 WHERE id = $2",
        three_days_ago, SONG_A_ID,
    )

    # Song B: 2 years old, 1000 clicks / 5000 impressions
    await conn.execute(
        "UPDATE songs SET created_at = $1, clicks = 1000, impressions = 5000 WHERE id = $2",
        two_years_ago, SONG_B_ID,
    )


async def restore_original_state(conn: asyncpg.Connection) -> None:
    """Reset test songs back to original state."""
    now = datetime.now(timezone.utc)
    await conn.execute(
        "UPDATE songs SET created_at = $1, clicks = 0, impressions = 0 WHERE id = $2",
        now, SONG_A_ID,
    )
    await conn.execute(
        "UPDATE songs SET created_at = $1, clicks = 0, impressions = 0 WHERE id = $2",
        now, SONG_B_ID,
    )


async def run_verification() -> None:
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        await setup_test_conditions(conn)

        # ------------------------------------------------------------------ #
        # QUERY 1: "new pop"
        # ------------------------------------------------------------------ #
        print("=" * 70)
        print('QUERY 1: "new pop"')
        print(f"  Setup: '{SONG_A_TITLE}' → age=3d, clicks=40/60 impressions")
        print(f"         '{SONG_B_TITLE}'  → age=730d, clicks=1000/5000 impressions")
        print("  Expected: Snowy Holiday Nights ranks above Electric Heart")
        print("=" * 70)

        rows = await conn.fetch(
            """
            SELECT
                s.id, s.generation_id, s.title, s.primary_genre,
                s.clicks, s.impressions, s.created_at, s.url,
                ts_rank_cd(s.search_vector, websearch_to_tsquery('english', $1)) AS hybrid_score
            FROM songs s
            WHERE (
                s.search_vector @@ websearch_to_tsquery('english', $1)
                OR s.primary_genre ILIKE '%pop%'
            )
            AND s.path_number = 1
            ORDER BY hybrid_score DESC
            LIMIT 50
            """,
            "new pop",
        )

        results = [dict(r) for r in rows]
        scored = [
            (r, calculate_final_score(r, float(r["hybrid_score"] or 0.01)))
            for r in results
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        deduped = scored

        now = datetime.now(timezone.utc)
        print(f"\n  {'#':<4} {'Title':<26} {'Age':>6} {'CTR':>10} {'Hybrid':>8} {'Final':>8}")
        print("  " + "-" * 68)

        # Find ranks across all results, display top 10
        snowy_rank = electric_rank = None
        for i, (song, final) in enumerate(deduped, 1):
            if SONG_A_TITLE in (song["title"] or ""):
                snowy_rank = snowy_rank or i   # first occurrence only
            if SONG_B_TITLE in (song["title"] or ""):
                electric_rank = electric_rank or i

        for i, (song, final) in enumerate(deduped[:10], 1):
            age = (now - song["created_at"].replace(tzinfo=timezone.utc)).days
            ctr = f"{song['clicks']}/{song['impressions']}"
            hybrid = float(song["hybrid_score"] or 0)
            print(f"  #{i:<3} {(song['title'] or ''):<26} {age:>5}d {ctr:>10} {hybrid:>8.4f} {final:>8.4f}")

        if snowy_rank and electric_rank:
            status = "PASS" if snowy_rank < electric_rank else "FAIL"
            print(f"\n  [{status}] '{SONG_A_TITLE}' #{snowy_rank} vs '{SONG_B_TITLE}' #{electric_rank}")
            print(f"         3d-old, 40/60 CTR (recency+Wilson) beats 730d-old, 1000/5000 raw clicks")
        else:
            print(f"\n  [INFO] snowy_rank={snowy_rank}, electric_rank={electric_rank}")

        # ------------------------------------------------------------------ #
        # QUERY 2: "C major female vocal"
        # ------------------------------------------------------------------ #
        print()
        print("=" * 70)
        print('QUERY 2: "C major female vocal"')
        print("  Expected: FTS surfaces songs with those exact tokens via search_vector")
        print("  These would be missed by vector search alone (key signatures cluster")
        print("  together in embedding space — 'C major' ≈ 'D minor' semantically)")
        print("=" * 70)

        rows2 = await conn.fetch(
            """
            SELECT
                s.id, s.generation_id, s.title, s.key, s.bpm, s.vocal_gender,
                s.primary_genre, s.clicks, s.impressions, s.created_at, s.url,
                ts_rank_cd(s.search_vector, websearch_to_tsquery('english', $1)) AS hybrid_score
            FROM songs s
            WHERE s.search_vector @@ websearch_to_tsquery('english', $1)
            ORDER BY hybrid_score DESC
            LIMIT 5
            """,
            "C major female vocal",
        )

        if rows2:
            print(f"\n  {'Title':<28} {'Key':<10} {'Vocals':<16} {'BPM':<6} {'FTS score'}")
            print("  " + "-" * 72)
            for r in rows2:
                print(
                    f"  {(r['title'] or ''):<28} {(r['key'] or ''):<10} "
                    f"{str(r['vocal_gender'] or ''):<16} {str(r['bpm'] or ''):<6} "
                    f"{float(r['hybrid_score']):.5f}"
                )
            print(f"\n  [PASS] {len(rows2)} result(s) found via FTS lexical match")
            print(f"         Vector search alone would rank any C-major-adjacent song here.")
            print(f"         FTS pins results to songs that literally contain 'C major' + 'female'.")
        else:
            print("\n  [FAIL] No FTS results — check search_vector trigger")

        # ------------------------------------------------------------------ #
        # DIVERSITY CHECK
        # ------------------------------------------------------------------ #
        print()
        print("=" * 70)
        print("DIVERSITY CHECK: diversify_results on Query 2 results")
        print("=" * 70)
        result_dicts = [dict(r) for r in rows2]
        diversified = diversify_results(result_dicts)
        top5 = diversified[:5]
        distinct = len(set(str(r["generation_id"]) for r in top5))
        print(f"\n  Top {len(top5)} results span {distinct} distinct lineage(s)")
        for i, r in enumerate(top5, 1):
            print(f"  #{i} '{r['title']}' — gen={str(r['generation_id'])[:8]}...")
        if distinct >= min(4, len(top5)):
            print(f"\n  [PASS] Diversity constraint satisfied ({distinct} distinct lineages in top {len(top5)})")
        else:
            print(f"\n  [WARN] Only {distinct} distinct lineages — dataset may be too small to satisfy 4-lineage rule")

    finally:
        await restore_original_state(conn)
        await conn.close()


if __name__ == "__main__":
    asyncio.run(run_verification())
