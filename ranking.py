"""
Part 3 — Re-Ranking: calculate_final_score
Part 5 — Diversity: diversify_results

Re-Ranking Design
=================

Formula:
    final_score = 0.6 * hybrid_score + 0.2 * recency(age_days) + 0.2 * wilson_ctr(clicks, impressions)

Three components:

1. hybrid_score (weight 0.6)
   Relevance is the dominant signal. A song must be relevant before recency
   or engagement matter. 60% keeps the search system honest.

2. recency(age_days) (weight 0.2)
   Exponential decay with half-life of 30 days:
       recency = exp(-age_days * ln(2) / 30)
   At 1 day  → 0.977 (barely penalized)
   At 30 days → 0.500
   At 2 years → ~0.000 (effectively zeroed out)

   Why exponential? Linear decay would still give 2-year-old content a nonzero
   score (e.g. day 730 / max_day = 0.2), which doesn't match user expectation
   for "new pop". Exponential reflects how content ages in practice: it drops
   fast initially and then asymptotes, so the 30-day cliff is meaningful but
   a 3-month-old song isn't treated the same as a 3-year-old song.

   Why 30-day half-life? A month is a natural cycle for "new" content in music
   streaming. Songs older than 90 days get <12.5% of the recency score of
   fresh content.

3. wilson_ctr(clicks, impressions) (weight 0.2)
   Wilson score lower bound at 95% confidence:
       p_hat = clicks / impressions
       wilson = (p_hat + z²/2n - z*sqrt((p_hat*(1-p_hat) + z²/4n)/n)) / (1 + z²/n)

   Why Wilson instead of raw CTR?
   - 40/60 (66.7% CTR) has a Wilson lower bound of ~55.8% — high confidence.
   - 1/1 (100% CTR) has a Wilson lower bound of ~20.7% — we don't trust it.
   - 1000/5000 (20% CTR) has a Wilson lower bound of ~18.9% — well-characterized.

   This naturally solves both the confidence problem (C ≠ A despite 100% CTR)
   and prevents small-sample noise from dominating.

   Cold start (impressions=0): use a prior of 0.3 CTR (neutral assumption,
   equivalent to "average platform CTR"). This ensures new songs with no data
   get a fair but not inflated engagement score. Penalizing them to 0 would
   create a bootstrapping failure where new content can never surface.

Validation against test cases:
    Song A (3d, 40/60, 0.72) → 0.432 + 0.187 + 0.112 = 0.731
    Song B (2y, 1000/5000, 0.80) → 0.480 + 0.000 + 0.038 = 0.518
    Song C (1d, 1/1, 0.75) → 0.450 + 0.195 + 0.041 = 0.686
    Song D (6m, 0/0, 0.68) → 0.408 + 0.003 + 0.060 = 0.471
    Order: A > C > B > D ✓ (A beats B; C doesn't beat A)
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Part 3 — Re-ranking
# ---------------------------------------------------------------------------

_HALF_LIFE_DAYS = 30.0
_RECENCY_DECAY = math.log(2) / _HALF_LIFE_DAYS
_WILSON_Z = 1.96          # 95% confidence
_WILSON_Z2 = _WILSON_Z ** 2
_COLD_START_CTR = 0.3     # neutral prior for songs with no impressions

_W_RELEVANCE = 0.6
_W_RECENCY   = 0.2
_W_ENGAGEMENT = 0.2


def _recency_score(age_days: float) -> float:
    """Exponential decay. Returns 1.0 for age_days=0, 0.5 at 30 days, ~0 at 2y."""
    return math.exp(-_RECENCY_DECAY * max(0.0, age_days))


def _wilson_lower_bound(clicks: int, impressions: int) -> float:
    """
    Wilson score lower bound for a binomial proportion.
    Handles zero impressions with a neutral prior.
    Handles impossible values (clicks > impressions) defensively.
    """
    if impressions <= 0:
        return _COLD_START_CTR
    clicks = max(0, min(clicks, impressions))  # clamp — guard against dirty data
    p_hat = clicks / impressions
    n = impressions
    z2 = _WILSON_Z2
    centre = p_hat + z2 / (2 * n)
    margin = _WILSON_Z * math.sqrt((p_hat * (1 - p_hat) + z2 / (4 * n)) / n)
    denom = 1 + z2 / n
    return (centre - margin) / denom


def calculate_final_score(song: dict[str, Any], hybrid_score: float) -> float:
    """
    Compute the final ranking score for a song after hybrid retrieval.

    Args:
        song: Dict with at least 'created_at' (datetime), 'clicks' (int),
              'impressions' (int). Missing keys are handled defensively.
        hybrid_score: RRF hybrid score from search.py (0.0–1.0 range approx).

    Returns:
        Final float score. Higher is better.
    """
    now = datetime.now(timezone.utc)
    created_at: datetime | None = song.get("created_at")
    if created_at is None:
        age_days = 365.0  # assume old if unknown — safe conservative default
    else:
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        age_days = max(0.0, (now - created_at).total_seconds() / 86400)

    recency = _recency_score(age_days)
    engagement = _wilson_lower_bound(
        song.get("clicks", 0) or 0,
        song.get("impressions", 0) or 0,
    )

    return _W_RELEVANCE * hybrid_score + _W_RECENCY * recency + _W_ENGAGEMENT * engagement


# ---------------------------------------------------------------------------
# Part 5 — Diversity
# ---------------------------------------------------------------------------

def diversify_results(ranked_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Enforce lineage diversity across the final result list.

    Rules:
    - The top 5 results must represent at least 4 distinct generation lineages.
    - If two results share a generation_id, the second is pushed past position 5.
    - If there aren't enough distinct lineages to fill top 5 (e.g., only 3
      distinct lineages in the entire result set), the constraint is relaxed
      and we fill with the best available candidates.

    Algorithm: Single greedy pass. O(n) time, O(k) space where k = distinct
    lineages seen. For a 1,000-item result set this is 1,000 dict lookups and
    set operations — negligible. We do NOT re-sort the list, so the relative
    order of non-penalized items is preserved.

    Time complexity: O(n). For n=1000, this is fast.
    """
    TOP_N = 5
    MAX_PER_LINEAGE_IN_TOP = 1  # only 1 song per generation in top-5

    top: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    lineage_count: dict[str, int] = defaultdict(int)

    for song in ranked_list:
        gen_id = str(song.get("generation_id", "unknown"))

        if len(top) < TOP_N:
            if lineage_count[gen_id] < MAX_PER_LINEAGE_IN_TOP:
                top.append(song)
                lineage_count[gen_id] += 1
            else:
                deferred.append(song)
        else:
            # Already filled top-5 — remaining songs go in order after top
            deferred.append(song)

    # Check if top-5 has at least 4 distinct lineages.
    # If not (fewer distinct lineages exist in the whole list), backfill
    # from deferred — allow repeats only if unavoidable.
    distinct_in_top = len(set(str(s.get("generation_id", "unknown")) for s in top))
    if distinct_in_top < 4 and len(top) < TOP_N:
        # Not enough distinct lineages — fill from deferred allowing repeats
        needed = TOP_N - len(top)
        top.extend(deferred[:needed])
        deferred = deferred[needed:]

    return top + deferred


# ---------------------------------------------------------------------------
# Verification table
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from datetime import timedelta

    now = datetime.now(timezone.utc)

    test_cases = [
        {"name": "A", "age_days": 3,   "clicks": 40,   "impressions": 60,   "hybrid": 0.72},
        {"name": "B", "age_days": 730, "clicks": 1000, "impressions": 5000, "hybrid": 0.80},
        {"name": "C", "age_days": 1,   "clicks": 1,    "impressions": 1,    "hybrid": 0.75},
        {"name": "D", "age_days": 180, "clicks": 0,    "impressions": 0,    "hybrid": 0.68},
    ]

    print(f"{'Song':<6} {'Age':<10} {'Clicks':<8} {'Imp':<8} {'Hybrid':<8} "
          f"{'Recency':<9} {'Wilson':<9} {'Final':<8}")
    print("-" * 68)

    results = []
    for tc in test_cases:
        song = {
            "created_at": now - timedelta(days=tc["age_days"]),
            "clicks": tc["clicks"],
            "impressions": tc["impressions"],
        }
        recency = _recency_score(tc["age_days"])
        wilson = _wilson_lower_bound(tc["clicks"], tc["impressions"])
        final = calculate_final_score(song, tc["hybrid"])
        results.append((tc["name"], final))
        print(
            f"{tc['name']:<6} {tc['age_days']:<10} {tc['clicks']:<8} "
            f"{tc['impressions']:<8} {tc['hybrid']:<8.2f} "
            f"{recency:<9.4f} {wilson:<9.4f} {final:<8.4f}"
        )

    results.sort(key=lambda x: x[1], reverse=True)
    print(f"\nRanking: {' > '.join(r[0] for r in results)}")
    print("Expected: A > C > B > D")
    assert results[0][0] == "A", "A should rank first"
    assert results[1][0] == "C", "C should rank second"
    assert results[2][0] == "B", "B should rank third"
    print("All assertions passed.")
