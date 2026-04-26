# DECISIONS.md

## Part 1 — Schema Design

### Decision: Two-table model (`generations` + `songs`)

Each source record has one `id` and two audio paths (`conversion_path_1`, `conversion_path_2`). These are sibling audio variants of the same generation event — they share all metadata but are independently searchable. I model them as:
- `generations`: one row per source record; holds the generation-level metadata (title, prompt, raw_metadata JSONB)
- `songs`: one row per audio path; holds embedding, search_vector, clicks, impressions

**Why not a single table?** The diversity requirement (Part 5) requires detecting that two search results share a lineage. If both paths were columns on the same row, they could never both appear in results independently — but the problem statement says they do appear as separate results that crowd the top 10. The two-table model makes this explicit: `generation_id` is a first-class FK that the diversity function checks.

**Considered and rejected:** A single `songs` table with a `sibling_id` self-referential FK. This is messier — queries need a self-join, and inserting in the right order matters. The parent-child model is cleaner.

### Decision: Field promotion strategy

**Promoted to first-class columns:**
- `title` — primary FTS signal (weight A); needed for display
- `acoustic_prompt` — verbatim technical terms ("128 BPM", "C major"); FTS weight B
- `acoustic_prompt_descriptive` — embedding source; FTS weight D
- `all_tags` — categorical tags for FTS (weight C) and potential filter queries
- `primary_genre`, `primary_mood` — high-value filter columns; common query predicates
- `bpm`, `key` — searchable integers/strings that appear in acoustic_prompt
- `vocal_gender` — common filter attribute
- `clicks`, `impressions` — hot-path counter columns; need to be first-class for ranking

**Left in JSONB (`raw_metadata`):**
- `core_attributes` nested structure (instruments, context, secondary moods, etc.)
- `algo_extra_tags`
- `technical.stereo_profile`, `technical.duration_type`
- `sounds_1`, `sounds_2` (stored as `sounds_desc` per song row, not in JSONB)

Rationale: JSONB is appropriate for fields that are rarely queried directly or whose structure may evolve. The nested DynamoDB map for instruments, context, and SFX details has sparse and inconsistent keys across records — promoting all of them would require many nullable columns that might never be queried. If a future filter on `stereo_profile` is needed, it can be promoted via a migration without schema-level changes to the JSONB blob.

**Known limitation:** GIN indexes on JSONB are not created. Queries into `raw_metadata` will be sequential scans. If JSONB filtering becomes common, add `CREATE INDEX CONCURRENTLY` on specific JSONB paths then.

### Decision: TSVECTOR weights

```
title (A) > acoustic_prompt (B) > all_tags (C) > acoustic_prompt_descriptive (D)
```

- `title` at A: a title match is the highest-precision signal — user searched for a specific song
- `acoustic_prompt` at B: verbatim technical terms live here ("C major", "128 BPM", "female vocal") — these exact tokens should outrank generic tag matches
- `all_tags` at C: broad categorical tags; a match here is meaningful but less precise than a BPM/key match
- `acoustic_prompt_descriptive` at D: richest prose but lowest precision — "a warm folk acoustic track at 100 bpm" overlaps with many songs

`all_tags` uses `'simple'` text search config (not `'english'`) because tags are controlled vocabulary — we don't want "dancing" stemmed to "danc" when the tag is literally "dance". `'simple'` preserves the exact token.

### Decision: Missing `acoustic_prompt_descriptive`

For records where this field is absent, the embedding falls back to:
1. `acoustic_prompt` (short technical string, e.g. "folk acoustic, 100 bpm, c major, i-iv-v, sarangi")
2. Constructed string from structured fields: `"{title}, {genre}, {mood}, {key}, {bpm} BPM"`

Records with no embeddable text at all get `embedding = NULL` and participate in FTS-only search. They are not excluded from the index — a relevant FTS match can still surface them.

**Considered and rejected:** Skipping records with missing `acoustic_prompt_descriptive`. This would silently drop valid records from vector search, which is worse than a degraded embedding.

### Decision: Indexing strategy

| Index | Type | Why |
|---|---|---|
| `search_vector` | GIN | Required for `@@` operator; FTS would be a seqscan without it |
| `embedding` | HNSW (m=16, ef_construction=64) | ANN search; HNSW outperforms IVFFlat for small-to-medium datasets (no `SELECT setval` needed) |
| `generation_id` | btree | Diversity lookup: check lineage in O(1) |
| `created_at` | btree | Recency filter and sort |
| `primary_genre`, `primary_mood` | btree | Common filter predicates |
| `url`, `lyrics`, `sounds_desc` | not indexed | Display fields; never in WHERE clause |
| `raw_metadata` | not indexed | Queries go through promoted columns |
| `all_tags` | not indexed (GIN) | GIN on array is useful for `@>` operator; deferred until a filter like `all_tags @> '{pop}'` is in production use |

---

## Part 2 — Bug Diagnosis

### Bug 1: `to_tsquery` with raw user input (FTS)

`to_tsquery('english', $2)` requires tsquery syntax — tokens joined by `&`, `|`, `!`, or `<->`. Passing a raw natural-language string like `"C major"` or `"female vocal"` throws a Postgres syntax error at runtime. The error causes `fts_ranked` to return zero rows — the FTS branch is dead for every query, single-word or multi-word alike.

**User experience:** Keyword-specific searches ("C major", "128 BPM", "female vocal") return results that feel random — the right songs exist verbatim in the data but only surface if they happen to be nearest neighbours by vector similarity. Vibe queries like "sad rainy piano" appear to work not because they survive `to_tsquery`, but because vector embeddings handle semantic/emotional descriptions well enough that users don't notice the missing FTS signal. "C major" in embedding space sits near all key signatures — without FTS to match the exact token, there is no way to filter precisely to songs in that key.

**Fix:** `websearch_to_tsquery('english', $2)` accepts natural-language input exactly as users type it. "C major female vocal" becomes `'c' & 'major' & 'femal' & 'vocal'` automatically. No syntax errors regardless of input.

### Bug 2: `LIMIT` without `ORDER BY` in candidate retrieval (fusion logic)

Both CTEs fetch candidates with `LIMIT 100` but no `ORDER BY` on the outer query:

```sql
SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> $1) AS rank
FROM songs
LIMIT 100   -- no ORDER BY here
```

PostgreSQL evaluates window functions before LIMIT, but LIMIT without ORDER BY returns rows in heap storage order (physical insertion order on disk). The `ROW_NUMBER()` labels are computed correctly across all rows, but LIMIT then cuts to the first 100 rows encountered during the heap scan — not the 100 nearest neighbours. The actual nearest neighbour (rank=1) may live on a later heap page and never be fetched. The same applies to `fts_ranked`: 100 arbitrary FTS-matching rows are returned, not the top 100 by ts_rank. RRF fusion is then computed over the wrong candidate pool.

**User experience:** Results are unpredictable and inconsistent. A song that is an exact keyword match and a strong vector match may never appear because it was inserted later and sits beyond the heap pages that LIMIT reaches. The system appears to ignore the query entirely for those songs.

**Fix:** Add `ORDER BY` before `LIMIT` in both CTEs so Postgres sorts the full result set before truncating to the top-N candidates:

```sql
FROM songs
WHERE embedding IS NOT NULL
ORDER BY embedding <=> $1
LIMIT $3
```

---

## Part 3 — Re-Ranking

### Decision: Formula components and weights

```
final = 0.6 × hybrid_score + 0.2 × recency + 0.2 × wilson_ctr
```

**Relevance at 60%:** Search relevance is the dominant purpose of the system. Reducing it below 50% risks surfacing recent but irrelevant content.

**Recency decay — exponential, half-life 30 days:**
Exponential decay was chosen over linear because content aging in music streaming is non-linear: a song loses most of its "new" status in the first month, then decays slowly. Linear decay would give 2-year-old content a score proportional to `(730-0)/(730+1)` which is near 0 anyway, but the shape between 0–90 days would be too gradual. With a 30-day half-life: 1 day → 0.977, 30 days → 0.500, 6 months → 0.016, 2 years → ~0.000. This matches the intuition that "new" means within the last month.

**Wilson CTR at 20%:**
Wilson score lower bound (95% CI) penalizes small samples without completely ignoring them. A 100% CTR from 1 impression gets a Wilson lower bound of ~0.21. A 66.7% CTR from 60 impressions gets ~0.56. This correctly orders A > C in the test cases. Raw CTR (clicks/impressions) would give C = 1.0, A = 0.67 — backwards.

**Cold start:** Songs with zero impressions receive a prior CTR of 0.30 (approximate platform average). This is preferable to 0 (which would permanently bury new songs before they get impressions) or 1 (which would incorrectly boost unproven songs). The exact value should be tuned to the platform's average CTR.

**Known limitation:** The 30-day half-life is a parameter that should be tuned on real user engagement data. A music platform may find that "new" means something different from "new" in news search.

---

## Part 4 — Concurrency

### Decision: In-process buffer + periodic batch UPSERT

**Committed approach:** Buffer click/impression deltas in-process (Python `defaultdict(int)`) and flush every 5 seconds via a single `INSERT ... ON CONFLICT DO UPDATE` using additive arithmetic in Postgres.

**Why not per-event UPDATE:** `UPDATE songs SET clicks = clicks + 1` at 5k rps causes row-level lock contention on hot rows — multiple connections block each other waiting for the same row's lock.

**Why not Redis:** Adds infra dependency. At 5k rps with 5s intervals, we're batching ~25k events into one SQL statement. That's easily within Postgres's capacity. Redis is justified if we need cross-process coordination (multiple API server instances) — if deploying multiple instances, replace the in-process buffer with a Redis INCR and a separate flush worker.

**Staleness:** At most `FLUSH_INTERVAL_SECONDS` (default 5s). Re-ranker sees counts that are ≤5s stale. This is acceptable: engagement re-ranking is a coarse signal and a 5s lag doesn't change which songs have fundamentally better CTR.

**Process restart:** Buffered increments since the last flush are lost. Worst case: 5s × 5k rps = 25,000 events. For engagement-based re-ranking this is tolerable. If exact count durability is required, add signal handlers (`SIGTERM`) to drain the buffer before shutdown.

**Next bottleneck after row-lock:** The bulk UPSERT updates `N` rows in `songs` per flush (N = distinct songs clicked in 5s). At peak traffic with many distinct songs, this becomes a B-tree update bottleneck on the songs PK index. Mitigation: introduce a separate `feedback_buffer` table with no indexes (just append), and merge it into `songs` with a background job — separating the hot append path from the hot read path.

---

## Part 5 — Diversity

### Decision: Greedy single-pass selection

A greedy pass through the ranked list, tracking a `lineage_count[generation_id]` counter. Songs are placed in the top-5 positions if their lineage hasn't already occupied one slot there. Overflow songs are deferred to positions 6+.

**Why greedy instead of penalized scoring:** A score penalty (e.g., multiply score by 0.5 for second sibling) requires choosing a penalty factor, which affects all positions uniformly. The greedy approach enforces a hard constraint on the top-5 positions exactly as specified, without introducing a tunable parameter.

**Edge case — fewer than 4 distinct lineages:** If the entire result set has only 3 distinct lineages, we cannot satisfy the "4 distinct in top 5" constraint. The implementation detects this and fills remaining top-5 slots with the best-scored duplicates from the deferred list (constraint relaxed gracefully rather than returning fewer than 5 results).

**Time complexity:** O(n) — single pass through the ranked list, O(1) dict lookups. For n=1000 items this is negligible (microseconds). A sort-based approach (O(n log n)) is unnecessary since the input is already ordered.

**Known limitation:** This approach only prevents the _same_ generation from filling the top 5. If there are 10 distinct generations but they all produce the same genre/mood, the result set may still feel repetitive. A more sophisticated approach would cluster by genre/mood and apply the diversity constraint at multiple levels. Deferred for now as it requires domain data not available in the current schema.
