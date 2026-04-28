# MusicGPT — Search Integrity Challenge

Hybrid search system over a generative music dataset. Combines pgvector (HNSW) for semantic similarity and Postgres full-text search (GIN/tsvector) for exact lexical matching, fused via Reciprocal Rank Fusion. Results are re-ranked by relevance, recency, and engagement confidence (Wilson CTR), then diversity-enforced across generation lineages.

---

## Prerequisites

- Docker + Docker Compose
- Python 3.12+
- `sentence-transformers` requires no API key — embedding runs locally

---

## Setup

**1. Start Postgres with pgvector:**
```bash
docker compose up -d
```
Waits for the healthcheck to pass before proceeding.

**2. Install dependencies:**
```bash
pip install -r requirements.txt
pip install sentence-transformers
```

**3. Run migrations:**
```bash
alembic upgrade head
```
Creates the `generations` and `songs` tables, HNSW index, GIN index, and the `search_vector` trigger.

**4. Seed the database:**
```bash
python seed.py
```
Fetches the dataset from S3, generates 384-dim embeddings locally via `all-MiniLM-L6-v2`, and inserts 48 generations (96 songs) into Postgres.

**5. Run verification:**
```bash
python verify.py
```
Runs two queries and a diversity check. Expected output: two `[PASS]` lines and one `[WARN]` (dataset too small for the 4-lineage diversity constraint — expected behaviour).

---

## File overview

| File | Purpose |
|---|---|
| `models.py` | SQLAlchemy models — `generations` + `songs` schema |
| `alembic/` | Migrations — schema, indexes, search_vector trigger |
| `search.py` | Hybrid search query (vector + FTS → RRF) |
| `ranking.py` | Re-ranking (`calculate_final_score`) + diversity (`diversify_results`) |
| `feedback.py` | High-throughput click/impression ingestion — in-process buffer + batch UPSERT |
| `seed.py` | Dataset fetch, DynamoDB unwrapping, embedding generation, DB insert |
| `verify.py` | Verification script — proves re-ranking and FTS correctness |
| `DECISIONS.md` | Design decisions, tradeoffs, and known limitations for all 5 parts |

---

## Database connection

```
postgresql://musicgpt:musicgpt@localhost:5432/musicgpt
```
