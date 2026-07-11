# Embeddings & Vector Search Intro

**Section:** Foundations | **Module:** RAG & Retrieval Foundations | **Est. time:** 2.5 hrs | **Exam mapping:** Supports Data Preparation (14%) & Application Development (30%)

> ⚠️ Fast-evolving: verify against current official docs before relying on this. As of mid-2026 Databricks renamed **Vector Search → "Databricks AI Search"** and the SDK from `databricks-vectorsearch`/`VectorSearchClient` to `databricks-ai-search`/`AISearchClient`. The concepts and index types below are unchanged, but the package name, class name, and endpoint SKU labels (Standard / Storage Optimized) have moved. The Mar-2026 exam guide and most existing material still say "Vector Search" — treat the two names as synonyms.

---

## TL;DR

Chapter 1 told you *why* RAG retrieves; this chapter is the *machinery*. Text becomes a dense vector (an embedding), similar meanings land near each other in vector space, and a vector index uses an approximate-nearest-neighbour algorithm (HNSW) to find the closest vectors fast instead of scanning every row. On Databricks this is Mosaic AI Vector Search: you point an index at a Delta table, choose who computes the embeddings (Databricks-managed vs self-managed) and how the index stays fresh (Delta Sync vs Direct Vector Access), then query it with `similarity_search`.

**The one thing to remember: retrieval quality is set by three coupled choices — the embedding model, the similarity metric, and the chunking — and the index type only decides how your data *gets in and stays fresh*, not how well it retrieves.**

---

## ELI5 — Explain It Like I'm 5

Imagine a giant library where books are shelved not by alphabet but by *what they are about*: cookbooks sit near each other, space books sit near each other, and a book about "the science of baking bread" sits somewhere between chemistry and cooking. To find books about your question, you don't read every spine — you walk to the region of the library where that topic lives and grab the nearest few. An embedding is the shelf coordinate the library gives each book based on meaning; the vector index is the map that lets you jump to the right region instead of walking every aisle. The common mistake is thinking the library matches your exact words — it doesn't; it matches *meaning*, so "how do I fix a dripping tap" lands next to a plumbing book that never uses the word "tap."

---

## Learning Objectives

By the end of this chapter you will be able to:
- [ ] Explain how embeddings encode semantic similarity and compute distance with cosine, dot product, and L2 metrics.
- [ ] Compare exact (brute-force) search with ANN/HNSW and articulate the recall-vs-latency trade-off.
- [ ] Distinguish a Delta Sync Index from a Direct Vector Access Index and choose between them for a given ingestion pattern.
- [ ] Configure managed vs self-managed embeddings and continuous vs triggered sync in Mosaic AI Vector Search.
- [ ] Design a dense / sparse / hybrid retrieval strategy and diagnose why chunking changes retrieval quality.

---

## Visual Overview

### Embedding Space — Near vs Far

```
                 vector space (each ● is one chunk's embedding)

        "refund policy" ●        ● "return a purchase"   ◄ close = similar meaning
                          ●  ● "money-back guarantee"

   query ▼ "how do I get my money back"
        ✚ ──► lands here, nearest neighbours are the 3 ● above

                                              ● "server latency tuning"  ◄ far = unrelated
                                        ● "GPU memory limits"
```

### Query Flow — Embed ──► ANN ──► Top-k

```
 user query text
      │
      ▼
 embed query  ──►  [0.12, -0.44, 0.98, ...]  (same model + dim as the index)
      │
      ▼
 HNSW graph walk over the index  ──►  approximate nearest neighbours
      │
      ▼
 apply filters (e.g. language='en')  ──►  optional reranker
      │
      ▼
 top-k rows (id, text, metadata)  ──►  stuffed into the LLM prompt
```

### Index-Type Decision Tree

```
Do you already have your data in (or can land it in) a Delta table?
├── Yes ──► Do you want Databricks to keep the index in sync automatically?
│           ├── Yes ──► DELTA SYNC INDEX
│           │            ├── Databricks computes embeddings? ──► managed (embedding_source_column)
│           │            └── You precomputed vectors?        ──► self-managed (embedding_vector_column)
│           └── No, I will push/patch vectors myself via API ──► (rare) still Delta Sync + Triggered
└── No — vectors are produced by an external system and pushed directly
            └──────────────────────────────────────────────► DIRECT VECTOR ACCESS INDEX (upsert/delete via SDK/REST)
```

---

## Key Concepts

### Embeddings (Dense Vectors)

**What it is.** An embedding is a fixed-length array of floats (typically 384–1536 dimensions) that a model assigns to a piece of text or an image so that semantically similar inputs get numerically close vectors.

**How it works under the hood.** A transformer embedding model maps tokens through learned weights into a single pooled vector; training pulls vectors of related text together and pushes unrelated text apart. Because meaning is encoded as *position*, "distance" between two vectors becomes a proxy for "difference in meaning." The dimensionality is a fixed property of the model — an `e5-small-v2` vector is not comparable to a 1536-dim vector.

**Where in Databricks.** You either let Databricks generate embeddings from a Delta table text column (managed) via a Foundation Model APIs or model-serving endpoint such as `databricks-qwen3-embedding-0-6b`, or you precompute vectors yourself and store them as an `array<float>` column (self-managed).

### Similarity Metrics — Cosine, Dot Product, L2

**What it is.** The rule used to score how close two vectors are: cosine similarity (angle), dot product (angle + magnitude), and L2 / Euclidean distance (straight-line distance).

**How it works under the hood.** Cosine ignores vector length and measures direction only — ideal when magnitude is noise. Dot product rewards both alignment *and* magnitude. L2 measures geometric distance; smaller is closer. Key fact: when vectors are **normalized to unit length**, the ranking produced by L2 distance is identical to the ranking produced by cosine similarity — they only differ in the scores, not the order.

**Where in Databricks.** Mosaic AI Vector Search uses **L2 distance** internally with HNSW. If you want cosine behaviour, you **normalize your embeddings before ingest and query**; then L2 ranking matches cosine ranking. The returned relevance score is computed as `1 / (1 + dist²)`, so higher score = closer.

### ANN & HNSW vs Exact Search — Recall/Latency Trade-off

**What it is.** Exact (brute-force) search compares the query to *every* vector and is guaranteed to return the true nearest neighbours; Approximate Nearest Neighbour (ANN) search trades a small amount of accuracy for a large speed gain. HNSW (Hierarchical Navigable Small World) is the specific ANN algorithm Databricks uses.

**How it works under the hood.** HNSW builds a layered graph of vectors: sparse long-range links at the top for fast jumps, dense short-range links at the bottom for precision. A query greedily hops toward closer neighbours layer by layer, visiting a tiny fraction of the index. **Recall** is the fraction of the true top-k that ANN actually returned; pushing recall toward 100% means visiting more of the graph, which raises latency. That is the core recall-vs-latency dial.

**Where in Databricks.** HNSW + ANN is the default `query_type="ann"`. You rarely tune graph internals directly; you trade recall/latency through choices like endpoint SKU, embedding dimensionality, and `num_results`.

### Vector Index / Vector Database — Delta Sync vs Direct Vector Access

**What it is.** A vector index is the searchable structure (HNSW graph + stored vectors + metadata) served by a **Vector Search endpoint**. Databricks offers two index types: **Delta Sync Index** and **Direct Vector Access Index**.

**How it works under the hood.** A **Delta Sync Index** is bound to a source Delta table and updates itself incrementally as the table changes (standard endpoints require Change Data Feed on the source). A **Direct Vector Access Index** has no source table — *you* own writing vectors into it with `upsert`/`delete` via the SDK or REST API, and it cannot be created from the UI.

**Where in Databricks.** Both live under Unity Catalog with a three-level name `catalog.schema.index`. Delta Sync is created with `create_delta_sync_index(...)`; Direct Vector Access with `create_direct_access_index(...)`.

### Managed vs Self-Managed Embeddings & Sync Modes

**What it is.** Two orthogonal choices for a Delta Sync Index: *who computes the vectors* (managed = Databricks, self-managed = you) and *how often the index refreshes* (`pipeline_type` CONTINUOUS or TRIGGERED).

**How it works under the hood.** With **managed embeddings** you pass `embedding_source_column` + `embedding_model_endpoint_name`; Databricks embeds each row at ingest and embeds the query at search time (adds a serving call to query latency). With **self-managed** you pass `embedding_vector_column` + `embedding_dimension` and supply the vectors. **CONTINUOUS** keeps latency to seconds by running a streaming pipeline (higher cost, a cluster stays up); **TRIGGERED** processes changes only when you call `index.sync()` (cheaper, staler between runs).

**Where in Databricks.** Both are arguments to `create_delta_sync_index`. Managed-embedding queries pass `query_text`; self-managed queries can pass either `query_text` (if a query model is configured) or a raw `query_vector`.

### Dense vs Sparse vs Hybrid Retrieval

**What it is.** Dense retrieval = vector/semantic (ANN). Sparse retrieval = keyword scoring (BM25 / full-text). Hybrid = both, fused together.

**How it works under the hood.** Sparse search (Databricks uses Okapi BM25) matches exact tokens — great for SKUs, error codes, proper nouns. Dense search matches meaning — great for paraphrases. **Hybrid** runs ANN and keyword search in parallel and merges them with **Reciprocal Rank Fusion (RRF)**, which re-scores each doc by `1/(rrf_param + rank)` (rrf_param = 60) and sums the contributions. Hybrid is the recommended general-purpose default but uses roughly 2× the resources of ANN.

**Where in Databricks.** Set `query_type="hybrid"` (or `"ann"`, or `"FULL_TEXT"` for keyword-only, Beta) on `similarity_search`. An optional cross-encoder reranker (`DatabricksReranker`) adds ~10% quality for extra latency.

### Chunking and Its Effect on Retrieval

**What it is.** Chunking is splitting source documents into the units that get embedded and indexed; each chunk becomes one row/vector.

**How it works under the hood.** The embedding of a chunk is an *average* of its content, so an oversized chunk blurs multiple topics into one vector and dilutes similarity, while a too-small chunk loses the surrounding context needed to answer. Overlap between adjacent chunks preserves continuity across split points. Because retrieval returns whole chunks, chunk boundaries directly set what context the LLM can ever see.

**Where in Databricks.** Chunking happens upstream in your Delta table (one row per chunk) before the Vector Search index reads the text column; it is a data-preparation decision, not an index setting.

### Key Parameters / Configuration Knobs

| Parameter | What it controls | Decision rule |
|---|---|---|
| Index type (`create_delta_sync_index` vs `create_direct_access_index`) | Whether the index tracks a Delta table or is written directly by you | Data lives in Delta → Delta Sync; vectors produced by an external system pushed via API → Direct Vector Access |
| `embedding_source_column` vs `embedding_vector_column`+`embedding_dimension` | Managed (Databricks embeds) vs self-managed (you supply vectors) | Want simplicity / no embedding infra → managed; need lowest query latency or a specific pre-computed model → self-managed |
| `pipeline_type` (`CONTINUOUS` / `TRIGGERED`) | Freshness vs cost of a Delta Sync Index | Sub-minute freshness required → CONTINUOUS; periodic/batch updates acceptable → TRIGGERED (cheaper) |
| `query_type` (`ann` / `hybrid` / `FULL_TEXT`) | Semantic vs keyword vs fused retrieval | Mixed conceptual + keyword workload → hybrid; pure semantic + max QPS → ann; exact-term lookup → FULL_TEXT |
| `num_results` | How many neighbours are returned | Keep 10–100; raising it deep-scans the index (10× can 2× latency, −3× QPS) |
| `columns` / `columns_to_sync` | Which fields are indexed/returned/filterable | Sync only PK + embedding + fields you actually filter/display on to cut cost and latency |
| Endpoint SKU (Standard / Storage Optimized) | Latency vs scale/cost | Latency-critical and < ~320M vectors → Standard; 10M+ vectors, cost-sensitive, tolerate ~300–500 ms → Storage Optimized |

### Worked Example: Requirement → Decision

**Given:** A retailer keeps a product catalogue (~2M products) in a Unity Catalog Delta table. Descriptions are updated in a nightly batch job. They want semantic search over descriptions for a support chatbot and have no embedding infrastructure of their own.

**Step 1 — Identify the goal.** Serve semantically-relevant product chunks to a RAG chatbot, kept in step with a once-a-day catalogue refresh, with minimal operational overhead.

**Step 2 — Define inputs.** A governed Delta table with a `product_id` primary key and a `description` text column; no precomputed vectors; nightly (not real-time) update cadence; team wants Databricks to own embeddings.

**Step 3 — Define outputs.** A queryable Vector Search index returning top-k `(product_id, description)` rows via `similarity_search(query_text=...)`.

**Step 4 — Apply constraints.** Data already in Delta; freshness requirement is "within a day," not seconds; no self-hosted embedding model; ~2M vectors fits a Standard endpoint's single vector-search-unit sweet spot for low latency.

**Step 5 — Select the approach.** Use a **Delta Sync Index with managed embeddings** (`embedding_source_column="description"`) on a **Standard endpoint**, with **`pipeline_type="TRIGGERED"`** fired at the end of the nightly job. *Rationale vs alternatives:* Direct Vector Access would force the team to build and run their own embedding + upsert pipeline for data that is already in Delta — pure overhead. CONTINUOUS sync would keep a streaming cluster running 24/7 for data that only changes once a day — unnecessary cost. Self-managed embeddings would require standing up an embedding model they explicitly don't have.

---

## Implementation

```python
# Scenario: Stand up semantic search over a nightly-updated product catalogue
# already in Unity Catalog, letting Databricks own embeddings, refreshed once per
# night to avoid paying for a 24/7 streaming sync.
from databricks.ai_search.client import AISearchClient  # formerly VectorSearchClient

client = AISearchClient()

client.create_endpoint(name="catalog_search_ep", endpoint_type="STANDARD")

index = client.create_delta_sync_index(
    endpoint_name="catalog_search_ep",
    source_table_name="prod.retail.products",
    index_name="prod.retail.products_idx",
    pipeline_type="TRIGGERED",                 # nightly batch, not continuous
    primary_key="product_id",
    embedding_source_column="description",     # managed embeddings
    embedding_model_endpoint_name="databricks-qwen3-embedding-0-6b",
    columns_to_sync=["product_id", "description", "category"],  # only what we filter/show
)

# Query: same model embeds the query text; filter on a synced metadata column.
results = index.similarity_search(
    query_text="waterproof hiking jacket",
    columns=["product_id", "description"],
    filters={"category": "outerwear"},
    num_results=10,          # stay in the 10–100 sweet spot
    query_type="hybrid",     # SKUs/brand names benefit from keyword + semantic
)
```

```python
# Anti-pattern: forcing freshness by dropping and rebuilding a Delta Sync Index on
# every catalogue change, and querying with a DIFFERENT embedding model than the one
# used at ingest. Both break retrieval.
client.delete_index("prod.retail.products_idx")          # throws away the HNSW graph
index = client.create_delta_sync_index(..., embedding_model_endpoint_name="e5-small-v2")  # 384-dim
# ...then querying self-managed vectors from a 1024-dim model:
index.similarity_search(query_vector=[0.1] * 1024, num_results=10)  # DIMENSION MISMATCH

# What breaks:
#  1) A Delta Sync Index already updates incrementally — manual rebuilds waste compute,
#     cause downtime, and defeat the point of "sync". Use pipeline_type + index.sync().
#  2) Query vectors MUST come from the SAME model/dimension as the indexed vectors.
#     A 1024-dim query against a 384-dim index is meaningless (or errors); the "nearest"
#     neighbours are noise.

# Correct approach: keep the index, trigger an incremental sync, and let the SAME
# managed model embed both rows and queries.
index = client.get_index(index_name="prod.retail.products_idx")
index.sync()                                             # incremental, no rebuild, no downtime
results = index.similarity_search(query_text="rain jacket", columns=["product_id"], num_results=10)
```

---

## Common Pitfalls & Misconceptions

- **Assuming Databricks uses cosine similarity by default** — because tutorials everywhere say "cosine," beginners expect it out of the box. Databricks uses **L2 distance** with HNSW; to get cosine behaviour you must **normalize vectors before ingest and query**, after which L2 ranking equals cosine ranking.
- **Treating the index type as a quality lever** — newcomers pick Delta Sync vs Direct Vector Access hoping for better answers. Index type only governs *how data enters and stays fresh*; retrieval quality is driven by the embedding model, similarity metric, and chunking.
- **Mixing embedding models between ingest and query** — it seems harmless to swap the query-side model. Vectors are only comparable within the *same model and dimension*; a mismatch returns nonsense, so pin one model (or a matched query model) end to end.
- **Cranking `num_results` "to be safe"** — more results feels like more recall. It deep-scans the HNSW graph, so 10× `num_results` can double latency and cut QPS ~3×; keep it 10–100 and rerank instead.
- **Believing ANN returns the exact nearest neighbours** — the word "nearest" misleads. ANN is *approximate*: recall < 100% by design, and you buy recall back with latency, not for free.
- **Choosing CONTINUOUS sync for batch data** — "real-time sounds better" keeps a streaming cluster billing 24/7. Match sync mode to update cadence: TRIGGERED for periodic/batch, CONTINUOUS only when sub-minute freshness is required.

---

## Key Definitions

| Term | Definition |
|---|---|
| Embedding | A fixed-length dense float vector encoding the semantic content of text/image; similar meaning → nearby vectors. |
| Similarity metric | The scoring rule (cosine, dot product, L2) that quantifies closeness between two vectors. |
| L2 / Euclidean distance | Straight-line distance between vectors; the metric Mosaic AI Vector Search uses internally. |
| ANN (Approximate Nearest Neighbour) | Search that returns *approximately* the closest vectors, trading a little recall for large speed gains. |
| HNSW | Hierarchical Navigable Small World — the layered-graph ANN algorithm Databricks uses. |
| Recall | Fraction of the true top-k neighbours that the ANN search actually returned. |
| Vector Search endpoint | The served compute (Standard or Storage Optimized) that hosts and answers queries for one or more indexes. |
| Delta Sync Index | An index bound to a source Delta table that updates incrementally (continuous or triggered). |
| Direct Vector Access Index | An index with no source table; vectors are written directly via `upsert`/`delete`. |
| Managed embeddings | Databricks computes embeddings from a Delta text column at ingest and query time. |
| Self-managed embeddings | You precompute vectors and store them in an `array<float>` column. |
| Hybrid search | Runs ANN + BM25 keyword search in parallel and merges with Reciprocal Rank Fusion (RRF). |
| Chunking | Splitting documents into the units that get embedded and indexed (one chunk = one vector). |

---

## Summary / Quick Recall

- Embeddings turn meaning into position; distance ≈ semantic difference.
- Databricks uses **L2 + HNSW**; normalize vectors to get cosine ranking.
- ANN trades recall for latency — "nearest" is approximate by design.
- **Delta Sync Index** = tracks a Delta table (continuous/triggered); **Direct Vector Access** = you write vectors yourself.
- **Managed** embeddings = Databricks embeds from a text column; **self-managed** = you supply the `array<float>`.
- Query with `similarity_search`; `query_type` = ann / hybrid / FULL_TEXT; keep `num_results` 10–100.
- Hybrid (dense + BM25 via RRF) is the safe general-purpose default; reranker adds ~10% quality for latency.
- Index type controls freshness/ingestion, not retrieval quality — that's the model, metric, and chunking.

---

## Self-Check Questions

1. Which similarity metric does Databricks Mosaic AI Vector Search use internally, and what must you do to make its ranking equivalent to cosine similarity?

   <details><summary>Answer</summary>

   It uses **L2 (Euclidean) distance** with HNSW. To make L2 ranking equivalent to cosine similarity you must **normalize the embeddings to unit length before ingest and query** — once normalized, L2 and cosine produce the same ordering (only the scores differ). The tempting wrong answer is "cosine by default"; cosine is what most tutorials assume, but Databricks does not apply it unless your vectors are normalized.

   </details>

2. Your data already lives in a Unity Catalog Delta table that updates every few minutes, and you want the index to stay fresh automatically with the lowest latency. Which index type and sync mode fit best?

   <details><summary>Answer</summary>

   A **Delta Sync Index** with **`pipeline_type="CONTINUOUS"`**. Delta Sync binds to the source table and updates incrementally; CONTINUOUS keeps latency to seconds via a streaming pipeline, which matches the "every few minutes / automatic" requirement. A Direct Vector Access Index is wrong here — it has no source table and would force you to write vectors manually, discarding the fact that your data is already in Delta. TRIGGERED would leave the index stale between manual syncs.

   </details>

3. **Which TWO** of the following are valid reasons to choose **self-managed embeddings** over managed embeddings in a Delta Sync Index?
   - A. You want Databricks to embed the query text automatically at search time.
   - B. You need the lowest possible query latency by passing precomputed `query_vector`s.
   - C. You have already computed embeddings with a specific model and stored them as `array<float>`.
   - D. You have no embedding model or serving endpoint of your own.
   - E. You want the index to sync automatically from a Delta table.

   <details><summary>Answer</summary>

   **B and C.** Self-managed embeddings avoid a runtime serving call, so passing a precomputed `query_vector` gives the fastest retrieval (B), and it is the right choice when you already produced vectors with a chosen model and stored them as `array<float>` (C). A and D describe *managed* embeddings (Databricks embeds rows and queries for you, ideal when you have no embedding infrastructure). E is irrelevant to the managed/self-managed choice — automatic sync is a property of Delta Sync Indexes regardless of who computes the embeddings, so it's the most tempting distractor but does not distinguish the two.

   </details>

4. A team raises `num_results` from 10 to 200 "to improve answer quality" and reports that latency roughly doubled while QPS fell sharply, yet answers didn't improve. What is happening and what's the better fix?

   <details><summary>Answer</summary>

   Large `num_results` forces a **deeper scan of the HNSW graph**, which raises latency and cuts QPS (per the performance guide, 10× results ≈ 2× latency and ~−3× QPS) without helping the LLM, which only needs a few relevant chunks. The better fix is to keep `num_results` in the **10–100 range** and, if precision is the real problem, add a **reranker** or switch to **hybrid** search. The tempting wrong conclusion is "the index is too small / needs more compute" — the bottleneck is the query parameter, not capacity.

   </details>

5. You must choose between a Standard endpoint and a Storage Optimized endpoint for an index that will hold ~80M vectors and is cost-sensitive, tolerating a few hundred milliseconds of latency. Which do you pick, and what's the trade-off you're accepting?

   <details><summary>Answer</summary>

   Pick **Storage Optimized**. It's designed for 10M+ (up to ~1B) vectors and is markedly cheaper per vector, at the cost of higher latency (~300–500 ms) and lower peak QPS than Standard. You're explicitly accepting higher latency in exchange for scale and cost efficiency, which the requirement allows. **Standard** would be the wrong pick here: it targets latency-critical workloads under ~320M vectors but is more expensive per vector and provides no benefit when a few hundred ms is acceptable and cost matters. Note Storage Optimized supports only TRIGGERED sync, which you'd also need to plan for.

   </details>

---

## Further Reading

- [Databricks AI Search (overview)](https://docs.databricks.com/aws/en/ai-search/ai-search) — *verified 2026-07-11* — HNSW + L2 mechanics, cosine-via-normalization, hybrid/BM25/RRF, and the managed-vs-self-managed embedding options.
- [Create AI Search endpoints and indexes](https://docs.databricks.com/aws/en/ai-search/create-ai-search) — *verified 2026-07-11* — Delta Sync vs Direct Vector Access, `create_delta_sync_index`, `pipeline_type`, `columns_to_sync`, and Unity Catalog requirements.
- [Query an AI Search index](https://docs.databricks.com/aws/en/ai-search/query-ai-search) — *verified 2026-07-11* — `similarity_search`, `query_type` (ann/hybrid/FULL_TEXT), filters, reranking, and the retrieval-algorithm decision table.
- [AI Search performance guide](https://docs.databricks.com/aws/en/ai-search/best-practices) — *verified 2026-07-11* — recall/latency, embedding dimensionality, `num_results` guidance, SKU (Standard vs Storage Optimized) sizing, and managed-vs-self-managed latency.
