# Databricks AI Search — Vector Search Index & Configuration

**Section:** 05-Assembling & Deploying Applications | **Module:** 02-Vector Search & Serving | **Est. time:** 3 hrs | **Exam mapping:** Assembling & Deploying (22%)

> ⚠️ **Fast-evolving:** Databricks AI Search (formerly Vector Search) is under active development. API names, SDK package names, and feature availability change frequently. Verify against current official docs before relying on any detail here. Last verified: 2026-07-11.

---

## TL;DR

Databricks AI Search (formerly Databricks Vector Search) is a managed approximate-nearest-neighbor search service built directly into the Databricks platform, turning a Delta table into a queryable similarity index in a few SDK calls. Two index types exist — Delta Sync (automatic replication from a Delta table) and Direct Vector Access (manual upsert) — each with distinct sync modes and embedding ownership models. The `AISearchClient` Python SDK is the primary programmatic interface for creating, syncing, and querying indexes. **The one thing to remember: a Vector Search index sits on top of an endpoint; the endpoint is the compute layer, the index is the data layer — you need both before you can run a single query.**

---

## ELI5 — Explain It Like I'm 5

Imagine a library where every book has a special scent attached to its cover. When you walk in and ask "find me something that smells like adventure," the librarian sniffs the room and hands you the three books whose scents are closest to your description — without reading a single word of any book. That is what vector search does: each document gets a numeric "scent" (its embedding), and when you query, the system finds the documents whose numbers are nearest to yours. The most common misconception is that the search reads and compares text directly — it does not; it compares floating-point number arrays, and the text comparison is an illusion produced upstream by the embedding model that converted language into those numbers. The endpoint is like the librarian's desk — it must exist and be running before the librarian can help anyone. The index is the catalogue of scent cards; creating it once is enough, but it needs to stay fresh as new books arrive.

---

## Learning Objectives

By the end of this chapter you will be able to:

- [ ] Distinguish between a Delta Sync Index and a Direct Vector Access Index, and select the right type given a scenario's update-frequency and ownership requirements.
- [ ] Configure an AI Search index with the correct embedding source (Databricks-managed vs. self-managed), `pipeline_type`, `primary_key`, and endpoint name.
- [ ] Create an endpoint and an index programmatically using `AISearchClient` and trigger a manual sync.
- [ ] Execute a `similarity_search()` query with column filters and choose the correct `query_type` (`ann`, `hybrid`, `FULL_TEXT`) for a given retrieval scenario.
- [ ] Explain how the HNSW algorithm, Reciprocal Rank Fusion, and the writeback table each contribute to index behavior.

---

## Visual Overview

### AI Search Architecture: Endpoint → Index → Query

```
┌─────────────────────────────────────────────────────────────┐
│  DATABRICKS WORKSPACE                                       │
│                                                             │
│  ┌──────────────────┐      ┌──────────────────────────┐    │
│  │  Source Delta    │      │   AI Search Endpoint     │    │
│  │  Table (UC)      │      │  (Compute / serving)     │    │
│  │  + Change Data   │      │  STANDARD or             │    │
│  │    Feed enabled  │      │  STORAGE_OPTIMIZED       │    │
│  └────────┬─────────┘      └───────────┬──────────────┘    │
│           │  sync (TRIGGERED /         │                    │
│           │  CONTINUOUS)               │                    │
│           ▼                            ▼                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  AI Search Index  (Delta Sync OR Direct Vector)      │  │
│  │  - HNSW vectors   - metadata columns                 │  │
│  │  - primary key    - optional writeback table         │  │
│  └──────────────────────────┬───────────────────────────┘  │
│                             │                               │
│              similarity_search() API                        │
│                             │                               │
│                             ▼                               │
│                  ┌──────────────────┐                       │
│                  │  RAG Retriever   │                       │
│                  │  / Application   │                       │
│                  └──────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

### Index Type Decision Tree

```
Need to create a Vector Search index
│
├── Does the data already live in a Delta table?
│   ├── YES ──► Delta Sync Index
│   │           │
│   │           ├── How often does the data change?
│   │           │   ├── Seconds latency required ──► pipeline_type = CONTINUOUS
│   │           │   └── Batch / on-demand refresh  ──► pipeline_type = TRIGGERED
│   │           │
│   │           └── Who computes embeddings?
│   │               ├── Databricks (text column) ──► embedding_source_column
│   │               └── You (precomputed floats)  ──► embedding_vector_column
│   │
│   └── NO / need direct write control ──► Direct Vector Access Index
│                                          (manual upsert via SDK / REST API)
```

### Hybrid Search Scoring Pipeline

```
Query text
    │
    ├──────────────────────────────────┐
    ▼                                  ▼
ANN (HNSW vector search)          BM25 keyword search
L2 distance ──► similarity score   token matching ──► BM25 score
    │                                  │
    └──────────┬───────────────────────┘
               ▼
    Reciprocal Rank Fusion (RRF)
    score = 1 / (60 + rank_ann) + 1 / (60 + rank_bm25)
               │
               ▼
    Normalized merged result list ──► top-k returned
```

### Sync Mode Comparison

```
TRIGGERED sync                           CONTINUOUS sync
─────────────────────────────────────    ──────────────────────────────────────
Delta table change                       Delta table change
        │                                        │
        │  (no automatic reaction)               │  (Spark Structured Streaming)
        │                                        ▼
  index.sync() call                     Streaming pipeline detects CDF
        │                               entries within seconds
        ▼                                        │
  Incremental rebuild starts                     ▼
        │                               Index updated continuously
        ▼
  Index updated (minutes to hours
  depending on size)

Cost:  lower                             Cost:  higher (dedicated cluster)
Best:  batch ETL, nightly refreshes     Best:  real-time RAG, live catalogs
```

---

## Key Concepts

### AI Search Endpoint

An AI Search endpoint is the managed compute infrastructure that hosts one or more indexes and serves query traffic. It is the layer that accepts `similarity_search()` REST/SDK calls and returns ranked results. Under the hood, Databricks provisions serverless compute to handle the HNSW index structure and the query-scoring logic; you never see or manage the underlying nodes. There are two endpoint types: **STANDARD** (incremental sync, lower cost, recommended for most use cases) and **STORAGE_OPTIMIZED** (partial rebuild on each sync, optimized for very large indexes or full-text-only workloads). In the Databricks UI, endpoints appear under **Compute → AI Search**; via SDK they are created with `AISearchClient().create_endpoint(name=..., endpoint_type="STANDARD")`.

### Delta Sync Index

A Delta Sync Index is an AI Search index that automatically mirrors a Unity Catalog Delta table, maintaining a vector representation of the table's contents. It tracks table changes using Delta's Change Data Feed (CDF) and propagates them into the HNSW index either continuously (via a Spark Streaming pipeline) or on demand (triggered mode). Under the hood, when `pipeline_type="TRIGGERED"`, calling `index.sync()` launches a Delta Live Tables-style incremental job that reads only changed rows since the last sync watermark, recomputes embeddings for those rows if Databricks-managed, and replaces the stale HNSW entries. The index is identified by a three-part Unity Catalog name (`catalog.schema.index_name`) and appears in Catalog Explorer alongside its source table.

### Direct Vector Access Index

A Direct Vector Access Index gives the application full control over what enters the vector store: you push rows — including pre-computed embedding arrays — directly via `index.upsert(...)` or the REST API, with no source Delta table required. Under the hood, the index stores whatever vectors you supply alongside metadata columns and updates them synchronously on each upsert call. There is no automatic sync; freshness is entirely the caller's responsibility. This type is mandatory when the source data is not a Delta table (e.g., real-time streamed embeddings computed externally) or when you need sub-second index updates without waiting for a sync pipeline. It must be created via the SDK or REST API — the UI does not support it.

### Embedding Source Options

The embedding source determines who converts raw text into floating-point vectors. **Databricks-managed embeddings** (`embedding_source_column`) point the index at a text column in the Delta table; Databricks calls a specified (or default) Foundation Model API endpoint at sync time to produce the vectors. **Self-managed embeddings** (`embedding_vector_column`) point the index at a column that already contains `array<float>` vectors; the application is responsible for generating and keeping those vectors current. Under the hood, Databricks-managed embeddings invoke the embedding model endpoint during each sync job batch; self-managed embeddings skip that call and use whatever floats are stored. The choice affects cost (managed embedding calls cost tokens), latency at sync time, and who owns embedding-model versioning. In the SDK, the choice is expressed by which keyword argument you provide: `embedding_source_column="text"` vs. `embedding_vector_column="text_vector"`.

### HNSW and the Similarity Search Algorithm

HNSW (Hierarchical Navigable Small World) is the graph-based approximate nearest neighbor algorithm that powers AI Search. It builds a multi-layer graph of vector nodes where each node connects to a small number of neighbors; queries traverse the graph from coarse upper layers to fine lower layers, finding approximate nearest neighbors orders of magnitude faster than brute-force search. AI Search uses **L2 (Euclidean) distance** as its metric; if you need cosine similarity, normalize your vectors before indexing — normalized L2 produces the same ranking as cosine similarity. The search score returned is `1 / (1 + L2_distance²)`, so higher scores are better. In Databricks, you do not configure HNSW parameters directly; they are managed by the platform.

### Hybrid Search and Reciprocal Rank Fusion

Hybrid search runs an ANN (vector) query and a BM25 keyword query in parallel, then merges their ranked lists using **Reciprocal Rank Fusion (RRF)**. Each document's RRF score is the sum of `1 / (60 + rank)` from each method; results are then normalized so the highest possible score is 1. This approach handles queries that contain both conceptual meaning (served by ANN) and specific tokens like SKUs, product codes, or proper nouns (served by BM25) — making it the recommended default for most RAG retrievers. Activated by passing `query_type="hybrid"` to `similarity_search()`.

### Index Sync Triggers

For TRIGGERED indexes, the sync must be initiated explicitly — it does not fire automatically when the Delta table changes. You start it by calling `index.sync()` on the index object returned by `get_index()`, or via the REST API `POST /api/2.0/vector-search/indexes/{index_name}/sync`, or from the Catalog Explorer UI with the **Sync now** button. Databricks computes only the delta since the last watermark (incremental for STANDARD endpoints; partial rebuild for STORAGE_OPTIMIZED endpoints). Orchestrating the sync trigger is the developer's responsibility — a common pattern is to add an `index.sync()` call at the end of the data pipeline that populates the source Delta table.

### RAG Retriever Integration

In a RAG pipeline, the AI Search index serves as the retrieval layer. A LangChain or LangGraph retriever wraps `similarity_search()` and converts the returned rows into `Document` objects that the LLM prompt template ingests. Databricks provides a `DatabricksVectorSearch` retriever class in `langchain-databricks` that accepts the index name and `text_column`; internally it calls the same SDK `similarity_search()` method. The integration point is the endpoint's REST URL, which the SDK discovers automatically from the index metadata. Filters can be passed to narrow retrieval to specific document subsets (e.g., per-tenant data) — a critical pattern for multi-tenant RAG security.

---

## Key Parameters / Configuration Knobs

| Parameter | What it controls | Decision rule |
|---|---|---|
| `endpoint_name` | Which AI Search endpoint hosts the index | Use one endpoint per team or environment; do not share a production endpoint with development indexes because endpoint downtime affects all hosted indexes |
| `pipeline_type` | `TRIGGERED` (manual sync) vs `CONTINUOUS` (streaming sync) | Choose `CONTINUOUS` only when staleness measured in minutes is unacceptable for the use case; otherwise use `TRIGGERED` to avoid the cost of a persistent streaming cluster |
| `embedding_source_column` | Text column Databricks will embed at sync time | Use when you do not want to manage embedding pipelines; requires specifying `embedding_model_endpoint_name` for production (do not rely on the default model for production workloads) |
| `embedding_vector_column` | Pre-computed `array<float>` column in the source table | Use when you control the embedding model version, need custom fine-tuned embeddings, or need to swap models without a full re-sync |
| `embedding_dimension` | Dimensionality of self-managed embedding vectors | Must match exactly the dimension of the model that generated them; mismatch causes sync failure |
| `primary_key` | Column used as a row identifier for upsert/delete operations | Must be unique and stable; avoid columns that change value over time (e.g., timestamps) because changing the primary key effectively orphans old vector entries |
| `columns_to_sync` | Subset of columns to replicate from the source table into the index | Omit to sync all columns; specify to reduce index storage and limit which metadata fields can be used as filters or returned in results |
| `index_name` | Three-part UC name (`catalog.schema.name`) | Follow the same naming convention as your source table to make lineage obvious; only alphanumeric characters and underscores are allowed |
| `endpoint_type` | `STANDARD` vs `STORAGE_OPTIMIZED` for the endpoint | Use `STORAGE_OPTIMIZED` only for very large indexes (billions of vectors) or dedicated full-text search workloads; `STANDARD` is the right default |
| `query_type` | `ann` (default), `hybrid`, `FULL_TEXT` | Default to `hybrid` for general RAG; use `ann` when queries are always semantic; use `FULL_TEXT` only when exact keyword matching is the only requirement |

---

## Worked Example: Requirement → Decision

**Given:** A product team is building a customer-support chatbot for an e-commerce platform. The knowledge base is a Delta table `catalog.support.articles` with 200,000 rows, updated nightly via a batch ETL job. Each row has a `content` (text) column and metadata columns `category` and `product_sku`. The team wants semantic search over `content` with the ability to filter by `product_sku` at query time. The chatbot's LangGraph agent must retrieve relevant articles before generating a response.

**Step 1 — Identify the goal:**
Expose the `support.articles` Delta table as a queryable semantic search index that a LangGraph retriever node can call with a natural-language query and optional `product_sku` filter.

**Step 2 — Define inputs:**
- Source table: `catalog.support.articles` with columns `id` (int, PK), `content` (string), `category` (string), `product_sku` (string)
- Embedding model: use Databricks Foundation Model API (managed embeddings) to avoid maintaining a separate embedding pipeline
- Update cadence: nightly batch ETL

**Step 3 — Define outputs:**
- An AI Search index `catalog.support.articles_index` on a STANDARD endpoint
- `similarity_search()` results containing `id`, `content`, `category`, `product_sku` ranked by relevance
- Filtered results when `product_sku` is known

**Step 4 — Apply constraints:**
- CDF must be enabled on `catalog.support.articles` (required for STANDARD endpoint)
- `TRIGGERED` sync mode is appropriate because data changes nightly, not in real-time
- `embedding_source_column="content"` is appropriate because the team does not manage their own embedding model
- `columns_to_sync` should include at minimum `id`, `content`, `category`, `product_sku` (all columns needed for filtering and result display)
- `primary_key="id"` because `id` is the stable unique identifier
- The ETL pipeline must call `index.sync()` after each batch load

**Step 5 — Select the approach:**
Create a **Delta Sync Index with Databricks-managed embeddings, TRIGGERED pipeline, and STANDARD endpoint**, then add `index.sync()` as the final step of the nightly ETL job. Using `CONTINUOUS` would be wasteful for a nightly-update cadence and would provision a persistent streaming cluster unnecessarily; using a Direct Vector Access Index would require building and maintaining a separate embedding pipeline and update job with no benefit.

---

## Implementation

```python
# Scenario: Create a nightly-refreshed semantic search index for a customer support
# knowledge base stored in a Unity Catalog Delta table, where the team wants
# Databricks to manage all embedding computation.

%pip install databricks-ai-search
dbutils.library.restartPython()

from databricks.ai_search.client import AISearchClient

client = AISearchClient()

# Step 1 — Create the endpoint (run once; skip if it already exists)
client.create_endpoint(
    name="support_search_endpoint",
    endpoint_type="STANDARD"
)

# Step 2 — Create the Delta Sync Index
index = client.create_delta_sync_index(
    endpoint_name="support_search_endpoint",
    source_table_name="catalog.support.articles",
    index_name="catalog.support.articles_index",
    pipeline_type="TRIGGERED",          # manual sync; triggered at end of ETL
    primary_key="id",
    embedding_source_column="content",  # Databricks computes embeddings
    embedding_model_endpoint_name="databricks-qwen3-embedding-0-6b",
    columns_to_sync=["id", "content", "category", "product_sku"]
)

print(f"Index created: {index.name}")
```

```python
# Scenario: At the end of the nightly ETL pipeline, trigger the index sync so
# newly loaded articles are searchable by the chatbot within minutes.

from databricks.ai_search.client import AISearchClient

client = AISearchClient()
index = client.get_index(index_name="catalog.support.articles_index")

# Trigger incremental sync — only changed rows since last watermark are reprocessed
index.sync()
print("Sync triggered successfully.")
```

```python
# Scenario: A LangGraph retriever node queries the index with a user question
# and constrains results to a specific product SKU so only relevant articles appear.

from databricks.ai_search.client import AISearchClient

client = AISearchClient()
index = client.get_index(index_name="catalog.support.articles_index")

results = index.similarity_search(
    query_text="How do I return a damaged item?",
    columns=["id", "content", "category", "product_sku"],
    filters={"product_sku": "SKU-42"},   # filter metadata at query time
    num_results=5,
    query_type="hybrid"                  # recommended default for RAG
)

for hit in results["result"]["data_array"]:
    print(hit)
```

```python
# Anti-pattern: Creating a Direct Vector Access Index and manually pushing
# embeddings when a Delta Sync Index with managed embeddings would do the job.
# Problem: this duplicates the embedding pipeline, breaks if the embedding model
# version changes, and requires a separate orchestration job to stay in sync.

# client.create_direct_access_index(...)          # <-- forces you to manage embeddings
# embed_and_upsert_all_articles(index, articles)  # <-- fragile, maintenance burden

# Correct approach: use Delta Sync Index with embedding_source_column so Databricks
# manages embedding computation, versioning, and incremental updates automatically.
index = client.create_delta_sync_index(
    endpoint_name="support_search_endpoint",
    source_table_name="catalog.support.articles",
    index_name="catalog.support.articles_index",
    pipeline_type="TRIGGERED",
    primary_key="id",
    embedding_source_column="content",   # Databricks handles embeddings
    embedding_model_endpoint_name="databricks-qwen3-embedding-0-6b"
)
```

```python
# Scenario: Self-managed embeddings are required because the team uses a custom
# fine-tuned embedding model and wants to control exactly which model version
# produces each row's vector, storing embeddings in the source table itself.

index = client.create_delta_sync_index(
    endpoint_name="support_search_endpoint",
    source_table_name="catalog.support.articles_with_embeddings",
    index_name="catalog.support.articles_custom_emb_index",
    pipeline_type="TRIGGERED",
    primary_key="id",
    embedding_dimension=768,              # must match model output dimension exactly
    embedding_vector_column="emb_vector"  # column of type array<float> in source table
)
```

---

## Common Pitfalls & Misconceptions

- **Confusing endpoint type with index type** — Beginners often use "endpoint" and "index" interchangeably because they are configured together in the UI. An endpoint is the always-on compute layer that can host many indexes; an index is a single dataset's vector representation. Deleting the endpoint deletes everything it hosts.

- **Forgetting to enable Change Data Feed on the source table** — CDF is required for STANDARD endpoint Delta Sync Indexes, but it is not enabled by default on Delta tables. Beginners discover this only when the index creation call fails with a cryptic error. Run `ALTER TABLE catalog.support.articles SET TBLPROPERTIES (delta.enableChangeDataFeed = true)` before creating the index.

- **Assuming TRIGGERED sync is automatic** — The word "triggered" implies something fires the sync; in practice, the developer must call `index.sync()` explicitly (or click Sync Now in the UI). Forgetting this means the index silently falls behind the source table without any warning.

- **Mismatching embedding dimension for self-managed embeddings** — When using `embedding_vector_column`, the `embedding_dimension` parameter must exactly match the dimensionality of the stored vectors. A mismatch does not fail loudly at index creation time; it surfaces as poor recall or a runtime error on the first sync.

- **Using cosine similarity without normalizing vectors** — AI Search uses L2 distance internally. If your downstream code or evaluation tooling expects cosine similarity, you must normalize each vector to unit length before inserting it. The platform does not do this automatically, and un-normalized L2 does not produce cosine rankings.

- **Querying with `query_type="ann"` for keyword-heavy workloads** — ANN is the default, and beginners often leave it unchanged. For queries that include product codes, model numbers, or exact phrases, ANN may miss documents that contain those exact tokens. Use `query_type="hybrid"` as the safe default for RAG.

- **Expecting sub-second freshness from a TRIGGERED index** — A TRIGGERED index only updates when `index.sync()` is called and the sync job completes (which can take minutes for large datasets). Use CONTINUOUS mode when the application requires near-real-time freshness.

---

## Key Definitions

| Term | Definition |
|---|---|
| **AI Search (formerly Vector Search)** | Databricks' managed similarity-search service that indexes Delta table rows as embedding vectors and serves approximate nearest neighbor queries |
| **Endpoint** | The always-on serverless compute unit that hosts one or more AI Search indexes and accepts query traffic; created with `create_endpoint()` |
| **Delta Sync Index** | An AI Search index that automatically or on-demand replicates a source Delta table's rows into an HNSW vector structure |
| **Direct Vector Access Index** | An AI Search index populated entirely by explicit upsert calls; no source Delta table is required |
| **TRIGGERED pipeline** | Sync mode where the developer manually initiates replication from the Delta table to the index by calling `index.sync()` |
| **CONTINUOUS pipeline** | Sync mode where a persistent Spark Structured Streaming job replicates Delta CDF events into the index with seconds of latency |
| **Databricks-managed embeddings** | Embedding computation performed by Databricks at sync time using a specified Foundation Model API endpoint; configured via `embedding_source_column` |
| **Self-managed embeddings** | Pre-computed `array<float>` vectors stored in the source Delta table and referenced via `embedding_vector_column`; the developer owns model versioning |
| **HNSW** | Hierarchical Navigable Small World — the graph-based approximate nearest neighbor algorithm used by AI Search for sub-linear query time |
| **Hybrid search** | A search mode that runs ANN and BM25 in parallel and merges results with Reciprocal Rank Fusion (`query_type="hybrid"`) |
| **Reciprocal Rank Fusion (RRF)** | Score-merging formula `sum(1 / (60 + rank_i))` used to combine ANN and keyword rankings in hybrid search |
| **Change Data Feed (CDF)** | Delta table feature that records row-level inserts, updates, and deletes; required for STANDARD endpoint Delta Sync Indexes |
| **AISearchClient** | The primary Python SDK class for creating, managing, and querying AI Search endpoints and indexes; installed via `pip install databricks-ai-search` |
| **`similarity_search()`** | The SDK method that queries an index; accepts `query_text` or `query_vector`, `filters`, `columns`, `num_results`, and `query_type` |

---

## Summary / Quick Recall

- An **endpoint** is compute; an **index** is data. Both are required before any query can run.
- **Delta Sync Index** = automatic source-of-truth from a Delta table; **Direct Vector Access** = manual upsert control.
- `pipeline_type="TRIGGERED"` requires an explicit `index.sync()` call; it does NOT fire automatically on table updates.
- `embedding_source_column` delegates embedding computation to Databricks; `embedding_vector_column` requires pre-computed `array<float>` in the table.
- AI Search uses **L2 distance + HNSW**; normalize vectors before indexing if your application logic assumes cosine similarity.
- `query_type="hybrid"` is the recommended default for RAG — it combines ANN semantic search with BM25 keyword matching via Reciprocal Rank Fusion.
- CDF (`delta.enableChangeDataFeed = true`) must be enabled on the source table before creating a STANDARD endpoint Delta Sync Index.

---

## Self-Check Questions

1. What must a developer do after calling `client.create_delta_sync_index(pipeline_type="TRIGGERED", ...)` to ensure the index contains data?

   <details><summary>Answer</summary>

   The developer must explicitly call `index.sync()` (or use the UI's "Sync now" button, or call the REST API sync endpoint). TRIGGERED mode does NOT automatically sync when the Delta table changes — it only performs a sync when the sync operation is manually invoked. Many beginners assume "triggered" means "triggered by a table change," but in Databricks AI Search it means "triggered by the developer." Until `sync()` is called, the index is empty.

   </details>

2. A team stores 512-dimensional embedding vectors in a Delta table column `emb` of type `array<float>`. They create a Delta Sync Index with `embedding_dimension=1024`. What happens?

   <details><summary>Answer</summary>

   The sync job will fail or produce incorrect results because the declared `embedding_dimension=1024` does not match the actual vector dimensionality of 512. The HNSW index structure is built around the declared dimension; mismatching it means the system tries to interpret 512-float arrays as 1024-float arrays, causing runtime errors or silent data corruption. The fix is to set `embedding_dimension=512` to match the actual output of the embedding model.

   </details>

3. **Which TWO** of the following are requirements for creating a Delta Sync Index on a STANDARD endpoint?
   - A. Change Data Feed must be enabled on the source Delta table
   - B. The source table must be in the `hive_metastore` catalog
   - C. The endpoint must be of type STORAGE_OPTIMIZED
   - D. The workspace must have serverless compute enabled
   - E. The embedding column must be of type `string`

   <details><summary>Answer</summary>

   **A and D** are correct.

   **A** — Change Data Feed (CDF) is explicitly required for STANDARD endpoint Delta Sync Indexes. The sync pipeline reads CDF entries to identify changed rows; without it, the sync job cannot determine what has changed.

   **D** — AI Search endpoints require serverless compute to be enabled in the workspace. This is a workspace-level prerequisite listed in the official documentation.

   **B** is wrong — the source table must be in **Unity Catalog**, not `hive_metastore`. UC governance is a prerequisite for AI Search.

   **C** is wrong — STANDARD is the default and most common endpoint type; STORAGE_OPTIMIZED is a separate, higher-cost option. You do not need STORAGE_OPTIMIZED for a basic Delta Sync Index.

   **E** is wrong — the embedding column for self-managed embeddings must be `array<float>`, not `string`. For Databricks-managed embeddings, the *source* column is `string`, but the embedding column itself is managed internally.

   </details>

4. A product manager asks why the production RAG chatbot sometimes fails to find articles that contain the exact product code "XP-7742" even though those articles are in the index. The chatbot uses `query_type="ann"`. What is the root cause and the fix?

   <details><summary>Answer</summary>

   The root cause is that ANN (approximate nearest neighbor) search operates on semantic similarity between embedding vectors. Product codes like "XP-7742" are out-of-vocabulary tokens for most embedding models — the model may not reliably embed them close to queries that mention them, so ANN misses the exact match. The fix is to switch to `query_type="hybrid"`, which adds BM25 keyword matching to the ANN search. BM25 will find documents that literally contain "XP-7742" even if the embedding vectors are not close. The hybrid RRF merge then combines the semantic and keyword rankings, ensuring both conceptual and exact-token matches surface in the results.

   </details>

5. A team must choose between `pipeline_type="CONTINUOUS"` and `pipeline_type="TRIGGERED"` for their AI Search index. Their source Delta table receives 10–20 new rows per hour from a production operational database, and the RAG application can tolerate up to 15 minutes of staleness. What should they choose and why?

   <details><summary>Answer</summary>

   They should choose **`TRIGGERED`**. CONTINUOUS mode spins up a persistent Spark Structured Streaming cluster that runs 24/7, incurring constant compute costs regardless of ingestion volume — an expensive choice when only 10–20 rows arrive per hour. Since the application can tolerate 15 minutes of staleness, a scheduled `index.sync()` call every 10–15 minutes via a lightweight Databricks Job is sufficient and far more cost-efficient. CONTINUOUS mode is the right choice only when staleness must be measured in seconds, which this use case does not require.

   </details>

---

## Further Reading

- [Databricks AI Search overview](https://docs.databricks.com/aws/en/ai-search/ai-search) — *verified 2026-07-11* — Architecture, endpoint options, and HNSW algorithm details
- [Create AI Search endpoints and indexes](https://docs.databricks.com/aws/en/ai-search/create-ai-search) — *verified 2026-07-11* — SDK, UI, and REST API walkthroughs for endpoint and index creation
- [Query an AI Search index](https://docs.databricks.com/aws/en/ai-search/query-ai-search) — *verified 2026-07-11* — `similarity_search()` API, filters, hybrid search, reranker, and pagination reference
