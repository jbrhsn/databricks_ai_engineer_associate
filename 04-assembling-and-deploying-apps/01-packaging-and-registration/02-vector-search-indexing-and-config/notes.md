# Vector Search Indexing and Config

**Section:** Assembling and Deploying Apps | **Module:** Packaging and Registration | **Est. time:** 2.5 hrs
**Exam mapping:** Databricks GenAI Engineer Associate — Assembling and Deploying Applications domain (~20%); creating and configuring AI Search (Vector Search) indexes, sync modes, and query types are directly tested. Overlaps with the Data Preparation domain (~20%).

## Learning Objectives

By the end of this chapter, you will be able to:

- Distinguish an AI Search **endpoint** from an **index** and choose the right endpoint type
- Create a **Delta Sync** index (managed and self-managed embeddings) and a **Direct Access** index
- Choose **TRIGGERED vs CONTINUOUS** sync for a given freshness/cost requirement
- Configure the prerequisites (Change Data Feed, UC permissions, supported embedding models)
- Query an index with **ANN, full-text, and hybrid** search, apply filters, and add reranking
- Explain the current product naming (Databricks **AI Search**, formerly Vector Search)

## Core Concepts

### Naming: AI Search is the new name for Vector Search

As of 2026 docs, the product formerly called **Databricks Vector Search** (and before that "Mosaic
AI Vector Search") is now **Databricks AI Search**. Every doc page carries the banner *"Databricks AI
Search was formerly known as Databricks Vector Search."*

But — and this matters for the exam and for real code — **the API surface still says
`vector-search`**. REST paths are `/api/2.0/vector-search/...`, the SQL function is `vector_search()`,
and the LangChain tool is `VectorSearchRetrieverTool`. So you may see either name; treat them as
synonyms. (The current SDK docs standardize on the `databricks-ai-search` package / `AISearchClient`
class, but the legacy `databricks-vectorsearch` / `VectorSearchClient` has the **same method names**,
so examples are interchangeable in practice.)

### Endpoints vs. indexes

These are two different objects and the exam tests whether you can keep them straight:

- An **endpoint** is the **compute** that hosts and serves indexes for querying. You create it once
  and it can host **multiple indexes**. Managed on the Compute tab.
- An **index** is the **searchable structure** — the embedded vectors plus metadata — built from a
  Delta table. It lives *on* an endpoint.

Analogy: the endpoint is the database server; the index is a table on it.

### Endpoint types: STANDARD vs STORAGE_OPTIMIZED

When you create an endpoint you pick `endpoint_type`:

| | STANDARD | STORAGE_OPTIMIZED |
|---|---|---|
| Best for | General-purpose, low-latency | Large datasets at lower cost |
| Sync modes | **TRIGGERED and CONTINUOUS** | **TRIGGERED only** |
| Query types | ANN, full-text, hybrid | ANN, full-text (Beta), hybrid |
| Filter syntax | **Dictionary** (`{"id <": 200}`) | **SQL-like string** (`"id < 200"`) |
| Self-managed embedding dim | any | **must be divisible by 16** |
| Notable | supports `target_qps` scaling (Preview) | partial rebuild each sync; batch embedding at ingest |

```python
from databricks.vector_search.client import VectorSearchClient  # or databricks.ai_search
client = VectorSearchClient()

client.create_endpoint(
    name="prod_agents_endpoint",
    endpoint_type="STANDARD",   # or "STORAGE_OPTIMIZED"
)
```

### Index types

**1. Delta Sync Index** — automatically and incrementally syncs with a source Delta table. This is
the workhorse for RAG. It comes in two flavors:

- **Managed embeddings:** Databricks computes the embeddings for you. You give it a **text column**
  (`embedding_source_column`) and an **embedding model endpoint** (`embedding_model_endpoint_name`).
  This is the simplest and most common choice.
- **Self-managed embeddings:** your source table *already* has a precomputed vector column
  (`embedding_vector_column`, type `array<float>`), and you tell it the `embedding_dimension`.
  Use this when you generate embeddings yourself.

**2. Direct Vector Access Index** — no source Delta table. You write vectors and metadata directly
via the SDK (`upsert`, `delete`). Use it for real-time writes / streaming inserts or when you fully
manage the lifecycle in code. **It cannot be created in the UI** — SDK or REST only.

```python
# Delta Sync with MANAGED embeddings — the common RAG case
index = client.create_delta_sync_index(
    endpoint_name="prod_agents_endpoint",
    source_table_name="prod.docs.chunks",          # Delta table of text chunks
    index_name="prod.agents.databricks_docs_index", # UC 3-level name
    pipeline_type="TRIGGERED",                      # or "CONTINUOUS"
    primary_key="chunk_id",
    embedding_source_column="text",
    embedding_model_endpoint_name="databricks-gte-large-en",
)

# Delta Sync with SELF-MANAGED embeddings — table already has vectors
index = client.create_delta_sync_index(
    endpoint_name="prod_agents_endpoint",
    source_table_name="prod.docs.chunks",
    index_name="prod.agents.databricks_docs_index",
    pipeline_type="TRIGGERED",
    primary_key="chunk_id",
    embedding_dimension=1024,
    embedding_vector_column="text_vector",
)

# Direct Access — you upsert vectors yourself
index = client.create_direct_access_index(
    endpoint_name="prod_agents_endpoint",
    index_name="prod.agents.realtime_index",
    primary_key="id",
    embedding_dimension=1024,
    embedding_vector_column="text_vector",
    schema={"id": "int", "text": "string", "text_vector": "array<float>"},
)
index.upsert([{"id": 1, "text": "hello", "text_vector": [0.1] * 1024}])
```

### TRIGGERED vs CONTINUOUS sync

For Delta Sync indexes, `pipeline_type` controls how the index keeps up with the source table:

- **TRIGGERED** — sync runs **on demand**: `index.sync()`, the UI "Sync now" button, or a scheduled
  job. Compute spins up per sync, so it's **cheaper**. Best when data changes on a schedule and a few
  minutes/hours of staleness is fine. **This is the only mode on storage-optimized endpoints and for
  full-text indexes.**
- **CONTINUOUS** — the index stays fresh within **seconds** via an always-on streaming pipeline. More
  expensive (a cluster runs continuously). Use it for near-real-time freshness. **Standard endpoints
  only.**

Both modes are **incremental** — only changed rows are processed, not the whole table.

### Prerequisites (frequently tested)

- **Unity Catalog-enabled workspace** + **serverless compute** enabled.
- **Change Data Feed (CDF) must be enabled on the source Delta table** for Delta Sync on standard
  endpoints. Without it, the index has no way to detect incremental changes. Enable with
  `ALTER TABLE ... SET TBLPROPERTIES (delta.enableChangeDataFeed = true)`.
- **`CREATE TABLE`** privilege on the target UC schema to create an index.
- The index name is a **UC three-level namespace** (`catalog.schema.name`), alphanumeric + underscores.
- **To query an index you don't own:** `USE CATALOG` + `USE SCHEMA` + **`SELECT` on the index**.
- **Supported embedding models** (Foundation Model APIs, 2026):

| Model endpoint | Dim | Context | Notes |
|---|---|---|---|
| `databricks-gte-large-en` | 1024 | 8192 tokens | not normalized |
| `databricks-bge-large-en` | 1024 | 512 tokens | normalized |
| `databricks-qwen3-embedding-0-6b` | up to 1024 | ~32K tokens | multilingual, Preview |

### Querying the index

```python
index = client.get_index(index_name="prod.agents.databricks_docs_index")

# ANN (default) — approximate nearest neighbor, semantic similarity
results = index.similarity_search(
    query_text="How do I create a vector index?",
    columns=["chunk_id", "text", "doc_uri"],
    num_results=5,
)

# Hybrid — ANN + full-text merged with Reciprocal Rank Fusion; the recommended default
results = index.similarity_search(
    query_text="create vector index",
    columns=["chunk_id", "text"],
    num_results=5,
    query_type="hybrid",
)
```

Query types:

- **ANN** (default) — HNSW approximate nearest neighbor over **L2 distance**. Semantic matching.
- **Full-text** (Beta) — BM25 keyword scoring over text columns. Exact-term matching.
- **Hybrid** — runs ANN and full-text in parallel and merges with **Reciprocal Rank Fusion (RRF)**.
  The recommended general-purpose default because it catches both semantic and keyword matches.

**Filters differ by endpoint type** — a genuine exam gotcha:
- **Standard endpoints** use **dictionary** syntax with operator suffixes:
  `filters={"year >=": 2024, "category": "docs"}`, `{"title NOT": "draft"}`, `{"tag LIKE": "delta"}`.
- **Storage-optimized endpoints** use a **SQL-like filter string**:
  `filters="year >= 2024 AND category = 'docs'"`.

### Reranking (GA in 2026)

For higher precision, add a cross-encoder reranker after retrieval — typically ~10% quality gain at
the cost of ~1.5s extra latency:

```python
from databricks.vector_search.reranker import DatabricksReranker

results = index.similarity_search(
    query_text="How do I create a vector index?",
    columns=["chunk_id", "text", "doc_summary"],
    num_results=10,
    query_type="hybrid",
    reranker=DatabricksReranker(columns_to_rerank=["text", "doc_summary"]),
)
```

Notes: `columns_to_rerank` **order matters**, only the first **2000 chars** per column are considered,
and `num_results` is the final count returned (not the rerank candidate pool size).

## Deep Dive / Advanced Topics

### Why L2 distance, and the cosine trap

AI Search uses **L2 (Euclidean) distance** as its metric with an **HNSW** graph index. This surprises
people who expect cosine similarity. For **normalized** embeddings (unit length), L2 ranking and
cosine ranking produce the *same order*, so it doesn't matter. But `databricks-gte-large-en` produces
**non-normalized** embeddings, while `databricks-bge-large-en` is **normalized**. If you need
cosine-equivalent ranking with a non-normalized model, normalize the vectors before indexing.

### Schema is fixed at creation — plan for rebuilds

An index's schema (columns, embedding config) is **fixed when you create it**. You cannot add a column
or change the embedding model in place. The zero-downtime pattern to change schema is:

1. Update the source table.
2. Create a **new** index with the new schema.
3. Switch application traffic (e.g., flip which index the agent's retriever tool points to).
4. Delete the old index.

This is analogous to the alias-based model promotion from the previous chapter — you build the new
thing beside the old thing and cut over.

### Managed-embeddings writeback table, and the reserved `_id`

- If you enable "save computed embeddings" on a managed Delta Sync index, Databricks creates a UC
  table `<index_name>_writeback_table`. Don't drop or edit it; it's deleted with the index.
- The column name **`_id` is reserved** — if your source has an `_id` column, rename it before
  creating the index or creation fails.

### Integrating with LangGraph (connecting the dots to Section 03)

The index you build here is consumed by the agent's retriever tool from Section 03. For a **managed**
Delta Sync index, the tool needs only the index name; for **self-managed or Direct Access** indexes,
you must also specify the text column and embedding model:

```python
from databricks_langchain import VectorSearchRetrieverTool, DatabricksEmbeddings

# Managed embeddings — minimal config
tool = VectorSearchRetrieverTool(
    index_name="prod.agents.databricks_docs_index",
    num_results=5,
    query_type="HYBRID",
    tool_name="databricks_docs_retriever",
    tool_description="Retrieves relevant Databricks documentation.",
)

# Self-managed / Direct Access — must supply text_column and embedding
tool = VectorSearchRetrieverTool(
    index_name="prod.agents.realtime_index",
    columns=["id", "text"],
    text_column="text",
    embedding=DatabricksEmbeddings(endpoint="databricks-bge-large-en"),
    tool_name="realtime_retriever",
    tool_description="Retrieves real-time indexed content.",
)
```

This is also exactly the index you declared in `resources=[DatabricksVectorSearchIndex(...)]` when
logging the agent in the previous chapter — closing the loop between index config and agent packaging.

## Worked Examples & Practice

### End-to-end: from Delta table to queryable index

You have a Delta table `prod.docs.chunks` of chunked documentation (`chunk_id`, `text`, `doc_uri`).
Stand up a production-ready managed index and query it.

```python
from databricks.vector_search.client import VectorSearchClient

client = VectorSearchClient()

# 0. Prereq: enable Change Data Feed on the source (run in SQL/Spark first)
#    ALTER TABLE prod.docs.chunks SET TBLPROPERTIES (delta.enableChangeDataFeed = true)

# 1. Create the endpoint (once)
client.create_endpoint(name="prod_agents_endpoint", endpoint_type="STANDARD")

# 2. Create a managed Delta Sync index, TRIGGERED (docs update nightly)
index = client.create_delta_sync_index(
    endpoint_name="prod_agents_endpoint",
    source_table_name="prod.docs.chunks",
    index_name="prod.agents.databricks_docs_index",
    pipeline_type="TRIGGERED",
    primary_key="chunk_id",
    embedding_source_column="text",
    embedding_model_endpoint_name="databricks-gte-large-en",
)

# 3. After the nightly ETL updates prod.docs.chunks, sync the index
index.sync()

# 4. Query with hybrid search + a metadata filter
results = index.similarity_search(
    query_text="How do I enable Change Data Feed?",
    columns=["chunk_id", "text", "doc_uri"],
    num_results=5,
    query_type="hybrid",
    filters={"doc_uri LIKE": "delta-lake"},
)
for row in results["result"]["data_array"]:
    print(row)
```

**Failure mode to observe:** skip step 0 (don't enable CDF), then create the index. On a standard
endpoint with a Delta Sync index, the sync fails because there is no change feed to read incremental
updates from — the index cannot track what changed in the source table. Enabling CDF and re-syncing
fixes it. This is the single most common "my index won't sync" support issue.

## Common Pitfalls & Misconceptions

- **Pitfall:** Forgetting to enable Change Data Feed on the source Delta table → **Why it happens:** the table works fine for normal queries without it → **Fix:** Delta Sync indexes on standard endpoints require CDF to detect incremental changes. Run `ALTER TABLE ... SET TBLPROPERTIES (delta.enableChangeDataFeed = true)` before creating the index.

- **Pitfall:** Choosing CONTINUOUS sync on a storage-optimized endpoint → **Why it happens:** CONTINUOUS sounds strictly better (fresher data) → **Fix:** storage-optimized endpoints support **TRIGGERED only**. CONTINUOUS is a standard-endpoint feature. Also, CONTINUOUS runs an always-on cluster and costs more — only use it when you genuinely need seconds-level freshness.

- **Pitfall:** Trying to create a Direct Access index in the UI → **Why it happens:** Delta Sync indexes *can* be created in the UI, so users assume all types can → **Fix:** Direct Access indexes are SDK/REST only. If you need UI creation, use a Delta Sync index instead.

- **Pitfall:** Using dictionary filters on a storage-optimized endpoint (or SQL-string filters on a standard one) → **Why it happens:** the two endpoint types use different filter syntaxes → **Fix:** standard = dictionary (`{"id <": 200}`); storage-optimized = SQL string (`"id < 200"`). Match the syntax to the endpoint type.

- **Pitfall:** Expecting cosine ranking from an ANN query with a non-normalized embedding model → **Why it happens:** most tutorials use cosine similarity → **Fix:** AI Search ranks by **L2 distance**. With normalized embeddings (`bge-large-en`) the order matches cosine; with non-normalized ones (`gte-large-en`) it may not. Normalize vectors if you specifically need cosine behavior.

## Key Definitions

| Term | Definition |
|---|---|
| Databricks AI Search | The current product name for what was formerly Databricks Vector Search; the API surface still uses `vector-search` identifiers |
| Endpoint | The compute that hosts and serves one or more AI Search indexes; created with `endpoint_type` STANDARD or STORAGE_OPTIMIZED |
| Index | The searchable structure (embedded vectors + metadata) built from a source, hosted on an endpoint |
| Delta Sync Index | An index that automatically and incrementally syncs with a source Delta table; supports managed or self-managed embeddings |
| Direct Vector Access Index | An index you write to directly via `upsert`/`delete` with no source Delta table; SDK/REST only, not creatable in the UI |
| TRIGGERED / CONTINUOUS | Delta Sync `pipeline_type`: TRIGGERED syncs on demand (cheaper); CONTINUOUS keeps the index fresh within seconds (always-on, costlier, standard endpoints only) |
| Change Data Feed (CDF) | A Delta table feature that records row-level changes; required on the source for Delta Sync on standard endpoints |
| RRF (Reciprocal Rank Fusion) | The algorithm hybrid search uses to merge ANN and full-text result rankings |
| Reranker | A cross-encoder applied after retrieval to reorder results for higher precision (~10% quality gain, ~1.5s latency) |

## Summary / Quick Recall

- **AI Search = former Vector Search**; API/SQL identifiers still say `vector-search` / `vector_search()`
- **Endpoint** = compute (hosts many indexes); **index** = the searchable structure. Don't conflate them
- Endpoint types: **STANDARD** (TRIGGERED + CONTINUOUS, dict filters) vs **STORAGE_OPTIMIZED** (TRIGGERED only, SQL-string filters, dim divisible by 16)
- Index types: **Delta Sync** (managed or self-managed embeddings, auto-syncs a Delta table) and **Direct Access** (you upsert vectors; SDK/REST only)
- **TRIGGERED** = on-demand, cheaper; **CONTINUOUS** = seconds-fresh, always-on, standard endpoints only. Both are incremental
- **Change Data Feed must be enabled** on the source table for Delta Sync (standard endpoints) — top cause of sync failures
- Query types: **ANN** (default, L2/HNSW), **full-text** (BM25), **hybrid** (ANN+full-text via RRF — recommended default)
- To query someone else's index: `USE CATALOG` + `USE SCHEMA` + `SELECT` on the index
- Three supported embedding models: `databricks-gte-large-en`, `databricks-bge-large-en`, `databricks-qwen3-embedding-0-6b`

## Self-Check Questions

1. Your Delta Sync index on a standard endpoint refuses to sync, even though the source table is populated and you have permissions. What is the most likely missing prerequisite?

<details>
<summary>Answer</summary>
Change Data Feed is not enabled on the source Delta table. Delta Sync on standard endpoints reads incremental changes from the CDF; without it there is no change stream to sync from. Fix with `ALTER TABLE <table> SET TBLPROPERTIES (delta.enableChangeDataFeed = true)`, then re-sync.
</details>

2. A team needs an index over a very large corpus, wants to minimize cost, and is fine with syncing once nightly after their ETL job. Which endpoint type and sync mode should they choose, and what constraint must they accept?

<details>
<summary>Answer</summary>
Use a **STORAGE_OPTIMIZED** endpoint with **TRIGGERED** sync — it's the lower-cost option for large datasets and nightly refresh fits triggered sync well. The constraints they accept: storage-optimized supports TRIGGERED only (no CONTINUOUS), uses SQL-string filter syntax rather than dictionaries, and self-managed embedding dimensions must be divisible by 16.
</details>

3. What is the practical difference between a managed-embeddings Delta Sync index and a self-managed one, and how does it change the `create_delta_sync_index` call?

<details>
<summary>Answer</summary>
With **managed** embeddings, Databricks generates the vectors — you pass `embedding_source_column` (the text column) and `embedding_model_endpoint_name` (the model). With **self-managed** embeddings, your source table already contains a precomputed vector column — you pass `embedding_vector_column` and `embedding_dimension` instead, and no model endpoint. Managed is simpler and most common; self-managed is for when you control embedding generation.
</details>

4. You need to reduce hallucinated / off-topic answers by improving retrieval precision without re-indexing. What query-time options do you have, and what's the cost?

<details>
<summary>Answer</summary>
Switch from ANN to **hybrid** search (ANN + full-text merged via RRF) to catch both semantic and keyword matches, and add a **reranker** (`DatabricksReranker`) to reorder the top results with a cross-encoder. Hybrid is essentially free; the reranker adds ~1.5s latency for roughly a 10% quality improvement. Neither requires rebuilding the index.
</details>

5. Why can't you just add a new metadata column to an existing AI Search index, and what's the recommended way to change an index's schema with no downtime?

<details>
<summary>Answer</summary>
An index's schema is fixed at creation — the embedding config and columns can't be altered in place. The zero-downtime pattern is to update the source table, create a **new** index with the new schema, switch application traffic to it (e.g., repoint the retriever tool), then delete the old index — the same build-beside-then-cut-over approach used for alias-based model promotion.
</details>

## Further Reading

- [Databricks — AI Search overview](https://docs.databricks.com/aws/en/ai-search/ai-search) — endpoints, index types, concepts (updated 2026-06-25)
- [Databricks — Create an AI Search index](https://docs.databricks.com/aws/en/ai-search/create-ai-search) — SDK signatures, sync modes, prerequisites (updated 2026-06-17)
- [Databricks — Query an AI Search index](https://docs.databricks.com/aws/en/ai-search/query-ai-search) — ANN/full-text/hybrid, filters, reranking (updated 2026-06-17)
- [Databricks — Unstructured retrieval agent tools](https://docs.databricks.com/aws/en/agents/agent-framework/unstructured-retrieval-tools) — `VectorSearchRetrieverTool` integration (updated 2026-06-30)
- [Databricks — Foundation Model APIs supported models](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/supported-models) — embedding model specs (updated 2026-07-02)
