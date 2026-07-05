# Delta Lake and Unity Catalog for GenAI Data

**Section:** Data Preparation | **Module:** Document Processing | **Est. time:** 4.5 hrs
**Exam mapping:** Databricks GenAI Engineer Associate — Data Preparation domain (~20%); Delta Lake, Unity Catalog Volumes, and AI Search are directly tested; Auto Loader and medallion architecture appear in pipeline design questions

## Learning Objectives

By the end of this chapter, you will be able to:

- Explain Delta Lake's key properties (ACID transactions, schema enforcement, time travel) and why they matter for GenAI data pipelines
- Design a medallion architecture (bronze/silver/gold) for an unstructured GenAI data pipeline
- Use Auto Loader to incrementally ingest new documents into a Delta table as they arrive in cloud storage
- Create managed and external Unity Catalog Volumes and explain the differences
- Write to and read from Unity Catalog Volumes from a Databricks notebook
- Create a Databricks AI Search index backed by a Delta table and configure it for automatic sync
- Explain how hybrid keyword-similarity search and Reciprocal Rank Fusion (RRF) work in AI Search
- Query an AI Search index programmatically with optional metadata filters

## Core Concepts

### Delta Lake — The Storage Backbone of GenAI Pipelines

**Delta Lake is the default table format on Databricks.** Every `spark.write.saveAsTable()` call,
every `CREATE TABLE` SQL statement, produces a Delta table unless you explicitly specify another
format. You do not need to add `format("delta")` — it is assumed.

Delta Lake extends Parquet with:

#### ACID Transactions

Delta Lake writes are atomic — either the entire write succeeds or nothing is committed. This
prevents the partial-write problem that plagues raw Parquet data lakes: a failed job cannot
leave the table in a partially-written state.

For GenAI pipelines, this matters for:
- **Re-indexing runs:** A pipeline that refreshes chunk embeddings with a new model can be
  interrupted mid-run without corrupting the table
- **Concurrent access:** Multiple jobs can read a table while another is writing — readers see
  a consistent snapshot

#### Schema Enforcement

Delta Lake validates that data written to a table matches the table's schema. Columns with the
wrong type, missing required columns, or extra unexpected columns trigger a write failure with
a clear error message.

```python
# Schema enforcement in action:
# If your pipeline adds a new column "embedding_model_version" without schema evolution,
# the write fails rather than silently corrupting the table.
try:
    df_new.write.mode("append").saveAsTable("main.rag.document_chunks")
except AnalysisException as e:
    print(f"Schema mismatch: {e}")
    # Fix: use .option("mergeSchema", "true") for intentional schema evolution
```

#### Time Travel (Table History)

Every write to a Delta table creates a new version. You can query any previous version:

```python
# Read the table as it was 3 versions ago
df_v3 = spark.read.format("delta").option("versionAsOf", 3).table("main.rag.document_chunks")

# Read as of a specific timestamp
df_yesterday = spark.read.format("delta") \
    .option("timestampAsOf", "2026-07-04 00:00:00") \
    .table("main.rag.document_chunks")
```

For GenAI pipelines, time travel enables:
- **Rollback after a bad re-embedding run:** If a new embedding model degrades retrieval
  quality, revert to the previous version's embeddings with a single SQL statement:
  `RESTORE TABLE main.rag.document_chunks TO VERSION AS OF 5`
- **Debugging:** Understand exactly what the index contained at the time a bad response
  was generated

#### Auto Loader — Incremental Document Ingestion

Auto Loader (`cloudFiles` format) monitors a cloud storage path (S3, ADLS, GCS) and
automatically ingests new files as they arrive. It is the Databricks-recommended approach for
production document pipelines.

```python
# Auto Loader: stream new PDFs from a Volume into a Delta staging table
(spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "binaryFile")   # ingest raw file bytes
    .option("cloudFiles.schemaLocation", "/tmp/autoloader_schema")
    .load("/Volumes/main/rag_demo/raw_docs/")     # UC Volume path
    .writeStream
    .format("delta")
    .option("checkpointLocation", "/tmp/autoloader_checkpoint")
    .trigger(availableNow=True)                   # process all pending files, then stop
    .table("main.rag_demo.raw_documents_bronze")
)
```

Key Auto Loader behaviors:
- **Incremental:** Only processes new files; never re-processes files already ingested
- **Fault-tolerant:** Checkpoints track which files have been processed; safe to restart after failure
- **Schema inference:** Can infer schema from a sample of incoming files
- `trigger(availableNow=True)` — process all pending files then stop; use for batch-style
  scheduled runs. Use `trigger(processingTime="10 minutes")` for continuous streaming.

---

### Medallion Architecture for GenAI Data Pipelines

The medallion architecture organizes data into three layers of increasing quality and specificity.
Databricks encourages its use for all production pipelines, including GenAI.

```
BRONZE (Raw)                SILVER (Processed)            GOLD (Feature-Ready)
────────────────────────    ────────────────────────────   ────────────────────────────
UC Volume:                  Delta table:                   Delta table:
  raw PDF/HTML/text files     parsed text + doc metadata     chunks + embeddings + metadata
  (unchanged originals)       deduped and filtered           ready for AI Search indexing
                              one row per source document    one row per chunk

Table: raw_documents_bronze   Table: documents_silver        Table: document_chunks_gold
```

**Why three layers?**
- **Bronze** preserves originals — you can always reparse with a better library
- **Silver** gives you clean, document-level text for analysis and audit
- **Gold** is the operational layer — directly backs the AI Search index

**Delta table schema for each layer:**

```sql
-- Bronze (Auto Loader output)
CREATE TABLE main.rag_demo.raw_documents_bronze (
    path        STRING,   -- full Volume path to the raw file
    content     BINARY,   -- raw file bytes (for re-parsing)
    length      BIGINT,
    modificationTime TIMESTAMP,
    _ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
) USING DELTA;

-- Silver (after parsing)
CREATE TABLE main.rag_demo.documents_silver (
    doc_id      STRING,   -- UUID
    source_path STRING,
    title       STRING,
    author      STRING,
    created_date DATE,
    parsed_text STRING,   -- clean extracted text
    num_tokens  INT,
    _parsed_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
) USING DELTA;

-- Gold (after chunking + embedding)
CREATE TABLE main.rag_demo.document_chunks_gold (
    chunk_id          STRING,   -- UUID
    doc_id            STRING,   -- FK to silver
    chunk_index       INT,
    chunk_text        STRING,
    token_count       INT,
    embedding         ARRAY<FLOAT>,  -- vector for AI Search
    source_path       STRING,
    title             STRING,
    section_heading   STRING,   -- structural metadata
    doc_version       STRING,
    _embedded_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
) USING DELTA;
```

---

### Unity Catalog Volumes — Governed File Storage

Volumes are Unity Catalog objects for governing non-tabular data (files). They sit at the third
level of the UC namespace: `catalog.schema.volume`.

**Path format:**
```
/Volumes/<catalog>/<schema>/<volume>/<path>/<filename>
```

**Two types (from Databricks docs, updated 2026-06-25):**

| Feature | Managed volumes | External volumes |
|---------|----------------|------------------|
| Storage location | UC-managed cloud storage | Existing cloud path you specify |
| Data lifecycle | UC controls; 7-day retention on delete | Data stays in cloud storage when dropped |
| Access control | All access through UC | UC governs; external tools can bypass via direct URI |
| Typical use case | Databricks-only workloads | Mixed Databricks + external system access |

**Creating and using volumes:**
```python
# SQL — create a managed volume
spark.sql("""
    CREATE VOLUME IF NOT EXISTS main.rag_demo.raw_docs
    COMMENT 'Raw PDF and HTML documents for RAG pipeline'
""")

# Write a file to a volume from Python
with open("/Volumes/main/rag_demo/raw_docs/test.txt", "w") as f:
    f.write("This is a test document.")

# Read it back
with open("/Volumes/main/rag_demo/raw_docs/test.txt", "r") as f:
    content = f.read()

# List volume contents
dbutils.fs.ls("/Volumes/main/rag_demo/raw_docs/")
```

**Compute requirements (from Databricks docs 2026-06-25):**
- Volumes require **Databricks Runtime 13.3 LTS or above**
- Must use a Unity Catalog-enabled cluster
- Not supported on legacy (pre-UC) clusters

**Key limitation:** You cannot register files in volumes as Delta tables. Volumes are for
path-based file access. Use Delta tables for tabular/structured data.

---

### Databricks AI Search (Formerly Vector Search)

**Databricks AI Search** (product name as of 2026) is the managed vector index built into
Databricks. It was formerly called "Databricks Vector Search." The exam may use either name —
know both.

#### How It Works

AI Search creates a vector index from a Delta table. The index includes:
- Embedded chunk vectors
- Chunk text
- Metadata columns for filtering

Queries to the index return the most similar vectors using the **HNSW algorithm** (Hierarchical
Navigable Small World) for approximate nearest-neighbor (ANN) search with **L2 distance**.

**Cosine similarity note:** AI Search uses L2 distance internally. If you want cosine
similarity ranking, normalize your embedding vectors before inserting them — when vectors are
L2-normalized, L2 distance ranking and cosine similarity ranking produce identical results.

#### Hybrid Keyword-Similarity Search

AI Search supports hybrid search: **vector similarity + keyword (BM25) search**, combined with
**Reciprocal Rank Fusion (RRF)**.

- **Vector search** finds chunks semantically similar to the query (even with different words)
- **Keyword search** (BM25) finds chunks containing exact query keywords — critical for unique
  identifiers like SKUs, product codes, or proper nouns not well-represented in training data
- **RRF** merges the two ranked lists into a single ranked list; no manual weight tuning required

**When to use hybrid search:** When your corpus contains unique identifiers, product codes,
or domain-specific terms that pure semantic search handles poorly. For general prose corpora,
pure semantic search is often sufficient.

#### Creating an AI Search Index

```python
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient(
    workspace_url=f"https://{spark.conf.get('spark.databricks.workspaceUrl')}",
    personal_access_token=dbutils.secrets.get(scope="genai-keys", key="databricks-token")
)

# Step 1: Create a Vector Search endpoint (one endpoint can serve multiple indexes)
vsc.create_endpoint(
    name="rag_demo_endpoint",
    endpoint_type="STANDARD"
)

# Step 2: Create an index backed by the gold Delta table
vsc.create_delta_sync_index(
    endpoint_name="rag_demo_endpoint",
    source_table_name="main.rag_demo.document_chunks_gold",
    index_name="main.rag_demo.document_chunks_gold_index",
    pipeline_type="TRIGGERED",              # TRIGGERED = manual sync; CONTINUOUS = auto-sync
    primary_key="chunk_id",
    embedding_source_column="chunk_text",   # AI Search handles embedding internally
    embedding_model_endpoint_name="databricks-bge-large-en"
    # OR: use embedding_vector_column="embedding" if you pre-computed embeddings
)

# Step 3: Trigger a sync (for TRIGGERED pipelines)
vsc.get_index("rag_demo_endpoint", "main.rag_demo.document_chunks_gold_index").sync()
```

**Two embedding approaches:**
1. **Managed embeddings** (`embedding_source_column`): AI Search embeds chunks automatically
   using a specified endpoint. Simpler, but the embedding model must be available as a
   Databricks serving endpoint.
2. **Pre-computed embeddings** (`embedding_vector_column`): You embed chunks in your pipeline
   and store vectors in the Delta table. AI Search indexes the pre-computed vectors. More
   control; required when using a custom or externally-hosted embedding model.

#### Querying the Index

```python
index = vsc.get_index("rag_demo_endpoint", "main.rag_demo.document_chunks_gold_index")

# Pure similarity search
results = index.similarity_search(
    query_text="What is the token limit for bge-large-en?",
    columns=["chunk_text", "source_path", "section_heading"],
    num_results=5
)

# Hybrid search (vector + keyword)
results = index.similarity_search(
    query_text="What is the token limit for bge-large-en?",
    query_type="HYBRID",    # enables RRF fusion
    columns=["chunk_text", "source_path", "section_heading"],
    num_results=5
)

# With metadata filtering (pre-filter before vector search)
results = index.similarity_search(
    query_text="Installation steps",
    filters={"section_heading": "Installation Guide"},
    columns=["chunk_text", "source_path"],
    num_results=5
)

# Display results
for result in results["result"]["data_array"]:
    print(f"Score: {result[-1]:.4f} | {result[0][:100]}...")
```

**`TRIGGERED` vs `CONTINUOUS` pipeline type:**

| Type | When index updates | Use case |
|------|-------------------|---------|
| `TRIGGERED` | Only when you call `.sync()` | Batch pipelines; nightly refresh |
| `CONTINUOUS` | Automatically when source Delta table changes | Near-real-time index freshness |

For most RAG applications with daily document ingestion cycles, `TRIGGERED` is sufficient and
more cost-effective than `CONTINUOUS`.

---

## Deep Dive / Advanced Topics

### Schema Evolution in Delta Lake

Delta Lake is strict about schema by default. When your pipeline adds a new metadata column
(e.g., you decide to start tracking `language` for multilingual documents), you have two options:

**Option 1 — Manual schema evolution (recommended for intentional changes):**
```python
# Add the new column explicitly
spark.sql("ALTER TABLE main.rag_demo.documents_silver ADD COLUMN language STRING")

# Then write with the new schema — append works normally
df_new_docs.write.mode("append").saveAsTable("main.rag_demo.documents_silver")
```

**Option 2 — Auto merge schema (for exploratory work):**
```python
df_new_docs.write \
    .mode("append") \
    .option("mergeSchema", "true") \
    .saveAsTable("main.rag_demo.documents_silver")
```

`mergeSchema=True` adds new columns automatically. Use with caution in production — it can
silently alter a table schema if a column is misspelled or has an unexpected type.

### Change Data Feed (CDF) for Incremental Re-Indexing

Delta Lake's Change Data Feed tracks row-level changes (inserts, updates, deletes) between
table versions. For AI Search with a `TRIGGERED` pipeline, you can use CDF to re-embed only
the changed chunks rather than the entire table:

```python
# Enable CDF on the gold table
spark.sql("""
    ALTER TABLE main.rag_demo.document_chunks_gold
    SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

# Read only changes since the last sync
changes = (spark.readStream
    .format("delta")
    .option("readChangeFeed", "true")
    .option("startingVersion", last_sync_version)
    .table("main.rag_demo.document_chunks_gold")
    .filter(F.col("_change_type").isin(["insert", "update_postimage"]))
)
```

This reduces re-indexing cost from O(total_corpus) to O(changed_documents) for large corpora.

### Lakeflow Spark Declarative Pipelines (LFP) — Overview

Lakeflow Spark Declarative Pipelines (formerly Delta Live Tables / DLT) is Databricks'
orchestration layer for building data pipelines. Rather than writing Spark jobs that call each
other explicitly, you declare pipeline tables and Databricks manages execution order, incremental
processing, and error handling.

**For GenAI pipelines, LFP provides:**
- Automatic incremental processing (only reprocesses changed upstream data)
- Built-in error handling and data quality expectations
- Visual pipeline lineage in the UI
- Managed Auto Loader integration

```python
# LFP pipeline definition (simplified)
import dlt

@dlt.table(comment="Raw document bytes ingested from Volume")
def raw_documents_bronze():
    return (spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "binaryFile")
        .load("/Volumes/main/rag_demo/raw_docs/"))

@dlt.table(comment="Parsed and cleaned text, one row per document")
def documents_silver():
    return (dlt.read_stream("raw_documents_bronze")
        .select(parse_document_udf("content").alias("parsed_text"), "*"))
```

For this exam, you need to know LFP exists and what it provides — deep implementation knowledge
is not required for the Associate exam.

## Worked Examples & Practice

### Complete Pipeline: Bronze → Silver → Gold with AI Search

This example builds a complete three-layer pipeline and creates an AI Search index.

```python
# ── SETUP ────────────────────────────────────────────────────────────────────
CATALOG = "main"
SCHEMA  = "rag_demo"
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# ── BRONZE: Ingest raw files with Auto Loader ─────────────────────────────────
(spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "binaryFile")
    .option("cloudFiles.schemaLocation", f"/Volumes/{CATALOG}/{SCHEMA}/checkpoints/schema")
    .load(f"/Volumes/{CATALOG}/{SCHEMA}/raw_docs/")
    .writeStream
    .format("delta")
    .option("checkpointLocation", f"/Volumes/{CATALOG}/{SCHEMA}/checkpoints/bronze")
    .trigger(availableNow=True)
    .table(f"{CATALOG}.{SCHEMA}.raw_documents_bronze")
).awaitTermination()

# ── SILVER: Parse bronze → clean text ─────────────────────────────────────────
from pyspark.sql.functions import udf, col, current_timestamp, expr
from pyspark.sql.types import StringType
import hashlib

@udf(returnType=StringType())
def parse_pdf_bytes(content_bytes):
    """Parse PDF bytes to text (simplified — real impl uses unstructured)."""
    try:
        from pypdf import PdfReader
        import io
        reader = PdfReader(io.BytesIO(bytes(content_bytes)))
        return "\n\n".join(p.extract_text() or "" for p in reader.pages)
    except Exception as e:
        return None  # flag for error tracking

bronze_df = spark.table(f"{CATALOG}.{SCHEMA}.raw_documents_bronze")
silver_df = (bronze_df
    .withColumn("parsed_text", parse_pdf_bytes(col("content")))
    .withColumn("doc_id", expr("sha2(path, 256)"))  # stable UUID from path
    .filter(col("parsed_text").isNotNull())          # drop parse failures
    .select("doc_id", "path", "parsed_text", "length", "modificationTime")
    .withColumn("_parsed_at", current_timestamp())
)

(silver_df.write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(f"{CATALOG}.{SCHEMA}.documents_silver")
)

# ── GOLD: Chunk + embed ────────────────────────────────────────────────────────
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pyspark.sql.types import ArrayType, FloatType, StructType, StructField, StringType, IntegerType
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=60,
    length_function=lambda t: len(enc.encode(t)),
    separators=["\n\n", "\n", ". ", " "]
)

def process_document(doc_id, path, parsed_text):
    chunks = splitter.split_text(parsed_text)
    return [
        {
            "chunk_id": hashlib.sha256(f"{doc_id}_{i}".encode()).hexdigest(),
            "doc_id": doc_id,
            "chunk_index": i,
            "chunk_text": chunk,
            "token_count": len(enc.encode(chunk)),
            "source_path": path,
        }
        for i, chunk in enumerate(chunks)
    ]

silver_rows = silver_df.collect()
all_chunks = []
for row in silver_rows:
    all_chunks.extend(process_document(row.doc_id, row.path, row.parsed_text))

# Embed chunks in batches
from openai import OpenAI
client = OpenAI(
    api_key=dbutils.secrets.get(scope="genai-keys", key="databricks-token"),
    base_url=f"https://{spark.conf.get('spark.databricks.workspaceUrl')}/serving-endpoints"
)

BATCH_SIZE = 50
for i in range(0, len(all_chunks), BATCH_SIZE):
    batch = all_chunks[i:i+BATCH_SIZE]
    response = client.embeddings.create(
        model="databricks-bge-large-en",
        input=[c["chunk_text"] for c in batch]
    )
    for chunk, item in zip(batch, response.data):
        chunk["embedding"] = item.embedding

# Write gold table
gold_df = spark.createDataFrame(all_chunks)
(gold_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOG}.{SCHEMA}.document_chunks_gold")
)

# Enable CDF for incremental re-indexing
spark.sql(f"""
    ALTER TABLE {CATALOG}.{SCHEMA}.document_chunks_gold
    SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

# ── AI SEARCH INDEX ────────────────────────────────────────────────────────────
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

# Create endpoint (once — reuse for all indexes)
try:
    vsc.create_endpoint(name="rag_demo_endpoint", endpoint_type="STANDARD")
    print("Endpoint created")
except Exception:
    print("Endpoint already exists")

# Create index (using pre-computed embeddings from gold table)
vsc.create_delta_sync_index(
    endpoint_name="rag_demo_endpoint",
    source_table_name=f"{CATALOG}.{SCHEMA}.document_chunks_gold",
    index_name=f"{CATALOG}.{SCHEMA}.document_chunks_gold_index",
    pipeline_type="TRIGGERED",
    primary_key="chunk_id",
    embedding_vector_column="embedding",   # use pre-computed embeddings
    embedding_dimension=1024               # bge-large-en output dimension
)

# Sync the index
index = vsc.get_index("rag_demo_endpoint", f"{CATALOG}.{SCHEMA}.document_chunks_gold_index")
index.sync()
print("Index synced — pipeline complete.")
```

**Failure mode to observe:** Drop the `primary_key` argument from `create_delta_sync_index`.
The call fails with a clear error: `primary_key is required`. This is an easy mistake when
your gold table does not have an explicit primary key column — design your schema with a
`chunk_id` UUID column from the start.

## Common Pitfalls & Misconceptions

- **Pitfall:** Assuming Delta Lake must be specified explicitly with `format("delta")` → **Why it happens:** Other data platforms require explicit format specification → **Fix:** On Databricks, Delta is the default. `spark.write.saveAsTable("mydb.mytable")` already produces a Delta table. Adding `format("delta")` is harmless but redundant.

- **Pitfall:** Using `TRIGGERED` AI Search pipelines without calling `.sync()` after writing new chunks → **Why it happens:** Index creation feels like a one-time operation; updates are easy to forget → **Fix:** Add an explicit `index.sync()` call at the end of every pipeline run that modifies the gold table; otherwise the index serves stale data

- **Pitfall:** Creating a new AI Search endpoint for every index → **Why it happens:** The API shows both endpoint and index creation; learners create one per application → **Fix:** One endpoint can serve multiple indexes. Create one endpoint per environment (dev/prod), then create multiple indexes on it. Endpoints have significant startup costs.

- **Pitfall:** Storing raw document files directly in Delta tables as binary columns → **Why it happens:** Delta is the storage tool for everything → **Fix:** Raw files belong in Unity Catalog Volumes (path-based access); Delta tables hold the tabular data derived from those files (parsed text, chunks, embeddings). The Volume → Delta pipeline is the correct pattern.

- **Pitfall:** Forgetting that Unity Catalog Volumes require DBR 13.3 LTS or above → **Why it happens:** Learners test on older clusters or don't check runtime version → **Fix:** Always use DBR 13.3 LTS ML or higher for GenAI work; Volumes silently write to ephemeral disk (not persistent cloud storage) on older runtimes

- **Pitfall:** Conflating AI Search with a general-purpose vector database like Pinecone or Weaviate → **Why it happens:** They serve similar functions → **Fix:** AI Search is Databricks-native and deeply integrated with Delta Lake, Unity Catalog governance, and Databricks Model Serving. For Databricks RAG applications, it is the preferred and exam-expected solution.

## Key Definitions

| Term | Definition |
|---|---|
| Delta Lake | An open-source storage layer that adds ACID transactions, schema enforcement, and time travel to Parquet data files; the default table format on Databricks |
| ACID transactions | Atomicity, Consistency, Isolation, Durability — guarantees that database operations are reliable; Delta Lake's atomic writes prevent partial-write corruption |
| Time travel | The ability to query a Delta table at any previous version using `versionAsOf` or `timestampAsOf`; enables rollback after bad pipeline runs |
| Auto Loader | A Databricks feature (`cloudFiles` format) that incrementally ingests new files from cloud storage as they arrive, with fault-tolerant checkpointing |
| Medallion architecture | A data organization pattern with three layers: bronze (raw), silver (cleaned/parsed), gold (feature-ready for consumption) |
| Unity Catalog Volumes | UC objects for governing non-tabular files; available as managed (UC-controlled storage) or external (existing cloud storage path); accessed via `/Volumes/<catalog>/<schema>/<volume>/` paths |
| Managed volume | A UC Volume where Databricks manages the underlying cloud storage location; simplest option for Databricks-only workloads |
| External volume | A UC Volume that adds UC governance to an existing cloud storage path; use when data must also be accessed by external systems |
| Databricks AI Search | The Databricks-managed vector search service (formerly Vector Search); creates HNSW-based approximate nearest-neighbor indexes backed by Delta tables |
| HNSW | Hierarchical Navigable Small World — the graph-based ANN algorithm used by AI Search for fast approximate nearest-neighbor queries |
| Hybrid search | A search mode combining vector similarity (semantic) with keyword BM25 ranking, merged via Reciprocal Rank Fusion |
| RRF (Reciprocal Rank Fusion) | A rank-fusion algorithm that merges multiple ranked lists (e.g., vector search + keyword search) into a single list without requiring manual weight tuning |
| TRIGGERED pipeline | An AI Search index that only syncs when `.sync()` is explicitly called; lower cost than CONTINUOUS; suitable for batch pipeline runs |
| CONTINUOUS pipeline | An AI Search index that automatically syncs when the source Delta table changes; suitable for near-real-time freshness requirements |
| Change Data Feed (CDF) | A Delta Lake feature that tracks row-level changes (insert/update/delete) between versions; enables incremental re-indexing |
| Lakeflow Spark Declarative Pipelines (LFP) | Databricks' managed ETL orchestration framework (formerly Delta Live Tables); supports declarative pipeline definitions with automatic incremental processing |

## Summary / Quick Recall

- Delta Lake is the **default format** on Databricks — no `format("delta")` needed
- Key Delta properties for GenAI pipelines: **ACID writes** (no partial corruption), **schema enforcement** (catch pipeline bugs early), **time travel** (rollback bad re-embedding runs)
- **Medallion architecture:** bronze (raw files in Volume) → silver (parsed text in Delta) → gold (chunks + embeddings in Delta)
- **Auto Loader** (`cloudFiles`) = incremental, fault-tolerant file ingestion; use `trigger(availableNow=True)` for scheduled batch runs
- **UC Volumes:** managed (simplest) vs. external (existing storage); path = `/Volumes/<catalog>/<schema>/<volume>/`; requires DBR 13.3 LTS+
- **AI Search** = managed HNSW vector index backed by Delta table; formerly called Vector Search
- **Hybrid search** = vector similarity + BM25 keyword search, merged by RRF — use when corpus has unique identifiers
- **TRIGGERED** vs **CONTINUOUS**: TRIGGERED syncs on-demand (batch); CONTINUOUS syncs automatically (streaming)
- One AI Search **endpoint** per environment; many **indexes** per endpoint
- **Change Data Feed** enables incremental re-indexing (only re-embed changed rows)

## Self-Check Questions

1. A colleague says "I need to specify `format("delta")` to make sure my table is stored as Delta Lake." Is this correct? Why or why not?

<details>
<summary>Answer</summary>
Not necessary on Databricks. Delta Lake is the default format for all tables. `spark.write.saveAsTable("mydb.mytable")` and `CREATE TABLE mydb.mytable AS SELECT ...` both produce Delta tables without specifying `format("delta")`. Adding it is harmless but redundant. This is a key Databricks distinction — traditional Spark environments default to Parquet, but Databricks defaults to Delta.
</details>

2. Your nightly pipeline adds new document chunks to the gold Delta table. Users report that new documents are not showing up in RAG answers even though the pipeline ran successfully. What is the most likely cause?

<details>
<summary>Answer</summary>
The AI Search index was not synced after the pipeline wrote to the gold table. For a `TRIGGERED` pipeline, the index only updates when `.sync()` is explicitly called. Adding `index.sync()` at the end of the pipeline job resolves the issue. Alternatively, switch to a `CONTINUOUS` pipeline type for automatic near-real-time syncing (at higher cost).
</details>

3. You have a 20 TB document corpus shared between Databricks pipelines and an external data science team's Python scripts that access S3 directly. Should you use a managed or external Unity Catalog Volume? Why?

<details>
<summary>Answer</summary>
Use an **external volume**. Managed volumes store data in UC-controlled storage that is only accessible through Databricks. An external volume registers governance (UC access controls, lineage, audit logging) over an existing cloud storage path (your S3 bucket) without requiring a data copy. The external data science team can continue accessing S3 directly using their existing credentials, while your Databricks pipelines access the same data through the UC-governed path. For mixed-system access, external volumes are always the correct choice.
</details>

4. Explain Reciprocal Rank Fusion (RRF) in the context of AI Search hybrid search. When would you choose hybrid search over pure vector search?

<details>
<summary>Answer</summary>
RRF is a rank-fusion algorithm that merges two independently-ranked result lists (vector similarity ranking + BM25 keyword ranking) into a single list. Each document receives a score from both searches; RRF combines the scores using a formula that rewards documents ranked highly in either list without requiring manually-tuned weights. Choose hybrid search when your corpus contains unique identifiers (product SKUs, error codes, version numbers, proper nouns) that are not well-represented in the embedding model's training data — pure vector search may miss an exact match for "Error E4029" because the query and document representations diverge, while BM25 will find it by exact keyword match.
</details>

5. You run a pipeline that re-embeds all 500,000 chunks in your corpus with a new embedding model. After reviewing the new model's retrieval quality, you decide it is worse than the previous model. How do you roll back?

<details>
<summary>Answer</summary>
Use Delta Lake time travel to restore the gold table to the previous version: `RESTORE TABLE main.rag_demo.document_chunks_gold TO VERSION AS OF <previous_version_number>`. Then trigger a re-sync of the AI Search index: `index.sync()`. The index will update to reflect the restored table contents. To find the correct version number, use `DESCRIBE HISTORY main.rag_demo.document_chunks_gold` and identify the version before your re-embedding run.
</details>

## Further Reading

- [Databricks — What is Delta Lake?](https://docs.databricks.com/en/delta/index.html) — official Delta Lake documentation, updated 2026-06-17
- [Databricks AI Search documentation](https://docs.databricks.com/en/ai-search/ai-search.html) — official AI Search reference including HNSW, hybrid search, and RRF details, updated 2026-06-25
- [Databricks — Unity Catalog Volumes](https://docs.databricks.com/en/connect/unity-catalog/volumes.html) — managed vs. external volumes, path conventions, compute requirements, updated 2026-06-25
- [Databricks — What is Auto Loader?](https://docs.databricks.com/en/ingestion/cloud-object-storage/auto-loader/index.html) — incremental cloud file ingestion reference
- [Databricks — Medallion architecture](https://docs.databricks.com/en/lakehouse/medallion.html) — official description of the bronze/silver/gold pattern
