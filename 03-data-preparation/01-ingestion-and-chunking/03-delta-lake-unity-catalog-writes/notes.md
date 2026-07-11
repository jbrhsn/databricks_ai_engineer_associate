# Delta Lake and Unity Catalog Writes for GenAI Applications

**Section:** Ingestion and Chunking | **Module:** Data Preparation | **Est. time:** 1 hr | **Exam mapping:** Data Preparation (14%)

---

## TL;DR

In Databricks GenAI pipelines, raw unstructured files are read from Unity Catalog Volumes, chunked into smaller text segments, and written to Delta Tables in Unity Catalog. The single most important configuration step is enabling Change Data Feed (CDF) on these Delta Tables, as it is a strict requirement for Databricks Vector Search to continuously synchronize new chunks into the vector index. **The one thing to remember: Always set `delta.enableChangeDataFeed = true` when writing chunked text to a Delta Table destined for Vector Search.**

---

## ELI5 — Explain It Like I'm 5

Think of Unity Catalog Volumes as the mailroom where raw, unopened packages (like PDF documents) arrive. Delta Tables are the organized filing cabinets where each package has been opened, cut into neat, readable pages (text chunks), and filed with a specific reference number. Change Data Feed (CDF) is the automated alert system that tells the librarian (Vector Search) exactly which pages were added, removed, or updated today — without it, the librarian would have to manually re-read the entire filing cabinet every single time. A common misconception is that you can simply save text chunks as JSON files back into a Volume — in reality, once data is structured (has a chunk_id, text, and metadata), it must live in a Delta Table to gain the ACID guarantees and Change Data Feed that Vector Search depends on.

---

## Learning Objectives

By the end of this chapter you will be able to:
- [ ] Distinguish the roles of Unity Catalog Volumes and Delta Tables in a GenAI ingestion pipeline.
- [ ] Convert chunked application objects (e.g., LangChain Documents) into Spark DataFrames.
- [ ] Write chunked data to Unity Catalog Delta Tables using standard PySpark APIs.
- [ ] Configure Delta Table properties to enable continuous Databricks Vector Search synchronization.
- [ ] Diagnose sync failures caused by missing Change Data Feed configurations.

---

## Visual Overview

### The RAG Ingestion Pipeline Flow

```
Raw Documents                  Chunking Process                Delta Lake                       Vector Database
┌───────────────┐              ┌────────────────┐              ┌────────────────────────┐       ┌─────────────────┐
│  UC Volume    │ ──► Read ──► │  LangChain /   │ ──► Write ──►│  Delta Table           │ ───►  │ Databricks      │
│  (PDFs, TXT)  │              │  Spark UDF     │              │  (doc_id, chunk_text)  │ Sync  │ Vector Search   │
└───────────────┘              └────────────────┘              │  [CDF Enabled]         │       └─────────────────┘
                                                               └────────────────────────┘
```

### Change Data Feed Row-Level Change Tracking

```
WITHOUT CDF enabled:                      WITH CDF enabled (delta.enableChangeDataFeed = true):

  Delta Table write                           Delta Table write
  ┌───────────────────────────┐               ┌───────────────────────────┐
  │  _delta_log/              │               │  _delta_log/              │
  │  ├── 000001.json          │               │  ├── 000001.json          │
  │  └── part-0000.parquet    │               │  ├── part-0000.parquet    │
  └───────────────────────────┘               │  └── _change_data/        │
                                              │      ├── chunk_id = "1-a" │
  Vector Search sees only:                   │      │   _change_type:     │
  "table changed" → must                     │      │   insert            │
  full-scan to find new rows                 │      ├── chunk_id = "1-b"  │
                                              │      │   _change_type:     │
                                              │      │   update_postimage  │
                                              │      └── chunk_id = "1-c"  │
                                              │          _change_type:     │
                                              │          delete            │
                                              └───────────────────────────┘

  _change_type values: insert | update_preimage | update_postimage | delete
  Vector Search reads _change_data → updates only affected vectors
```

### Vector Search Sync Modes

```
Continuous Sync                                Triggered Sync
──────────────────────────────                 ──────────────────────────────
Delta Table write occurs                       Delta Table write occurs
        │                                              │
        ▼                                              ▼ (on schedule / manual)
CDF streaming reader watches                   CDF query API reads all rows
_change_data folder in real-time               changed since last watermark:
        │                                        spark.read.format("delta")
        ▼                                          .option("readChangeFeed","true")
Row-level change events emitted                        │
(insert/update/delete per chunk_id)                    ▼
        │                                      Batch of changed rows identified
        ▼                                              │
Vector index updated                                   ▼
within seconds of write                        Vector index updated on trigger
        │                                              │
        ▼                                              ▼
Near-real-time freshness                       Lower freshness; lower cost
Requires: CDF enabled                          Requires: CDF enabled
Best for: low-latency RAG                      Best for: overnight batch jobs
```

---

## Key Concepts

### Unity Catalog Volumes for Raw Data

**What is it?** Unity Catalog Volumes are a governance construct that provide secure access to non-tabular data (unstructured files) in cloud object storage.

**How does it work mechanistically?** A volume creates a logical path (`/Volumes/catalog/schema/volume_name/file.pdf`) mapped to a physical cloud URI (like an S3 bucket). It allows PySpark, Python's `os` module, and external libraries to read files securely using Databricks credentials without exposing cloud access keys. The volume object is stored as metadata in the Unity Catalog metastore; Databricks translates path reads into signed cloud-storage requests transparently.

**Where does it appear?** In the Data Explorer under a schema, or accessed programmatically via the path `/Volumes/<catalog>/<schema>/<volume>/`.

### Delta Tables for Chunked Data

**What is it?** A Delta Table is the foundational structured storage layer in Databricks, providing ACID transactions, scalable metadata handling, and time travel over Parquet files.

**How does it work mechanistically?** When unstructured text is chunked, it becomes highly structured (e.g., `chunk_id`, `parent_doc_id`, `text_content`). Delta Lake stores this tabular data as versioned Parquet files, writing a transaction log (`_delta_log`) that tracks exactly which files belong to which version of the table. Every write — append, overwrite, or merge — is recorded as an atomic JSON entry in the transaction log, guaranteeing consistency even if a job fails mid-write.

**Where does it appear?** Written via `df.write.format("delta").saveAsTable("catalog.schema.table")` and queried using standard SQL or PySpark.

### Change Data Feed (CDF)

**What is it?** CDF is a Delta Lake feature that records row-level changes (inserts, updates, deletes) made to a Delta Table.

**How does it work mechanistically?** When enabled, Delta Lake writes extra metadata files alongside the transaction log that explicitly detail which rows changed and how (e.g., `update_preimage`, `update_postimage`, `insert`, `delete`). These are stored in a `_change_data` subfolder. Databricks Vector Search reads these CDF files to incrementally update its vector index rather than recomputing the entire index from scratch, making syncs faster by orders of magnitude for large tables.

**Where does it appear?** It is enabled using the table property `delta.enableChangeDataFeed = true` during table creation or via an `ALTER TABLE` SQL command. CDF data is queryable via `spark.read.format("delta").option("readChangeFeed", "true").option("startingVersion", 0).table("catalog.schema.table")`.

### Databricks Vector Search Sync Modes

**What is it?** Two modes control how a Delta Table-backed Vector Search index stays current: Continuous Sync and Triggered Sync.

**How does it work mechanistically?** Continuous Sync uses a streaming reader that watches the `_change_data` folder of the source Delta Table in near-real-time. Each row-level CDF event (insert/update/delete, keyed on the primary key column) is translated into a corresponding vector operation, keeping the index updated within seconds. Triggered Sync instead uses the CDF query API — `spark.read.format("delta").option("readChangeFeed", "true")` — to batch-read all rows changed since the last recorded watermark, then applies those changes to the vector index in one pass. Both modes require CDF to be enabled on the source table; neither can function on a plain Delta Table without `delta.enableChangeDataFeed = true`.

**Where does it appear?** The sync mode is configured during Vector Search index creation in the Databricks UI (Catalog Explorer → Vector Search Indexes → Create) or via the Python SDK (`VectorSearchClient.create_delta_sync_index(pipeline_type="CONTINUOUS" | "TRIGGERED")`).

### Table Schema Design for Vector Search

**What is it?** The columns in the source Delta Table directly determine what Databricks Vector Search can index and filter on; schema design is therefore a prerequisite to index creation, not an afterthought.

**How does it work mechanistically?** Vector Search requires two mandatory columns: (1) a unique primary key column (non-nullable, any data type) used to track row identity across CDF events — inserts, updates, and deletes are matched by this key; and (2) the text (or embedding) source column whose content gets embedded into the vector space. Any additional columns become filterable metadata fields in the index, enabling pre-filter queries (e.g., filter by `doc_id` before vector similarity search). The primary key must be deterministic and stable — regenerating it from different data will create phantom duplicates in the index.

**Where does it appear?** The schema is defined when creating the Delta Table in PySpark (`StructType`) or SQL (`CREATE TABLE`). During Vector Search index creation in the UI or SDK, you specify the primary key column name and the embedding source column name explicitly.

---

## Key Parameters / Configuration Knobs

| Parameter | What it controls | Decision rule |
|---|---|---|
| `delta.enableChangeDataFeed` | Whether Delta logs row-level change events (inserts, updates, deletes) in a `_change_data` subfolder. | Set to `true` on any table that will be the source for a Databricks Vector Search index; leave as `false` to save storage on tables not feeding downstream sync consumers. |
| `write mode (append vs. overwrite vs. merge)` | Controls whether new data is added to the table or the entire table is replaced. | Use `append` for incremental daily runs; use `MERGE` (upsert) when chunks may be re-processed and duplicates must be avoided; never use `overwrite` in production as it triggers a full Vector Search re-index. |
| `delta.autoOptimize.optimizeWrite` | Auto-compacts small files during write by coalescing output files to the target file size. | Set to `true` on tables with frequent small appends (daily ingestion jobs) to prevent the "small file problem" degrading query performance. |
| `delta.autoOptimize.autoCompact` | Triggers background compaction after write operations complete. | Set to `true` alongside `optimizeWrite` for large-scale ingestion tables where write throughput is high and you want Delta to manage file sizes automatically. |
| `primary key column` | The column used as the unique identifier for MERGE operations and Vector Search change tracking. | Design the schema with a deterministic `chunk_id` (e.g., hash of `doc_id + chunk_index`) so Vector Search can efficiently track changes per chunk and avoid duplicates on re-ingestion. |

---

## Worked Example: Requirement → Decision

**Given:** An enterprise pipeline receives 10,000 new PDFs daily. You need to parse these into chunks, store them, and ensure the GenAI application's vector search index stays up-to-date within 5 minutes of a new PDF arriving.

- **Step 1 — Identify the goal:** Store text chunks and keep a vector index continuously updated with near-real-time freshness.
- **Step 2 — Define inputs:** Raw PDFs arriving in a UC Volume; each PDF may be re-processed if its source changes.
- **Step 3 — Define outputs:** A structured Delta Table containing `chunk_id`, `doc_id`, and `text_content`, usable as a Continuous Sync source for a Vector Search index.
- **Step 4 — Apply constraints:** The vector index must update quickly (incrementally, within minutes), meaning it cannot afford to scan all historical chunks to find the new ones. Re-processed PDFs mean chunks may need updating, not just inserting, so duplicates must be handled.
- **Step 5 — Select the approach:** Write the chunks to a Unity Catalog Delta Table with `delta.enableChangeDataFeed = true`, use a deterministic `chunk_id` (hash of `doc_id + chunk_index`), and use Delta `MERGE` on `chunk_id` for upserts. Configure the Vector Search index as Continuous Sync. **Rationale:** CDF enables incremental sync (avoiding full re-index); MERGE prevents duplicates when PDFs are re-processed; Continuous Sync satisfies the 5-minute freshness requirement that Triggered Sync on a schedule could not guarantee.

---

## Implementation

```python
# Scenario: Writing LangChain Document chunks to a Delta Table ready for Vector Search.
# Constraint: Table must have CDF enabled before the Vector Search index is created.

from pyspark.sql.types import StructType, StructField, StringType
from langchain_core.documents import Document

# 1. Assume 'chunks' is a list of LangChain Document objects
chunks = [
    Document(page_content="Databricks is a unified data analytics platform.", metadata={"doc_id": "1", "chunk_id": "1-a"}),
    Document(page_content="Unity Catalog provides unified governance.", metadata={"doc_id": "1", "chunk_id": "1-b"})
]

# 2. Extract data into a list of dictionaries
chunk_data = [{"chunk_id": doc.metadata["chunk_id"],
               "doc_id": doc.metadata["doc_id"],
               "text_content": doc.page_content} for doc in chunks]

# 3. Define schema and create DataFrame
schema = StructType([
    StructField("chunk_id", StringType(), False),  # non-nullable primary key
    StructField("doc_id", StringType(), True),
    StructField("text_content", StringType(), True)
])
df = spark.createDataFrame(chunk_data, schema=schema)

# 4. Write to Delta Table with Change Data Feed enabled
(df.write
  .format("delta")
  .mode("append")
  .option("delta.enableChangeDataFeed", "true")
  .saveAsTable("main.genai_docs.chunked_text"))
```

```python
# Anti-pattern: Overwriting a table without setting CDF, then trying to create a
# continuous Vector Search index. This fails at index creation time and forces
# expensive re-ingestion if the table already has production data.

df.write.format("delta").mode("overwrite").saveAsTable("main.genai_docs.chunked_text")
# Vector Search creation will fail with:
# "Change Data Feed is not enabled on the source table."
# Additionally, overwrite drops all previous table history, making incremental
# sync impossible until a full re-index completes.

# Correct approach:
# If the table already exists but lacks CDF, enable it dynamically via SQL
# without re-ingesting any data:
spark.sql("""
  ALTER TABLE main.genai_docs.chunked_text
  SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")
```

```python
# Scenario: Upserting chunks to a Delta Table to handle re-processed documents
# without duplicates. (Correct mental model from Pitfall 3 — never blindly overwrite;
# always MERGE on chunk_id so re-processed chunks update existing rows rather than
# creating duplicates that corrupt the vector index.)

chunk_data_df.createOrReplaceTempView("new_chunks")
spark.sql("""
  MERGE INTO main.genai_docs.chunked_text AS target
  USING new_chunks AS source
  ON target.chunk_id = source.chunk_id
  WHEN MATCHED THEN UPDATE SET *
  WHEN NOT MATCHED THEN INSERT *
""")
```

---

## Common Pitfalls & Misconceptions

- **Forgetting Change Data Feed (CDF)** — Beginners assume Vector Search magically knows when a Delta Table updates, so they skip the table property. Databricks Vector Search strictly requires `delta.enableChangeDataFeed = true` to perform continuous incremental updates; without it, index creation throws an error and the only workaround is altering the table property and triggering a full re-index.

- **Writing chunks back to Volumes** — Beginners treat text chunks as "files" and try to save JSON or TXT chunks back to a UC Volume, following the same pattern used for raw source documents. Once data is chunked, it is highly structured (chunk_id, text, metadata) and must live in a Delta Table to gain SQL querying, ACID transactions, and the CDF mechanism that Vector Search depends on.

- **Using PySpark `overwrite` mode carelessly** — Beginners use `mode("overwrite")` to avoid handling duplicate logic, assuming it is a safe way to "refresh" the table. Overwriting completely drops the old table version and forces Vector Search to perform an expensive full re-index of every row. The correct mental model is to design a deterministic `chunk_id` and use Delta `MERGE` (upsert) so only changed or new chunks are processed.

---

## Key Definitions

| Term | Definition |
|---|---|
| **Change Data Feed (CDF)** | A Delta Lake feature that logs row-level inserts, updates, and deletes in a `_change_data` subfolder, providing the necessary metadata for downstream systems to perform incremental updates. |
| **Unity Catalog Volume** | A logical object representing a directory of non-tabular, unstructured files (like PDFs or images) in cloud storage, governed by Unity Catalog. |
| **Databricks Vector Search** | A fully managed vector database in Databricks that can automatically sync from a CDF-enabled Delta Table to keep its vector index incrementally up to date. |
| **Unity Catalog Three-Level Namespace** | The path format `catalog.schema.table` (or `/Volumes/catalog/schema/volume/path`) that uniquely identifies any data asset in Databricks, enabling cross-workspace governance. |
| **Delta MERGE (Upsert)** | A Delta Lake SQL operation that atomically updates existing rows and inserts new rows in a single pass, keyed on a matching condition — the correct pattern for handling re-processed chunks without duplicates. |

---

## Summary / Quick Recall

- Raw unstructured files (PDFs) belong in Unity Catalog Volumes; chunked text data belongs in Unity Catalog Delta Tables.
- Databricks Vector Search continuously syncs Delta Tables to a vector index; this requires `delta.enableChangeDataFeed = true` on the source table.
- CDF records `insert`, `update_preimage`, `update_postimage`, and `delete` events per row — Vector Search reads these to update only affected vectors.
- Continuous Sync uses a streaming CDF reader (near-real-time); Triggered Sync batch-reads changed rows since a watermark (scheduled/manual) — both require CDF.
- Schema design matters: Vector Search requires a non-nullable primary key column and an embedding source column; use a deterministic `chunk_id` (hash of `doc_id + chunk_index`).
- Use Delta `MERGE` on `chunk_id` for re-processed documents; never use `overwrite` in production as it forces a full re-index.
- LangChain `Document` objects must be extracted into dictionaries and converted to Spark DataFrames before writing to Delta.

---

## Self-Check Questions

1. What is the primary purpose of a Unity Catalog Volume in a RAG ingestion pipeline?
   - A. To store the final vector embeddings for similarity search.
   - B. To provide secure, governed access to raw, unstructured files like PDFs.
   - C. To log row-level changes for Delta Lake.
   - D. To host the Large Language Model endpoints.

   <details><summary>Answer</summary>

   **B is correct.** Volumes are designed for non-tabular, unstructured files (like raw PDFs); they provide governed access to cloud object storage without exposing credentials.
   A is wrong because embeddings belong in Vector Search or Delta Tables. C describes Change Data Feed, which is a Delta Lake feature. D describes Model Serving endpoints, which are an entirely separate Databricks service.

   </details>

2. You have a Delta Table containing 5 million text chunks. You want to configure a Databricks Vector Search index to update continuously whenever new chunks are appended. What table property is strictly required?
   - A. `delta.appendOnly = true`
   - B. `delta.minReaderVersion = 2`
   - C. `delta.enableChangeDataFeed = true`
   - D. `delta.autoOptimize.optimizeWrite = true`

   <details><summary>Answer</summary>

   **C is correct.** Databricks Vector Search requires Change Data Feed (CDF) to be enabled on the source Delta Table to perform continuous, incremental syncs — without it, the index creation call fails with an explicit error.
   A (`appendOnly`) prevents deletes and updates but does not enable change tracking. B sets the reader protocol version and is unrelated to Vector Search sync. D (`optimizeWrite`) controls file compaction during writes and has no effect on Vector Search synchronization.

   </details>

3. **Which TWO** statements accurately describe the data movement in a standard Databricks GenAI ingestion pipeline?
   - A. Chunks are extracted from Delta Tables and written to Unity Catalog Volumes as JSON.
   - B. Raw PDFs are read from Unity Catalog Volumes.
   - C. Chunked text is converted to a Spark DataFrame and saved as a Delta Table.
   - D. Vector Search indexes read directly from Unity Catalog Volumes to generate embeddings.
   - E. Unity Catalog Volumes require `enableChangeDataFeed` for continuous syncing.

   <details><summary>Answer</summary>

   **B and C are correct.** The standard flow is: Volume (raw PDFs) → processing/chunking → Delta Table (structured chunks) → Vector Search. B correctly identifies Volumes as the raw-data source. C correctly identifies that chunks must be converted to a Spark DataFrame before writing to Delta.
   A is backwards — structured chunks do not go back to Volumes. D is wrong because Vector Search syncs from Delta Tables via CDF, not from Volumes. E is wrong because Volumes are non-tabular storage objects; the CDF property applies only to Delta Tables.

   </details>

4. You try to create a Continuous Vector Search Index from a Delta Table, but receive an error stating: `Change Data Feed is not enabled on the source table.` The table already has 100 GB of chunked data. What is the most efficient way to resolve this?
   - A. Delete the table, recreate it with the property, and rerun the 100 GB ingestion.
   - B. Run an `ALTER TABLE ... SET TBLPROPERTIES (delta.enableChangeDataFeed = true)` command.
   - C. Copy the table into a Unity Catalog Volume.
   - D. Change the Vector Search sync type to Triggered; Triggered syncs do not require CDF.

   <details><summary>Answer</summary>

   **B is correct.** `ALTER TABLE ... SET TBLPROPERTIES` enables CDF on an existing table dynamically without touching the data, saving the compute cost of reprocessing 100 GB.
   A works but is highly inefficient — it discards an existing table and re-incurs full ingestion cost. C is architecturally incorrect; moving structured data to a Volume removes ACID guarantees and breaks Vector Search entirely. D is wrong because both Continuous and Triggered sync modes in Databricks Vector Search require CDF to perform efficient incremental updates.

   </details>

5. Your ingestion pipeline re-processes source PDFs nightly because source documents can be updated. After several nights you notice the Vector Search index contains duplicate chunks for re-processed documents. Which write strategy eliminates duplicates without forcing a full re-index?
   - A. Switch the write mode to `overwrite` so the table is replaced nightly with a fresh set of chunks.
   - B. Use Delta `MERGE` on `chunk_id` so matching chunks are updated and new chunks are inserted.
   - C. Disable CDF and use a Triggered sync, which scans the full table each night.
   - D. Add a `DISTINCT` clause to the DataFrame before writing to remove duplicates at write time.

   <details><summary>Answer</summary>

   **B is correct.** Delta `MERGE` on `chunk_id` is the correct pattern: it updates existing rows when a chunk is re-processed and inserts rows for genuinely new chunks, producing zero duplicates while preserving CDF-enabled incremental syncing.
   A (`overwrite`) eliminates duplicates but drops all prior table versions and forces Vector Search to perform a full re-index every night, which is expensive and slow. C would disable the very mechanism that makes sync efficient and would also still not eliminate duplicates — a full-table scan finds changes but does not deduplicate. D (`DISTINCT`) only deduplicates within the current batch being written, not against rows already in the table from previous nights.

   </details>

---

## Further Reading

- [Unity Catalog Volumes](https://docs.databricks.com/en/volumes/index.html) — *verified 2026-07-11* — Official documentation on managing unstructured data in Unity Catalog.
- [Use Delta Lake Change Data Feed](https://docs.databricks.com/en/delta/delta-change-data-feed.html) — *verified 2026-07-11* — How to enable and query CDF, including `_change_type` column values.
- [Databricks Vector Search](https://docs.databricks.com/en/generative-ai/vector-search.html) — *verified 2026-07-11* — Delta Table requirements, sync modes, and index creation for vector indexing.
- [Delta Lake MERGE INTO](https://docs.databricks.com/en/delta/merge.html) — *verified 2026-07-11* — Reference for upsert patterns including matched/not-matched clauses.
