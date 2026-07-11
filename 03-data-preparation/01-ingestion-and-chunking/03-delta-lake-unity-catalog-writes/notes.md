# Delta Lake and Unity Catalog Writes for GenAI Applications

**Section:** Ingestion and Chunking | **Module:** Data Preparation | **Est. time:** 1 hr | **Exam mapping:** Data Preparation (14%)

---

## TL;DR

In Databricks GenAI pipelines, raw unstructured files are read from Unity Catalog Volumes, chunked into smaller text segments, and written to Delta Tables in Unity Catalog. The single most important configuration step is enabling Change Data Feed (CDF) on these Delta Tables, as it is a strict requirement for Databricks Vector Search to continuously synchronize new chunks into the vector index. **The one thing to remember: Always set `delta.enableChangeDataFeed = true` when writing chunked text to a Delta Table destined for Vector Search.**

---

## ELI5 — Explain It Like I'm 5

Think of Unity Catalog Volumes as the mailroom where raw, unopened packages (like PDF documents) arrive. Delta Tables are the organized filing cabinets where each package has been opened, cut into neat, readable pages (text chunks), and filed with a specific reference number. Change Data Feed (CDF) is the automated alert system that tells the librarian (Vector Search) exactly which pages were added, removed, or updated today. Without this alert system, the librarian would have to manually re-read the entire filing cabinet every time a single page changes to keep the index up to date.

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

```text
Raw Documents                  Chunking Process                Delta Lake                       Vector Database
┌───────────────┐              ┌────────────────┐              ┌────────────────────────┐       ┌─────────────────┐
│  UC Volume    │ ──► Read ──► │  LangChain /   │ ──► Write ──►│  Delta Table           │ ───►  │ Databricks      │
│  (PDFs, TXT)  │              │  Spark UDF     │              │  (doc_id, chunk_text)  │ Sync  │ Vector Search   │
└───────────────┘              └────────────────┘              │  [CDF Enabled]         │       └─────────────────┘
                                                               └────────────────────────┘
```

---

## Key Concepts

### Unity Catalog Volumes for Raw Data

1. **What is it?** Unity Catalog Volumes are a governance construct that provide secure access to non-tabular data (unstructured files) in cloud object storage.
2. **How does it work mechanistically?** A volume creates a logical path (`/Volumes/catalog/schema/volume_name/file.pdf`) mapped to a physical cloud URI (like an S3 bucket). It allows PySpark, Python's `os` module, and external libraries to read files securely using Databricks credentials without exposing cloud access keys.
3. **Where does it appear?** In the Data Explorer under a schema, or accessed programmatically via the path `/Volumes/<catalog>/<schema>/<volume>/`.

### Delta Tables for Chunked Data

1. **What is it?** A Delta Table is the foundational structured storage layer in Databricks, providing ACID transactions, scalable metadata handling, and time travel over Parquet files.
2. **How does it work mechanistically?** When unstructured text is chunked, it becomes highly structured (e.g., `chunk_id`, `parent_doc_id`, `text_content`). Delta Lake stores this tabular data as versioned Parquet files, writing a transaction log (`_delta_log`) that tracks exactly which files belong to which version of the table.
3. **Where does it appear?** Written via `df.write.format("delta").saveAsTable("catalog.schema.table")` and queried using standard SQL or PySpark.

### Change Data Feed (CDF)

1. **What is it?** CDF is a Delta Lake feature that records row-level changes (inserts, updates, deletes) made to a Delta Table.
2. **How does it work mechanistically?** When enabled, Delta Lake writes extra metadata files alongside the transaction log that explicitly detail which rows changed and how (e.g., `update_preimage`, `update_postimage`, `insert`, `delete`). Databricks Vector Search reads these CDF files to incrementally update its vector index rather than recomputing the entire index from scratch.
3. **Where does it appear?** It is enabled using the table property `delta.enableChangeDataFeed = true` during table creation or via an `ALTER TABLE` SQL command.

### Key Parameters / Configuration Knobs

| Parameter | What it controls | Decision rule |
|---|---|---|
| `delta.enableChangeDataFeed` | Whether Delta logs row-level change events. | Set to `true` if this table will be the source for a Databricks Vector Search index; otherwise, leave as `false` to save storage. |

### Worked Example: Requirement → Decision

**Given:** An enterprise pipeline receives 10,000 new PDFs daily. You need to parse these into chunks, store them, and ensure the GenAI application's vector search index stays up-to-date within 5 minutes of a new PDF arriving.
- **Step 1 — Identify the goal:** Store text chunks and keep a vector index continuously updated.
- **Step 2 — Define inputs:** Raw PDFs arriving in a UC Volume.
- **Step 3 — Define outputs:** A syncable structured storage format containing `chunk_id` and `text`.
- **Step 4 — Apply constraints:** The vector index must update quickly (incrementally), meaning it cannot afford to scan all historical chunks to find the new ones.
- **Step 5 — Select the approach:** Write the chunks to a Unity Catalog Delta Table with `delta.enableChangeDataFeed = true`. **Rationale:** Without CDF, Databricks Vector Search cannot perform continuous incremental syncs, forcing expensive and slow full-table recalculations.

---

## Implementation

```python
# Scenario: Writing LangChain Document chunks to a Delta Table ready for Vector Search

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
    StructField("chunk_id", StringType(), False),
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
# Anti-pattern: Overwriting a table without setting CDF, then trying to create a continuous Vector Search index

df.write.format("delta").mode("overwrite").saveAsTable("main.genai_docs.chunked_text")
# Vector Search creation will fail with an error stating continuous sync requires Change Data Feed.

# Correct approach:
# If the table already exists but lacks CDF, enable it via SQL before configuring Vector Search:
spark.sql("ALTER TABLE main.genai_docs.chunked_text SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
```

---

## Common Pitfalls & Misconceptions

- **Forgetting Change Data Feed (CDF)** — Beginners assume Vector Search magically knows when a Delta Table updates. Databricks Vector Search strictly requires CDF to be set to `true` to perform continuous incremental updates; otherwise, it throws an error during index creation.
- **Writing chunks back to Volumes** — Beginners treat chunks as "files" and try to save JSON/TXT chunks back to a UC Volume. Once data is chunked, it is highly structured (ID, text, metadata) and should always be written to a Delta Table to leverage SQL querying, ACID transactions, and Vector Search syncing.
- **Using PySpark `overwrite` mode carelessly** — Beginners use `mode("overwrite")` to handle duplicates. Overwriting completely drops the old table version. While Vector Search can handle overwrites if CDF is enabled, it's highly inefficient. The correct mental model is to use Delta `MERGE` (upsert) to handle duplicates incrementally.

---

## Key Definitions

| Term | Definition |
|---|---|
| **Change Data Feed (CDF)** | A Delta Lake feature that logs row-level inserts, updates, and deletes, providing the necessary metadata for downstream systems to perform incremental updates. |
| **Unity Catalog Volume** | A logical object representing a directory of non-tabular, unstructured files (like PDFs or images) in cloud storage, governed by Unity Catalog. |
| **Databricks Vector Search** | A fully managed vector database in Databricks that can automatically sync from a Delta Table to keep its vector index up to date. |

---

## Summary / Quick Recall

- Raw unstructured files (PDFs) belong in Unity Catalog Volumes.
- Chunked text data belongs in Unity Catalog Delta Tables.
- Databricks Vector Search continuously syncs Delta Tables to a vector index.
- Continuous sync **requires** `delta.enableChangeDataFeed = true` on the source table.
- LangChain `Document` objects must be extracted into dictionaries and converted to Spark DataFrames before writing to Delta.

---

## Self-Check Questions

1. What is the primary purpose of a Unity Catalog Volume in a RAG ingestion pipeline?
   - A. To store the final vector embeddings for similarity search.
   - B. To provide secure, governed access to raw, unstructured files like PDFs.
   - C. To log row-level changes for Delta Lake.
   - D. To host the Large Language Model endpoints.

   <details><summary>Answer</summary>
   
   **B is correct.** Volumes are designed for non-tabular, unstructured files (like raw PDFs). 
   A is wrong because embeddings belong in Vector Search or Delta Tables. C describes Change Data Feed. D describes Model Serving.
   
   </details>

2. You have a Delta Table containing 5 million text chunks. You want to configure a Databricks Vector Search index to update continuously whenever new chunks are appended. What table property is strictly required?
   - A. `delta.appendOnly = true`
   - B. `delta.minReaderVersion = 2`
   - C. `delta.enableChangeDataFeed = true`
   - D. `delta.autoOptimize.optimizeWrite = true`

   <details><summary>Answer</summary>
   
   **C is correct.** Databricks Vector Search requires Change Data Feed (CDF) to be enabled on the source Delta Table to perform continuous, incremental syncs. 
   A, B, and D are valid Delta properties but have nothing to do with enabling row-level change tracking for vector indexing.
   
   </details>

3. **Which TWO** statements accurately describe the data movement in a standard Databricks GenAI ingestion pipeline?
   - A. Chunks are extracted from Delta Tables and written to Unity Catalog Volumes as JSON.
   - B. Raw PDFs are read from Unity Catalog Volumes.
   - C. Chunked text is converted to a Spark DataFrame and saved as a Delta Table.
   - D. Vector Search indexes read directly from Unity Catalog Volumes to generate embeddings.
   - E. Unity Catalog Volumes require `enableChangeDataFeed` for continuous syncing.

   <details><summary>Answer</summary>
   
   **B and C are correct.** The standard flow is Volume (raw) → Processing → Delta Table (chunked). 
   A is backwards (we don't write structured chunks back to Volumes). D is wrong because Vector Search syncs from Delta Tables, not Volumes. E is wrong because Volumes do not have the CDF property; only Delta Tables do.
   
   </details>

4. You try to create a Continuous Vector Search Index from a Delta Table, but receive an error stating: `Change Data Feed is not enabled on the source table.` The table already has 100GB of chunked data. What is the most efficient way to resolve this?
   - A. Delete the table, recreate it with the property, and rerun the 100GB ingestion.
   - B. Run an `ALTER TABLE ... SET TBLPROPERTIES (delta.enableChangeDataFeed = true)` command.
   - C. Copy the table into a Unity Catalog Volume.
   - D. Change the Vector Search sync type to Triggered; Triggered syncs do not require CDF.

   <details><summary>Answer</summary>
   
   **B is correct.** You can enable CDF on an existing table dynamically via SQL `ALTER TABLE`, saving the compute cost of reprocessing 100GB of data. 
   A works but is highly inefficient. C is architecturally incorrect. D is wrong because even Triggered syncs in Databricks Vector Search require CDF to do incremental updates efficiently.
   
   </details>

5. Your ingestion pipeline processes thousands of text files and outputs LangChain `Document` objects. What is the necessary transformation step before these can be saved to a Delta Table?
   - A. Serialize the Documents directly using `df.write.format("langchain").save()`.
   - B. Save the Documents to a local temporary JSON file and load them using `spark.read.json()`.
   - C. Extract the `page_content` and `metadata` into Python dictionaries, then construct a Spark DataFrame using `spark.createDataFrame()`.
   - D. Loop through the Documents and run an `INSERT INTO` SQL command for each one.

   <details><summary>Answer</summary>
   
   **C is correct.** LangChain Document objects must be mapped into standard data structures (like dictionaries) to be converted into a native PySpark DataFrame, which can then be written to Delta. 
   A is wrong because there is no "langchain" write format in Spark. B is highly inefficient and breaks distributed processing paradigms. D is an anti-pattern that will cause massive performance bottlenecks due to thousands of tiny single-row transactions.
   
   </details>

---

## Further Reading

- [Unity Catalog Volumes](https://docs.databricks.com/en/volumes/index.html) — *verified 2026-07-11* — Official documentation on managing unstructured data.
- [Use Delta Lake Change Data Feed](https://docs.databricks.com/en/delta/delta-change-data-feed.html) — *verified 2026-07-11* — How to enable and query CDF.
- [Databricks Vector Search](https://docs.databricks.com/en/generative-ai/vector-search.html) — *verified 2026-07-11* — Delta table requirements for vector indexing.