# Delta Lake and Unity Catalog Writes for GenAI Applications — Interview Prep

**Section:** Ingestion and Chunking | **Role target:** GenAI Engineer, Data Engineer, Solutions Architect

## Core Conceptual Questions

These test whether you understand the fundamentals.

| Question | Key points to cover | Common weak-answer trap |
|---|---|---|
| What is the difference between a Unity Catalog Volume and a Delta Table in a RAG pipeline? | Volumes hold raw, unstructured files (PDFs, images). Delta Tables hold structured, tabular data (parsed text chunks, metadata, doc IDs). | Saying "Volumes are for big data and Delta is for small data" or failing to distinguish unstructured vs structured. |
| Why is Change Data Feed (CDF) critical for Databricks Vector Search? | CDF logs row-level inserts, updates, and deletes. Vector Search relies on these logs to incrementally update the vector index without recalculating the entire table. | Explaining what CDF is generally, but failing to connect it to the *incremental sync* requirement of Vector Search. |
| How do you convert LangChain Documents into a format Databricks can store efficiently? | Extract the `page_content` and `metadata` dictionary from the LangChain objects, map them to a list of dicts, and convert to a Spark DataFrame to write as Parquet/Delta. | Claiming you can just `spark.write(langchain_docs)`. |

## Applied / Scenario Questions

**Q:** Your team has an ingestion pipeline that runs nightly, parsing updated financial PDFs and overwriting a Delta table with the new text chunks. However, the downstream Databricks Vector Search index is failing to sync continuously. How do you diagnose and fix this?

**Strong answer framework:**
- **Diagnose:** The failure is likely caused by two things: using `overwrite` mode and missing Change Data Feed (CDF).
- **Fix 1:** Run `ALTER TABLE ... SET TBLPROPERTIES (delta.enableChangeDataFeed = true)` on the Delta Table. Vector Search strictly requires this for continuous sync.
- **Fix 2 (Tradeoff awareness):** Stop using `overwrite` mode for the nightly job. Overwriting drops the table history and forces massive, inefficient re-syncs. Instead, refactor the pipeline to use a `MERGE` (upsert) operation, so only the actual new or changed chunks are processed by the Vector Search index.

## System Design / Architecture Questions (if applicable)

**Q:** Design a data ingestion layer for a company that receives 50,000 mixed documents (PDFs, Word, TXT) daily that need to be searchable by a GenAI chatbot.

**Approach:**
1. **Clarify requirements:** Is real-time search required, or is batch processing acceptable? (Assume hourly batch is fine).
2. **Propose structure:** 
   - **Landing:** Auto Loader streams raw files into a Unity Catalog Volume.
   - **Processing:** A Spark job reads the Volume, uses a Python UDF (e.g., PyMuPDF or LangChain loaders) to extract text and chunk it.
   - **Storage:** The output is appended to a Delta Table with `delta.enableChangeDataFeed = true`.
   - **Serving:** Databricks Vector Search is configured for continuous sync pointing at that Delta Table.
3. **Justify choices and name tradeoffs explicitly:** Using Delta Lake as the intermediate layer adds a storage hop, but provides a durable source of truth. If the embedding model needs to be swapped later, we don't have to re-parse 50,000 documents a day from scratch; we just point a new vector index at the existing Delta table.

## Vocabulary That Signals Expertise

Use these terms naturally — don't force them:
- **Change Data Feed (CDF)** — Use when discussing how to keep vector indexes up to date efficiently without full rebuilds.
- **Incremental Sync** — Use when discussing the alternative to expensive full-table scans.
- **Structured vs Unstructured** — Use when justifying the boundary between UC Volumes and Delta Tables.

## Vocabulary That Signals Weakness

Avoid these — they signal outdated or shallow understanding:
- **"Saving chunks to JSON files"** — Red flag. Once text is chunked and has metadata, it is structured data and belongs in a Delta Table, not loose files.
- **"Writing directly to Vector Search"** — In Databricks, the best practice is to write to a Delta table and let the managed Vector Search sync it. Bypassing Delta removes governance, time-travel, and easy model swapping.

## STAR Answer Frame

**Situation:** Our GenAI application was serving stale data because the vector database took 6 hours to rebuild every night after the PDF parsing job finished.  
**Task:** I needed to reduce the data freshness latency from 24 hours to under 15 minutes.  
**Action:** I redesigned the ingestion pipeline. I routed the parsed chunks into a Delta Table and enabled `enableChangeDataFeed`. I then switched our vector database to Databricks Vector Search, configuring it to continuously sync from the CDF logs.  
**Result:** The vector index now updates incrementally within 5 minutes of a new PDF being processed, and our compute costs dropped by 40% because we eliminated the daily full-index rebuilds.

## Red Flags Interviewers Watch For

- Candidates who treat GenAI data engineering as entirely separate from traditional data engineering. A strong candidate knows that a text chunk is just a string column in a database, and the standard rules of ACID transactions, merges, and change data capture still apply.
- Candidates who don't know how to bridge the gap between Python-native GenAI libraries (LangChain/LlamaIndex) and scalable Spark DataFrames.