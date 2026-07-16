# Databricks Generative AI Engineer Associate — Practice Questions (D2 Data Preparation: Delta Live Tables for RAG)

*Focus: DLT pipelines, data quality expectations, incremental ingestion, Vector Search integration. Difficulty: medium-to-hard. 6 questions, mixed single/multi-select.*

---

## Q1: DLT vs. Databricks Workflows for RAG Ingestion

Your team ingests documents into chunks every hour, embeds chunks, and syncs to a Vector Search index. The pipeline has clear dependencies: `raw_docs → chunks → embeddings → vector_sync`. Should you use Delta Live Tables (DLT) or Databricks Workflows?

   A. Use Databricks Workflows—it is the only framework that supports scheduled execution and allows manual retries of failed stages.
   
   B. Use DLT—it automatically infers dependencies from table references, provides built-in data quality monitoring, and integrates tightly with Vector Search Delta Sync.
   
   C. Use either one—both provide identical orchestration, scheduling, and lineage tracking capabilities.
   
   D. Use DLT only if all transformations are written in SQL; Python transformations require Databricks Workflows.

**A: B**. DLT is the right choice for declarative, dependency-driven pipelines. DLT automatically analyzes your source code and builds a directed acyclic graph (DAG) of dependencies based on which tables reference which; it then orchestrates execution in the correct order without explicit job definition. It provides built-in `@dp.expect()` decorators for data quality checks (e.g., chunk length validation) and tight integration with Vector Search Delta Sync indexing. Workflows (option A) are lower-level orchestration and require explicit job definitions for each stage, making them better suited for heterogeneous workloads or complex retry logic, not declarative ETL. Option C is wrong—Workflows and DLT have very different models. Option D is false—DLT supports both Python (via `@dp.table`, `@dp.expect_or_drop()`) and SQL (`CONSTRAINT ... EXPECT ...`).

---

## Q2: Data Quality Expectations in DLT Pipelines

**Which TWO of the following are valid expectation types or actions in a Delta Live Tables pipeline?**

   A. `@dp.expect_or_fail()` — rejects invalid rows and stops the entire pipeline on violation.
   
   B. `@dp.expect_warn()` — logs a warning but always drops invalid rows before writing.
   
   C. `@dp.expect_or_drop()` — drops invalid rows and allows the pipeline to continue, logging the drop count.
   
   D. `@dp.expect()` — the default action; invalid rows are written to the target table and metrics are logged.
   
   E. `@dp.expect_or_quarantine()` — routes invalid rows to a separate quarantine table for manual review.

**A: A and C**. **A qualifies** because `@dp.expect_or_fail()` (or `EXPECT ... ON VIOLATION FAIL UPDATE` in SQL) is a documented DLT expectation action that atomically rolls back the transaction if any record violates the constraint, stopping downstream flows. This is appropriate for critical data (e.g., chunk embedding must succeed or the pipeline fails). **C qualifies** because `@dp.expect_or_drop()` (or `EXPECT ... ON VIOLATION DROP ROW` in SQL) is the documented action that filters out invalid records before writing—essential for cleaning chunks before Vector Search ingestion without halting the pipeline. **B is wrong** because there is no `@dp.expect_warn()` decorator; the warn/retain action uses plain `@dp.expect()`. **D is incomplete** for this question (it is a valid action but only one of five options is allowed per multi-select). **E is wrong** because DLT does not have a built-in `@dp.expect_or_quarantine()` decorator; quarantine logic must be implemented manually with separate tables and business logic, not as a standard expectation action.

---

## Q3: Incremental Ingestion of Documents in DLT

Your DLT pipeline chunks documents. After the first run, 1,000 documents are chunked and stored in a `chunks` table. The next day, 50 new documents arrive in the source `raw_docs` table. How should you write the chunking stage to process only the new 50 documents on the next run (not rechunk all 1,050)?

   A. Use a standard materialized view: `@dp.table def chunks(): return spark.read.table('raw_docs').transform(chunk_fn)` — DLT automatically handles incremental updates.
   
   B. Use `@dlt.incremental()` or `dlt.read_stream()` to read only new rows since the last successful pipeline run, avoiding full reprocessing.
   
   C. Query the pipeline event log manually to find the last offset and filter `raw_docs` by that offset in your transformation.
   
   D. Re-run the full pipeline each time—DLT does not support incremental ingestion and reprocessing all rows is the standard pattern.

**A: B**. The DLT Python API provides the `@dlt.incremental()` decorator (or `dlt.read_stream()` to read a streaming table) that automatically reads only rows added since the last successful pipeline update. Internally, DLT tracks the processed state and filters the source accordingly. For a source table like `raw_docs`, you would write `@dlt.table @dlt.incremental() def chunks(): return dlt.read('raw_docs').filter(...).transform(chunk_fn)` or use a streaming table as the source. This ensures only the 50 new documents are chunked on the second run, dramatically improving efficiency. Option A is wrong—a standard materialized view will reprocess the entire source on each run unless you explicitly use incremental semantics. Option C is overly complex and error-prone; DLT abstracts this for you. Option D is false; DLT's incremental support is a core feature for RAG pipelines where documents arrive continuously.

---

## Q4: Python vs. SQL for Chunking in DLT

Your chunking stage uses LangChain's `RecursiveCharacterTextSplitter` to split documents. The splitting logic cannot be expressed in SQL. Should this stage be written as a SQL or Python DLT table?

   A. Write it in SQL using a DLT SQL view; Databricks Spark functions like `split()` and `explode()` can replicate LangChain's behavior.
   
   B. Write it in Python using `@dp.table`; Python UDFs or libraries like LangChain can be called within the transformation, but expectations must still use SQL syntax.
   
   C. Write it in SQL with a call to an external REST API; DLT supports HTTP requests within SQL queries.
   
   D. Use Databricks Workflows instead; DLT does not support custom Python libraries or external dependencies.

**A: B**. DLT Python tables (decorated with `@dp.table`) can execute arbitrary Python code, including UDFs, imports of LangChain or other libraries, and complex business logic. Within a Python DLT table, you call external libraries directly: `@dp.table def chunks(): from langchain.text_splitter import RecursiveCharacterTextSplitter; splitter = RecursiveCharacterTextSplitter(...); return spark.read.table('raw_docs').mapInPandas(lambda pdf: split_documents(pdf, splitter), schema).to_df()`. Expectations defined on the resulting table can use SQL syntax (e.g., `@dp.expect('chunk_length', 'length(text) BETWEEN 100 AND 2000')`), but the transformation itself is pure Python. Option A is wrong—SQL's native string functions cannot replicate LangChain's semantic-aware splitting. Option C is wrong; DLT does not directly support HTTP calls within SQL (and it would be inefficient). Option D is false; DLT supports Python transformations with external libraries just fine.

---

## Q5: Observability and Debugging in DLT Pipelines

**Which TWO observability features in DLT help you debug where a pipeline failure occurred (e.g., embedding model error in the embeddings stage vs. a chunking error)?**

   A. The DLT pipeline **lineage graph** in the UI shows which datasets feed which; if `embeddings` depends on `chunks`, and `embeddings` fails, you know `chunks` succeeded.
   
   B. The **Data Quality tab** in the pipeline UI displays expectation metrics (rows passed/failed) for each dataset, identifying where validation failed.
   
   C. The **event log** (queryable table or viewable in the Observability tab) records flow-level details: dataset name, operation (compute/stream), duration, exception messages, and row counts.
   
   D. The **vectorSearch.sync()** API call always logs the embedding model name and error code to Databricks workspace logs for manual review.
   
   E. Databricks automatically rolls back the entire pipeline and resets all tables to their previous state on any error, allowing you to rerun without debugging.

**A: A and C**. **A qualifies** because DLT's lineage graph (visible in the Pipeline Editor or Observability view) shows dependencies. If you see that `embeddings` failed but `chunks` is green (completed successfully), you know the error is downstream in the embedding stage, not in chunking. **C qualifies** because the DLT event log (queryable via a Delta table at `<pipeline-root>/_event_log/`) records flow execution details including dataset name, status, duration, exception messages, and row counts. You can query it to find which specific flow failed and see the root cause. **B is incorrect** for the primary failure scenario (though data quality tab does help identify constraint violations, not general errors). **D is incorrect** because Vector Search sync is not a built-in DLT feature and does not automatically log to workspace logs in the way described. **E is wrong**; DLT does not auto-rollback and reset—you must manually investigate and fix the pipeline code.

---

## Q6: Vector Search Integration with DLT Pipeline Output

Your DLT pipeline outputs a `chunks` table with columns: `id` (string), `text` (string), `embedding` (array<float>). You want to create a Vector Search Delta Sync index on this table. What is the MINIMUM configuration required in the Vector Search index definition?

   A. The index must specify the source Delta table path, the embedding column name (`embedding`), the text/source column name (`text`), and a primary key column (`id`); it then auto-syncs when the source table updates.
   
   B. The index requires a REST API endpoint URL pointing to a running embedding model; Vector Search cannot sync from DLT output directly.
   
   C. The index must be manually refreshed using the `refresh()` API; Delta Sync does not exist for DLT tables.
   
   D. The index requires both pre-computed embeddings in the `chunks` table AND a separate embedding model endpoint for live query-time re-embedding of search vectors.

**A: A**. Vector Search's Delta Sync index is designed exactly for this use case. The minimum configuration is: (1) source Delta table URI (e.g., `catalog.schema.chunks`), (2) the embedding column name (`embedding`), which must be `array<float>` or `array<double>`; (3) optionally, a text/source column (`text`) for retrieval and display; and (4) a primary key or unique identifier column (`id`). Once these are set, Vector Search handles sync automatically—when the `chunks` table is updated (e.g., new documents chunked by the DLT pipeline), the index is refreshed. Option B is wrong; Delta Sync does not require an external embedding model endpoint because embeddings are pre-computed and stored in the table. Option C is false; Delta Sync automatically refreshes. Option D is incorrect; pre-computed embeddings in the table are sufficient for similarity search; live re-embedding is not required for sync indexing.

---

## Summary of Question Types

- **Single-select:** Q1, Q3, Q4, Q6 (4 questions)
- **Multi-select:** Q2, Q5 (2 questions)
- **Cognitive coverage:** Q1 (recall/framework choice), Q2 (application of decorators), Q3 (incremental design pattern), Q4 (language selection), Q5 (observability/debugging), Q6 (integration)

## Key Assumptions Made

1. **DLT is the modern Databricks branding** for declarative pipelines (formerly "Delta Live Tables"; docs now refer to "Lakeflow pipelines" in recent 2026 releases). The exam and documentation use "DLT" interchangeably.

2. **Incremental reading is core to DLT:** The `@dlt.incremental()` decorator (or `dlt.read_stream()`) is the standard way to avoid full reprocessing. Some sources refer to this as "reading only rows modified since the last successful update."

3. **Expectations language is documented:** `@dp.expect()` (retain), `@dp.expect_or_drop()` (drop), `@dp.expect_or_fail()` (fail pipeline). No built-in quarantine action exists; that requires custom logic.

4. **Python UDFs and external libraries are fully supported** within `@dp.table` transformations, including libraries like LangChain for text splitting.

5. **Vector Search Delta Sync is the standard integration:** The index definition requires source table, embedding column, and (optionally) text/metadata columns. Pre-computed embeddings are stored in the Delta table.

6. **Observability features include lineage graph and event log**, both queryable for debugging pipeline failures at the stage level.

---

## References Verified

- Databricks Lakeflow Pipelines Concepts (Jul 10, 2026): https://docs.databricks.com/aws/en/ldp/concepts/pipelines.html — *verified 2026-07-16*
- Manage Data Quality with Pipeline Expectations (Jul 14, 2026): https://docs.databricks.com/aws/en/ldp/expectations.html — *verified 2026-07-16*
- Databricks AI Search / Vector Search (Jun 25, 2026): https://docs.databricks.com/aws/en/ai-search/ai-search.html — *verified 2026-07-16*
