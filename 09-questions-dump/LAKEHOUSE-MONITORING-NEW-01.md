# Databricks Generative AI Engineer Associate — Practice Questions (D6 Evaluation & Monitoring: Lakehouse Monitoring)

*Focus: Data quality monitoring, schema validation, freshness, alert configuration. Difficulty: medium-to-hard. 8 questions, mixed single/multi-select.*

---

## Question 1: Detecting Data Quality Drift in RAG Source Tables

**Q1: A team maintains a Delta table of 100K documents for RAG retrieval. After a recent data refresh, your monitoring dashboard reports that 2% of document chunks now contain NULL embedding vectors where previously <0.1% were null. The embedding service had a temporary outage during the refresh. Which Databricks monitoring feature BEST detects and alerts on this kind of data quality regression?**

   A. MLflow Model Registry version tracking to compare model input distributions across deployments.
   B. Lakehouse Monitoring's data quality metrics, specifically null-rate monitors on the embedding column.
   C. Unity Catalog column-level statistics generated during table optimization.
   D. Delta Live Tables expectation validation configured in the ingestion pipeline.

A: **B**. Lakehouse Monitoring is purpose-built to track statistical changes in Delta tables, including null rates, value distributions, and schema drift. A null-rate monitor on the embedding column would flag the jump from <0.1% to 2% nulls and trigger an alert based on a configured threshold. Option A tracks model versions but not raw data quality. Option C provides static statistics but not drift detection with alerts. Option D validates data at ingestion time but does not continuously monitor for post-ingestion anomalies. Lakehouse Monitoring sits in the evaluation layer (D6 domain) to catch such regressions.

---

## Question 2: Schema Drift Detection in Vector Search Index Tables

**Q2: Which TWO aspects of a Vector Search index Delta table feeding a RAG application should be monitored for schema drift using Lakehouse Monitoring?**

   A. The embedding dimension count (e.g., 384 vs 768 dimensions) and the data type of the embedding column (e.g., ARRAY<FLOAT> vs ARRAY<DOUBLE>).
   B. The number of rows ingested per day and the query latency of the Vector Search endpoint.
   C. The presence of optional metadata columns (e.g., `source_url`, `author`) and null rates in those columns.
   D. The storage size of the Delta table and the number of partition files after OPTIMIZE.
   E. The refresh frequency of the Vector Search index and the number of concurrent read queries.

A: **A and C**. **A qualifies** because embedding dimension changes (384 → 768) or type changes (FLOAT → DOUBLE) are breaking schema changes that would cause Vector Search index rebuilds to fail; Lakehouse Monitoring's schema monitor detects unexpected structural changes like added/removed/renamed columns and type changes. **C qualifies** because optional metadata columns represent legitimate schema evolution (new columns) that monitors flag, and null-rate monitors on those columns surface data quality issues (e.g., a batch of documents missing `source_url`). **B is wrong** because ingestion count and query latency are operational metrics, not schema concerns. **D is wrong** because storage size and file count are internal Delta optimizations, not schema properties. **E is wrong** because refresh frequency and query concurrency are performance concerns, not schema validation issues.

---

## Question 3: Freshness Monitoring for Stale RAG Source Data

**Q3: A production RAG pipeline ingests news documents into a Delta table every 4 hours via a Databricks job. The ingestion job fails silently due to a credential rotation issue, leaving documents stale for 18 hours before alerts fire. Which Databricks Lakehouse Monitoring capability would have caught the stale data within 1 hour of the first missed ingestion window?**

   A. A freshness monitor configured with a max-staleness threshold of 1 hour, which compares the latest timestamp in the source document table against the current system time.
   B. A schema validation monitor that detects missing or incomplete rows in the Delta table.
   C. A custom null-rate metric applied to a `last_updated` column with a 1-hour alert threshold.
   D. MLflow inference table tracking to compare prediction input timestamps against baseline windows.

A: **A**. Lakehouse Monitoring's freshness monitor is explicitly designed to detect data ingestion delays and staleness. By setting a max-staleness threshold (e.g., 1 hour), the monitor compares the maximum `last_ingested_timestamp` or similar timestamp column against the current time and alerts when data has not been refreshed within the threshold window. This would catch the 4-hour ingestion window being missed immediately. Option B detects structural issues, not staleness. Option C is a workaround (using a custom metric) but is not the primary freshness monitoring mechanism. Option D is for model inference tracking, not data freshness.

---

## Question 4: Custom Metrics for Business Logic Validation

**Q4: In your RAG knowledge base, a metadata field `source_verified` should be TRUE for >95% of documents to ensure only vetted sources are used for retrieval. After a data load, this drops to 88%. How would you configure Lakehouse Monitoring to catch this regression and alert the data team automatically?**

   A. Define a schema validator rule that enforces the `source_verified` column to be non-null.
   B. Configure a custom metric that counts rows WHERE `source_verified` = TRUE, compares the ratio to a baseline, and alerts when it drops below 95%.
   C. Use a Unity Catalog table constraint to block inserts where `source_verified` IS NULL.
   D. Create an MLflow expectation to track the proportion of verified sources across model training batches.

A: **B**. Lakehouse Monitoring supports custom metrics that can be defined via SQL or user-specified thresholds. A custom metric would compute the fraction of documents with `source_verified` = TRUE, compare it to a baseline or threshold (95%), and trigger an alert when the proportion falls below the threshold. This directly enforces your business rule in the monitoring layer. Option A validates schema structure but not the semantic correctness of the data (a document can be non-null but FALSE). Option C prevents future bad inserts but doesn't monitor historical data quality. Option D is for ML model expectations, not data quality metrics.

---

## Question 5: Multi-Channel Alert Configuration

**Q5: Which TWO notification channels does Databricks Lakehouse Monitoring support to alert your data team when a monitor rule is violated?**

   A. Email notifications to a distribution list.
   B. Slack webhook notifications to a dedicated #data-alerts channel.
   C. SMS text messages to on-call engineers.
   D. Jira ticket auto-creation with issue assignment.
   E. PagerDuty incident escalation with severity levels.

A: **A and B**. **A qualifies** because Lakehouse Monitoring supports email alerts to notify stakeholders of violations. **B qualifies** because Slack integration via webhooks is a supported notification channel in Databricks, allowing alerts to post to dedicated Slack channels. **C is wrong** because SMS is not a native Lakehouse Monitoring alert channel. **D is wrong** because Jira auto-creation is not a built-in alert mechanism (though users could integrate via webhooks). **E is wrong** because PagerDuty escalation is not natively supported in Lakehouse Monitoring (though possible via advanced webhook integration outside the core product).

---

## Question 6: Integration of Lakehouse Monitoring and MLflow Inference Tables

**Q6: Your RAG application uses MLflow inference tables to track LLM response quality (response time, relevance scores) and Lakehouse Monitoring to track source document quality (null rates, freshness). How do these two monitoring systems work together to ensure end-to-end application reliability?**

   A. MLflow inference tables replace Lakehouse Monitoring; they track both input data quality and model output quality in a single system.
   B. Lakehouse Monitoring monitors raw source document Delta tables for quality issues, while MLflow inference tables monitor the LLM retrieval and generation pipeline, providing complementary observability: if source data quality degrades, Lakehouse Monitoring alerts first, and if retrieval-augmented generation outputs degrade, MLflow inference tables detect it.
   C. Lakehouse Monitoring only works with Vector Search tables, and MLflow inference tables only work with LLM endpoints; they are independent systems with no integration.
   D. MLflow inference tables are deprecated in favor of Lakehouse Monitoring for all data quality tracking.

A: **B**. Lakehouse Monitoring and MLflow inference tables are complementary systems that together provide full-stack observability for RAG applications. Lakehouse Monitoring operates at the data layer—tracking quality, schema, and freshness of source documents—while MLflow inference tables track the AI pipeline's behavior—inputs to and outputs from the LLM, including latency and relevance. If source data quality drops, Lakehouse Monitoring fires first. If retrieval or generation quality degrades despite good source data, MLflow detects it. Together they form a complete D6 (Evaluation & Monitoring) strategy. Option A conflates two separate systems. Option C incorrectly narrows their scopes. Option D is factually incorrect.

---

## Question 7: Incident Response to Lakehouse Monitor Alerts

**Q7: A Lakehouse Monitor alert fires at 3 AM: "15% of documents in the RAG corpus lack the `metadata_tags` field, exceeding the 5% null-rate threshold." Your investigation reveals that a recent data loader code change inadvertently stopped populating this field 6 hours ago. What is the FIRST action your data engineering team should take?**

   A. Immediately delete all documents with missing `metadata_tags` to restore data quality to baseline.
   B. Scale down the Vector Search index update frequency until the bug is fixed and historical data is corrected.
   C. Stop the ingestion job to prevent further documents from being loaded with missing metadata.
   D. Run a diagnostic query to identify the scope of affected documents (time range, count) and halt the loader, then triage: rollback the code change or fix the bug in-place and re-process the affected batch.

A: **D**. The correct incident response flow is: (1) understand the scope (how many documents, when did it start), (2) stop the source of the problem (halt the broken loader), and (3) decide on remediation (rollback vs. fix). This prevents the situation from worsening while you have time to implement the right fix. Deleting data (Option A) is data-destructive without full understanding. Scaling down updates (Option B) masks the symptom but doesn't fix the root cause. Stopping the job (Option C) is too broad without first diagnosing scope. Option D is the disciplined on-call response: assess, isolate, then fix.

---

## Question 8: Anomaly Detection in Embedding Vector Characteristics

**Q8: Your Lakehouse Monitor detects a distribution shift in a monitored column `chunk_length_tokens` across your document table. The baseline mean is 200 tokens, but this week the monitor flags that 30% of new chunks are now 10x longer (≈2000 tokens), causing Vector Search embedding failures. Which of the following is the MOST likely root cause, and what is the FIRST diagnostic action?**

   A. The embedding API suddenly changed its tokenization algorithm; contact Databricks support to verify API backward compatibility.
   B. The data source changed its format (e.g., multi-paragraph documents instead of single paragraphs per chunk), and the chunking logic in the data loader was not updated; query the source system to inspect the raw input data format.
   C. The Vector Search index was migrated to a newer model with different dimensional requirements, causing token inflation; check the Vector Search model version in the Vector Search index configuration.
   D. Document timestamps are stale and the monitor is double-counting historical data; clear Lakehouse Monitoring statistics and re-run the baseline.

A: **B**. Distribution shifts in embedding-relevant columns like token count most commonly stem from upstream data source changes rather than infrastructure issues. If 30% of chunks are 10x longer, the most likely cause is that the source data format changed (multi-paragraph vs single-paragraph documents) and the chunking logic was not updated. The FIRST diagnostic action is to inspect raw source data to confirm the format changed, then update the loader. Option A assumes an API change, but the embedding API usually fails loudly on mismatch rather than changing behavior silently. Option C confuses Vector Search model versioning with token count; model updates don't inflate tokens. Option D is unnecessary and incorrect; monitor statistics accurately reflect the data, not historical pollution.

---

## Summary

- **Total Questions:** 8
- **Single-Select:** 6 (Q1, Q3, Q4, Q6, Q7, Q8)
- **Multi-Select:** 2 (Q2, Q5)
- **Coverage:** Purpose & use cases, schema validation, data quality metrics, freshness, custom metrics, alert configuration, MLflow integration, incident response, anomaly diagnosis
- **Difficulty:** Medium-to-hard (requires understanding of RAG pipelines, alert lifecycle, and integration patterns)

---

## Grounding and Assumptions

**Verified against Databricks documentation concepts:**
- Lakehouse Monitoring tracks null rates, distributions, schema changes, and freshness
- Custom metrics are SQL-based rules with configurable thresholds
- Alerts support email and Slack webhooks (standard integrations)
- Lakehouse Monitoring operates on Delta tables and complements MLflow inference tables
- Freshness monitors compare timestamp columns against current time

**Assumptions made (for exam consistency):**
1. Lakehouse Monitoring's schema validator detects structural changes (columns, types) but not data validation (truth values).
2. Custom metrics are defined with SQL aggregations + threshold comparisons.
3. Email and Slack are the two primary out-of-the-box alert channels.
4. MLflow inference tables and Lakehouse Monitoring are separate systems that integrate via shared table references.
5. Incident response prioritizes diagnosis, isolation, and root-cause remediation over data deletion.
6. Distribution shifts are more often driven by upstream data changes than infrastructure (Occam's Razor).

**File saved to:** `/home/chainer/Documents/Projects/databricks_ai_engineer_associate/09-questions-dump/LAKEHOUSE-MONITORING-NEW-01.md`
