# Databricks AI Search — Interview Prep

**Section:** 05-Assembling & Deploying Applications | **Role target:** Senior ML Engineer, MLOps Engineer, Solutions Architect, Generative AI Engineer

> ⚠️ **Fast-evolving:** Databricks AI Search APIs and features change frequently. Re-verify specifics before an interview.

---

## Core Conceptual Questions

These test whether you understand the fundamentals.

| Question | Key points to cover | Common weak-answer trap |
|---|---|---|
| What is the difference between a Delta Sync Index and a Direct Vector Access Index? | Delta Sync mirrors a UC Delta table automatically (TRIGGERED or CONTINUOUS); Direct Access requires manual upsert via SDK/REST, no source table needed. Delta Sync is best for existing Delta data pipelines; Direct Access is for external data sources or real-time custom embeddings. | Saying "Direct Access is faster" — speed is about query time, not index type. Or conflating the index type with the sync mode. |
| What does `pipeline_type="TRIGGERED"` actually mean, and what must the developer do? | It means the sync does NOT fire automatically when the Delta table changes. The developer must call `index.sync()` explicitly (or use the UI/API). Without this call, the index is frozen at the last sync point. | Saying "the index updates automatically when the table changes" — this is exactly what TRIGGERED does NOT do. CONTINUOUS is the mode that reacts automatically. |
| What algorithm does AI Search use for similarity queries, and what distance metric? | HNSW (Hierarchical Navigable Small World) for approximate nearest neighbor, with L2 (Euclidean) distance. For cosine similarity, vectors must be normalized before indexing — normalized L2 and cosine similarity produce the same ranking. | Saying "cosine similarity" as the native metric without noting the normalization requirement. Or confusing HNSW with brute-force exact search. |
| What is Reciprocal Rank Fusion and when does AI Search use it? | RRF merges ranked lists from two independent retrievers — ANN (vector) and BM25 (keyword) — by summing `1/(60 + rank)` for each document across both lists. It is used in `query_type="hybrid"`. The rrf_param of 60 is fixed by the platform. | Vaguely saying "it combines the scores" without explaining that it operates on rank positions (not raw scores), making it robust to score scale differences between the two retrievers. |
| What is the requirement for using a STANDARD endpoint with a Delta Sync Index? | Change Data Feed (CDF) must be enabled on the source Delta table (`delta.enableChangeDataFeed = true`). The workspace must also have serverless compute enabled. CDF is how the sync pipeline identifies changed rows incrementally. | Forgetting CDF entirely, or listing only the serverless compute requirement. Both are required. |
| What is the difference between `embedding_source_column` and `embedding_vector_column`? | `embedding_source_column` is a text column; Databricks calls the specified embedding model endpoint at sync time to compute vectors. `embedding_vector_column` is an `array<float>` column already containing pre-computed vectors; Databricks does not compute embeddings. | Saying either one "stores the embeddings" — only `embedding_vector_column` stores them; `embedding_source_column` is the *input* to the embedding process. |

---

## Applied / Scenario Questions

**Q:** Your company's legal knowledge base is stored in a Unity Catalog Delta table and updated weekly via a batch pipeline. A LangGraph agent must retrieve relevant clauses before answering user questions. Users sometimes ask about very specific legal section numbers (e.g., "Section 12.4(b)"). How would you configure the AI Search index, and why?

**Strong answer framework:**
- **Index type:** Delta Sync Index with `pipeline_type="TRIGGERED"` — weekly updates mean continuous sync would waste cluster resources 24/7 for a batch workload.
- **Embedding source:** `embedding_source_column` pointing to the clause text column, with Databricks-managed embeddings — avoids maintaining a separate embedding pipeline for standard legal text.
- **Query type:** `query_type="hybrid"` — section numbers like "12.4(b)" are exact tokens that ANN may miss; hybrid combines BM25 keyword matching with ANN semantic search.
- **Sync orchestration:** Add `index.sync()` as the final step of the weekly ETL job.
- **Trade-off awareness:** If the legal team required finer-tuned embeddings for domain-specific legal language, self-managed embeddings with a fine-tuned model would be worth the operational overhead. For standard legal text with Foundation Model API models, managed embeddings are sufficient.

---

**Q:** A multi-tenant SaaS platform stores all customers' documents in a single AI Search index indexed with a `tenant_id` column. How do you ensure tenant isolation at query time?

**Strong answer framework:**
- Pass `filters={"tenant_id": current_user_tenant}` in every `similarity_search()` call in the application layer.
- AI Search does not enforce row-level security at the index layer automatically — isolation is entirely the application's responsibility.
- Consider adding a code review lint rule or wrapper function that enforces the filter is always present.
- Advanced: for strict isolation requirements, consider separate indexes per tenant (higher cost but eliminates application-layer mistakes) vs. shared index with filters (lower cost but requires careful engineering).
- Acknowledge the limitation: with storage-optimized endpoints, over-fetching during filter application means a misconfigured filter could theoretically surface cross-tenant results in edge cases; additional application-layer validation of returned rows' tenant_id field is a defense-in-depth measure.

---

**Q:** The AI Search index for your knowledge base is configured with TRIGGERED sync. A monitoring alert fires: the index is 48 hours behind the source table. Walk through your diagnosis and resolution steps.

**Strong answer framework:**
- Check the Databricks Job or workflow that should be calling `index.sync()` — did the job fail, or was the sync call never added to the pipeline?
- In Catalog Explorer, navigate to the index's Overview tab and check the last sync time and any error messages in the Data Ingest section.
- Check the Delta table's CDF status: `SHOW TBLPROPERTIES catalog.schema.table` — if CDF was accidentally disabled, the sync will fail.
- If the job failed partway, examine the job run logs for the specific error (common: embedding model endpoint scaled to zero and timed out).
- Resolution: fix the root cause, re-enable CDF if needed, ensure the embedding endpoint is not configured to scale to zero, then trigger `index.sync()` manually to catch up.
- Prevention: add monitoring that compares the index's last sync timestamp against the source table's latest modification time and pages on-call if the gap exceeds 2× the scheduled refresh interval.

---

## System Design / Architecture Questions

**Q:** Design a production RAG system that serves 10,000 users per day with a knowledge base of 5 million documents updated in near-real-time from a streaming data source.

**Approach:**
1. **Clarify requirements:** What is the definition of "near-real-time"? Seconds? Minutes? This determines sync mode choice. What is the acceptable query latency? What is the tenant model (single tenant or multi)?

2. **Propose structure:**
   - Source: streaming data arrives via Kafka → written to a Delta table with CDF enabled (via Spark Structured Streaming or a DLT pipeline)
   - Index type: Delta Sync Index, `pipeline_type="CONTINUOUS"` (given near-real-time requirement)
   - Endpoint type: STANDARD for streaming; consider STORAGE_OPTIMIZED only if the index exceeds ~10B vectors
   - Embedding: evaluate Databricks-managed embeddings if using a Foundation Model API endpoint; if latency at ingestion is a concern, use a GPU-backed provisioned throughput endpoint without Scale to Zero
   - Query: `query_type="hybrid"` as the default retriever in LangGraph; enable reranker for high-stakes answers
   - Scale: use `target_qps` on the endpoint for predictable high-throughput workloads

3. **Justify choices and name trade-offs explicitly:**
   - CONTINUOUS sync adds cost of a persistent streaming cluster; justified only by the near-real-time requirement
   - Disabling Scale to Zero on the embedding endpoint avoids cold-start timeouts at the cost of idle compute when ingestion pauses
   - Separate the embedding endpoint (ingestion) from the query endpoint if ingestion throughput is high — `model_endpoint_name_for_query` allows different endpoints for the two workloads
   - Add monitoring on index lag (source table watermark vs. last index sync) and query latency P95

---

## Vocabulary That Signals Expertise

Use these terms naturally — don't force them:

- **HNSW (Hierarchical Navigable Small World)** — the ANN algorithm; signals you know how the similarity search actually works, not just the API surface
- **Reciprocal Rank Fusion (RRF)** — the specific merging algorithm for hybrid search; signals you've read beyond the "hybrid = better" marketing
- **Change Data Feed (CDF)** — the Delta mechanism that enables incremental sync; signals you know the infrastructure requirement, not just the SDK call
- **`embedding_source_column` vs `embedding_vector_column`** — precise API parameters; signals you've actually written the code, not just read a summary
- **ANN vs. exact nearest neighbor** — framing similarity search as approximate signals awareness of the cost/accuracy trade-off
- **RRF parameter (60)** — knowing the fixed parameter in the RRF formula signals deep reading
- **Writeback table** — the optional Unity Catalog table where Databricks saves generated embeddings; relevant when audit or reuse of embeddings is needed
- **`query_type` parameter** — explicit knowledge of `ann`, `hybrid`, `FULL_TEXT` options and when each applies

---

## Vocabulary That Signals Weakness

Avoid these — they signal outdated or shallow understanding:

- **"Vector database"** as a generic term applied to Databricks AI Search — Databricks calls it "AI Search" (formerly Vector Search); using "vector database" suggests you've only read about the category, not the specific product
- **"The index updates automatically"** — this conflates CONTINUOUS and TRIGGERED; only CONTINUOUS mode auto-updates
- **"Cosine similarity"** as the native metric without mentioning L2 distance and the normalization requirement — signals you've copied from generic vector search content
- **"Just use the default settings"** — weak answer for any configuration question; shows you haven't thought about operational trade-offs
- **Describing the endpoint and the index as the same thing** — they are distinct resources with distinct lifecycles; conflating them signals fundamental conceptual gaps

---

## STAR Answer Frame

**Situation:** I was building a RAG system for an internal knowledge base with 500,000 documents. The initial implementation used a Direct Vector Access Index with a custom embedding pipeline managed by a separate team.

**Task:** I was responsible for reducing operational complexity and improving index freshness after several incidents where the custom embedding pipeline failed silently, leaving the index 3–5 days stale.

**Action:** I migrated the index to a Delta Sync Index with `embedding_source_column` (Databricks-managed embeddings) and `pipeline_type="TRIGGERED"`. I added `index.sync()` as the final step of the data pipeline and added a monitoring job that fired an alert if the index was more than 4 hours behind the source table's last modification time. I also changed the default `query_type` from `"ann"` to `"hybrid"` after observing that 30% of user queries contained internal project codes that ANN was systematically missing.

**Result:** Silent staleness incidents dropped to zero because the monitoring alert now catches any failed sync within 4 hours. The retrieval quality improvement from switching to hybrid search was measurable in our offline evaluation set — recall@5 increased by 12% for queries containing internal identifiers. The elimination of the custom embedding pipeline removed one failure mode and two operational runbooks.

---

## Red Flags Interviewers Watch For

- **Can't explain the difference between TRIGGERED and CONTINUOUS sync** — this is foundational to AI Search operation; inability to explain it suggests the candidate has not built a real system with the product.
- **Assumes index is always fresh** — any answer that ignores sync latency or staleness reveals a candidate who hasn't run a production index.
- **No awareness of the CDF requirement** — this is the most common misconfiguration in AI Search setup; not knowing about it suggests the candidate has only used a pre-configured environment, not built one from scratch.
- **Describes multi-tenant security as automatic** — AI Search does not enforce data isolation automatically; a candidate who doesn't know this would ship a data breach.
- **Can't name the query_type options** — `ann`, `hybrid`, `FULL_TEXT` are the three options; not knowing them suggests the candidate has only seen the default behavior and never tuned retrieval.
- **Conflates endpoint type (STANDARD / STORAGE_OPTIMIZED) with index type (Delta Sync / Direct Access)** — these are orthogonal dimensions; confusing them reveals a shallow read of the documentation.
