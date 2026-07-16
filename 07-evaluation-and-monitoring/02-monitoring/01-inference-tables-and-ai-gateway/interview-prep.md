# Inference Tables and AI Gateway Monitoring — Interview Prep

**Section:** 07 Evaluation and Monitoring | **Role target:** Senior ML/AI Engineer, AI Platform Engineer, Solutions Architect

---

## Core Conceptual Questions

| Question | Key points to cover | Common weak-answer trap |
|---|---|---|
| What is a Databricks inference table and how does it differ from a standard application log? | Delta table in Unity Catalog; at-least-once delivery within ~1 hour; stores raw JSON request + response; governed by UC ACLs; queryable with DBSQL; captures `databricks_request_id` for joining with traces; NOT the same as system logs which only capture errors/metrics | Saying "it's just a log file" — misses that it's a queryable Delta table with governance, structured schema, and join keys for downstream evaluation pipelines |
| What are the key schema differences between legacy inference tables and AI Gateway-enabled inference tables? | Legacy: `timestamp_ms` (LONG), `execution_time_ms`, `request_metadata` MAP; AI Gateway: `request_time` (TIMESTAMP), `request_date` (DATE), `execution_duration_ms` (BIGINT), `requester` (identity), `served_entity_id` (joins to system table), `logging_error_codes` (ARRAY for oversize payloads) | Saying "they're basically the same" — the column name changes break existing queries; the addition of `requester` and `served_entity_id` are architecturally significant for cost attribution and governance |
| How does MLflow 3 production monitoring relate to inference tables? | MLflow production monitoring runs scorers against *traces* logged to an MLflow experiment (via MLflow Tracing), not directly against the inference table. Inference table = payload audit log. MLflow traces = structured observability data. The two can be correlated via `databricks_request_id`. For agent deployments, traces are also stored in the inference table's `_payload_request_logs` derived table | Conflating the two as the same thing — they are complementary systems with different data models and different query patterns |
| What constraints apply to custom scorer functions in MLflow production monitoring? | Must use `@scorer` decorator (not class-based); must be registered from a Databricks notebook (serialization requires notebook env); all imports must be inline inside the function; no external variable references; no type hints requiring imports in the signature | Saying "you can use any Python function" — ignores the serialization constraints that will cause silent failures or hard-to-diagnose errors in production |
| When would you use Data Profiling's Inference profile type over the TimeSeries profile type? | Use Inference when the table is a model request log — it computes per-model-version metrics, compares predictions over time, and supports a training-baseline table for meaningful drift detection. Use TimeSeries for generic time-stamped Delta tables that are not model request logs | Choosing TimeSeries for inference tables "because it has time windows" — Inference profile is specifically designed for request logs and adds model performance semantics that TimeSeries lacks |

---

## Applied / Scenario Questions

**Q:** A production RAG chatbot's CSAT score dropped from 4.2 to 3.7 over two weeks. Inference tables are enabled. Walk me through how you would diagnose the root cause using the available data.

**Strong answer framework:**
- First, establish a baseline: query the inference table for the two-week period, extract prompts and responses via `get_json_object()`, sample ~1,000 rows.
- Check the distribution of `execution_duration_ms` — unusually fast responses may indicate retrieval failures (context not injected); unusually slow responses may indicate overloaded retriever.
- Run an MLflow quality scorer (correctness or faithfulness LLM judge) on the sample and compare the score distribution in week 1 vs week 2.
- Check Data Profiling drift metrics: look for shifts in the prompt token length distribution (users may be asking longer/more complex questions) and in the response length distribution (model may be truncating).
- Correlate by `requester` to see if the drop is uniform or concentrated in one user group / team.
- Show tradeoff awareness: if no inference table exists, you are down to CSAT survey data with no ground-truth signal about what the model actually said — this is why enabling inference tables before deployment matters.

---

**Q:** Your team needs to attribute GenAI API costs to business units. How would you use AI Gateway inference tables to build that attribution report?

**Strong answer framework:**
- The `requester` column in the AI Gateway schema records the caller's identity (user or service principal).
- Join the inference table with `system.serving.served_entities` on `served_entity_id` to get the foundation model name and pricing tier.
- Extract token counts from the `response` JSON (`get_json_object(response, '$.usage.total_tokens')` for OpenAI-compatible endpoints) — these are the billable unit.
- Map `requester` to business unit using a Unity Catalog identity table or a manually maintained mapping table.
- Aggregate by `request_date`, business unit, and foundation model to produce a weekly cost attribution table.
- Show tradeoff awareness: token-level billing data is in the `response` JSON for Foundation Model API and external model endpoints; for custom model endpoints, you may need to instrument the model to emit usage metadata in the response body.

---

## System Design / Architecture Questions

**Q:** Design a production quality monitoring system for a high-volume GenAI application (100K requests/day) that must: (1) detect answer quality degradation within 24 hours, (2) attribute quality scores by user department, (3) alert on-call via PagerDuty if quality drops below threshold, and (4) not exceed $50/day in monitoring compute costs.

**Approach:**
1. **Clarify requirements:** What is the quality metric? (LLM judge score, user thumbs-up rate, or both?) What is "degradation"? (Absolute drop? Relative to 7-day rolling average?) What is the alerting threshold?
2. **Propose structure:**
   - Enable AI Gateway inference table at `sampling_fraction=0.1` → 10K rows/day logged ($0 for storage at Delta table rates).
   - Register one MLflow safety scorer at `sample_rate=1.0` (evaluates all 10K logged traces) — safety is non-negotiable coverage.
   - Register one LLM correctness scorer at `sample_rate=0.1` (evaluates 1K traces/day at ~$0.04/trace = $40/day — within budget).
   - Attach a Data Profiling Inference monitor with `slicing_exprs=["requester"]` for department attribution.
   - Databricks Workflow runs nightly: queries drift metrics table, checks if 24h rolling quality score < threshold, sends PagerDuty alert via webhook.
3. **Justify choices and tradeoffs:**
   - `sampling_fraction=0.1` is the right starting point at 100K/day — represents statistical significance without storage explosion.
   - LLM judge at `sample_rate=0.1` of logged traffic = 1,000 evaluations/day at ~$40/day, leaving $10 headroom.
   - 24-hour detection window is achievable with a nightly job querying yesterday's data — near-real-time detection (< 1 hour) would require a streaming job or Databricks Workflows trigger on table update, which is more complex and costs more.
   - PagerDuty integration: Databricks Workflows supports outbound webhook tasks; the alert condition is a SQL query on the drift metrics table.

---

## Vocabulary That Signals Expertise

Use these terms naturally — don't force them:
- **at-least-once delivery** — when describing inference table reliability; signals you understand the durability model
- **`sampling_fraction` × `sample_rate` compounding** — when discussing monitoring cost vs coverage tradeoffs
- **`from_json()` / `get_json_object()`** — when discussing JSON extraction from inference tables; signals practical hands-on experience
- **`served_entity_id` join** — when discussing cost attribution or multi-model endpoints; signals knowledge of the AI Gateway schema specifically
- **Inference profile type** — when discussing Data Profiling; distinguishes you from someone who would use TimeSeries for an inference table
- **scorer serialization constraints** — when discussing production monitoring; signals you've actually tried to implement it
- **baseline table drift** — when discussing Data Profiling; signals you think about monitoring in terms of distribution shift, not just point-in-time metrics

---

## Vocabulary That Signals Weakness

Avoid these — they signal outdated or shallow understanding:
- **"Lakehouse Monitoring"** — the old name; the product is now called Data Profiling / data quality monitoring under Unity Catalog
- **"`auto_capture_config`"** — the legacy inference table API, archived and unsupported for new endpoints; using this term without qualification suggests you haven't checked the current docs
- **"real-time monitoring via inference tables"** — signals you don't know about the ~1-hour delivery latency; real-time guardrails use AI Gateway service policies, not inference tables
- **"just query the table after each request"** — same problem as above; demonstrates confusion between the inference serving hot path and the async logging pipeline
- **"MLflow evaluate on the full table every hour"** — signals lack of awareness of LLM judge cost; always discuss sampling when you discuss evaluation at scale

---

## STAR Answer Frame

**Situation:** Our team deployed a customer-facing document summarization agent on Databricks Model Serving. After three weeks in production, legal flagged that some summaries omitted important liability clauses from contract documents.

**Task:** I was responsible for diagnosing the failure scope and implementing a monitoring system that would catch similar issues within 24 hours going forward.

**Action:** I queried the inference table (which was enabled from day one) using `get_json_object()` to extract prompts and responses. I identified that 8% of requests contained multi-page contracts exceeding 6,000 tokens — the model was silently truncating the input. I then registered an MLflow content-length scorer using the `@scorer` decorator from a notebook, checking if response length was proportional to input length. I attached a Data Profiling Inference monitor with a baseline table from our validation set to detect future distribution shifts in document length. I set an alert on the drift metrics table to fire if input token length P95 exceeds the training distribution by 20%.

**Result:** The monitoring system detected the next batch of oversized documents (a new document type added by a partner) within 18 hours of first occurrence, before any legal escalation. The alert triggered an automatic routing rule to a higher-context model endpoint for oversized inputs.

---

## Red Flags Interviewers Watch For

- Saying "we use logging" without specifying whether inference tables are enabled — vague operational hygiene
- Conflating inference table availability (async, ~1 hour) with real-time monitoring — fundamental misunderstanding of the architecture
- Designing a monitoring system with 100% LLM judge coverage at scale without discussing cost — signals no production experience with LLM evaluation economics
- Not knowing that `request` and `response` are JSON strings, not structs — signals never actually queried an inference table
- Describing MLflow production monitoring without mentioning the notebook registration constraint — suggests reading docs but not implementing it
- Saying "we check quality during the request" — confuses guardrails (AI Gateway service policies, synchronous) with monitoring (inference tables + scorers, asynchronous)
- Using the legacy `auto_capture_config` terminology for new endpoints without acknowledging the migration to AI Gateway inference tables
