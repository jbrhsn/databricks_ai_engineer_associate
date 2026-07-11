# Foundation Model APIs and ai_query — Interview Prep

**Section:** 05-Assembling and Deploying Applications | **Role target:** Senior ML Engineer, MLOps Engineer, Solutions Architect, Data Engineer (AI/ML)

---

## Core Conceptual Questions

These test whether you understand the fundamentals.

| Question | Key points to cover | Common weak-answer trap |
|---|---|---|
| What are the three Foundation Model API endpoint modes and when do you use each? | (1) Pay-per-token: shared capacity, no setup, billed per token, dev/PoC/burst workloads; (2) Provisioned throughput: dedicated GPU capacity, tokens/sec guarantee, fine-tuned models, HIPAA-eligible, sustained production load; (3) External models: third-party providers proxied through Databricks, centralised credentials, same OpenAI-compatible interface | Saying "pay-per-token is for cheap workloads and provisioned is for expensive ones" — this misses the compliance and throughput guarantee dimensions entirely |
| How does the OpenAI-compatible interface work across different model backends? | Model Serving normalises requests to each backend's native inference API and normalises responses back to OpenAI format (`choices[0].message.content`). The caller only changes `base_url` and the `model` string (endpoint name). Auth uses a Databricks token as a Bearer token. | Saying "Databricks just wraps OpenAI" — this confuses the interface standard with the implementation; Databricks hosts its own models and proxies others independently |
| What is `ai_query()` and what are its compute requirements? | General-purpose SQL function that calls any Model Serving endpoint from a SQL `SELECT` statement. Requires Serverless compute or DBR 18.2+. Not available on Pro or Classic SQL warehouses. Parallelises internally; submit full dataset in one query. `failOnError => false` preserves successful rows in batch jobs. | Describing it as "a simple UDF" without mentioning the serverless requirement or internal parallelisation — interviewers will probe those specifics |
| What does Unity AI Gateway provide and how is it different from the old per-endpoint AI Gateway? | Unity AI Gateway is an account-level governance layer built on Unity Catalog providing rate limits, budget caps, per-principal usage tracking (system tables), and service-policy guardrails. The legacy per-endpoint AI Gateway configured `ai_gateway` fields on individual serving endpoints at workspace level. Both coexist during transition (Beta as of July 2026). | Saying "it's the same thing, just updated UI" — the architectural scope (account vs workspace, Unity Catalog integration) is a key differentiator |
| How does an external model endpoint handle API key security? | API keys are stored in Databricks Secrets using `{{secrets/scope/key}}` references in the endpoint configuration. Databricks encrypts and stores them; they are injected at request time and auto-deleted when the endpoint is deleted. The application developer and endpoint consumers never see the raw key. | Saying the key is stored in environment variables or passed in the request — this misses that Databricks handles credential injection at the platform level |

---

## Applied / Scenario Questions

**Q:** Your team has a nightly Databricks SQL job that enriches a 20-million-row customer table with AI-generated sentiment labels. A junior engineer has written Python code that reads the table in chunks of 50k rows, calls the OpenAI SDK in a loop, and writes results back. What would you change and why?

**Strong answer framework:**
- Identify the problem: manual batching prevents `ai_query()`'s internal parallelisation; Python loop is serial and fragile; no built-in retry or progress monitoring.
- Propose: replace with a single `CREATE TABLE AS SELECT ai_query(endpoint, CONCAT(prompt, text), failOnError => false)` SQL statement on a Serverless warehouse. Submit the full 20M rows in one query.
- Justify: `ai_query()` handles parallelisation, retries, and scaling automatically. The query profile view provides built-in monitoring of completed/failed inferences. `failOnError => false` preserves successful rows without aborting the job.
- Show tradeoff awareness: if the workload needed real-time scoring or complex multi-step agent logic, Python would be appropriate; for batch enrichment of a static table, `ai_query()` is the right tool.

---

**Q:** A colleague says you should always use pay-per-token endpoints in production to save money. How do you respond?

**Strong answer framework:**
- Acknowledge the partial truth: for bursty, low-volume, or development workloads, pay-per-token avoids idle provisioned capacity costs.
- Counter with compliance: pay-per-token runs on shared infrastructure and is not HIPAA-eligible. If the data is sensitive or regulated, provisioned throughput is mandatory regardless of cost.
- Counter with SLA: pay-per-token provides no latency or throughput guarantee. If your application has a p99 latency SLA, you need provisioned throughput.
- Counter with fine-tuned models: pay-per-token only serves base Databricks-hosted models; if you have a fine-tuned variant in Unity Catalog, you must use provisioned throughput.
- Conclude: the decision is compliance + SLA architecture first; cost is a secondary optimisation.

---

**Q:** A security audit finds that three different teams are each maintaining separate OpenAI API keys in different secret managers. How would you consolidate this using Databricks?

**Strong answer framework:**
- Create a single Databricks external model endpoint for OpenAI with the API key stored in Databricks Secrets.
- Grant teams `CAN_QUERY` permission on the endpoint through Unity Catalog instead of sharing the raw key.
- Configure Unity AI Gateway rate limits per team so no single team can exhaust the API quota.
- Enable usage tracking to produce per-team attribution reports from system tables.
- Result: one key rotation point, no raw key exposure in code, auditability.

---

## System Design / Architecture Questions

**Q:** Design a production RAG pipeline that serves 10,000 queries per day for a HIPAA-regulated healthcare company. The pipeline uses a fine-tuned Llama model and must be auditable.

**Approach:**
1. Clarify requirements: HIPAA compliance, fine-tuned model, audit trail, 10k queries/day (~0.1 req/sec average, likely spiky).
2. Propose structure:
   - Fine-tuned Llama model registered in Unity Catalog under `system.ai`.
   - Provisioned throughput endpoint (HIPAA-eligible, supports fine-tuned models); use the model optimisation info API to size model units for the traffic profile.
   - Application calls the endpoint via the OpenAI SDK with Databricks OAuth M2M auth (not PAT for production).
   - Unity AI Gateway enabled: rate limits per application, inference table logging for audit trail, budget cap.
   - Vector Search index (separate) for the retrieval step.
3. Justify choices:
   - Provisioned throughput: required for HIPAA and fine-tuned model support.
   - Inference table logging: provides the per-request audit trail required by HIPAA.
   - OAuth M2M: recommended over PAT for production service accounts.
4. Name tradeoffs explicitly: provisioned throughput costs more than pay-per-token at this traffic level; the compliance requirement makes it non-negotiable. GPU capacity may affect endpoint creation time (known limitation).

---

## Vocabulary That Signals Expertise

Use these terms naturally — don't force them:
- **"endpoint name"** (not "model name") when referring to the `model` parameter in FMAPI calls — distinguishes you from candidates who conflate model families with serving endpoints
- **"tokens per second"** and **"model units"** when discussing provisioned throughput sizing — shows you understand the capacity model
- **"model optimisation info API"** when describing how to calculate `min_provisioned_throughput` / `max_provisioned_throughput` — signals you have done this in practice
- **`failOnError => false`** as a deliberate production choice in batch `ai_query()` jobs — shows you understand fault tolerance vs. correctness tradeoffs
- **"Unity AI Gateway"** vs **"per-endpoint AI Gateway"** — distinguishes current architecture from legacy; signals you track feature evolution
- **"Databricks Secrets reference"** (`{{secrets/scope/key}}`) for external model credentials — signals security-aware design
- **"OpenAI-compatible interface"** (not "OpenAI API") — correctly describes the standard Databricks implements without implying Databricks is OpenAI

---

## Vocabulary That Signals Weakness

Avoid these — they signal outdated or shallow understanding:
- **"Just use the OpenAI API"** without acknowledging the `base_url` change to Databricks — suggests unfamiliarity with the Databricks serving layer
- **"ai_query is a UDF"** — technically imprecise; it is a built-in SQL function with special execution semantics, not a user-defined function
- **"Provisioned throughput is for big companies"** — the decision criteria are compliance and SLA, not company size
- **"DBRX is the main model"** — DBRX is no longer featured prominently in current FMAPI; the model list has expanded significantly and DBRX is not the default recommendation; using it as the canonical example suggests stale knowledge
- **"ai_query calls are slow"** without context — they are optimised for batch; latency at row level is a feature of the model, not `ai_query`
- **"You have to split batches manually"** — explicitly contradicted by Databricks documentation; signals you haven't read the best practices

---

## STAR Answer Frame

**Situation:** My team was building a document classification pipeline for a financial services client. We initially used a Python notebook with a manual loop calling the OpenAI SDK, batching 500 documents at a time. The job ran for 8 hours on a 20-million-document dataset.

**Task:** I was responsible for reducing job runtime and making the pipeline production-grade — retry-safe, monitorable, and compatible with the team's Databricks SQL-based data platform.

**Action:** I rewrote the pipeline as a single Databricks SQL query using `ai_query()` on a Serverless warehouse, pointing at a Databricks-hosted LLM endpoint. I removed the manual batching entirely and set `failOnError => false` so transient model errors would not abort the full run. I configured Unity AI Gateway on the endpoint for rate limiting and enabled inference table logging to capture every request for the client's audit requirements.

**Result:** Runtime dropped from 8 hours to 2.5 hours. The pipeline required zero retry-handling code. The inference table logs satisfied the client's audit requirement without additional instrumentation. The team's SQL engineers could read and modify the pipeline without any Python knowledge.

---

## Red Flags Interviewers Watch For

- **Cannot distinguish endpoint modes:** If you cannot explain when to use pay-per-token vs provisioned throughput beyond "cost," the interviewer will doubt you have built production FMAPI systems.
- **Unaware of `ai_query()` compute requirements:** Saying `ai_query()` works on any SQL warehouse signals you have not run it yourself and have not read the requirements.
- **Conflating the model name with the endpoint name:** Calling FMAPI with `model="llama-3.3-70b"` instead of the endpoint name is a basic error that fails immediately in practice.
- **Describing external model endpoints as "just a proxy":** This undersells credential centralisation, unified rate limiting, and AI Gateway integration — the actual reasons enterprises use them.
- **Not knowing that Unity AI Gateway is in Beta:** This is a fast-evolving area; interviewers evaluating for senior roles expect you to flag feature maturity and verify against current docs before making architectural commitments.
- **Recommending manual batch splitting for `ai_query()` jobs:** This directly contradicts Databricks best practices and signals you have read outdated blog posts rather than official documentation.
