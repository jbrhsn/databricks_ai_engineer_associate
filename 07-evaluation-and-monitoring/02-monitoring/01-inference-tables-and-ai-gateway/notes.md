# Inference Tables and AI Gateway Monitoring

**Section:** 07 Evaluation and Monitoring | **Module:** 02 Monitoring | **Est. time:** 2 hrs | **Exam mapping:** Evaluation & Monitoring domain (12%)

---

## TL;DR

Inference tables are Delta tables in Unity Catalog that automatically capture every request and response sent to a Databricks Model Serving endpoint, giving you a permanent, queryable record of your model's production behavior. AI Gateway extends this logging upstream — adding token counts, latency, safety scores, and requester identity — and routes the same data through Unity Catalog governance. Together they close the feedback loop: sample production traffic, run MLflow scorers over it, and compare live quality against your offline benchmark. **The one thing to remember: you cannot catch quality degradation in GenAI without a log of what the model actually said — inference tables are that log, and every monitoring strategy starts there.**

---

## ELI5 — Explain It Like I'm 5

Imagine a restaurant where every time a waiter brings a dish to a table, a clerk in the corner silently writes down the order and what came out of the kitchen — the time it took, whether the dish looked right, and the table number. That notebook is the inference table: a running record of every real interaction between a customer and the kitchen, stored forever. The most common mistake engineers make is thinking that because they tested the recipe in a kitchen trial (offline evaluation), the restaurant is fine — but the real customers order differently, at strange hours, and the kitchen occasionally burns things. Without the notebook, you only find out about quality problems when customers stop coming back. The inference table is that notebook, and the monitoring system is the manager who reads it every morning to spot trends before they become crises.

---

## Learning Objectives

By the end of this chapter you will be able to:
- [ ] Enable AI Gateway-enabled inference tables on a Model Serving endpoint and verify logging via Unity Catalog.
- [ ] Query an inference table in SQL, including parsing nested JSON `request` and `response` fields with `from_json()`.
- [ ] Explain the schema differences between legacy inference tables and AI Gateway-enabled inference tables.
- [ ] Design a production monitoring pipeline that samples inference table rows and runs MLflow scorers for continuous quality assessment.
- [ ] Configure Data Profiling on an inference table to detect data and response-quality drift over time.

---

## Visual Overview

### Inference Table Data Flow

```
User / Application
       │
       ▼
Model Serving Endpoint
       │  (every request/response logged automatically)
       ▼
┌─────────────────────────────────────────────────┐
│  AI Gateway Layer                               │
│  - token counts, latency, requester, safety     │
│  - sampling_fraction applied here               │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
         Unity Catalog Delta Table
         <catalog>.<schema>.<endpoint>_payload
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
  Data Profiling Monitor    Sampling + MLflow Scorers
  (drift metrics table)     (continuous quality assessment)
          │                     │
          ▼                     ▼
  Dashboard / Alerts      Feedback attached to traces
```

### Production Monitoring Feedback Loop

```
Offline Evaluation (dev)
  mlflow.genai.evaluate()
  benchmark score = 0.82
         │
         │   deploy endpoint
         ▼
  Inference Table (prod)
  rows accumulate with real traffic
         │
         │   scheduled job / production monitoring
         ▼
  MLflow Scorers run on sampled rows
  live score computed (e.g. 0.74)
         │
         │   score < threshold?
         ├── Yes ──► Alert (Slack / PagerDuty / Databricks Workflow)
         └── No  ──► Continue sampling
```

### Legacy vs AI Gateway Inference Table Schema

```
Legacy Inference Table                AI Gateway Inference Table
─────────────────────────────         ────────────────────────────────
databricks_request_id (STRING)        databricks_request_id (STRING)
client_request_id (STRING)            client_request_id (STRING)
date (DATE)                           request_date (DATE)
timestamp_ms (LONG)                   request_time (TIMESTAMP)
status_code (INT)                     status_code (INT)
execution_time_ms (LONG)              execution_duration_ms (BIGINT)
sampling_fraction (DOUBLE)            sampling_fraction (DOUBLE)
request (STRING)                      request (STRING)
response (STRING)                     response (STRING)
request_metadata (MAP<STRING,STRING>) served_entity_id (STRING)
                                      requester (STRING)       ← NEW
                                      logging_error_codes (ARRAY) ← NEW
```

### Data Profiling Monitor Architecture

```
Inference Table (primary)
       │
       │   profiling attaches here
       ▼
┌─────────────────────────────┐
│  Data Profile (Inference)   │
│  - timestamp col: request_time│
│  - model_id_col specified   │
│  - optional baseline table  │
└────────────────┬────────────┘
                 │  refresh on schedule
      ┌──────────┴──────────┐
      ▼                     ▼
Profile Metrics Table   Drift Metrics Table
(summary statistics)    (drift vs baseline / time window)
      │
      ▼
Auto-generated Dashboard + Alert Rules
```

---

## Key Concepts

### Inference Tables

**What it is:** An inference table is a Delta table in Unity Catalog that Databricks automatically populates with the raw JSON request and response payloads for every call made to a Model Serving endpoint. It is the permanent production log of your model's inputs and outputs.

**How it works mechanistically:** When a request reaches a Model Serving endpoint with inference tables enabled, Databricks intercepts the payload at the serving layer and writes a row asynchronously to the target Delta table. The write is decoupled from inference — the model response is returned to the caller immediately, and the log entry lands in the table within approximately one hour. Because the delivery mechanism is "at-least-once", duplicate rows are possible but rare; downstream deduplication on `databricks_request_id` is the correct pattern. The table is owned by the endpoint creator and governed by standard Unity Catalog ACLs.

**Where it appears in Databricks:** Enabled via the AI Gateway section of the Serving UI ("Enable inference tables") or programmatically in the endpoint configuration's `ai_gateway.inference_table_config`. The resulting table lives at `<catalog>.<schema>.<endpoint_name>_payload` and is visible in Catalog Explorer. Query it with `SELECT * FROM <catalog>.<schema>.<endpoint_name>_payload` in DBSQL or a notebook.

> ⚠️ Fast-evolving: As of July 2026, Databricks has deprecated the legacy inference table format. New endpoints on provisioned throughput and Foundation Model APIs only support AI Gateway-enabled inference tables. The legacy table format (using `auto_capture_config`) is archived and unsupported for new endpoints. Verify the current default at https://docs.databricks.com/aws/en/ai-gateway/inference-tables-serving-endpoints before authoring production code.

### AI Gateway Inference Table Schema

**What it is:** The AI Gateway-enabled inference table schema is the current (non-legacy) schema for logged payloads. It includes additional columns beyond the legacy schema: `requester` (the identity making the call), `served_entity_id` (links to the `system.serving.served_entities` system table), `logging_error_codes` (reasons a payload was not fully logged, e.g. `MAX_REQUEST_SIZE_EXCEEDED`), and `request_time` (a proper `TIMESTAMP` vs the legacy `timestamp_ms` epoch long).

**How it works mechanistically:** The AI Gateway sits in front of the actual model backend. Every request passes through the gateway, which stamps it with `databricks_request_id`, records `requester` from the authenticated identity, applies any configured rate limits or guardrails, and then forwards the request downstream. On response, the gateway records `execution_duration_ms`, `status_code`, and the raw JSON bodies before writing the row. The `sampling_fraction` column reflects whether the endpoint is configured to log only a fraction of traffic (for cost control on high-throughput endpoints).

**Where it appears in Databricks:** The schema is documented at `docs.databricks.com/aws/en/ai-gateway/inference-tables-serving-endpoints#schema`. To join inference table data with foundation model metadata: `SELECT * FROM <catalog>.<schema>.<endpoint>_payload p JOIN system.serving.served_entities se ON p.served_entity_id = se.served_entity_id`.

### Parsing Nested JSON in Inference Tables

**What it is:** The `request` and `response` columns are raw JSON strings. To extract individual fields (e.g. the prompt text or the generated answer), you must use SQL's `from_json()` or `get_json_object()` functions, or `pyspark.functions.from_json()` in Python.

**How it works mechanistically:** Databricks stores the entire request payload as a single STRING column to remain schema-agnostic across model types (chat, completion, embedding, custom). This means the same inference table pattern works for a foundation model endpoint (which uses OpenAI-compatible JSON) and a custom sklearn model endpoint (which uses a different schema). Consumers are responsible for declaring the target schema and calling `from_json()` with the appropriate `StructType`. Because the schema of `request` and `response` depends on the model type, you must inspect a sample row first and derive the schema.

**Where it appears in Databricks:** In DBSQL and notebooks. Example pattern for a chat-completion endpoint:

```sql
SELECT
  databricks_request_id,
  request_time,
  get_json_object(request,  '$.messages[0].content') AS user_prompt,
  get_json_object(response, '$.choices[0].message.content') AS model_answer,
  status_code,
  execution_duration_ms
FROM main.monitoring.my_endpoint_payload
WHERE request_date = current_date() - 1
  AND status_code = 200
```

### Production Monitoring with MLflow Scorers

**What it is:** MLflow 3 production monitoring is a service that runs registered scorers (LLM judges, guidelines judges, or custom `@scorer` functions) continuously against incoming traces from a GenAI app, attaching quality assessments to each evaluated trace without requiring a separate offline evaluation run.

**How it works mechanistically:** You register a scorer with a specific MLflow experiment (`.register(name="...")`) and then start it with a sampling configuration (`.start(sampling_config=ScorerSamplingConfig(sample_rate=0.5))`). The monitoring service polls for new traces in that experiment, selects a random sample at the configured rate, runs each scorer function (on serverless compute), and attaches the result as feedback to the trace. At most 20 scorers can be registered per experiment. The scorer function must be defined in a Databricks notebook because the serialization mechanism relies on the notebook environment.

**Where it appears in Databricks:** `mlflow.genai.scorers` module. Built-in scorers: `Safety`, `Guidelines`, `ConversationCompleteness`, `UserFrustration`. Custom scorers use the `@scorer` decorator. Results are visible in the MLflow Experiment UI under the **Traces** tab.

> ⚠️ Fast-evolving: MLflow 3 production monitoring is in Beta as of July 2026. The API (`.register()` / `.start()`) is stable for Beta but may change before GA. Verify at https://docs.databricks.com/aws/en/generative-ai/agent-evaluation/monitoring.html.

### Data Profiling on Inference Tables

**What it is:** Data Profiling (formerly Lakehouse Monitoring) is a Unity Catalog feature that attaches a monitor to any Delta table, computes summary statistics and drift metrics on a schedule, and generates a dashboard and alert rules. For inference tables, the "Inference" profile type is used.

**How it works mechanistically:** You attach a profile to the inference table, specifying `profile_type=InferenceLog`, the timestamp column (`request_time`), and optionally a baseline table (e.g. the validation set used to train the model). On each refresh, Databricks computes metrics over time-based windows (by default the last 30 days) and writes results to two metric tables: a profile metrics table (summary statistics per window) and a drift metrics table (drift vs baseline or vs prior window). The metric tables are ordinary Delta tables — you can query them in SQL, build dashboards, and set alerts.

**Where it appears in Databricks:** UI: Data tab → select the inference table → "Data quality" → "Create profile". API: `databricks.sdk` with `w.data_profiling.create(...)` or the REST API. Docs: `docs.databricks.com/aws/en/data-governance/unity-catalog/data-quality-monitoring/data-profiling`.

---

## Key Parameters / Configuration Knobs

| Parameter | What it controls | Decision rule |
|---|---|---|
| `sampling_fraction` (inference table config) | Fraction of requests logged to the table (0.0–1.0) | Set to 1.0 for endpoints with < 1K RPM; reduce to 0.1–0.3 for high-throughput endpoints (> 10K RPM) to control storage cost while retaining a statistically representative sample |
| `sample_rate` (`ScorerSamplingConfig`) | Fraction of traces sent to MLflow scorers for evaluation | Set to 1.0 for safety-critical scorers (toxicity, PII); set to 0.05–0.2 for expensive LLM-judge scorers; balance coverage vs serverless compute cost |
| `filter_string` (`ScorerSamplingConfig`) | Which traces are eligible for scoring (MLflow search syntax) | Use `"attributes.status = 'OK'"` to skip errored traces; add `timestamp_ms` bounds to re-evaluate only recent traffic after a model update |
| `catalog` / `schema` (inference table location) | Unity Catalog namespace where the `_payload` table is created | Place in the same catalog as your model assets so UC governance policies apply uniformly; use a dedicated schema (e.g. `monitoring`) to separate operational tables from feature tables |
| `profile_type` (Data Profiling) | Which profiling algorithm runs: `TimeSeries`, `Inference`, or `Snapshot` | Always use `Inference` for inference tables — it computes per-model-version metrics and drift relative to a training baseline; `Snapshot` would reprocess the entire table on every refresh |
| `baseline_table` (Data Profiling) | Reference distribution against which drift is measured | Set to the validation/test split used to train the current model version; this makes drift detection meaningful — drift from training distribution signals the model may need retraining |

---

## Worked Example: Requirement → Decision

**Given:** Your team has deployed a RAG-based customer-support chatbot on a Databricks Model Serving endpoint. After two weeks in production, customer escalation rates have risen 15% but your offline evaluation score is unchanged. You need to determine whether the model's answer quality has degraded on real traffic, identify which query types are failing, and set up automated alerting before the next sprint.

**Step 1 — Identify the goal:** Continuously measure answer quality on production traffic, surface degradation by query category, and alert if the weekly average quality score drops below a threshold.

**Step 2 — Define inputs:** The Model Serving endpoint (already deployed), the inference table `main.support_monitoring.chatbot_payload` (already enabled), the offline evaluation dataset used for the pre-deployment benchmark (stored in `main.support_eval.benchmark_v2`).

**Step 3 — Define outputs:** (a) A quality score time-series attached to production traces; (b) a drift metrics table showing distribution shift from the training baseline; (c) an alert that fires if the 7-day rolling quality average falls below 0.70.

**Step 4 — Apply constraints:** Scoring every trace with an LLM judge would cost ~$0.04/trace × 50K daily traces = $2,000/day — too expensive. Logs arrive with up to 1-hour delay. The `@scorer` function must be self-contained (all imports inline) and registered from a notebook.

**Step 5 — Select the approach:**
- Set `sampling_fraction=0.05` on the inference table to reduce logged volume to 2,500 rows/day.
- Use `ScorerSamplingConfig(sample_rate=0.2, filter_string="attributes.status='OK'")` so the LLM judge runs on ~500 traces/day (manageable cost).
- Attach a Data Profiling monitor (`InferenceLog` type) with `baseline_table=main.support_eval.benchmark_v2` to detect input distribution drift automatically.
- Create a Databricks Workflow that runs nightly, queries the drift metrics table, and sends a Slack notification via webhook if `score_7d_avg < 0.70`.

**Rationale vs alternatives:** Using `mlflow.genai.evaluate()` in a nightly batch job on a full inference table sample would work but adds operational overhead (cluster, job code). MLflow production monitoring's serverless execution and trace attachment is lower-maintenance and keeps quality scores co-located with traces for debuggability. Data Profiling handles input distribution drift separately, which the MLflow scorer alone does not detect.

---

## Implementation

```python
# Scenario: Enable AI Gateway inference tables on an existing endpoint
# so that every production request is logged to Unity Catalog for monitoring.
# Constraint: endpoint already deployed; we need zero-downtime config update.

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    AiGateway,
    InferenceTableConfig,
    PutAiGatewayRequest,
)

w = WorkspaceClient()

ENDPOINT_NAME = "chatbot-prod-v2"
CATALOG       = "main"
SCHEMA        = "support_monitoring"

# Enable AI Gateway inference tables via SDK (non-destructive update)
w.serving_endpoints.put_ai_gateway(
    name=ENDPOINT_NAME,
    request=PutAiGatewayRequest(
        inference_table_config=InferenceTableConfig(
            catalog_name=CATALOG,
            schema_name=SCHEMA,
            table_name_prefix=ENDPOINT_NAME,
            enabled=True,
        )
    ),
)
print(f"Inference table will land at: {CATALOG}.{SCHEMA}.{ENDPOINT_NAME}_payload")
```

```python
# Scenario: Register and start an LLM safety scorer on production traces
# so that 100% of requests are automatically evaluated for safety violations.
# Constraint: must be defined and registered from a Databricks notebook.

from mlflow.genai.scorers import Safety, ScorerSamplingConfig

# Step 1: register the scorer with this experiment
safety_judge = Safety().register(name="production_safety_check")

# Step 2: start continuous monitoring at 100% sample rate for safety (critical)
safety_judge = safety_judge.start(
    sampling_config=ScorerSamplingConfig(
        sample_rate=1.0,
        filter_string="attributes.status = 'OK'"
    )
)
print("Safety scorer registered and running. Results appear in Traces tab after ~15 min.")
```

```python
# Scenario: Sample production inference table rows and run a custom
# quality scorer for RAG answer completeness against the offline benchmark.
# Constraint: scorer must be self-contained (all imports inline).

from mlflow.genai.scorers import scorer, ScorerSamplingConfig

@scorer
def rag_completeness(inputs, outputs):
    """
    Returns 1.0 if the response contains at least one sentence from the retrieved
    context, 0.0 otherwise. Heuristic proxy for retrieval utilization.
    """
    import re
    response = (outputs or {}).get("response", "") or ""
    context  = (inputs  or {}).get("context",  "") or ""
    if not context or not response:
        return 0.0
    # Check if any sentence fragment from context appears verbatim
    sentences = [s.strip() for s in re.split(r'[.!?]', context) if len(s.strip()) > 20]
    for sentence in sentences[:5]:  # check top-5 context sentences
        if sentence[:40] in response:
            return 1.0
    return 0.0

completeness_scorer = rag_completeness.register(name="rag_completeness_v1")
completeness_scorer = completeness_scorer.start(
    sampling_config=ScorerSamplingConfig(sample_rate=0.2)
)
```

```sql
-- Scenario: Extract prompt and response from the inference table's nested JSON
-- to build a quality review dataset. Constraint: must work in DBSQL without a cluster.

SELECT
    databricks_request_id,
    request_time,
    get_json_object(request,  '$.messages[0].content')        AS user_prompt,
    get_json_object(response, '$.choices[0].message.content') AS model_answer,
    status_code,
    execution_duration_ms,
    requester
FROM main.support_monitoring.chatbot_prod_v2_payload
WHERE request_date BETWEEN current_date() - 7 AND current_date()
  AND status_code = 200
  AND get_json_object(request, '$.messages[0].content') IS NOT NULL
ORDER BY request_time DESC
LIMIT 500;
```

```python
# Anti-pattern: Querying the inference table directly inside the production
# request handler to implement real-time quality checks.
# What breaks: inference table rows land with up to 1-hour latency — the row
# for request N does not exist when request N is being answered. This pattern
# will silently find no rows and produce incorrect "no quality issues" results.

# WRONG — do not do this in the serving code:
def handle_request(user_input):
    response = model.predict(user_input)
    # BUG: the row for this request won't exist yet
    recent_rows = spark.sql("SELECT * FROM payload_table WHERE request_time > now() - interval 1 minute")
    if recent_rows.count() == 0:
        log.info("No quality issues detected")   # always true, always wrong
    return response

# Correct approach: treat inference tables as an asynchronous batch audit log.
# Quality checks run in a SEPARATE scheduled job (Databricks Workflow / Lakeflow Job)
# that reads from the table on a schedule (e.g. hourly or daily).
# Real-time guardrails belong in AI Gateway service policies, not inference table reads.

def scheduled_quality_job():
    # Run on a schedule, not in the hot path
    import mlflow
    rows = spark.sql("""
        SELECT databricks_request_id,
               get_json_object(request, '$.messages[0].content') AS prompt,
               get_json_object(response, '$.choices[0].message.content') AS answer
        FROM main.support_monitoring.chatbot_prod_v2_payload
        WHERE request_date = current_date() - 1
          AND status_code = 200
    """).limit(1000)
    # ... run mlflow.genai.evaluate() or custom scorers here
```

```python
# Scenario: Attach a Data Profiling monitor to the inference table to detect
# input distribution drift vs the training baseline.
# Constraint: must use the Inference profile type; baseline is the validation set.

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    MonitorInferenceLog,
    CreateMonitor,
    MonitorTimeSeries,
)

w = WorkspaceClient()

TABLE_NAME     = "main.support_monitoring.chatbot_prod_v2_payload"
BASELINE_TABLE = "main.support_eval.benchmark_v2"
OUTPUT_SCHEMA  = "main.support_monitoring"

w.quality_monitors.create(
    table_name=TABLE_NAME,
    inference_log=MonitorInferenceLog(
        timestamp_col="request_time",
        model_id_col="served_entity_id",
        prediction_col="response",      # raw JSON — post-process in custom metrics
        problem_type="question_answering",
    ),
    output_schema_name=OUTPUT_SCHEMA,
    baseline_table_name=BASELINE_TABLE,
    slicing_exprs=["requester"],        # drift metrics broken down per user/team
)
print(f"Profile created. Metric tables in: {OUTPUT_SCHEMA}")
```

---

## Common Pitfalls & Misconceptions

- **Treating inference table availability as real-time** — Engineers familiar with streaming pipelines assume a logged request is immediately queryable. Inference table delivery is asynchronous and best-effort, with up to 1-hour latency; building downstream logic that expects sub-minute freshness will silently fail or produce empty results.

- **Using `timestamp_ms` column from the legacy schema on a new AI Gateway table** — The AI Gateway schema replaced `timestamp_ms` (epoch long) with `request_time` (TIMESTAMP) and `request_date` (DATE); queries that reference `timestamp_ms` on an AI Gateway-enabled table will throw a column-not-found error, which is confusing because the same endpoint name is used.

- **Registering an MLflow scorer in a Python file instead of a notebook** — The production monitoring service serializes scorer functions for remote execution in a serverless environment; this serialization mechanism requires the notebook environment. Scorers defined in local `.py` files or IDEs cannot be registered and will raise a serialization error at `.register()` time.

- **Sampling at the inference table level AND the scorer level without accounting for compounding** — If `sampling_fraction=0.1` is set on the endpoint (logging 10% of traffic) and `sample_rate=0.1` is set on the scorer, only 1% of actual requests are evaluated. This is often far too sparse to detect meaningful quality trends on moderate-volume endpoints; calculate effective coverage before choosing both parameters.

- **Forgetting that `request` and `response` are raw JSON strings, not structs** — Beginners often try to access `payload.request.messages` as if `request` is a nested struct. It is a STRING; every field extraction requires `get_json_object(request, '$.messages[0].content')` or a full schema declaration with `from_json()`.

- **Using the legacy `auto_capture_config` API for new endpoints** — The legacy inference table format is archived and unsupported for new provisioned throughput and Foundation Model API endpoints. New endpoints must use the AI Gateway `inference_table_config` path. Using the archived API on a new endpoint will silently produce no table.

---

## Key Definitions

| Term | Definition |
|---|---|
| Inference table | A Unity Catalog Delta table automatically populated with the raw JSON request and response payloads for every call to a Model Serving endpoint, used as the primary production audit log for GenAI monitoring |
| AI Gateway | The Databricks governance layer that sits in front of Model Serving endpoints, adding identity tracking (`requester`), rate limits, guardrails (service policies), and the current (non-legacy) inference table schema |
| `sampling_fraction` | The fraction of endpoint requests logged to the inference table (0.0–1.0); recorded in the table row so downstream analytics can weight-correct the sample back to full-traffic estimates |
| Data Profiling (Inference profile) | A Unity Catalog feature that attaches a statistical monitor to a Delta table, computing time-windowed summary statistics and drift metrics; the Inference profile type is designed for request logs with a timestamp, a model ID, and optionally a baseline training distribution |
| `ScorerSamplingConfig` | The MLflow 3 configuration object that controls what fraction of traces (`sample_rate`) and which traces (`filter_string`) are sent to a registered scorer for production quality assessment |
| `@scorer` decorator | The MLflow 3 function decorator that marks a Python function as a scorer; required for custom scorers used in production monitoring (class-based `Scorer` subclasses cannot be serialized for remote execution) |
| `databricks_request_id` | A system-generated unique string identifier for each Model Serving request; the join key for linking inference table rows to MLflow traces, ground-truth labels, and assessment logs |
| `served_entity_id` | The unique ID of the served entity (model version / agent deployment) in the AI Gateway inference table schema; used to join with `system.serving.served_entities` for model metadata |
| Drift metrics table | A Delta table generated by Data Profiling that records statistical distance (e.g. Jensen-Shannon divergence) between the current time window and the baseline distribution, per column, per time slice |

---

## Summary / Quick Recall

- Inference tables automatically log every Model Serving request/response to a Unity Catalog Delta table; enable via AI Gateway section in the Serving UI.
- The current schema (AI Gateway-enabled) adds `requester`, `served_entity_id`, `logging_error_codes`, and uses `request_time` (TIMESTAMP) instead of `timestamp_ms` (LONG).
- `request` and `response` are raw JSON STRING columns — always use `get_json_object()` or `from_json()` to extract fields.
- Log delivery is asynchronous and best-effort, up to ~1 hour; never use inference tables for real-time quality checks in the hot path.
- MLflow production monitoring runs scorers on a configurable sample of traces; register from a Databricks notebook, start with `ScorerSamplingConfig`.
- Data Profiling (`InferenceLog` profile) detects input distribution drift vs a training baseline and generates dashboards and alert rules automatically.
- Set `sampling_fraction` (endpoint-level) and `sample_rate` (scorer-level) carefully — they multiply; 10% × 10% = 1% effective coverage.

---

## Self-Check Questions

1. Which column in the AI Gateway-enabled inference table schema records the identity of the caller making the request?

   <details><summary>Answer</summary>

   The correct answer is **`requester`**. This column, present in the AI Gateway schema but absent from the legacy schema, stores the user or service principal ID whose permissions are used for the invocation. The distractor is `databricks_request_id`, which is a system-generated request identifier (a UUID-like string), not an identity. `client_request_id` is an optional caller-supplied correlation ID, not an authenticated identity.

   </details>

2. Your team enables inference table logging at `sampling_fraction=0.05` on a high-throughput endpoint, then registers an MLflow safety scorer with `sample_rate=0.5`. What fraction of actual endpoint requests will be evaluated by the safety scorer?

   <details><summary>Answer</summary>

   **2.5%** (0.05 × 0.5 = 0.025). The `sampling_fraction` controls how many requests are written to the inference table; the `sample_rate` controls what fraction of *those logged traces* are sent to the scorer. The two rates are multiplicative, not additive. A common mistake is to assume `sample_rate` applies to all traffic — it applies only to traces that were already logged. For a safety-critical scorer, you should raise `sampling_fraction` closer to 1.0 or set `sample_rate=1.0` while lowering `sampling_fraction` only to the minimum acceptable for storage cost.

   </details>

3. **Which TWO** of the following correctly describe constraints that apply to custom `@scorer` functions used in MLflow 3 production monitoring?
   - A. The scorer must be defined and registered from a Databricks notebook.
   - B. The scorer can reference Python objects or modules defined outside the function body.
   - C. All imports required by the scorer must be done inline, inside the function body.
   - D. Class-based `Scorer` subclasses are supported as an alternative to the `@scorer` decorator.
   - E. The scorer can be defined in a local `.py` file and uploaded to DBFS before registering.

   <details><summary>Answer</summary>

   The correct answers are **A and C**. The monitoring service serializes the scorer function code for remote execution on serverless compute, and this serialization requires the Databricks notebook environment (A). Because the function is serialized as code, all imports must be inline inside the function body — external references are not captured (C). Option B is wrong: external references (variables, modules outside the function) are not captured during serialization and will cause runtime errors. Option D is wrong: class-based `Scorer` subclasses cannot be serialized for remote execution; only `@scorer` decorator functions are supported. Option E is wrong: local `.py` files and DBFS uploads do not satisfy the notebook environment requirement.

   </details>

4. A team discovers that their inference table's `status_code` column shows 200 for all logged rows, but customers report intermittent wrong answers. The team's first instinct is that the inference table must be missing error rows. What is the more likely explanation and what should they investigate?

   <details><summary>Answer</summary>

   The more likely explanation is that the model is returning **semantically incorrect but syntactically valid responses** — requests that succeed at the HTTP level (status 200) but produce wrong or hallucinated answers. The inference table faithfully records status_code=200 because the endpoint did not throw an error. The team should: (1) extract `response` text using `get_json_object()` and run quality scorers (e.g. an LLM judge for correctness or a RAG completeness scorer) over the logged rows; (2) check Data Profiling drift metrics for shifts in `execution_duration_ms` (unusually fast responses may indicate the model is not retrieving context); (3) look at `sampling_fraction` to confirm they are actually capturing enough traffic. The distractor reasoning — that inference tables hide 4xx/5xx rows — is a real limitation (errors may not be logged), but it does not explain wrong answers on successful requests.

   </details>

5. You are designing a production monitoring architecture for a GenAI app expecting 100K daily requests. You have two options: (A) run `mlflow.genai.evaluate()` nightly on a 10% random sample of the previous day's inference table rows; (B) use MLflow production monitoring with a registered LLM judge at `sample_rate=0.1`. What is the key operational difference, and when would you prefer option A?

   <details><summary>Answer</summary>

   **Key difference:** Option B (production monitoring) is lower-maintenance — the monitoring service handles scheduling, serverless execution, and attaches results directly to MLflow traces, making them queryable in the Experiment UI. Option A (nightly batch job) requires maintaining a Databricks Workflow with cluster provisioning or serverless job code, and results are stored separately from the traces (typically in a Delta table you manage). **Prefer option A when:** you need to use `mlflow.genai.evaluate()` with a ground-truth dataset (i.e. you have labels and need accuracy/F1-style metrics, not just LLM judge scores); when you need to join inference table rows with a separate labels table before scoring; or when your scoring logic is complex enough that it cannot be expressed in a self-contained `@scorer` function (e.g. requires reading from another Delta table). The distractor is assuming option A is always better because it is "more controlled" — for standard quality monitoring without ground truth, option B is simpler and more scalable.

   </details>

---

## Further Reading

- [Monitor served models using AI Gateway-enabled inference tables](https://docs.databricks.com/aws/en/ai-gateway/inference-tables-serving-endpoints) — *verified 2026-07-16* — Current (non-legacy) inference table schema, enable/disable instructions, sampling configuration, and limitations
- [Monitor GenAI apps in production](https://docs.databricks.com/aws/en/generative-ai/agent-evaluation/monitoring.html) — *verified 2026-07-16* — MLflow 3 production monitoring: scorer registration, sampling config, multi-turn judges, troubleshooting serialization issues
- [Data profiling (Lakehouse Monitoring)](https://docs.databricks.com/aws/en/data-governance/unity-catalog/data-quality-monitoring/data-profiling/) — *verified 2026-07-16* — Profile types (Inference/TimeSeries/Snapshot), metric tables, dashboard, alert setup
- [AI governance with Unity AI Gateway](https://docs.databricks.com/aws/en/ai-gateway/) — *verified 2026-07-16* — Unity AI Gateway overview: access control, traffic management, inference table logging, usage tracking system tables
- [Inference tables for monitoring and debugging models (archived legacy)](https://docs.databricks.com/aws/en/machine-learning/model-serving/inference-tables.html) — *verified 2026-07-16* — Legacy `auto_capture_config` schema and migration path to AI Gateway inference tables
