# LAB-21: Inference Tables and AI Gateway Monitoring

**Lab:** LAB-21 | **Section:** 07 Evaluation and Monitoring | **Module:** 02 Monitoring | **Est. time:** 1.5 hrs

> **Simulation mode:** This lab is illustrative — it does not require a live Databricks cluster or active endpoint. All SDK calls are shown with realistic parameters; the SQL and Python blocks can be run verbatim in a Databricks notebook when a cluster and endpoint are available. Expected output is annotated in comments.

---

## Objective

Enable AI Gateway inference tables on a Model Serving endpoint, send sample requests, query and parse the resulting Delta table in SQL, sample rows for offline evaluation with MLflow scorers, and attach a Data Profiling monitor for continuous drift detection.

---

## Prerequisites

- Completed LAB-20 (deploying a Model Serving endpoint) or have an existing endpoint available
- Unity Catalog enabled in the workspace; `USE CATALOG`, `USE SCHEMA`, `CREATE TABLE` permissions on your target catalog/schema
- `databricks-sdk>=0.20.0` and `mlflow>=3.0.0` installed in the notebook environment
- Familiarity with Delta SQL and Python notebook environments

---

## Setup

```python
# Setup: install required packages and configure workspace client
# Scenario: preparing a fresh notebook environment for the lab

%pip install databricks-sdk>=0.20 mlflow>=3.0 --quiet
dbutils.library.restartPython()
```

```python
# Setup: define lab constants — adapt to your environment
# Scenario: centralising all configuration to make the lab portable

CATALOG        = "main"
SCHEMA         = "lab21_monitoring"
ENDPOINT_NAME  = "lab21-rag-chatbot"            # existing or newly created endpoint
TABLE_PREFIX   = ENDPOINT_NAME.replace("-", "_")
PAYLOAD_TABLE  = f"{CATALOG}.{SCHEMA}.{TABLE_PREFIX}_payload"
BASELINE_TABLE = f"{CATALOG}.{SCHEMA}.baseline_eval_set"

print(f"Inference table will be: {PAYLOAD_TABLE}")
```

```sql
-- Setup: create the target schema if it doesn't exist
-- Scenario: provision the Unity Catalog namespace before enabling inference tables

CREATE SCHEMA IF NOT EXISTS main.lab21_monitoring
  COMMENT 'LAB-21 monitoring namespace for inference table and profiling outputs';
```

---

## Steps

### Step 1 — Enable inference tables on a Model Serving endpoint

```python
# Scenario: Enable AI Gateway inference tables on an existing endpoint so that
# every production request is logged to Unity Catalog for monitoring and debugging.
# Constraint: uses the current (non-legacy) AI Gateway API, not auto_capture_config.

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    AiGateway,
    InferenceTableConfig,
    PutAiGatewayRequest,
)

w = WorkspaceClient()

response = w.serving_endpoints.put_ai_gateway(
    name=ENDPOINT_NAME,
    request=PutAiGatewayRequest(
        inference_table_config=InferenceTableConfig(
            catalog_name=CATALOG,
            schema_name=SCHEMA,
            table_name_prefix=TABLE_PREFIX,
            enabled=True,
        )
    ),
)

print(f"AI Gateway updated. Inference table: {PAYLOAD_TABLE}")
print(f"State: {response.inference_table_config.state}")
# Expected output:
# AI Gateway updated. Inference table: main.lab21_monitoring.lab21_rag_chatbot_payload
# State: ACTIVE
```

**Verify in the UI:** Navigate to Serving → your endpoint → AI Gateway section. Confirm "Inference table: Enabled" and the table path shown matches `PAYLOAD_TABLE`.

---

### Step 2 — Send sample requests and verify logging

```python
# Scenario: Send ten realistic chatbot requests to the endpoint to populate
# the inference table with sample rows for the rest of the lab.
# Constraint: at-least-once delivery — rows appear within ~1 hour of the request.

import time
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

sample_prompts = [
    "What is the return policy for enterprise software licenses?",
    "How do I upgrade from version 3.1 to 3.2 without downtime?",
    "My invoice shows a line item labeled 'DBU-ML' — what does that mean?",
    "Can I use the API with a service principal instead of a personal token?",
    "What SLA applies to the support tier I'm on?",
    "Error code 429 appeared in the logs last night — what triggers it?",
    "How many seats does the team license include?",
    "Is there a sandbox environment for testing before production deployment?",
    "Where can I find the audit log for my workspace?",
    "What happens to my data if I cancel the subscription?",
]

for prompt in sample_prompts:
    w.serving_endpoints.query(
        name=ENDPOINT_NAME,
        messages=[ChatMessage(role=ChatMessageRole.USER, content=prompt)],
    )
    time.sleep(0.5)  # avoid rate limiting

print(f"Sent {len(sample_prompts)} requests. Rows will appear in {PAYLOAD_TABLE} within ~1 hour.")
```

---

### Step 3 — Query the inference table in SQL (parse nested JSON)

```sql
-- Scenario: Inspect the raw inference table to understand its structure before
-- building monitoring queries. Parse nested JSON request/response fields.
-- Constraint: request and response are STRING columns, not structs.

-- 3a. Raw table inspection
SELECT
    databricks_request_id,
    request_time,
    status_code,
    execution_duration_ms,
    requester,
    LEFT(request,  120) AS request_preview,
    LEFT(response, 120) AS response_preview
FROM main.lab21_monitoring.lab21_rag_chatbot_payload
ORDER BY request_time DESC
LIMIT 10;
```

```sql
-- Scenario: Extract the user prompt and model answer from nested JSON for
-- human review and downstream quality evaluation.

-- 3b. Parse chat-completion JSON structure
SELECT
    databricks_request_id,
    request_time,
    get_json_object(request,  '$.messages[0].content')          AS user_prompt,
    get_json_object(response, '$.choices[0].message.content')   AS model_answer,
    get_json_object(response, '$.usage.prompt_tokens')          AS prompt_tokens,
    get_json_object(response, '$.usage.completion_tokens')      AS completion_tokens,
    CAST(get_json_object(response, '$.usage.total_tokens') AS INT)
                                                                AS total_tokens,
    status_code,
    execution_duration_ms
FROM main.lab21_monitoring.lab21_rag_chatbot_payload
WHERE request_date = current_date()
  AND status_code = 200
ORDER BY request_time DESC;
```

```sql
-- Scenario: Identify anomalous requests — unusually fast responses may indicate
-- retrieval skipped (no context injected); long responses may need review.

-- 3c. Anomaly flagging query
SELECT
    databricks_request_id,
    request_time,
    execution_duration_ms,
    CASE
        WHEN execution_duration_ms < 300  THEN 'suspiciously_fast'
        WHEN execution_duration_ms > 8000 THEN 'slow'
        ELSE 'normal'
    END AS latency_flag,
    CAST(get_json_object(response, '$.usage.completion_tokens') AS INT) AS completion_tokens,
    get_json_object(request,  '$.messages[0].content') AS user_prompt
FROM main.lab21_monitoring.lab21_rag_chatbot_payload
WHERE request_date >= current_date() - 7
  AND status_code = 200
HAVING latency_flag IN ('suspiciously_fast', 'slow')
ORDER BY execution_duration_ms;
```

---

### Step 4 — Sample production requests for offline evaluation with MLflow scorers

```python
# Scenario: Register and start an LLM safety scorer and a custom RAG completeness
# scorer to continuously assess production traces, without evaluating every request.
# Constraint: all scorers must be registered from a Databricks notebook.

import mlflow
from mlflow.genai.scorers import Safety, Guidelines, ScorerSamplingConfig, scorer

# 4a. Register built-in Safety scorer at 100% sample rate (critical coverage)
safety_judge = Safety().register(name="lab21_safety")
safety_judge = safety_judge.start(
    sampling_config=ScorerSamplingConfig(
        sample_rate=1.0,
        filter_string="attributes.status = 'OK'"
    )
)
print("Safety scorer registered and running.")

# 4b. Register a Guidelines scorer at 50% sample rate (moderately expensive)
english_judge = Guidelines(
    name="english_only",
    guidelines=["The response must be in English and use professional language."]
).register(name="lab21_english_professional")

english_judge = english_judge.start(
    sampling_config=ScorerSamplingConfig(sample_rate=0.5)
)
print("English/professional tone scorer registered.")
```

```python
# Scenario: Define a custom RAG completeness scorer that checks whether the model
# response is substantively longer than a trivial acknowledgement.
# Constraint: self-contained — all imports must be inside the function body.

from mlflow.genai.scorers import scorer, ScorerSamplingConfig

@scorer
def response_substantiveness(outputs):
    """
    Heuristic: a response with fewer than 50 words is likely a refusal or
    a retrieval failure. Returns 1.0 (substantive) or 0.0 (thin response).
    """
    import re
    response = (outputs or {}).get("response", "") or ""
    if isinstance(response, dict):
        # Handle cases where response is already parsed
        response = response.get("content", "")
    word_count = len(re.findall(r'\w+', str(response)))
    return 1.0 if word_count >= 50 else 0.0

substantiveness_scorer = response_substantiveness.register(name="lab21_substantiveness")
substantiveness_scorer = substantiveness_scorer.start(
    sampling_config=ScorerSamplingConfig(sample_rate=0.3)
)
print("Substantiveness scorer registered at 30% sample rate.")
print("Allow 15-20 minutes for first results to appear in the Traces tab.")
```

```python
# Anti-pattern: Registering a scorer that references an external variable or
# module defined outside the function body.
# What breaks: the production monitoring service serializes the function code
# for remote serverless execution; external references are NOT captured and
# cause a runtime NameError when the scorer runs on a production trace.

import re  # defined outside the function — WRONG

# WRONG: this scorer will fail silently or raise NameError in production
@scorer
def bad_scorer(outputs):
    response = (outputs or {}).get("response", "")
    return 1.0 if len(re.findall(r'\w+', response)) > 50 else 0.0  # re not in scope

# Correct: import inside the function body
@scorer
def good_scorer(outputs):
    import re  # CORRECT: inline import, captured during serialization
    response = (outputs or {}).get("response", "")
    return 1.0 if len(re.findall(r'\w+', response)) > 50 else 0.0
```

---

### Step 5 — Set up a basic quality monitor using Data Profiling

```python
# Scenario: Attach a Data Profiling Inference monitor to the inference table
# so that Databricks automatically computes drift metrics vs the training baseline
# and generates a dashboard for ongoing quality tracking.

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    MonitorInferenceLog,
    CreateMonitor,
)

w = WorkspaceClient()

# Create baseline evaluation set (in a real scenario this is your validation split)
# Here we create a minimal representative table for illustration
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {BASELINE_TABLE}
    COMMENT 'Baseline evaluation set for Lab 21 — mirrors inference table schema'
    AS SELECT
        CAST(NULL AS TIMESTAMP)  AS request_time,
        CAST(NULL AS STRING)     AS served_entity_id,
        CAST(NULL AS STRING)     AS response,
        CAST(NULL AS INT)        AS status_code,
        CAST(NULL AS BIGINT)     AS execution_duration_ms
    WHERE 1=0
""")

# Attach an Inference profile to the payload table
monitor = w.quality_monitors.create(
    table_name=PAYLOAD_TABLE,
    inference_log=MonitorInferenceLog(
        timestamp_col="request_time",
        model_id_col="served_entity_id",
        prediction_col="response",
        problem_type="question_answering",
    ),
    output_schema_name=f"{CATALOG}.{SCHEMA}",
    baseline_table_name=BASELINE_TABLE,
    slicing_exprs=["requester"],   # break down drift metrics per calling identity
)

print(f"Data Profiling monitor created on {PAYLOAD_TABLE}")
print(f"Profile metrics table: {monitor.profile_metrics_table_name}")
print(f"Drift metrics table:   {monitor.drift_metrics_table_name}")
```

```python
# Scenario: Trigger an on-demand refresh of the Data Profiling monitor
# to compute metrics immediately (normally runs on a schedule).

w.quality_monitors.run_refresh(table_name=PAYLOAD_TABLE)
print("Refresh triggered. Navigate to the table in Catalog Explorer")
print("→ 'Data quality' tab to view the auto-generated dashboard.")
```

```sql
-- Scenario: Query the drift metrics table to check for input distribution shift.
-- Use this as the basis for an alerting query in a Databricks Workflow.

SELECT
    window_start_time,
    column_name,
    metric_name,
    metric_value,
    baseline_metric_value,
    ABS(metric_value - baseline_metric_value) AS drift_magnitude
FROM main.lab21_monitoring.lab21_rag_chatbot_payload_drift_metrics
WHERE metric_name = 'js_distance'    -- Jensen-Shannon divergence vs baseline
  AND column_name IN ('response', 'execution_duration_ms')
ORDER BY window_start_time DESC, drift_magnitude DESC
LIMIT 20;
```

---

## Validation

```python
# Validation: confirm inference table exists and has the expected columns

from pyspark.sql.functions import col

df = spark.table(PAYLOAD_TABLE)

expected_columns = {
    "databricks_request_id", "request_time", "request_date",
    "status_code", "execution_duration_ms", "request", "response",
    "requester", "served_entity_id", "sampling_fraction",
}

actual_columns = set(df.columns)
missing = expected_columns - actual_columns

if missing:
    print(f"FAIL — missing columns: {missing}")
else:
    row_count = df.count()
    print(f"PASS — all expected columns present. Row count: {row_count}")
    if row_count == 0:
        print("INFO — table is empty. Rows appear within ~1 hour of requests.")
```

```sql
-- Validation: confirm scorers are registered on the experiment

SELECT * FROM mlflow.genai.production_scorers
WHERE experiment_id = (
    SELECT experiment_id FROM mlflow.experiments
    WHERE name LIKE '%lab21%'
    ORDER BY last_update_time DESC
    LIMIT 1
)
-- Expected: rows for lab21_safety, lab21_english_professional, lab21_substantiveness
-- with status = 'ACTIVE'
```

---

## Teardown

```python
# Teardown: disable inference tables and delete the monitoring schema
# to avoid incurring ongoing storage and compute costs after the lab.

# 1. Disable inference tables on the endpoint
w.serving_endpoints.put_ai_gateway(
    name=ENDPOINT_NAME,
    request=PutAiGatewayRequest(
        inference_table_config=InferenceTableConfig(enabled=False)
    ),
)
print("Inference tables disabled on endpoint.")

# 2. Stop registered scorers (prevent serverless charges)
for scorer_name in ["lab21_safety", "lab21_english_professional", "lab21_substantiveness"]:
    try:
        registered = mlflow.genai.get_scorer(scorer_name)
        registered.stop()
        print(f"Stopped scorer: {scorer_name}")
    except Exception as e:
        print(f"Could not stop {scorer_name}: {e}")

# 3. Drop the monitoring schema (drops all tables including inference table)
# WARNING: this is irreversible — all logged data will be lost
spark.sql(f"DROP SCHEMA IF EXISTS {CATALOG}.{SCHEMA} CASCADE")
print(f"Schema {CATALOG}.{SCHEMA} dropped. Lab teardown complete.")
```

---

## Reflection Questions

1. The lab set `sampling_fraction=1.0` (default) on the inference table. At what daily request volume would you start reducing this, and what would you set it to? What statistical guarantees do you lose by sampling, and how would you correct for them in downstream analytics?

2. The custom `response_substantiveness` scorer is a simple word-count heuristic. What are three failure modes for this heuristic that a real RAG application would expose? How would you replace it with a more robust LLM-judge-based scorer, and what would the cost tradeoff be at 10K evaluated traces/day?

3. The lab attached the Data Profiling monitor with `slicing_exprs=["requester"]`. In a production scenario where `requester` values are service principal IDs (not human-readable names), how would you map these to business units for cost attribution? Which Databricks tables would you join, and what governance considerations apply?
