# Evaluation vs. Monitoring: The GenAI Application Lifecycle

**Section:** 04 Application Development | **Module:** 03 Agents, MLflow & Genie | **Est. time:** 3 hrs | **Exam mapping:** Application Development 30% — evaluate, deploy, and monitor AI agents across the full production lifecycle

---

## TL;DR

GenAI applications require two distinct quality assurance mechanisms that operate at different points in time: offline evaluation runs `mlflow.evaluate()` against a fixed golden dataset before deployment to gate model registration, while online monitoring continuously scores sampled live traffic after deployment using the same LLM judges running asynchronously over inference table rows. Inference tables (AI Gateway-enabled Delta Tables) automatically capture every request, response, latency, and token count from a Model Serving endpoint — they are the prerequisite for any production quality signal. The Evaluation → Monitoring feedback loop closes the lifecycle: production regressions surface new failure clusters, those clusters enrich the golden eval set, and the improved model is validated offline before re-deployment. **The one thing to remember: offline eval is a snapshot in time; only online monitoring over inference tables can catch the silent quality degradation that starts the day you deploy.**

---

## ELI5 — Explain It Like I'm 5

Imagine you are a chef who opens a new restaurant. Before you open, you give your dish to five food critics and ask them to score it — that is offline evaluation, a controlled test before anything is served to real customers. Once you open, some real diners love the food, but a few weeks later you notice that customers ordering the new seasonal menu are leaving half the dish uneaten. Nobody filed a formal complaint; no one walked out — you only notice because your manager quietly tracks every table's leftovers in a notebook. That notebook is the inference table, and the process of reading it every morning to spot patterns is online monitoring. The most common misconception is thinking the critic scores from before opening tell you everything you need to know about how real customers experience the restaurant — they do not, because real diners order dishes the critics never tried, and the kitchen staff change over time. Both the pre-opening critics and the daily leftovers notebook are necessary; neither alone is enough.

---

## Learning Objectives

By the end of this chapter you will be able to:

- [ ] Distinguish offline evaluation (`mlflow.evaluate()` on a golden dataset) from online monitoring (LLM judges running asynchronously over inference table rows) and explain what each mechanism can and cannot detect
- [ ] Enable AI Gateway-enabled inference tables on a Model Serving endpoint using both the Databricks UI and the SDK, and identify the columns captured in the resulting Delta Table
- [ ] Configure data profiling (formerly Lakehouse Monitoring) on an inference table using the `databricks.sdk` Python API with `InferenceLog` or `TimeSeries` profile types to detect quality drift
- [ ] Design an A/B test for two agent versions using `traffic_config` endpoint weights and explain the statistical significance requirement before declaring a winner
- [ ] Describe the closed-loop Evaluation → Monitoring feedback cycle and identify which Databricks services participate at each stage

---

## Visual Overview

### GenAI Application Lifecycle

```
┌────────────────────────────────────────────────────────────────────┐
│                  DEVELOPMENT (OFFLINE)                             │
│                                                                    │
│  Build agent  ──►  mlflow.evaluate()  ──►  Pass quality gate?      │
│                    (golden dataset)        │                       │
│                                           ├── No  ──►  Iterate    │
│                                           └── Yes ──►  Register   │
└───────────────────────────────────────┬────────────────────────────┘
                                        │ mlflow.register_model()
                                        ▼
┌────────────────────────────────────────────────────────────────────┐
│                  DEPLOYMENT                                        │
│                                                                    │
│  Unity Catalog Model Registry  ──►  Model Serving Endpoint         │
│  (@champion alias)                  (AI Gateway inference tables   │
│                                      enabled)                      │
└───────────────────────────────────────┬────────────────────────────┘
                                        │ live traffic
                                        ▼
┌────────────────────────────────────────────────────────────────────┐
│                  PRODUCTION (ONLINE MONITORING)                    │
│                                                                    │
│  Inference Table (Delta)  ──►  Data Profiling  ──►  Drift Alert    │
│  (requests, responses,         (InferenceLog        │              │
│   latency, tokens)              profile)            │              │
│                                                     ▼              │
│                             Add failure clusters to golden set     │
│                             ──► Re-run mlflow.evaluate() ──►       │
│                             Pass? ──► Register new version         │
└────────────────────────────────────────────────────────────────────┘
```

### Inference Table Data Flow

```
  User Request
       │
       ▼
┌─────────────────────────────────────┐
│     Model Serving Endpoint          │
│  (AI Gateway inference table on)    │
│                                     │
│  ┌─────────────────────────────┐   │
│  │     Agent / LLM             │   │
│  │  (LangGraph / LangChain)    │   │
│  └─────────────┬───────────────┘   │
└────────────────┼────────────────────┘
                 │ response + metadata
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  AI Gateway-enabled Inference Table (Unity Catalog Delta)   │
│                                                             │
│  databricks_request_id │ request_time │ status_code         │
│  execution_duration_ms │ request (JSON) │ response (JSON)   │
│  sampling_fraction     │ served_entity_id │ trace           │
└──────────────────────────────────┬──────────────────────────┘
                                   │
                                   ▼
                     Data Profiling (InferenceLog)
                     ──► Profile metric table
                     ──► Drift metric table
                     ──► Auto-generated dashboard
                     ──► Alerts on threshold breach
```

### A/B Testing Traffic Split

```
  Incoming traffic (100%)
          │
          ▼
┌─────────────────────────────────────────────┐
│           Model Serving Endpoint            │
│         traffic_config weights              │
│                                             │
│  90% ──► Champion served entity            │
│           ──► champion_payload table        │
│                                             │
│  10% ──► Challenger served entity          │
│           ──► challenger_payload table      │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
          Compare in MLflow Experiments UI
          (groundedness, correctness scores)
          Declare winner when p < 0.05
          over ≥ 1,000 requests per arm
```

---

## Key Concepts

### Offline Evaluation (Pre-deployment)

**What is it?** Offline evaluation is the practice of running `mlflow.evaluate()` against a fixed golden dataset before shipping a new model version, producing quantitative pass/fail scores on LLM-judge metrics to gate model registration in the CI/CD pipeline.

**How does it work mechanistically?** The engineer constructs a Pandas DataFrame with columns `request`, `expected_response`, and optionally `retrieved_context`, then calls `mlflow.evaluate(model=..., data=eval_df, model_type="databricks-agent")`. For each row, Databricks-hosted LLM judges evaluate groundedness (is the answer supported by the retrieved context?), answer correctness (does the answer match the expected response?), and chunk relevance (did the retriever fetch the right documents?). Each judge returns a binary yes/no score plus a written rationale; aggregate percentages (e.g., `agent/percent_grounded`) are logged as MLflow run metrics. The evaluation run links back to the registered model version so the scores are traceable to the exact artifact being gated.

**Where does it appear in Databricks?** Called inside a Databricks notebook or CI/CD job before `mlflow.register_model()`. Results appear under the **Metrics** and **Tables** tabs of the MLflow Experiment run; `result.tables["eval_results"]` returns the per-row scored DataFrame directly in the notebook. The Experiments UI supports side-by-side comparison of multiple evaluation runs to track improvement across versions.

---

### Online Monitoring (Post-deployment)

**What is it?** Online monitoring is the process of sampling live production traffic and scoring it asynchronously after deployment using the same LLM judges used during offline evaluation, without requiring any `expected_response` ground-truth labels.

**How does it work mechanistically?** Unlike offline evaluation, online monitoring operates on the inference table rather than a curated eval set. A data profiling job runs on a schedule (using serverless compute), reads newly arrived rows from the inference table, invokes LLM judges on the `request`/`response` pairs, and writes scored metric rows to a profile metric table. Because ground-truth labels are typically unavailable in production, judges that require `expected_response` (such as answer correctness) are skipped; groundedness and relevance judges operate on `request` + `response` + any extracted trace context. The profiling job tracks statistical profiles (mean, standard deviation, percentiles) over time windows and computes drift relative to either a prior window or a specified baseline table.

**Where does it appear in Databricks?** Configured as a data profiling monitor (formerly Lakehouse Monitoring) on the inference table via the Unity Catalog UI (**Data → [table] → Quality**) or programmatically using the `databricks.sdk` `WorkspaceClient().data_quality.create_monitor()` API. Dashboards are auto-generated in Databricks Dashboards and linked from the table's Quality tab. Alerts are configured via `notification_settings` on the monitor or as SQL alerts on the metric table.

---

### Inference Tables

**What is it?** Inference tables are AI Gateway-enabled Unity Catalog Delta Tables that automatically capture every request payload, response payload, execution duration, HTTP status code, sampling fraction, and (for AI agents) MLflow trace from a Model Serving endpoint, without any application-level code changes.

**How does it work mechanistically?** When inference tables are enabled for an endpoint, the AI Gateway layer intercepts each request-response cycle and writes a row to the inference table within approximately one hour of the request. For AI agents deployed via `mlflow.deploy()`, inference tables are enabled by default and three related tables are created: the payload table (raw JSON request/response), the payload request logs table (formatted messages + MLflow traces), and the payload assessment logs table (human feedback from the Review App). The table grows incrementally as an append-only Delta Table partitioned by `request_date`, making it efficient to query recent data. The `sampling_fraction` column records the fraction of requests logged (default 100%); lowering this reduces storage cost on high-throughput endpoints while preserving a representative sample.

**Where does it appear in Databricks?** Enabled via the **AI Gateway** section of the Model Serving endpoint creation or edit UI (select **Enable inference tables**, choose catalog/schema), or via the REST API with the `ai_gateway.inference_table_config` payload. The table is queryable as a standard Delta Table at `<catalog>.<schema>.<endpoint_name>_payload` in Unity Catalog. The endpoint page shows a direct link to the inference table in Catalog Explorer.

---

### Drift Detection for LLM Applications

**What is it?** Drift detection is the statistical identification of significant shifts in either input query distribution (what users ask) or output quality metrics (how well the agent answers) over time, triggering alerts before user-visible degradation becomes severe.

**How does it work mechanistically?** Data profiling computes rolling-window statistical profiles over the inference table columns on each scheduled refresh. For numerical columns (e.g., `execution_duration_ms`, computed `groundedness_score`), it computes mean, standard deviation, min, max, and percentiles within each time window and compares them against either the prior window or a specified baseline table. For categorical columns (e.g., extracted `request_topic`), it tracks value frequency distributions. Drift is quantified using statistical distance measures (e.g., population stability index or Jensen-Shannon divergence); if a column's drift metric exceeds a configured threshold, an alert fires. The `slicing_exprs` parameter lets you monitor drift separately by sub-populations (e.g., by user segment or product category).

**Where does it appear in Databricks?** Configured as a `TimeSeries` or `InferenceLog` profile via `WorkspaceClient().data_quality.create_monitor()` with the `baseline_table_name` parameter pointing to the golden eval set table. Drift metrics land in the auto-created drift metrics table (named `<output_schema>.<table_name>_drift_metrics`). Dashboards appear in Databricks Dashboards; alerts are configured under **Alerts** in Databricks SQL or via the `notification_settings` parameter.

---

### A/B Testing Agents

**What is it?** A/B testing in Model Serving is the practice of routing a configured percentage of live production traffic to a challenger model version while keeping the champion version serving the majority, so quality metrics can be compared under real-world conditions before a full promotion decision.

**How does it work mechanistically?** The Model Serving endpoint's `traffic_config` specifies a weight for each `served_model_name` in the endpoint configuration; weights must sum to 100. Each served model in the endpoint can be mapped to a different model version from the Unity Catalog Model Registry. Because each served model logs to its own section of the inference table (identified by `served_entity_id`), the challenger's and champion's requests are separable by joining on the `system.serving.served_entities` system table. LLM judge scores computed by data profiling over each `served_entity_id` slice provide the quality comparison. Statistical significance is required before declaring a winner: LLM judge scores have high per-row variance, so at least 1,000 requests per arm are needed to achieve p < 0.05 with typical effect sizes.

**Where does it appear in Databricks?** Configured in the Model Serving endpoint's **Served entities** section in the UI, or via the REST API endpoint configuration with `traffic_config` weights. The `served_entity_id` column in the inference table is the join key to `system.serving.served_entities` for splitting metrics by version.

---

### The Evaluation → Monitoring Feedback Loop

**What is it?** The closed-loop feedback process that connects production monitoring insights back to offline evaluation improvements, forming the continuous quality improvement cycle for a GenAI application.

**How does it work mechanistically?** The loop has four stages that repeat. (1) **Production monitoring** surfaces low-quality request clusters — for example, a data profiling alert fires because `groundedness_score` drops below 0.70 for requests containing a new product name. (2) **Root-cause analysis** queries the inference table to isolate the failing requests; the LLM judge rationale column identifies whether the failure is a retrieval miss (retrieved chunks did not contain the answer) or a generation failure (model hallucinated despite relevant chunks). (3) **Eval set enrichment** adds representative examples of the failing request type to the golden evaluation set as new rows, increasing coverage of the blind spot. (4) **Offline re-evaluation** runs `mlflow.evaluate()` against the enriched eval set on a candidate model version; the version is registered and deployed only if it passes the updated quality gate. The loop velocity is bounded by the data profiling refresh schedule (minimum granularity is 5 minutes) and the time required to collect statistically significant production data.

**Where does it appear in Databricks?** The loop spans three services: MLflow Experiments (offline eval), the inference table + data profiling (online monitoring), and the Unity Catalog Model Registry (deployment gate). In MLflow 3, `mlflow.genai.evaluate()` uses the same scorer configuration in both offline and online contexts, ensuring that the quality definition is consistent across the full lifecycle.

---

## Key Parameters / Configuration Knobs

| Parameter | What it controls | Decision rule |
|---|---|---|
| AI Gateway `inference_table_config.enabled` | Whether inference table logging is active for an endpoint | Always set to `true` before sending any production traffic; retroactive logging is impossible — data from before enablement is permanently lost |
| AI Gateway `inference_table_config.catalog_name` / `schema_name` / `table_name_prefix` | Where inference logs land in Unity Catalog | Set to the same catalog and schema as the agent's source Delta Tables for co-located governance and simplified access control grants |
| AI Gateway `inference_table_config.sampling_fraction` | What fraction of requests are logged (0–1, default 1.0) | Keep at 1.0 (100%) in production for full auditability; reduce only on CPU endpoints exceeding 70 MB/s throughput where log delivery degrades |
| Data profiling `slicing_exprs` | Which dimensional cuts to compute separate metric profiles for | Include expressions like `request_topic` or `user_segment` when you need per-topic quality visibility; each slice doubles metric table storage |
| Data profiling `baseline_table_name` | The reference distribution to compare drift against | Point to the golden eval set table to compare live distribution against expected; if omitted, drift is computed window-over-window only |
| Data profiling `granularities` | The time window size for rolling metric aggregation | Use `AGGREGATION_GRANULARITY_1_DAY` for daily monitoring; use `AGGREGATION_GRANULARITY_1_HOUR` only if SLA requires intra-day alerting (higher cost) |
| `traffic_config` weights | Traffic split percentage between served model versions | Start at 95/5 champion/challenger; increase challenger share only after statistical significance is confirmed (p < 0.05 over ≥ 1,000 requests per arm) |
| `mlflow.evaluate()` `evaluators` (via `model_type`) | Which LLM judge metrics to activate | Always include groundedness and answer correctness; add chunk relevance when debugging retrieval quality; these are activated via `model_type="databricks-agent"` |

---

## Worked Example: Requirement → Decision

**Given:** A RAG application answering product questions has been in production for 3 weeks. Users report that answers about a recently released product line are incomplete and sometimes wrong. The team suspects the knowledge base lacks sufficient coverage of the new product, but must diagnose the exact failure mode (retrieval miss vs. generation failure) before deciding on a fix.

**Step 1 — Identify the goal:** Detect the root cause of the quality regression for new-product queries, validate a fix offline, and deploy the improved version with a controlled rollout — without re-indexing the entire vector store or exposing more users to degraded quality.

**Step 2 — Define inputs:** The inference table with 3 weeks of production logs (`prod_catalog.agents.product_rag_payload`), the original golden eval set (`prod_catalog.agents.product_rag_eval`), and the data profiling metric table showing groundedness drift.

**Step 3 — Define outputs:** Root-cause classification (retrieval miss or generation failure), an enriched golden eval set with ≥ 10 new product Q&A pairs, a new registered model version with passing quality gate metrics, and a deployment plan using 90/10 A/B traffic split.

**Step 4 — Apply constraints:** The fix must not require full vector store re-indexing (too slow and costly); the new model version must achieve ≥ 85% groundedness on the enriched eval set before any production traffic is routed to it; the A/B test must run for at least 1,000 requests per arm before a promotion decision.

**Step 5 — Select the approach:**
- Query the inference table to identify low-groundedness requests mentioning the new product name (confirms retrieval miss because judge rationales cite "no relevant chunks found"):
  ```sql
  SELECT request, response, trace FROM prod_catalog.agents.product_rag_payload_request_logs
  WHERE request LIKE '%ProductX%' ORDER BY timestamp DESC LIMIT 100
  ```
- Confirm root cause: judge rationale in the `trace` column shows retriever returned 0 relevant chunks → retrieval miss (not generation failure).
- Add 10 synthetic Q&A pairs for ProductX to the golden eval set; run `mlflow.evaluate()` on a new model version that uses an updated vector index seeded with ProductX documentation.
- Gate promotion: if `agent/percent_grounded >= 0.85`, register new model version with `@challenger` alias.
- Deploy as 90/10 A/B split; monitor data profiling slice for `request ~ ProductX`; promote `@challenger → @champion` after 1,000+ requests and p < 0.05 quality improvement.

This approach is preferred over an emergency full-index rebuild because it targets the specific coverage gap, validates quality before any user exposure, and provides a statistically sound promotion criterion. Waiting for user complaints to accumulate (the alternative) delays the fix by days and has no quality gate.

---

## Implementation

```python
# Scenario: Enabling AI Gateway-enabled inference tables when creating a Model Serving
# endpoint so that every production request is logged to Unity Catalog for monitoring.
# Constraint: must be enabled at creation time — retroactive enablement loses prior data.

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedModelInput,
    ServedModelInputWorkloadSize,
    AiGatewayConfig,
    AiGatewayInferenceTableConfig,
)

w = WorkspaceClient()

endpoint_config = EndpointCoreConfigInput(
    name="product-rag-agent",
    served_models=[
        ServedModelInput(
            model_name="prod_catalog.agents.product_rag",
            model_version="3",
            workload_size=ServedModelInputWorkloadSize.SMALL,
            scale_to_zero_enabled=True,
        )
    ],
    traffic_config=None,  # 100% to single served model
)

# Enable inference table via AI Gateway configuration
ai_gateway = AiGatewayConfig(
    inference_table_config=AiGatewayInferenceTableConfig(
        catalog_name="prod_catalog",
        schema_name="agents",
        table_name_prefix="product_rag",
        enabled=True,
    )
)

w.serving_endpoints.create(
    name="product-rag-agent",
    config=endpoint_config,
    ai_gateway=ai_gateway,
)
# Inference table created at: prod_catalog.agents.product_rag_payload
```

```python
# Anti-pattern: Monitoring only system metrics (latency and error rate) while ignoring
# LLM quality metrics — this misses the entire class of "silent quality degradation"
# where the model halluccinates but returns HTTP 200 with fast responses.

# WRONG: Setting a Databricks SQL alert on p99 latency only
# CREATE ALERT latency_alert
#   ON QUERY: SELECT percentile(execution_duration_ms, 0.99) AS p99
#             FROM prod_catalog.agents.product_rag_payload
#   WHEN p99 > 5000 THEN NOTIFY "team@company.com"
#
# Why this fails: A hallucinating RAG agent returns fast, low-latency 200 responses.
# The p99 latency stays at 800ms. Zero alerts fire. Users receive confidently wrong
# answers for 3 weeks. The signal simply does not exist in the system metric.

# CORRECT: Query the inference table for LLM judge quality metrics after
# data profiling has scored them, and alert on groundedness degradation.

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.dataquality import (
    Monitor,
    DataProfilingConfig,
    InferenceLogConfig,
    InferenceProblemType,
    AggregationGranularity,
    NotificationSettings,
    NotificationDestination,
)

w = WorkspaceClient()

# Get the inference table's object ID
table = w.tables.get(full_name="prod_catalog.agents.product_rag_payload_request_logs")
schema = w.schemas.get(full_name="prod_catalog.agents")

config = DataProfilingConfig(
    output_schema_id=schema.schema_id,
    assets_dir="/Workspace/Shared/monitoring/product-rag",
    inference_log=InferenceLogConfig(
        problem_type=InferenceProblemType.INFERENCE_PROBLEM_TYPE_CLASSIFICATION,
        prediction_column="response",          # agent response to score
        model_id_column="served_entity_id",    # separates A/B versions
        timestamp_column="timestamp",
        granularities=[AggregationGranularity.AGGREGATION_GRANULARITY_1_DAY],
    ),
    # Monitor quality separately by topic if request_topic column is extracted
    slicing_exprs=["request_topic"],
    # Compare live distribution against the golden eval set
    baseline_table_name="prod_catalog.agents.product_rag_eval",
    notification_settings=NotificationSettings(
        on_failure=NotificationDestination(
            email_addresses=["ml-ops@company.com"]
        )
    ),
)

w.data_quality.create_monitor(
    monitor=Monitor(
        object_type="table",
        object_id=table.table_id,
        data_profiling_config=config,
    )
)
# Drift alerts are now configured; if groundedness_score drifts significantly
# from the baseline, the on_failure notification fires.
```

```sql
-- Scenario: Surfacing low-groundedness request clusters from the inference table
-- to identify which topics triggered a quality regression detected by data profiling.
-- This query is run after a groundedness drift alert fires, to find the root cause.

SELECT
    DATE_TRUNC('day', timestamp)                        AS request_day,
    -- Extract topic keyword from request (adjust regex to your domain)
    REGEXP_EXTRACT(request, '(?i)(ProductX|ProductY|ProductZ)', 1)
                                                        AS topic,
    COUNT(*)                                            AS request_count,
    -- groundedness_score is populated by the data profiling LLM judge job
    AVG(groundedness_score)                             AS avg_groundedness,
    SUM(CASE WHEN groundedness_score < 0.7 THEN 1 ELSE 0 END) AS low_quality_count,
    ROUND(
        SUM(CASE WHEN groundedness_score < 0.7 THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
        1
    )                                                   AS pct_low_quality
FROM prod_catalog.agents.product_rag_payload_request_logs
WHERE timestamp >= DATEADD(DAY, -7, CURRENT_TIMESTAMP())
  AND status_code = 200
GROUP BY 1, 2
HAVING request_count >= 10   -- filter out noise from single-request topics
ORDER BY avg_groundedness ASC
LIMIT 20;
-- Result: rows with topic = 'ProductX', avg_groundedness = 0.42
-- → confirms ProductX queries are the root cause of the regression
```

```python
# Scenario: Configuring a 90/10 A/B traffic split to test a challenger agent version
# against the current champion before committing to a full promotion, so quality
# improvement can be verified under real traffic without full user exposure.

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedModelInput,
    ServedModelInputWorkloadSize,
    TrafficConfig,
    Route,
)

w = WorkspaceClient()

# Update existing endpoint to route 10% traffic to challenger version
w.serving_endpoints.update_config(
    name="product-rag-agent",
    served_models=[
        ServedModelInput(
            name="champion-v3",
            model_name="prod_catalog.agents.product_rag",
            model_version="3",
            workload_size=ServedModelInputWorkloadSize.SMALL,
            scale_to_zero_enabled=True,
        ),
        ServedModelInput(
            name="challenger-v4",
            model_name="prod_catalog.agents.product_rag",
            model_version="4",    # new version with updated vector index
            workload_size=ServedModelInputWorkloadSize.SMALL,
            scale_to_zero_enabled=True,
        ),
    ],
    traffic_config=TrafficConfig(
        routes=[
            Route(served_model_name="champion-v3", traffic_percentage=90),
            Route(served_model_name="challenger-v4", traffic_percentage=10),
        ]
    ),
)
# Both served models log to the same inference table; differentiate by served_entity_id
# Run data profiling sliced by served_entity_id to compare quality metrics per version
```

---

## Common Pitfalls & Misconceptions

- **Treating offline eval as sufficient** — Beginners run `mlflow.evaluate()` once before deployment and assume the system is validated for production. The correct mental model: offline eval is a snapshot against a fixed dataset; real user query distributions shift daily as user populations grow and products evolve, so online monitoring is the only mechanism that can catch post-deployment regressions that the golden set never anticipated.

- **Not enabling inference tables before going live** — Teams plan to add inference logging after a quality issue surfaces, assuming they can turn it on retroactively. Retroactive logging is impossible — the AI Gateway only logs requests received after the table is enabled. Always set `inference_table_config.enabled = true` at endpoint creation time, before any production traffic is sent.

- **Monitoring token counts as a quality proxy** — Beginners assume shorter or longer responses correlate with quality, and set alerts on `execution_duration_ms` or response length. Response length is unrelated to groundedness; a two-sentence grounded answer scores 1.0 while a ten-sentence hallucinated answer scores 0.0. Monitor LLM judge groundedness and correctness scores from data profiling, not token counts or response length.

- **Running A/B tests for too short a period** — Engineers route 50% traffic to the challenger, collect 100 requests, observe a 3-point improvement in average groundedness, and declare the challenger the winner. The correct mental model: LLM judge scores have high per-row variance (they are binary yes/no per question); statistical significance for a typical 5% quality improvement requires at least 1,000 requests per arm to achieve p < 0.05. Declare a winner only after reaching this threshold.

- **Conflating drift in query distribution with model degradation** — When groundedness drops, beginners immediately assume the model has degraded and schedule a retraining run. Often the underlying cause is that users started asking about topics not covered by the knowledge base (query distribution shift), not that the model got worse on questions it was already handling. Always investigate the `request` column distribution in the inference table to distinguish query-side drift from model-side degradation before deciding on a fix.

---

## Key Definitions

| Term | Definition |
|---|---|
| Offline evaluation | The process of running LLM-judge scoring against a fixed golden dataset before deployment, using `mlflow.evaluate(model_type="databricks-agent")`; produces pass/fail metrics that gate model registration |
| Online monitoring | Asynchronous post-deployment scoring of sampled live traffic using the same LLM judges, operating on inference table rows without requiring ground-truth labels |
| Inference table | An AI Gateway-enabled Unity Catalog Delta Table that auto-captures every Model Serving endpoint request payload, response payload, execution duration, HTTP status code, and (for agents) MLflow trace |
| Data profiling | A Databricks Unity Catalog feature (formerly Lakehouse Monitoring) that computes rolling statistical profiles and drift metrics over Delta Tables on a schedule; for LLM apps, applied to the inference table |
| Golden eval set | A curated Pandas DataFrame or Delta Table of representative `request` / `expected_response` / `retrieved_context` rows used as the fixed reference dataset for offline evaluation |
| Groundedness | An LLM-judge metric scoring whether the agent's response is supported by the retrieved context (yes = response is factually grounded in the retrieved documents) |
| Drift detection | Statistical identification of significant distributional shifts in input queries or output quality metrics over time, measured via profile metric and drift metric tables |
| A/B testing (agents) | Routing a configured percentage of live traffic to a challenger model version via `traffic_config` weights while the champion serves the majority, enabling quality comparison under real-world conditions |
| Feedback loop | The closed-loop process where production monitoring regression signals enrich the golden eval set, which gates the next model version's registration and deployment |
| `traffic_config` | The Model Serving endpoint configuration field that assigns percentage weights to each served model (e.g., 90% champion / 10% challenger); weights must sum to 100 |
| Sampling fraction | The fraction of endpoint requests that are written to the inference table (0–1); default 1.0 for full logging; reducible to control storage cost on high-throughput endpoints |

---

## Summary / Quick Recall

- Offline evaluation (`mlflow.evaluate()`) gates release; online monitoring (data profiling over inference tables) catches post-release regressions — both are required.
- Enable AI Gateway inference tables **before** sending production traffic; retroactive logging is impossible.
- Inference table columns include `databricks_request_id`, `request_time`, `execution_duration_ms`, `status_code`, `request`, `response`, `served_entity_id`, and (for agents) `trace`.
- Data profiling uses `InferenceLog` profile type for inference tables; `baseline_table_name` points to the golden eval set for drift comparison.
- `slicing_exprs` in data profiling enables per-topic or per-segment quality visibility — critical for diagnosing which query types are degrading.
- A/B testing uses `traffic_config` weights in the Model Serving endpoint; differentiate versions in the inference table via `served_entity_id`.
- Require ≥ 1,000 requests per A/B arm before declaring a statistical winner (p < 0.05).
- The feedback loop: monitoring alert → query inference table → enrich golden set → offline eval gate → register → deploy.

---

## Self-Check Questions

1. **[Recall]** Which columns does an AI Gateway-enabled inference table automatically capture for every request to a Model Serving endpoint?

   - A. Only `request` and `response`
   - B. `request`, `response`, `execution_duration_ms`, `status_code`, `databricks_request_id`, `request_time`, `sampling_fraction`, and `served_entity_id`
   - C. `request`, `response`, `groundedness_score`, and `answer_correctness_score`
   - D. `request`, `response`, and the full MLflow trace for all endpoint types

   <details><summary>Answer</summary>

   **Correct: B.** The AI Gateway-enabled inference table schema captures `request`, `response`, `execution_duration_ms` (inference latency), `status_code`, `databricks_request_id`, `request_time`, `sampling_fraction`, `served_entity_id`, `client_request_id`, and `logging_error_codes`. Together these provide full request/response audit, latency telemetry, and version identification.

   **Why A is wrong:** Capturing only request and response would be sufficient for payload auditing but not for latency monitoring, error diagnosis, or A/B test version attribution — the table captures significantly more metadata.

   **Why C is wrong:** `groundedness_score` and `answer_correctness_score` are *not* auto-populated by the inference table — they are computed later by data profiling LLM judges running over the table rows. The raw table has no LLM judge scores.

   **Why D is wrong:** MLflow traces are captured for AI agents deployed via `mlflow.deploy()` (stored in the `_payload_request_logs` table's `trace` column), but standard custom model and foundation model endpoints do not automatically capture full traces — the trace capture requires `ENABLE_MLFLOW_TRACING=True` in the endpoint environment.

   </details>

2. **[Application]** Your groundedness scores drop from 0.85 to 0.62 over 2 weeks with no code changes to the agent or the vector index. What is the MOST likely FIRST diagnostic step?

   - A. Immediately schedule a model retraining run, because a score drop with no code changes must mean model weights have degraded
   - B. Query the inference table to examine whether the `request` column distribution has shifted — specifically whether users are now asking about topics not covered in the knowledge base
   - C. Increase the number of retrieved chunks from k=3 to k=10 to give the model more context
   - D. Roll back the serving endpoint to the previous model version

   <details><summary>Answer</summary>

   **Correct: B.** A drop in groundedness with no code changes is a classic symptom of query distribution shift — users are now asking about topics the knowledge base does not cover, so the retriever returns low-relevance chunks, and the model cannot ground its answer in them. The first diagnostic step is to examine the `request` column distribution in the inference table to identify what topics or keywords have increased in frequency, before taking any corrective action.

   **Why A is wrong:** Model weights on a deployed endpoint cannot degrade on their own; the serving endpoint serves a fixed registered model version. Retraining before diagnosing the root cause could waste significant compute and may not address the actual problem (a knowledge base gap).

   **Why C is wrong:** Increasing k before diagnosis treats the symptom (insufficient context) without confirming the root cause. If the topic is absent from the knowledge base entirely, retrieving 10 irrelevant chunks instead of 3 does not help and increases token cost.

   **Why D is wrong:** Rolling back assumes the current model version is the cause, but the score dropped without any code change — there is no prior model version to roll back to that would be different in a meaningful way. The correct action is to diagnose the distribution shift first.

   </details>

3. **[Application] Which TWO** of the following statements are correct about the difference between offline evaluation and online monitoring?

   - A. Offline evaluation requires `expected_response` labels; online monitoring does not require them because LLM judges can assess groundedness from `request` + `response` + trace context alone
   - B. Online monitoring replaces offline evaluation once the model is deployed, making the golden eval set unnecessary after release
   - C. Offline evaluation gates model registration before deployment; online monitoring detects regressions in already-deployed models
   - D. Both offline evaluation and online monitoring use identical LLM judges with identical scoring logic, making the quality definition consistent across the lifecycle
   - E. Inference tables must be manually populated by the application developer; they are not auto-populated by the serving endpoint

   <details><summary>Answer</summary>

   **Correct: A and C.**

   **Why A is correct:** In development, `expected_response` is optionally provided to unlock the answer correctness judge; in production, ground-truth labels are typically unavailable so the correctness judge is skipped, but groundedness and relevance judges operate on `request` + `response` alone (or with trace-extracted context).

   **Why C is correct:** This precisely describes the temporal division of labor: offline eval is a pre-deployment gate, online monitoring is a post-deployment regression detector.

   **Why B is wrong:** Online monitoring and offline evaluation serve complementary, not mutually exclusive, roles. The golden eval set remains the authoritative regression benchmark; monitoring alerts are signals to re-run offline eval on a new model version, not to replace it.

   **Why D is a tempting distractor:** While Databricks designs the same judge configuration to work in both contexts (especially in MLflow 3), D overstates the current guarantee — in MLflow 2, the APIs for offline eval (`mlflow.evaluate()`) and production monitoring (data profiling) are different subsystems. The claim of *identical* logic is aspirational for the MLflow 3 unified path.

   **Why E is wrong:** Inference tables are auto-populated by the AI Gateway layer at the endpoint level with no application-level code changes required — this is precisely their purpose.

   </details>

4. **[Analysis]** A team wants to A/B test two agent versions. They configure the endpoint to route 50% traffic to the challenger immediately on day 1. After 200 total requests (100 per arm), the challenger shows a 4-point higher average groundedness (0.81 vs. 0.77). They promote the challenger to champion. What is wrong with this approach?

   - A. The traffic split should always start at 99/1, not 50/50
   - B. The team promoted based on an insufficient sample size — 100 requests per arm is too few to establish statistical significance for LLM judge score differences, which have high per-row variance
   - C. A/B testing is not supported in Databricks Model Serving; shadow deployment must be used instead
   - D. The 4-point difference is below the minimum required improvement threshold of 10 points for a valid promotion

   <details><summary>Answer</summary>

   **Correct: B.** LLM judge scores are binary yes/no values per request; the average across requests has high variance, especially at small sample sizes. A 4-point difference (0.81 vs. 0.77) on 100 samples per arm is not statistically significant — the confidence interval is wide enough that the true challenger quality could be lower than the champion's. Best practice requires ≥ 1,000 requests per arm and confirmation of p < 0.05 before promoting.

   **Why A is wrong:** Starting at 50/50 is a valid choice when fast experimentation is a priority and the team accepts higher exposure of users to the challenger. The *start* of the split is not the problem — the sample size at decision time is. There is no universal rule requiring 99/1 as a starting point.

   **Why C is wrong:** A/B testing via `traffic_config` weights is an explicitly supported feature of Databricks Model Serving. Multiple `ServedModelInput` entries with `Route` weight assignments are documented and production-grade.

   **Why D is wrong:** There is no fixed minimum threshold of 10 points for a valid promotion. Promotion decisions are based on statistical significance at a chosen significance level (typically p < 0.05), not on an absolute point threshold. A 1-point difference that is statistically significant on 10,000 requests can justify promotion; a 20-point difference on 50 requests cannot.

   </details>

5. **[Trade-off]** An inference table has accumulated 10 million rows over 6 months. Running data profiling on the full table is slow and expensive. What is the BEST architectural approach to maintain quality monitoring at scale?

   - A. Delete the inference table and start a new one every month to keep row counts manageable
   - B. Switch from a `TimeSeries` or `InferenceLog` profile to a `Snapshot` profile, which processes all data at once more efficiently
   - C. Use the `TimeSeries` or `InferenceLog` profile type with change data feed (CDF) enabled on the table, so data profiling processes only newly appended rows on each refresh rather than re-scanning all 10 million rows
   - D. Disable data profiling and rely on manual SQL queries run ad hoc by the team

   <details><summary>Answer</summary>

   **Correct: C.** Databricks explicitly recommends enabling change data feed (CDF) on inference tables used with `TimeSeries` and `InferenceLog` profiles. When CDF is enabled, data profiling uses incremental processing — only newly appended rows since the last refresh are scanned, not the full historical table. This makes refresh cost roughly proportional to new daily volume, not total table size.

   **Why A is wrong:** Deleting the inference table destroys historical data needed for trend analysis, long-window drift detection, and retrospective debugging. The table is also the ground-truth audit log for the endpoint; destroying it is a governance risk.

   **Why B is wrong:** `Snapshot` profiles are explicitly described as less efficient for large tables: they re-process the *entire* table on every refresh. The Databricks docs note a 4TB maximum for snapshot profiles and recommend `TimeSeries` for large tables. Switching to Snapshot would make the performance problem *worse*, not better.

   **Why D is wrong:** Ad hoc SQL queries do not provide continuous monitoring, do not fire automated alerts, and do not track statistical drift over time. They are useful for investigation after an alert fires, not as a replacement for automated monitoring.

   </details>

---

## Further Reading

- [Agent Evaluation (MLflow 2)](https://docs.databricks.com/aws/en/agents/agent-evaluation/) — *verified 2026-07-11* — Overview of offline evaluation with `mlflow.evaluate(model_type="databricks-agent")`, LLM judge metrics, and evaluation sets
- [Evaluate and monitor AI agents (MLflow 3)](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/) — *verified 2026-07-11* — MLflow 3 unified evaluation and production monitoring overview
- [Monitor served models using AI Gateway-enabled inference tables](https://docs.databricks.com/aws/en/ai-gateway/inference-tables-serving-endpoints) — *verified 2026-07-11* — AI Gateway inference table schema, enablement steps, and agent-specific payload tables
- [Inference tables for monitoring and debugging models (legacy)](https://docs.databricks.com/aws/en/archive/machine-learning/inference-tables) — *verified 2026-07-11* — Legacy inference table reference; retained for migration context (Databricks now recommends AI Gateway inference tables)
- [Data profiling overview](https://docs.databricks.com/aws/en/data-governance/unity-catalog/data-quality-monitoring/data-profiling/) — *verified 2026-07-11* — Unity Catalog data profiling (formerly Lakehouse Monitoring): profile types, metric tables, dashboards, and alerts
- [Create a data profile using the API](https://docs.databricks.com/aws/en/data-governance/unity-catalog/data-quality-monitoring/data-profiling/create-monitor-api) — *verified 2026-07-11* — `databricks.sdk` API reference for `TimeSeries`, `InferenceLog`, and `Snapshot` profile creation with `slicing_exprs` and `baseline_table_name`
