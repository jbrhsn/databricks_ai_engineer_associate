# Unity Catalog Model Registration and Serving

**Section:** 05-assembling-and-deploying-applications | **Module:** 02-vector-search-and-serving | **Est. time:** 3 hrs | **Exam mapping:** Domain 3 — Application Development (supporting) / Domain 4 — Assembling & Deploying Applications (primary, 22%)

---

## TL;DR

Models registered in Unity Catalog (UC) live under a three-level namespace (`catalog.schema.model_name`) that gives them the same access-control, lineage, and auditing guarantees as Delta tables. Registration is a one-call operation — `mlflow.register_model()` with a UC three-part name — and deployment is a second call that creates a Model Serving endpoint backed by that version. The Champion/Challenger alias pattern decouples which model version is live from which version number it is, so traffic routing can be changed without redeploying code. **The one thing to remember: the UC three-part model name (`catalog.schema.model_name`) governs *where* a model lives and *who* can use it; everything downstream — serving, lineage, access control — derives from that single registration act.**

---

## ELI5 — Explain It Like I'm 5

Imagine a huge library where every book (your trained model) must first be checked in at the front desk. The librarian writes down the book's full address: which building it is in, which room, and what the title is — that is the three-part name. Once a book is checked in, the library automatically keeps a log of who donated it, which training data it was built from, and who is allowed to borrow it. When a teacher (your application) wants to use the book, they go to the serving window and say "give me the Champion copy of the earthquake prediction book"; the librarian fetches whichever physical copy currently has the Champion sticker — and that sticker can be moved to a new edition without telling the teacher anything changed. The most common mistake is thinking that registering a model *deploys* it; registering only checks the book in. Deployment is the separate step of opening a serving window so external applications can borrow it in real time.

---

## Learning Objectives

By the end of this chapter you will be able to:
- [ ] Register an MLflow model to Unity Catalog using the three-level namespace and explain why the registry URI must be set to `databricks-uc`.
- [ ] Distinguish between `mlflow.register_model()` and `MlflowClient.create_registered_model()` and choose the correct call for a given registration scenario.
- [ ] Apply the Champion/Challenger alias pattern to perform zero-code-change model promotions.
- [ ] Create a Model Serving endpoint via the Databricks SDK (`WorkspaceClient.serving_endpoints.create()`), configure traffic splits for A/B testing, and query the endpoint's `/invocations` path.
- [ ] Explain how Unity Catalog governance features (lineage, RBAC, audit logs) apply to registered models and serving endpoints.

---

## Visual Overview

### Registration Pipeline: Training Run → UC Model

```
MLflow Run
  │
  │  mlflow.sklearn.log_model(registered_model_name="prod.ml_team.iris_model")
  │  — OR —
  │  mlflow.register_model("runs:/<run_id>/model", "prod.ml_team.iris_model")
  │
  ▼
Unity Catalog Model Registry
  ┌─────────────────────────────────────┐
  │  catalog: prod                      │
  │    schema: ml_team                  │
  │      model: iris_model              │
  │        version 1 ── alias: Champion │
  │        version 2 ── alias: Challenger│
  └─────────────────────────────────────┘
```

### Serving Endpoint Creation and Query Flow

```
Databricks SDK / REST API
  │
  │  w.serving_endpoints.create(name="iris-endpoint", config=...)
  │
  ▼
Model Serving Control Plane
  │  resolves entity_name + entity_version
  ▼
Serving Endpoint (READY)
  │
  │  POST /serving-endpoints/iris-endpoint/invocations
  │  { "dataframe_split": { "columns": [...], "data": [[...]] } }
  │
  ▼
Response: { "predictions": [...] }
```

### Traffic Split for A/B Testing

```
Incoming Traffic (100%)
  │
  ├── 80% ──► served entity: iris_model-v1  (Champion)
  └── 20% ──► served entity: iris_model-v2  (Challenger)

traffic_config:
  routes:
    - served_model_name: iris_model-v1   traffic_percentage: 80
    - served_model_name: iris_model-v2   traffic_percentage: 20
```

### UC Governance Layer on a Registered Model

```
Unity Catalog Metastore
  │
  ├── GRANT EXECUTE ON MODEL prod.ml_team.iris_model TO analysts
  ├── Lineage tab  ──► upstream Delta tables logged via mlflow.log_input()
  ├── Audit log    ──► who registered, who queried endpoint
  └── Catalog Explorer  ──► model versions, aliases, tags, descriptions
```

---

## Key Concepts

### Unity Catalog Three-Level Namespace for Models

**What is it?** A registered model in Unity Catalog is identified by a three-part name — `catalog.schema.model_name` — that places the model inside the same hierarchical namespace used for tables, views, and functions. For example, `prod.ml_team.iris_model` means the model `iris_model` lives in schema `ml_team` of catalog `prod`.

**How does it work under the hood?** When you call `mlflow.register_model()` with a three-part name, the MLflow client (configured to target `databricks-uc`) posts a create-model-version request to the UC REST API. UC creates the registered model object (if it does not already exist) and stores model artifacts in the catalog's managed storage location. The model object is a first-class UC securable of subtype `FUNCTION`, which means every privilege operation that applies to a function (`USE CATALOG`, `USE SCHEMA`, `EXECUTE`, `CREATE MODEL`) also applies to the model.

**Where does it appear in Databricks?** The full three-part name is the value passed to `registered_model_name` in `mlflow.<flavor>.log_model()`, to the `name` argument of `mlflow.register_model()`, and to `entity_name` in the serving endpoint config. In Catalog Explorer, navigate to **Catalog → Schema → Models** to browse and manage registered models.

---

### `mlflow.register_model()` vs `MlflowClient.create_registered_model()`

**What is it?** These are two distinct operations. `mlflow.register_model(model_uri, name)` takes an existing logged-model artifact (identified by a `runs:/` or `models:/` URI) and creates a new numbered *version* under the named registered model, creating the registered model parent if it does not exist. `MlflowClient.create_registered_model(name)` creates only the registered model *shell* with no versions; a version must be separately added via `log_model(registered_model_name=...)` or `register_model()`.

**How does it work under the hood?** `register_model()` is a combined idempotent call: it issues a `POST /api/2.0/mlflow/registered-models/create` (creating the parent if absent) and then `POST /api/2.0/mlflow/model-versions/create` in sequence. `create_registered_model()` only issues the first request. When autologging is active, calling `mlflow.last_active_run()` or `mlflow.last_logged_model()` (MLflow 3) provides the URI, so a single subsequent `register_model()` call is sufficient.

**Where does it appear in Databricks?** Both calls are part of the `mlflow` and `mlflow.tracking.MlflowClient` Python APIs. For CI/CD pipelines, `register_model()` is the canonical path because it is idempotent — re-running it creates a new version rather than failing. `create_registered_model()` is useful when you want to pre-create a model shell before any training run has completed (for example, to set governance metadata in advance).

---

### Model Aliases and the Champion/Challenger Pattern

**What is it?** A model alias is a mutable, named pointer to a specific version number of a registered model. Common aliases include `Champion` (the current production version) and `Challenger` (a candidate version under evaluation). Aliases replace the old Workspace Model Registry concept of *stages* (Staging/Production), which are not supported in UC.

**How does it work under the hood?** Aliases are stored as metadata on the registered model object in UC. When you call `client.set_registered_model_alias("prod.ml_team.iris_model", "Champion", 2)`, UC atomically updates the alias pointer. Code that references `models:/prod.ml_team.iris_model@Champion` always resolves to whichever version currently holds that alias — no code change is required to promote version 3 when it replaces version 2 as Champion. Serving endpoints can also be updated to pin a specific version resolved from an alias at deployment time.

**Where does it appear in Databricks?** Aliases are managed via `MlflowClient.set_registered_model_alias()`, `delete_registered_model_alias()`, and `get_model_version_by_alias()`. In Catalog Explorer, aliases appear as badges on model version rows; clicking **Add alias** sets one interactively.

---

### Creating a Model Serving Endpoint (Databricks SDK)

**What is it?** A Model Serving endpoint is a managed, auto-scaling REST endpoint that loads a registered model version and exposes it at `POST /serving-endpoints/{name}/invocations`. The endpoint handles containerization, dependency resolution, and health checking automatically.

**How does it work under the hood?** When `w.serving_endpoints.create()` is called, the Databricks control plane: (1) resolves the `entity_name` + `entity_version` against UC and validates the creator's `EXECUTE` grant; (2) builds a container image with the model's logged dependencies; (3) provisions a compute replica of the requested `workload_size`; (4) waits until the health check passes before setting state to `READY`. Scale-to-zero suspends all replicas after inactivity, reducing cost but adding a cold-start latency (typically 30–60 s) on the next request. Updates follow the same lifecycle — old config keeps serving traffic until the new config reaches `READY`.

**Where does it appear in Databricks?** The primary SDK path is `from databricks.sdk import WorkspaceClient; w.serving_endpoints.create(name=..., config=EndpointCoreConfigInput(...))`. The REST equivalent is `POST /api/2.0/serving-endpoints`. In the UI, navigate to **Serving** in the sidebar and click **Create serving endpoint**.

---

### Endpoint Configuration: Served Entities, Traffic Splits, and Scale-to-Zero

**What is it?** An endpoint's `config` object specifies one or more *served entities* (each pointing to a UC model version with compute settings) and an optional `traffic_config` that routes a percentage of requests to each served entity — enabling A/B or shadow deployments.

**How does it work under the hood?** The control plane maintains a `traffic_config.routes` list. Each route pairs a `served_model_name` (the name of a served entity) with a `traffic_percentage`. The percentages must sum to 100. When a request arrives, the load balancer samples a uniform random number and routes to the entity whose cumulative percentage bucket it falls into. Each served entity independently autoscales between its `min_provisioned_concurrency` and `max_provisioned_concurrency` values (which must be multiples of 4).

**Where does it appear in Databricks?** In the SDK, `ServedEntityInput` accepts `entity_name`, `entity_version`, `workload_size` (`Small`/`Medium`/`Large`), `scale_to_zero_enabled`, and concurrency fields. In the REST API, use `PUT /api/2.0/serving-endpoints/{name}/config` to update the traffic split without recreating the endpoint.

---

### Querying the Serving Endpoint (`/invocations`)

**What is it?** Any READY endpoint accepts `POST /serving-endpoints/{endpoint_name}/invocations` with a JSON body containing the model inputs. The response contains a `predictions` key with the model's output.

**How does it work under the hood?** The invocations path normalizes the input against the model's logged signature (if present) and routes to the appropriate served entity replica. For pandas-based models, the recommended input format is `dataframe_split`: `{ "dataframe_split": { "columns": [...], "data": [[...]] } }`. For tensor-based models (PyTorch, TensorFlow), `instances` or `inputs` keys are used. Internally, the serving container calls `model.predict(input_df)` and serializes the result to JSON.

**Where does it appear in Databricks?** From Python, use the MLflow Deployments SDK: `client.predict(endpoint="iris-endpoint", inputs={"dataframe_split": {...}})`. From bash, `curl -X POST -u token:$TOKEN $HOST/serving-endpoints/<name>/invocations`. The Serving UI provides a **Query endpoint** button that accepts raw JSON.

---

### Unity Catalog Governance for Serving (Lineage, Access Control)

**What is it?** Because registered models are UC securables of type `FUNCTION`, every access event is captured in UC audit logs, every privilege follows UC RBAC (`USE CATALOG` → `USE SCHEMA` → `EXECUTE`), and upstream data lineage is automatically tracked when `mlflow.log_input()` is used during training.

**How does it work under the hood?** When a user invokes a serving endpoint, the control plane re-validates that the endpoint's *recorded creator* still holds `EXECUTE` on the model and `USE SCHEMA`/`USE CATALOG` on the enclosing hierarchy. Lineage data flows: `mlflow.log_input(dataset, "training")` writes the input table reference into the MLflow run metadata; when the run's model version is registered to UC, that reference is copied to the model version's lineage graph, which is visible in the **Lineage** tab in Catalog Explorer. Cross-workspace model access is granted by assigning privileges on the UC object in the shared metastore.

**Where does it appear in Databricks?** Audit events appear in the account-level audit log stream. The lineage graph is visible in **Catalog Explorer → model version → Lineage tab → See lineage graph**. RBAC is managed with `GRANT EXECUTE ON MODEL catalog.schema.model_name TO principal`.

---

## Key Parameters / Configuration Knobs

| Parameter | What it controls | Decision rule |
|---|---|---|
| `entity_name` | Which UC registered model the served entity is backed by (full three-part name required) | Always use `catalog.schema.model_name`; bare names silently target the workspace model registry (legacy) |
| `entity_version` | Which numbered version of the model to serve | Pin to a specific integer version in production; use an alias-resolved version in CD pipelines (`client.get_model_version_by_alias()`) |
| `workload_size` | CPU/memory replica size (`Small`, `Medium`, `Large`, `CPU_MEDIUM`, `CPU_LARGE`, or GPU variants) | Start with `Small` (4 vCPU, 16 GB); scale up only when p99 latency exceeds SLA or OOM errors appear in server logs |
| `scale_to_zero_enabled` | Whether the endpoint suspends all replicas after idle period | Set `False` for production SLA endpoints; set `True` for dev/staging to cut cost at the expense of cold-start latency |
| `min_provisioned_concurrency` | Floor on the number of in-flight requests the endpoint can handle without scaling | Set equal to your expected steady-state QPS × average model inference time (seconds); must be a multiple of 4 |
| `max_provisioned_concurrency` | Ceiling on concurrent requests (hard cap) | Set to peak QPS × inference time × 1.5 safety factor; must be a multiple of 4 |
| `traffic_percentage` (per route) | Fraction of requests directed to each served entity | Must sum to 100 across all routes; use 80/20 for conservative Challenger rollouts; shift to 100/0 after Challenger wins |
| `mlflow.set_registry_uri("databricks-uc")` | Directs all MLflow registry API calls to Unity Catalog instead of the legacy Workspace Model Registry | Required when the workspace default catalog is `hive_metastore` or when running DBR < 13.3 LTS; skip when DBR 13.3+ and default catalog is a UC catalog |

---

## Worked Example: Requirement → Decision

**Given:** A data science team has trained a new fraud detection model (`v5`) in the `staging` catalog. The current production model (`v3`) lives in `prod.risk.fraud_detection` with the `Champion` alias. The team wants to serve 10% of live traffic to `v5` while keeping 90% on `v3`, with zero changes to the calling application. The endpoint already exists as `fraud-endpoint`.

**Step 1 — Identify the goal:** Route a slice of production traffic to the Challenger model version to measure real-world performance, without modifying application code or recreating the endpoint.

**Step 2 — Define inputs:**
- Current endpoint name: `fraud-endpoint`
- Champion version: `prod.risk.fraud_detection` version 3 (alias `Champion`)
- Challenger version: `staging.risk.fraud_detection` version 5 (to be copied to `prod`)

**Step 3 — Define outputs:**
- Endpoint `fraud-endpoint` updated with two served entities: `fraud-v3` at 90% and `fraud-v5` at 10%
- Challenger alias `Challenger` set on version 5 in `prod.risk.fraud_detection`

**Step 4 — Apply constraints:**
- Endpoint creator must hold `EXECUTE` on both model versions in `prod`; the staging model must be copied to `prod` first (cross-catalog serving is not supported directly).
- Traffic percentages must sum to 100.
- Endpoint update keeps the old config live until the new config is `READY`.

**Step 5 — Select the approach:**
1. Copy `staging.risk.fraud_detection/5` to `prod.risk.fraud_detection` using `client.copy_model_version()`.
2. Set alias: `client.set_registered_model_alias("prod.risk.fraud_detection", "Challenger", <new_version>)`.
3. Update the endpoint config via `PUT /api/2.0/serving-endpoints/fraud-endpoint/config` with two `served_entities` entries and a `traffic_config` with routes `[{fraud-v3: 90}, {fraud-v5: 10}]`.

This is preferred over recreating the endpoint (which would cause downtime) or using feature flags in application code (which couples deployment to release cycles).

---

## Implementation

```python
# Scenario: Registering a newly-trained scikit-learn model to Unity Catalog so that
# governance, lineage, and serving can be applied without changing application URIs.

import mlflow
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer

# 1. Point the MLflow client at Unity Catalog (required unless DBR 13.3+ with UC default catalog)
mlflow.set_registry_uri("databricks-uc")

X, y = load_breast_cancer(return_X_y=True, as_frame=True)

with mlflow.start_run():
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=4)
    clf.fit(X, y)

    # input_example enables automatic signature inference and auto-generates
    # the input example shown in the Serving UI "Query endpoint" dialog.
    input_example = X.iloc[:3]

    mlflow.sklearn.log_model(
        sk_model=clf,
        artifact_path="model",
        input_example=input_example,
        # Three-part UC name — creates registered model if absent, always adds new version
        registered_model_name="prod.ml_team.cancer_detection",
    )
```

```python
# Scenario: Creating a Model Serving endpoint via the Databricks SDK to expose
# a UC-registered model for real-time inference, with scale-to-zero disabled
# to meet a sub-200 ms p99 SLA in production.

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

w = WorkspaceClient()

w.serving_endpoints.create(
    name="cancer-detection-endpoint",
    config=EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                name="cancer-v1",
                entity_name="prod.ml_team.cancer_detection",
                entity_version="1",
                workload_size="Small",
                scale_to_zero_enabled=False,       # keep warm for SLA
                min_provisioned_concurrency=4,
                max_provisioned_concurrency=16,
            )
        ]
    ),
)
```

```python
# Scenario: Promoting the Challenger model to Champion once it has proven superior
# accuracy in A/B traffic, with no application code change needed.

from mlflow import MlflowClient

client = MlflowClient()
model_name = "prod.ml_team.cancer_detection"

# Challenger (version 2) has won the A/B test; move the Champion alias atomically
client.set_registered_model_alias(model_name, "Champion", 2)
# Optionally demote old champion alias (alias per model, not per version)
# The previous version 1 simply loses its Champion sticker; it still exists.
```

```python
# Scenario: Querying the serving endpoint programmatically from another notebook or
# service, using the MLflow Deployments SDK for token-free authentication via the
# Databricks environment context.

from mlflow.deployments import get_deploy_client
import pandas as pd

client = get_deploy_client("databricks")

sample = pd.DataFrame({
    "mean radius": [17.99],
    "mean texture": [10.38],
    # ... remaining features omitted for brevity
})

response = client.predict(
    endpoint="cancer-detection-endpoint",
    inputs={
        "dataframe_split": {
            "columns": sample.columns.tolist(),
            "data": sample.values.tolist(),
        }
    },
)
print(response["predictions"])
```

```python
# Anti-pattern: Registering with a single-part model name when the registry URI
# is still pointing at the legacy Workspace Model Registry.
#
# What breaks: The model is silently registered in the Workspace Model Registry,
# not in Unity Catalog. It receives no UC lineage, no RBAC from catalog privileges,
# and is unreachable via the three-part namespace. Serving endpoints configured
# with "catalog.schema.model_name" will fail to resolve the model.

import mlflow

# WRONG — registry URI defaults to workspace model registry
# mlflow.register_model("runs:/abc123/model", "cancer_detection")

# Correct approach: set the registry URI first, then use the three-part name.
mlflow.set_registry_uri("databricks-uc")
mlflow.register_model(
    "runs:/abc123/model",
    "prod.ml_team.cancer_detection",   # three-part name required
)
```

---

## Common Pitfalls & Misconceptions

- **Registering a model deploys it** — Beginners conflate `register_model()` with making the model callable via HTTP, because both happen in sequence during onboarding tutorials. Registration is an artifact-cataloguing step; it stores the model and its metadata in UC but provisions no compute. A separate `serving_endpoints.create()` call is required to expose it to traffic.

- **Using the two-part name accidentally targets the legacy registry** — When the workspace default catalog is still `hive_metastore` and `set_registry_uri("databricks-uc")` has not been called, passing `"my_schema.my_model"` silently creates a model in the workspace registry under that exact two-part string, bypassing UC entirely. Always verify with `mlflow.get_registry_uri()` before calling `register_model()` in a new environment.

- **Aliases are mutable; version numbers are immutable** — Engineers sometimes set a serving endpoint to always resolve the `Champion` alias, then are surprised when the endpoint still serves the old model after re-assigning the alias. The endpoint resolves the alias at *creation/update time* to a fixed version number unless the endpoint config explicitly references an alias URI. To pick up alias changes without an endpoint update, use batch-inference code that calls `get_model_version_by_alias()` dynamically and then updates the endpoint config programmatically.

- **Scale-to-zero is on by default and causes silent SLA violations** — Because scale-to-zero is convenient for development, teams often forget to disable it before going to production. The first request after an idle period hits a cold-start delay of 30–90 seconds, breaching any sub-second SLA. Always set `scale_to_zero_enabled=False` for production endpoints.

- **Traffic percentages that do not sum to 100 are rejected at update time** — When adding a Challenger route by updating only its percentage without adjusting the existing Champion route, the API returns a validation error. Plan both route values before issuing the `PUT /config` call.

- **The endpoint creator identity is immutable** — The UC identity that created the endpoint is permanently recorded and re-validated on every config update. If that service principal is deactivated or loses `EXECUTE` on the model, all subsequent config updates fail with `PERMISSION_DENIED` even if the caller has valid permissions. Always use a long-lived team-owned service principal, not a personal account, as the endpoint creator.

---

## Key Definitions

| Term | Definition |
|---|---|
| Registered Model | A named container in the MLflow Model Registry (UC-backed) that groups one or more versioned model artifacts under a three-level UC identifier (`catalog.schema.model_name`). |
| Model Version | A single, immutable, numbered snapshot of a model artifact within a registered model; created by each call to `register_model()` or `log_model(registered_model_name=...)`. |
| Model Alias | A mutable named pointer to a specific model version (e.g., `Champion`, `Challenger`) that allows code to reference models by role rather than by version number. |
| Model Serving Endpoint | A Databricks-managed, auto-scaling REST endpoint that loads a specific model version and exposes it at `POST /serving-endpoints/{name}/invocations`. |
| Served Entity | A single model version + compute configuration block within an endpoint's config; multiple served entities per endpoint enable A/B traffic splitting. |
| Traffic Config | The routing table on an endpoint that maps each served entity name to a percentage of incoming requests; all percentages must sum to 100. |
| `databricks-uc` | The registry URI string passed to `mlflow.set_registry_uri()` that routes all MLflow Model Registry API calls to the Unity Catalog-hosted registry. |
| Scale-to-Zero | An endpoint mode that suspends all compute replicas after a configurable idle period, reducing cost but introducing cold-start latency on resume. |
| `CREATE MODEL` | The UC privilege (on a schema) required to register a new model; equivalent to `CREATE FUNCTION` and subject to the same schema-level grants. |
| `EXECUTE` | The UC privilege (on a registered model) required to load model artifacts for batch inference or to validate endpoint access at creation/update time. |
| Champion/Challenger | A deployment pattern where `Champion` alias points to the current production version and `Challenger` alias points to the candidate version; traffic is split between them during evaluation. |

---

## Summary / Quick Recall

- Always call `mlflow.set_registry_uri("databricks-uc")` before registering unless on DBR 13.3+ with a UC default catalog.
- The three-part name `catalog.schema.model_name` is the single source of truth for identity, access control, and lineage of a model.
- `mlflow.register_model()` creates or increments a version; `create_registered_model()` creates only the shell.
- Aliases (`Champion`, `Challenger`) replace stages and are the recommended way to manage deployment state in UC.
- A serving endpoint is independent of registration; `serving_endpoints.create()` is a separate, compute-provisioning call.
- Traffic splits require all `traffic_percentage` values to sum to 100; the split is applied at the load balancer, not in model code.
- Set `scale_to_zero_enabled=False` for any endpoint with a latency SLA.
- The endpoint's recorded creator identity is permanent; use a service principal, not a personal account.

---

## Self-Check Questions

1. What is the minimum set of MLflow API calls required to make a newly-trained scikit-learn model queryable via a Databricks Model Serving endpoint?

   <details><summary>Answer</summary>

   Three steps are required: (1) Log the model with `mlflow.sklearn.log_model(..., registered_model_name="catalog.schema.model")` or separately call `mlflow.register_model(run_uri, "catalog.schema.model")` — this registers the model version in UC. (2) Create a serving endpoint with `w.serving_endpoints.create(name=..., config=EndpointCoreConfigInput(served_entities=[ServedEntityInput(entity_name="catalog.schema.model", entity_version="1", ...)]))`. (3) Wait for the endpoint state to reach `READY`, then `POST /serving-endpoints/{name}/invocations` with the input payload.

   A common wrong answer is "just call `mlflow.register_model()`", which only catalogues the artifact — it does not provision any compute or HTTP interface. Serving is a separate control-plane operation.

   </details>

2. Your team's nightly batch job loads a model using `mlflow.pyfunc.load_model("models:/prod.ml_team.fraud@Champion")`. After a Challenger model beats the Champion in offline evaluation, you update the alias: `client.set_registered_model_alias("prod.ml_team.fraud", "Champion", 7)`. What happens the next time the batch job runs?

   <details><summary>Answer</summary>

   The batch job automatically loads version 7, because the `@Champion` URI resolves the alias at load time — every call to `mlflow.pyfunc.load_model()` performs a live alias lookup. No code change is required. This is the key benefit of the Champion/Challenger pattern over pinning version numbers in code.

   A tempting distractor is thinking the job still loads the old version because serving endpoints cache the version at creation time. That is true for serving endpoints (which must be explicitly updated to pick up alias changes), but batch inference code that calls `load_model` directly always re-resolves the alias on each invocation.

   </details>

3. **Which TWO** of the following statements correctly describe the relationship between UC privileges and Model Serving endpoint access?
   - A. The caller who sends the `/invocations` request must hold `EXECUTE` on the registered model.
   - B. The endpoint's recorded creator must hold `EXECUTE` on the registered model; this is re-validated on every config update.
   - C. `CAN_QUERY` permission on the serving endpoint is sufficient for end-users to invoke it; no UC model privilege is required for the end-user.
   - D. Stages (Staging/Production) control which versions can be served; aliases are only for documentation.
   - E. The `CREATE MODEL` privilege on the catalog is required to create a serving endpoint.

   <details><summary>Answer</summary>

   **B and C** are correct.

   **B** is correct: the recorded creator's `EXECUTE` grant on the model is re-validated every time the endpoint config is updated; if the creator loses the grant or is removed from the workspace, config updates fail with `PERMISSION_DENIED`.

   **C** is correct: end-users querying the endpoint only need `CAN_QUERY` on the *serving endpoint* object, not `EXECUTE` on the underlying UC model. The endpoint acts as a proxy; the model-level access check is against the creator, not the caller.

   **A** is wrong: the end-user (invocations caller) does not need `EXECUTE` on the UC model — only the endpoint creator does.

   **D** is wrong: stages are not supported in UC. Aliases are the replacement for stages and are functionally significant (not decorative).

   **E** is wrong: `CREATE MODEL` is needed to *register* a model, not to create a serving endpoint. Endpoint creation requires `EXECUTE` on the model (held by the creator) and workspace `CAN_MANAGE` serving endpoint permissions.

   </details>

4. A production serving endpoint is exhibiting 45-second first-response latency on occasional requests, followed by normal sub-200 ms responses. Logs show no model inference errors. What is the most likely cause and how should it be fixed?

   <details><summary>Answer</summary>

   The most likely cause is `scale_to_zero_enabled=True` on the endpoint. When no requests arrive for the idle period, Databricks suspends all compute replicas. The next request triggers a cold start — the control plane must spin up a container, load the model weights, and warm the JVM/Python runtime — which typically takes 30–90 seconds. The correct fix is to update the endpoint config to set `scale_to_zero_enabled=False` and set `min_provisioned_concurrency` to at least 4, keeping at least one warm replica at all times.

   A plausible distractor is "the model is too large for the workload size" — but model-size issues manifest as consistently high latency or OOM errors, not intermittent 45-second spikes followed by fast responses.

   </details>

5. You need to run an A/B test between model version 3 (current Champion) and version 5 (Challenger) on a live endpoint. Version 5 is already registered in `prod.ml_team.recommender`. Which approach introduces the least operational risk and why?

   <details><summary>Answer</summary>

   The safest approach is to update the existing endpoint's config in-place using `PUT /api/2.0/serving-endpoints/{name}/config` (or the SDK equivalent), adding `fraud-v5` as a second served entity with `traffic_percentage: 10` and reducing the existing route to `traffic_percentage: 90`. The control plane performs a rolling update — the old config continues serving 100% of traffic until the new config is `READY`, so there is no gap in service.

   Deleting and recreating the endpoint is far riskier: the endpoint is unavailable during recreation, any calling applications that use the endpoint URL will receive 404 errors, and previously configured settings (permissions, tags, inference table config) are lost. Deploying a new endpoint under a different name and shifting DNS/URL is operationally complex and forces all callers to update their endpoint URL.

   The in-place config update with traffic split is the canonical Databricks recommended pattern for Challenger evaluation.

   </details>

---

## Further Reading

- [Manage model lifecycle in Unity Catalog](https://docs.databricks.com/en/machine-learning/manage-model-lifecycle/index.html) — *verified 2026-07-11* — Primary reference for registering models, aliases, lineage, and access control in UC.
- [Create custom model serving endpoints](https://docs.databricks.com/en/machine-learning/model-serving/create-manage-serving-endpoints.html) — *verified 2026-07-11* — Full endpoint creation reference including SDK, REST API, and UI flows.
- [Query serving endpoints for custom models](https://docs.databricks.com/en/machine-learning/model-serving/score-custom-model-endpoints.html) — *verified 2026-07-11* — Supported scoring formats (`dataframe_split`, `instances`, `inputs`) and authentication patterns.
- [Manage model serving endpoints](https://docs.databricks.com/en/machine-learning/model-serving/manage-serving-endpoints.html) — *verified 2026-07-11* — Traffic updates, permissions, status checks, and debug logs.
