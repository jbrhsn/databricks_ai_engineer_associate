# LAB-14: Unity Catalog Model Registration and Serving

**Lab:** LAB-14 | **Section:** 05-assembling-and-deploying-applications | **Module:** 02-vector-search-and-serving | **Chapter:** 02-unity-catalog-registration-serving | **Est. time:** 90 min

---

## Objective

By the end of this lab you will have:
1. Registered an MLflow model to Unity Catalog using a three-level namespace.
2. Applied Champion and Challenger aliases to model versions.
3. Created a Model Serving endpoint via the Databricks SDK with explicit concurrency settings.
4. Issued a scored inference request against the `/invocations` endpoint using the MLflow Deployments SDK.
5. Updated the endpoint to implement a 90/10 A/B traffic split between Champion and Challenger.

---

## Prerequisites

- A Databricks workspace with Unity Catalog enabled (DBR 13.3 LTS ML or above recommended).
- A UC catalog named `dev` (or adjust the variable `UC_CATALOG` to an available catalog).
- A UC schema named `lab14` inside that catalog, or `CREATE SCHEMA` privilege on the catalog.
- `CREATE MODEL` privilege on `dev.lab14` (or equivalent).
- Workspace access to create Model Serving endpoints (workspace admin or CAN_MANAGE on serving endpoints).
- Python packages: `mlflow>=2.10`, `scikit-learn>=1.3`, `databricks-sdk>=0.20`.

---

## Setup

```python
# Cell 1 — Variable configuration
# Adjust these to match your workspace topology.
UC_CATALOG   = "dev"
UC_SCHEMA    = "lab14"
MODEL_NAME   = f"{UC_CATALOG}.{UC_SCHEMA}.wine_quality"
ENDPOINT_NAME = "lab14-wine-endpoint"
```

```python
# Cell 2 — Ensure the schema exists
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {UC_CATALOG}.{UC_SCHEMA}")
print(f"Schema ready: {UC_CATALOG}.{UC_SCHEMA}")
```

```python
# Cell 3 — Install / upgrade MLflow if needed (run only on older runtimes)
# %pip install --upgrade "mlflow-skinny[databricks]"
# dbutils.library.restartPython()
```

```python
# Cell 4 — Point MLflow at Unity Catalog
import mlflow
mlflow.set_registry_uri("databricks-uc")
print(f"Registry URI: {mlflow.get_registry_uri()}")
# Expected output: databricks-uc
```

---

## Steps

### Step 1 — Train and Register Model Version 1 (Champion candidate)

```python
# Cell 5 — Train v1: Ridge regression on the UCI Wine Quality dataset
from sklearn.datasets import load_wine
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

X, y = load_wine(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="wine-ridge-v1") as run_v1:
    alpha = 1.0
    mlflow.log_param("alpha", alpha)

    model_v1 = Ridge(alpha=alpha)
    model_v1.fit(X_train, y_train)

    mse = mean_squared_error(y_test, model_v1.predict(X_test))
    mlflow.log_metric("mse", mse)
    print(f"v1 MSE: {mse:.4f}")

    input_example = X_train.iloc[:3]

    mlflow.sklearn.log_model(
        sk_model=model_v1,
        artifact_path="model",
        input_example=input_example,
        registered_model_name=MODEL_NAME,
    )

run_id_v1 = run_v1.info.run_id
print(f"Run ID v1: {run_id_v1}")
```

### Step 2 — Train and Register Model Version 2 (Challenger candidate)

```python
# Cell 6 — Train v2: lower regularization, different hyperparameters
with mlflow.start_run(run_name="wine-ridge-v2") as run_v2:
    alpha = 0.1
    mlflow.log_param("alpha", alpha)

    model_v2 = Ridge(alpha=alpha)
    model_v2.fit(X_train, y_train)

    mse = mean_squared_error(y_test, model_v2.predict(X_test))
    mlflow.log_metric("mse", mse)
    print(f"v2 MSE: {mse:.4f}")

    mlflow.sklearn.log_model(
        sk_model=model_v2,
        artifact_path="model",
        input_example=X_train.iloc[:3],
        registered_model_name=MODEL_NAME,
    )

run_id_v2 = run_v2.info.run_id
print(f"Run ID v2: {run_id_v2}")
```

### Step 3 — Assign Aliases

```python
# Cell 7 — Set Champion (v1) and Challenger (v2) aliases
from mlflow import MlflowClient

client = MlflowClient()

client.set_registered_model_alias(MODEL_NAME, "Champion", 1)
client.set_registered_model_alias(MODEL_NAME, "Challenger", 2)

champion = client.get_model_version_by_alias(MODEL_NAME, "Champion")
challenger = client.get_model_version_by_alias(MODEL_NAME, "Challenger")
print(f"Champion → version {champion.version}")
print(f"Challenger → version {challenger.version}")
```

### Step 4 — Create the Serving Endpoint

```python
# Cell 8 — Create a single-entity serving endpoint for Champion
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

w = WorkspaceClient()

# Delete endpoint if it already exists from a previous lab run
existing = [ep.name for ep in w.serving_endpoints.list()]
if ENDPOINT_NAME in existing:
    w.serving_endpoints.delete(name=ENDPOINT_NAME)
    import time; time.sleep(10)
    print(f"Deleted existing endpoint: {ENDPOINT_NAME}")

w.serving_endpoints.create(
    name=ENDPOINT_NAME,
    config=EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                name="wine-champion",
                entity_name=MODEL_NAME,
                entity_version=str(champion.version),
                workload_size="Small",
                scale_to_zero_enabled=True,   # acceptable for lab (not production)
            )
        ]
    ),
)
print(f"Endpoint creation initiated: {ENDPOINT_NAME}")
print("Wait ~5-10 minutes for READY state before proceeding to Step 5.")
```

```python
# Cell 9 — Poll until the endpoint is READY
import time

for attempt in range(30):
    ep = w.serving_endpoints.get(name=ENDPOINT_NAME)
    state = ep.state.ready.value if ep.state and ep.state.ready else "UNKNOWN"
    print(f"[{attempt+1}/30] Endpoint state: {state}")
    if state == "READY":
        print("Endpoint is READY.")
        break
    time.sleep(30)
else:
    raise TimeoutError("Endpoint did not reach READY within 15 minutes.")
```

### Step 5 — Query the Endpoint

```python
# Cell 10 — Score the Champion model via the invocations endpoint
from mlflow.deployments import get_deploy_client

deploy_client = get_deploy_client("databricks")

payload = {
    "dataframe_split": {
        "columns": X_test.columns.tolist(),
        "data": X_test.iloc[:5].values.tolist(),
    }
}

response = deploy_client.predict(endpoint=ENDPOINT_NAME, inputs=payload)
print("Predictions:", response["predictions"])

# Expected: a list of 5 float predictions (wine quality scores)
```

### Step 6 — Update to A/B Traffic Split (90% Champion / 10% Challenger)

```python
# Cell 11 — Add Challenger as a second served entity with 10% traffic
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    TrafficConfig,
    Route,
)

w.serving_endpoints.update_config(
    name=ENDPOINT_NAME,
    served_entities=[
        ServedEntityInput(
            name="wine-champion",
            entity_name=MODEL_NAME,
            entity_version=str(champion.version),
            workload_size="Small",
            scale_to_zero_enabled=True,
        ),
        ServedEntityInput(
            name="wine-challenger",
            entity_name=MODEL_NAME,
            entity_version=str(challenger.version),
            workload_size="Small",
            scale_to_zero_enabled=True,
        ),
    ],
    traffic_config=TrafficConfig(
        routes=[
            Route(served_model_name="wine-champion",    traffic_percentage=90),
            Route(served_model_name="wine-challenger",  traffic_percentage=10),
        ]
    ),
)
print("Endpoint update initiated: 90/10 traffic split.")
```

```python
# Cell 12 — Poll until update is complete
for attempt in range(30):
    ep = w.serving_endpoints.get(name=ENDPOINT_NAME)
    update_state = ep.state.config_update.value if ep.state and ep.state.config_update else "UNKNOWN"
    print(f"[{attempt+1}/30] Config update state: {update_state}")
    if update_state == "NOT_UPDATING":
        print("Config update complete.")
        break
    time.sleep(30)
else:
    raise TimeoutError("Endpoint update did not complete within 15 minutes.")
```

### Step 7 — Verify Traffic Split by Inspecting the Config

```python
# Cell 13 — Print the active traffic config
ep = w.serving_endpoints.get(name=ENDPOINT_NAME)
for route in ep.config.traffic_config.routes:
    print(f"  {route.served_model_name}: {route.traffic_percentage}%")

# Expected output:
#   wine-champion: 90%
#   wine-challenger: 10%
```

---

## Validation

Run each check below. All should pass before marking the lab complete.

```python
# Validation Cell — Automated checks

errors = []

# Check 1: Both model versions exist in UC
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
version_numbers = {int(v.version) for v in versions}
if {1, 2}.issubset(version_numbers):
    print("PASS: Versions 1 and 2 registered in UC")
else:
    errors.append(f"FAIL: Expected versions 1 and 2; found {version_numbers}")

# Check 2: Aliases are correctly set
champ = client.get_model_version_by_alias(MODEL_NAME, "Champion")
chall = client.get_model_version_by_alias(MODEL_NAME, "Challenger")
if int(champ.version) == 1 and int(chall.version) == 2:
    print("PASS: Champion=v1, Challenger=v2")
else:
    errors.append(f"FAIL: Champion={champ.version}, Challenger={chall.version}")

# Check 3: Endpoint is READY
ep = w.serving_endpoints.get(name=ENDPOINT_NAME)
if ep.state.ready.value == "READY":
    print("PASS: Endpoint is READY")
else:
    errors.append(f"FAIL: Endpoint state is {ep.state.ready.value}")

# Check 4: Traffic split is 90/10
routes = {r.served_model_name: r.traffic_percentage for r in ep.config.traffic_config.routes}
if routes.get("wine-champion") == 90 and routes.get("wine-challenger") == 10:
    print("PASS: Traffic split is 90/10")
else:
    errors.append(f"FAIL: Traffic config is {routes}")

# Check 5: Inference returns predictions
response = deploy_client.predict(endpoint=ENDPOINT_NAME, inputs=payload)
if "predictions" in response and len(response["predictions"]) == 5:
    print("PASS: Endpoint returns 5 predictions")
else:
    errors.append(f"FAIL: Unexpected response: {response}")

if errors:
    for e in errors:
        print(e)
    raise AssertionError("One or more validation checks failed.")
else:
    print("\nAll validation checks passed.")
```

---

## Teardown

Run these cells after the lab to avoid incurring unnecessary compute charges.

```python
# Teardown Cell 1 — Delete the serving endpoint
w.serving_endpoints.delete(name=ENDPOINT_NAME)
print(f"Deleted endpoint: {ENDPOINT_NAME}")
```

```python
# Teardown Cell 2 — Delete registered model (all versions and aliases)
client.delete_registered_model(name=MODEL_NAME)
print(f"Deleted registered model: {MODEL_NAME}")
```

```python
# Teardown Cell 3 — (Optional) Drop the schema
# Uncomment only if you want to remove all lab artifacts including the schema.
# spark.sql(f"DROP SCHEMA IF EXISTS {UC_CATALOG}.{UC_SCHEMA} CASCADE")
# print(f"Dropped schema: {UC_CATALOG}.{UC_SCHEMA}")
```

---

## Reflection Questions

1. In Step 4, the lab uses `scale_to_zero_enabled=True`. Under what production conditions would this cause an SLA violation, and what configuration change would you make to prevent it?

2. In Step 7, the traffic split is controlled by `traffic_config.routes`. What would happen if you set `wine-champion` to `90` and `wine-challenger` to `15` (a total of 105)? Where would this error surface — at the SDK call, the REST API, or at inference time?

3. The lab assigns the `Champion` alias to version 1 and the `Challenger` alias to version 2. If you now call `client.set_registered_model_alias(MODEL_NAME, "Champion", 2)` *without* updating the endpoint config, does the endpoint start serving version 2? Explain why or why not, and describe what additional step would be required to serve the new Champion version from the endpoint.
