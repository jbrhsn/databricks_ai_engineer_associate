# LAB-15: Foundation Model APIs and ai_query

**Lab:** LAB-15 | **Section:** 05-Assembling and Deploying Applications | **Module:** 02-Vector Search and Serving | **Est. time:** 1.5 hrs

---

## Objective

Call a Databricks Foundation Model API endpoint from Python using the OpenAI SDK, run batch LLM inference using `ai_query()` in a SQL statement, and configure an external model endpoint with centrally managed credentials — demonstrating all three FMAPI interaction patterns in a single end-to-end workflow.

---

## Prerequisites

- Databricks workspace with Model Serving enabled in a supported region
- Serverless compute enabled (required for `ai_query()`)
- A personal access token (PAT) or OAuth M2M token with `CAN_QUERY` permission on Model Serving endpoints
- `databricks-openai` Python package available (install in the notebook)
- A Unity Catalog with write access to create a table (e.g., `main.lab15`)
- Optional for Step 4: An OpenAI or Anthropic API key stored in a Databricks Secret scope (e.g., `secrets/lab15/openai_key`)

---

## Setup

Create the lab schema and a sample input table.

```sql
-- Run in a Databricks SQL editor or notebook cell
CREATE SCHEMA IF NOT EXISTS main.lab15;

CREATE OR REPLACE TABLE main.lab15.feedback (
  feedback_id  INT,
  text         STRING
) USING DELTA;

INSERT INTO main.lab15.feedback VALUES
  (1, 'The product arrived on time but the packaging was damaged.'),
  (2, 'Absolutely loved the experience — fast, friendly, and professional.'),
  (3, 'I had an issue with my order and the support team resolved it immediately.'),
  (4, 'Terrible service, waited 3 weeks and received the wrong item.'),
  (5, 'Good value for money. Would consider buying again.');
```

Install the Python package in a notebook:

```python
# Setup: install the databricks-openai package for auto-configured auth
%pip install -U databricks-openai
dbutils.library.restartPython()
```

---

## Steps

### Step 1 — Explore Available Pay-per-Token Endpoints

Navigate to **Serving** in the workspace sidebar. Observe the preconfigured Foundation Model API endpoints at the top of the list. Note the endpoint names — they follow the pattern `databricks-<model-name>`.

In a notebook cell, confirm you can list endpoints programmatically:

```python
# Scenario: discover available FMAPI endpoint names before coding them into a pipeline
import requests, os

host  = os.environ["DATABRICKS_HOST"]
token = os.environ["DATABRICKS_TOKEN"]

resp = requests.get(
    f"{host}/api/2.0/serving-endpoints",
    headers={"Authorization": f"Bearer {token}"},
)

endpoints = resp.json().get("endpoints", [])
fmapi_endpoints = [e["name"] for e in endpoints if e["name"].startswith("databricks-")]
print(f"Found {len(fmapi_endpoints)} FMAPI endpoints:")
for name in fmapi_endpoints[:10]:   # print first 10
    print(f"  {name}")
```

**Expected output:** A list of endpoint names beginning with `databricks-`, such as `databricks-meta-llama-3-3-70b-instruct` and `databricks-bge-large-en`.

---

### Step 2 — Call FMAPI from Python Using the OpenAI SDK

Use a pay-per-token endpoint to classify the five feedback rows as `complaint`, `query`, or `compliment`.

```python
# Scenario: classify customer feedback using FMAPI pay-per-token endpoint
# from Python, to validate prompt and model before committing to batch SQL inference.

import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DATABRICKS_TOKEN"],
    base_url=f"{os.environ['DATABRICKS_HOST']}/serving-endpoints",
)

ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"  # pay-per-token endpoint

test_texts = [
    "The product arrived on time but the packaging was damaged.",
    "Absolutely loved the experience — fast, friendly, and professional.",
]

for text in test_texts:
    response = client.chat.completions.create(
        model=ENDPOINT,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a customer feedback classifier. "
                    "Classify the following feedback as exactly one of: "
                    "complaint, query, or compliment. "
                    "Reply with only the single label word, lowercase."
                ),
            },
            {"role": "user", "content": text},
        ],
        max_tokens=5,
        temperature=0.0,
    )
    label = response.choices[0].message.content.strip()
    print(f"[{label}] {text[:60]}...")
```

**Expected output:**
```
[complaint] The product arrived on time but the packaging was dam...
[compliment] Absolutely loved the experience — fast, friendly, an...
```

---

### Step 3 — Batch Inference with `ai_query()` in SQL

Scale the classification to all rows in the Delta table using a single SQL statement.

```sql
-- Scenario: classify all 5 feedback rows in one SQL batch using ai_query(),
-- leveraging Databricks internal parallelisation instead of a Python loop.
-- failOnError=false preserves successful rows if any row produces a model error.

CREATE OR REPLACE TABLE main.lab15.classified_feedback AS
SELECT
  feedback_id,
  text,
  ai_query(
    'databricks-meta-llama-3-3-70b-instruct',
    CONCAT(
      'Classify the following feedback as exactly one of: ',
      'complaint, query, or compliment. ',
      'Reply with only the single label word, lowercase. ',
      'Feedback: ', text
    ),
    modelParameters => named_struct('max_tokens', 5, 'temperature', 0.0),
    failOnError => false
  ) AS classification
FROM main.lab15.feedback;

-- Inspect results
SELECT * FROM main.lab15.classified_feedback ORDER BY feedback_id;
```

**Expected output:** Five rows with `classification` values of `complaint`, `compliment`, `query`, `complaint`, and `compliment` (exact labels depend on the model).

---

### Step 4 — Create an External Model Endpoint (Optional — requires API key)

If you have an OpenAI API key stored in a Databricks Secret scope (`secrets/lab15/openai_key`), create an external model endpoint that routes to OpenAI through Databricks, enabling centralised credential management.

```python
# Scenario: create an external model endpoint for OpenAI so the team accesses
# GPT models through Databricks with centralised API key management,
# without distributing the raw key to application developers.

import mlflow.deployments

deploy_client = mlflow.deployments.get_deploy_client("databricks")

deploy_client.create_endpoint(
    name="lab15-openai-chat",
    config={
        "served_entities": [
            {
                "name": "openai-gpt-entity",
                "external_model": {
                    "name": "gpt-4o-mini",
                    "provider": "openai",
                    "task": "llm/v1/chat",
                    "openai_config": {
                        # Key injected at request time — never visible to callers
                        "openai_api_key": "{{secrets/lab15/openai_key}}"
                    },
                },
            }
        ]
    },
)
print("External endpoint created: lab15-openai-chat")
```

Query the external endpoint using the same OpenAI SDK pattern as Step 2 — only the `model` string changes:

```python
# Scenario: confirm the external endpoint behaves identically to a Databricks-hosted
# endpoint from the caller's perspective — only the endpoint name changes.

import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DATABRICKS_TOKEN"],
    base_url=f"{os.environ['DATABRICKS_HOST']}/serving-endpoints",
)

response = client.chat.completions.create(
    model="lab15-openai-chat",    # external endpoint name, not the OpenAI model name
    messages=[{"role": "user", "content": "What is 2 + 2?"}],
    max_tokens=10,
)
print(response.choices[0].message.content)
```

---

### Step 5 — Inspect Usage Tracking (if Unity AI Gateway is enabled)

If your workspace has Unity AI Gateway Beta enabled, check the usage system table:

```sql
-- Scenario: audit FMAPI token consumption for the lab to understand
-- billing and usage attribution before moving to production.

SELECT
  timestamp,
  endpoint_name,
  request_id,
  total_token_count,
  input_token_count,
  output_token_count
FROM system.serving.served_entities_usage_logs
WHERE endpoint_name = 'databricks-meta-llama-3-3-70b-instruct'
  AND timestamp >= current_timestamp() - INTERVAL 1 HOUR
ORDER BY timestamp DESC
LIMIT 20;
```

---

## Validation

Run all checks below. Each should return a non-error result.

```sql
-- Check 1: classified_feedback table was created with all 5 rows
SELECT COUNT(*) AS row_count FROM main.lab15.classified_feedback;
-- Expected: 5

-- Check 2: all rows have a non-null classification
SELECT COUNT(*) AS null_labels
FROM main.lab15.classified_feedback
WHERE classification IS NULL;
-- Expected: 0

-- Check 3: all labels are in the expected set
SELECT DISTINCT classification FROM main.lab15.classified_feedback;
-- Expected: values drawn from {complaint, query, compliment}
```

```python
# Check 4: Python OpenAI SDK call returns a valid response object
assert response.choices[0].message.content is not None, "No content in response"
assert len(response.choices) == 1, "Expected exactly one choice"
print("Python SDK check: PASS")
```

---

## Teardown

Remove all lab resources to avoid unnecessary storage costs. Delete the external endpoint only if you created it in Step 4.

```sql
-- Remove lab tables
DROP TABLE IF EXISTS main.lab15.feedback;
DROP TABLE IF EXISTS main.lab15.classified_feedback;
DROP SCHEMA IF EXISTS main.lab15;
```

```python
# Remove the external model endpoint (Step 4 only)
import mlflow.deployments
deploy_client = mlflow.deployments.get_deploy_client("databricks")
try:
    deploy_client.delete_endpoint("lab15-openai-chat")
    print("External endpoint deleted.")
except Exception as e:
    print(f"Endpoint may not exist or already deleted: {e}")
```

---

## Reflection Questions

1. In Step 3, you submitted all 5 rows in a single `ai_query()` call. If you had instead split the table into 5 individual single-row queries and called them sequentially, how would that affect throughput, and what Databricks feature would you lose?

2. You need to swap the classification model from `databricks-meta-llama-3-3-70b-instruct` to a new provisioned throughput endpoint for a fine-tuned variant. What is the minimum code change required in Step 3's SQL, and what would you need to do in the Databricks Serving UI first?

3. How does the external model endpoint in Step 4 improve the security posture compared to each team member storing their own OpenAI API key in a notebook, and what additional governance capability does this enable when Unity AI Gateway is applied to the endpoint?
