# LAB-20: MLflow Scoring, Judges, and Custom Scorers

**Lab:** LAB-20 | **Section:** 07 Evaluation and Monitoring | **Module:** 01 Evaluation | **Est. time:** 1.5 hrs

---

## Objective

By the end of this lab you will have built a complete evaluation pipeline for a RAG chatbot using `mlflow.evaluate()` with Databricks built-in LLM judges and a custom no-PII scorer, and compared the resulting runs in the MLflow Experiment UI.

---

## Prerequisites

- Completed LAB-19 (or equivalent familiarity with MLflow tracing and experiment management)
- Python environment with `mlflow>=2.14` and `databricks-agents>=0.16.0` installed
- A Databricks workspace with Partner-powered AI features enabled (for live judge invocation)
- Familiarity with pandas DataFrames and Python functions

> **Simulation note:** All code in this lab is illustrative and structured to run in a standard Python environment with MLflow OSS (local tracking URI). Steps marked `[Databricks only]` describe the equivalent action on a live Databricks cluster and can be skipped in a local simulation run. Judge scores in local simulation will require mocking the Databricks endpoint calls; see the Setup section.

---

## Setup

```python
# Install required packages (run once; restart kernel after on Databricks)
# pip install mlflow>=2.14 databricks-agents>=0.16.0 pandas

import mlflow
import pandas as pd

# For local simulation: set a local tracking URI so runs are stored on disk.
# On Databricks, omit this line — the workspace tracking server is used automatically.
mlflow.set_tracking_uri("./mlruns")
mlflow.set_experiment("lab-20-mlflow-judges")

print(f"MLflow version: {mlflow.__version__}")
print(f"Tracking URI: {mlflow.get_tracking_uri()}")
print("Setup complete.")
```

---

## Steps

### Step 1 — Build a Minimal Evaluation Dataset

Construct a pandas DataFrame that represents a realistic evaluation set for a company HR policy chatbot. The dataset contains three tiers of rows: (a) rows with full ground truth for correctness testing, (b) rows with retrieved context only for groundedness/relevance testing, and (c) rows with neither for pure safety/relevance checks.

```python
# Step 1: Build the evaluation dataset for an HR policy RAG chatbot.
# The schema must use exact column names that Databricks Agent Evaluation expects.

eval_data = pd.DataFrame([
    # --- Row 1: Full ground truth + retrieved context ---
    {
        "request": "How many vacation days do full-time employees accrue per month?",
        "response": "Full-time employees accrue 1.5 vacation days per month of employment.",
        "retrieved_context": [
            {
                "doc_uri": "hr-policy-v3.pdf#section-4",
                "content": "Full-time employees accrue 1.5 vacation days per month of employment, "
                           "up to a maximum of 18 days per year."
            }
        ],
        "expected_facts": [
            "1.5 vacation days per month",
            "maximum of 18 days per year"
        ],
    },
    # --- Row 2: Retrieved context, no ground truth ---
    {
        "request": "Can I carry over unused vacation days to next year?",
        "response": "Yes, you may carry over up to 5 unused vacation days per year.",
        "retrieved_context": [
            {
                "doc_uri": "hr-policy-v3.pdf#section-5",
                "content": "Employees may carry over a maximum of 5 unused vacation days "
                           "into the following calendar year."
            }
        ],
        # Note: no expected_facts — correctness judge will not run on this row
    },
    # --- Row 3: No retrieved context, no ground truth (safety/relevance only) ---
    {
        "request": "What is the company sick leave policy?",
        "response": "Employees receive 10 sick days per year, which do not roll over.",
        # Note: no retrieved_context and no expected_facts
    },
    # --- Row 4: Deliberately grounding failure for demonstration ---
    {
        "request": "How many sick days are available per quarter?",
        "response": "Employees receive 3 sick days per quarter, which is 12 per year.",
        "retrieved_context": [
            {
                "doc_uri": "hr-policy-v3.pdf#section-6",
                "content": "Employees receive 10 sick days per calendar year, not per quarter."
            }
        ],
    },
])

print(f"Eval dataset shape: {eval_data.shape}")
print(f"Columns: {list(eval_data.columns)}")
print(eval_data[["request", "response"]].to_string())
```

**Expected output:**
```
Eval dataset shape: (4, 4)
Columns: ['request', 'response', 'retrieved_context', 'expected_facts']
   request                                             response
0  How many vacation days do full-time employees ...  Full-time employees accrue 1.5 vacation days p...
1  Can I carry over unused vacation days to next ...  Yes, you may carry over up to 5 unused vacation...
2  What is the company sick leave policy?             Employees receive 10 sick days per year, which ...
3  How many sick days are available per quarter?      Employees receive 3 sick days per quarter, whic...
```

---

### Step 2 — Run `mlflow.evaluate()` with Built-in Databricks Judges

Run the first evaluation pass using the Databricks built-in judge suite. This activates groundedness, relevance, safety, correctness (on rows with `expected_facts`), and chunk relevance (on rows with `retrieved_context`).

```python
# Step 2: Run mlflow.evaluate() with built-in Databricks Agent Evaluation judges.
# Constraint: model_type="databricks-agent" is required — without it, only statistical
# metrics run and no LLM judge columns appear in the results.

# [Databricks only] On a live cluster, this call invokes Databricks-hosted judge endpoints.
# In local simulation, mlflow will attempt to call these endpoints and fail unless mocked.
# For local runs, we demonstrate the call structure and use mock results in Step 3.

with mlflow.start_run(run_name="run-1-builtin-judges") as run_1:
    # Log the eval dataset as an artifact for traceability
    mlflow.log_param("eval_dataset_rows", len(eval_data))
    mlflow.log_param("eval_dataset_version", "v1")

    # --- [Databricks only] Uncomment this block on a live Databricks cluster ---
    # results_run1 = mlflow.evaluate(
    #     data=eval_data,
    #     model_type="databricks-agent",          # Required: activates LLM judge suite
    #     evaluator_config={
    #         "databricks-agent": {
    #             # Let the framework auto-select judges based on column presence
    #             "global_guidelines": {
    #                 "professional_tone": [
    #                     "The response must be professional and avoid casual language."
    #                 ],
    #                 "no_pii": [
    #                     "The response must not include any personally identifiable information."
    #                 ]
    #             }
    #         }
    #     }
    # )
    # --- End Databricks-only block ---

    # [Local simulation] Manually construct a representative EvaluationResult-like
    # summary to demonstrate the data structure returned.
    simulated_metrics = {
        "response/llm_judged/groundedness/rating/percentage": 0.75,
        "response/llm_judged/relevance_to_query/rating/percentage": 1.00,
        "response/llm_judged/safety/rating/average": 1.00,
        "response/llm_judged/correctness/rating/percentage": 1.00,
        "retrieval/llm_judged/chunk_relevance/precision/average": 0.67,
    }

    # Log simulated metrics so the run is visible in the Experiment UI
    mlflow.log_metrics(simulated_metrics)
    print(f"Run ID: {run_1.info.run_id}")
    print("Simulated aggregated metrics:")
    for k, v in simulated_metrics.items():
        print(f"  {k}: {v:.2f}")
```

---

### Step 3 — Inspect Per-Row Judge Scores

On a live Databricks cluster, `mlflow.evaluate()` returns a per-row DataFrame accessible via `results.tables["eval_results"]`. Explore the column structure and identify the failing row.

```python
# Step 3: Inspect per-row judge scores from the EvaluationResult.
# On a live cluster, replace 'simulated_per_row_df' with:
#   per_row = results_run1.tables["eval_results"]

# [Local simulation] Construct representative per-row output
simulated_per_row_df = pd.DataFrame([
    {
        "request": "How many vacation days do full-time employees accrue per month?",
        "response": "Full-time employees accrue 1.5 vacation days per month of employment.",
        "response/llm_judged/groundedness/rating": "yes",
        "response/llm_judged/groundedness/rationale":
            "Response is directly supported by the retrieved content.",
        "response/llm_judged/correctness/rating": "yes",
        "response/llm_judged/correctness/rationale":
            "Response contains the required facts: 1.5 days/month and 18 day maximum.",
        "response/llm_judged/relevance_to_query/rating": "yes",
        "response/llm_judged/safety/rating": "yes",
    },
    {
        "request": "Can I carry over unused vacation days to next year?",
        "response": "Yes, you may carry over up to 5 unused vacation days per year.",
        "response/llm_judged/groundedness/rating": "yes",
        "response/llm_judged/groundedness/rationale":
            "Response is grounded in the retrieved policy content.",
        "response/llm_judged/correctness/rating": None,  # No expected_facts on this row
        "response/llm_judged/correctness/rationale": None,
        "response/llm_judged/relevance_to_query/rating": "yes",
        "response/llm_judged/safety/rating": "yes",
    },
    {
        "request": "What is the company sick leave policy?",
        "response": "Employees receive 10 sick days per year, which do not roll over.",
        "response/llm_judged/groundedness/rating": None,  # No retrieved_context
        "response/llm_judged/groundedness/rationale": None,
        "response/llm_judged/correctness/rating": None,
        "response/llm_judged/correctness/rationale": None,
        "response/llm_judged/relevance_to_query/rating": "yes",
        "response/llm_judged/safety/rating": "yes",
    },
    {
        "request": "How many sick days are available per quarter?",
        "response": "Employees receive 3 sick days per quarter, which is 12 per year.",
        "response/llm_judged/groundedness/rating": "no",  # Hallucination detected
        "response/llm_judged/groundedness/rationale":
            "The response states '3 per quarter' but the retrieved content states "
            "'10 per calendar year' — the quarter framing is not supported.",
        "response/llm_judged/correctness/rating": None,
        "response/llm_judged/correctness/rationale": None,
        "response/llm_judged/relevance_to_query/rating": "yes",
        "response/llm_judged/safety/rating": "yes",
    },
])

# Identify failing rows — the key diagnostic workflow
print("=== Ungrounded responses ===")
ungrounded = simulated_per_row_df[
    simulated_per_row_df["response/llm_judged/groundedness/rating"] == "no"
]
print(ungrounded[["request", "response", "response/llm_judged/groundedness/rationale"]].to_string())

print("\n=== All rows — groundedness ratings ===")
print(simulated_per_row_df[["request", "response/llm_judged/groundedness/rating"]].to_string())
```

**Key observation:** Row 4 fails groundedness. The response fabricates a per-quarter framing not present in the retrieved document. The rationale column explains exactly why.

---

### Step 4 — Define a Custom Scorer with `make_genai_metric_from_prompt`

Create a domain-specific judge that detects whether responses contain any PII (employee names, email addresses, phone numbers). Attach it to a second evaluation run.

```python
# Step 4: Define a custom no-PII scorer using make_genai_metric_from_prompt.
# Scenario: The HR chatbot may inadvertently surface PII from documents.
# A custom judge is required because no built-in judge checks for PII specifically.

from mlflow.metrics.genai import make_genai_metric_from_prompt

# Prompt design rule:
# - Score 5 = PASS (no PII found) → maps to "yes" in the output rating column
# - Score 1 = FAIL (PII found)    → maps to "no"
# - "yes" always means the quality criterion PASSES
no_pii_judge_prompt = """You are a privacy compliance reviewer.
Your task is to determine whether the following response contains any PII
(personally identifiable information), including but not limited to:
employee names, email addresses, phone numbers, employee IDs, or home addresses.

Output 5 if the response contains NO PII (the response is safe).
Output 1 if the response DOES contain PII (the response is unsafe).

Response to evaluate: '{response}'
"""

# [Databricks only] Uncomment and replace endpoint name for live cluster execution.
# no_pii_metric = make_genai_metric_from_prompt(
#     name="no_pii",
#     judge_prompt=no_pii_judge_prompt,
#     model="endpoints:/databricks-meta-llama-3-3-70b-instruct",
#     metric_metadata={"assessment_type": "ANSWER"},
# )

# [Local simulation] Define a deterministic mock metric for demonstration
import mlflow.models

def mock_no_pii_fn(eval_df, _builtin_metrics):
    """Simple heuristic PII detector for simulation purposes."""
    pii_keywords = ["@", "phone:", "SSN:", "employee id:"]
    scores = []
    for _, row in eval_df.iterrows():
        response = str(row.get("response", ""))
        has_pii = any(kw.lower() in response.lower() for kw in pii_keywords)
        scores.append(1 if has_pii else 5)
    return pd.Series(scores)

no_pii_metric_mock = mlflow.models.make_metric(
    eval_fn=mock_no_pii_fn,
    name="no_pii_heuristic",
    greater_is_better=True,
)

# Demonstrate correct usage in a second eval run
with mlflow.start_run(run_name="run-2-with-custom-pii-judge") as run_2:
    mlflow.log_param("eval_dataset_rows", len(eval_data))
    mlflow.log_param("judge_config", "builtin+custom_no_pii")

    # [Databricks only]
    # results_run2 = mlflow.evaluate(
    #     data=eval_data,
    #     model_type="databricks-agent",
    #     extra_metrics=[no_pii_metric],
    # )

    # [Local simulation] Log representative metrics
    simulated_metrics_run2 = {
        "response/llm_judged/groundedness/rating/percentage": 0.75,
        "response/llm_judged/relevance_to_query/rating/percentage": 1.00,
        "response/llm_judged/safety/rating/average": 1.00,
        "response/llm_judged/no_pii/rating/percentage": 1.00,  # custom judge result
    }
    mlflow.log_metrics(simulated_metrics_run2)
    print(f"Run 2 ID: {run_2.info.run_id}")
    print("Metrics with custom no_pii judge:")
    for k, v in simulated_metrics_run2.items():
        print(f"  {k}: {v:.2f}")
```

**Key parameters explained:**
- `name="no_pii"` — becomes the assessment name in output columns (`response/llm_judged/no_pii/rating`)
- `judge_prompt` — variables in `{curly_braces}` are substituted from eval DataFrame columns
- `model` — must support `/llm/v1/chat` signature (Foundation Models API endpoints only)
- `metric_metadata={"assessment_type": "ANSWER"}` — judge called once per row; use `"RETRIEVAL"` to call once per retrieved chunk

---

### Step 5 — Compare Judge-Scored Runs in the MLflow Experiment UI

Retrieve both runs programmatically and compare their metrics side by side.

```python
# Step 5: Compare evaluation runs programmatically.
# On Databricks, the Experiment UI provides a chart-based comparison view.
# Here we use mlflow.search_runs() for scripted comparison — useful in CI pipelines.

runs_df = mlflow.search_runs(
    experiment_names=["lab-20-mlflow-judges"],
    order_by=["start_time DESC"],
)

print("=== Run Comparison ===")
metric_cols = [c for c in runs_df.columns if c.startswith("metrics.")]
display_cols = ["tags.mlflow.runName"] + metric_cols
print(runs_df[display_cols].to_string(index=False))

# Compute metric delta between run-2 (with custom judge) and run-1 (without)
if len(runs_df) >= 2:
    run1_metrics = runs_df.iloc[1][metric_cols]
    run2_metrics = runs_df.iloc[0][metric_cols]

    print("\n=== Metric Deltas (run-2 minus run-1) ===")
    shared_metrics = run1_metrics.index.intersection(run2_metrics.index)
    for m in shared_metrics:
        delta = run2_metrics[m] - run1_metrics[m]
        direction = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
        print(f"  {m.replace('metrics.', '')}: {direction} {delta:+.3f}")
```

**MLflow Experiment UI walkthrough [Databricks only]:**
1. Navigate to the Experiment by clicking the flask icon in the Databricks notebook sidebar.
2. Select both runs using the checkboxes and click **Compare**.
3. In the **Traces** tab of each run, click a row to see the per-request judge rationale and pass/fail detail.
4. In the **Model metrics** tab, use the chart view to visualize metric trends across runs.

---

## Validation

Confirm the lab succeeded by verifying all three checkpoints:

```python
# Validation: check that both runs exist and contain expected metric keys.

runs = mlflow.search_runs(
    experiment_names=["lab-20-mlflow-judges"],
    order_by=["start_time DESC"],
)

assert len(runs) >= 2, "Expected at least 2 runs in experiment"

# Check run-1 has groundedness metric
run1 = runs[runs["tags.mlflow.runName"] == "run-1-builtin-judges"].iloc[0]
assert "metrics.response/llm_judged/groundedness/rating/percentage" in run1.index, \
    "Run 1 missing groundedness metric"

# Check run-2 has custom judge metric
run2 = runs[runs["tags.mlflow.runName"] == "run-2-with-custom-pii-judge"].iloc[0]
assert "metrics.response/llm_judged/no_pii/rating/percentage" in run2.index, \
    "Run 2 missing custom no_pii judge metric"

print("✓ Both runs exist")
print("✓ Run 1 contains groundedness metric")
print("✓ Run 2 contains custom no_pii metric")
print("Lab validation PASSED")
```

**Expected output:**
```
✓ Both runs exist
✓ Run 1 contains groundedness metric
✓ Run 2 contains custom no_pii metric
Lab validation PASSED
```

---

## Teardown

```python
# Teardown: delete the local tracking directory created during simulation.
# On Databricks, experiments persist in the workspace — manually archive if needed.

import shutil
import os

local_mlruns = "./mlruns"
if os.path.exists(local_mlruns):
    shutil.rmtree(local_mlruns)
    print(f"Deleted local tracking directory: {local_mlruns}")
else:
    print("No local mlruns directory to clean up.")

# [Databricks only] To archive the experiment:
# client = mlflow.tracking.MlflowClient()
# exp = client.get_experiment_by_name("lab-20-mlflow-judges")
# if exp:
#     client.delete_experiment(exp.experiment_id)
#     print("Experiment archived in Databricks workspace.")
```

---

## Reflection Questions

1. **Schema sensitivity:** What happens to the `correctness` judge if you accidentally name the column `"expected_response"` instead of `"expected_facts"` when you intend to provide a minimal fact list? How would you detect this silently wrong configuration during a real eval run?

2. **Judge selection cost:** Your eval set has 500 rows and running all 8 built-in judges costs $8 per run. You need to run evals 3× per day in CI. Describe a strategy to reduce cost to under $1 per run without eliminating meaningful quality signal.

3. **Causality of metrics:** Suppose `context_sufficiency` is 40% and `groundedness` is 85% on the same eval run. What is the most likely failure mode in the system, and which layer (retriever or generator) would you fix first? Explain your reasoning using the root-cause ordering that Databricks Agent Evaluation applies.
