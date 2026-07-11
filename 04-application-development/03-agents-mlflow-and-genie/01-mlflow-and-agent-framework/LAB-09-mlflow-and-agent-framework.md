# LAB-09: MLflow Logging, Tracing, and Evaluation for a LangChain RAG Agent

**Lab:** LAB-09 | **Section:** 04 Application Development | **Module:** 03 Agents, MLflow & Genie | **Est. time:** 90 min

---

## Objective

Log a LangChain RAG chain to MLflow with full dependency capture, instrument it with manual tracing using `@mlflow.trace`, run `mlflow.evaluate()` against a 5-question eval set with groundedness scoring, and inspect span hierarchies in the MLflow UI Traces tab.

---

## Prerequisites

- Completed LAB-08 (Vector Search index created; `hr_catalog.agents.hr_policy_index` available) or equivalent context vector search index
- Databricks Runtime 15.4 ML or later (includes MLflow ≥ 2.15)
- Unity Catalog privileges: `CREATE MODEL` on `hr_catalog.agents`, `USE CATALOG hr_catalog`, `USE SCHEMA hr_catalog.agents`
- LLM serving endpoint available (e.g., `databricks-meta-llama-3-1-70b-instruct` or equivalent Foundation Model endpoint)

---

## Setup

Open a new Databricks notebook and run the following cell to install the required packages:

```python
# Setup: install required packages (Runtime 15.4 ML already includes mlflow>=2.15)
%pip install --upgrade \
    "langchain>=0.3.0" \
    "langgraph>=0.2.0" \
    "databricks-langchain>=0.3.0" \
    "databricks-agents>=0.16.0"

dbutils.library.restartPython()
```

Set your experiment and imports:

```python
# Setup: configure experiment and imports for the entire lab
import mlflow
import pandas as pd
from mlflow.entities import SpanType

# All runs in this lab go to a dedicated experiment
mlflow.set_experiment("/Shared/lab-09-mlflow-rag-agent")
print(f"MLflow version: {mlflow.__version__}")
```

---

## Steps

### Step 1 — Build a Simple LangChain RAG Chain

Build a minimal RAG chain using `ChatDatabricks` and a mock retriever. In a production setting, the retriever would call Unity Catalog Vector Search; here we use a stub to keep the lab self-contained.

```python
# Step 1: Build a minimal LangChain RAG chain
# Goal: create a runnable chain we can log to MLflow without requiring
# a live vector search index (uses a mock retriever for portability).

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from databricks_langchain import ChatDatabricks

# --- Mock retriever (replace with Vector Search retriever in production) ---
MOCK_DOCS = {
    "parental leave": "Employees receive 12 weeks of paid parental leave.",
    "vacation": "New employees receive 15 vacation days in their first year.",
    "fmla": "FMLA requests must be submitted to HR at least 30 days in advance.",
    "remote work": "Remote work eligibility depends on role classification.",
    "expenses": "Business expenses must be submitted within 30 days with receipts.",
}

def mock_retrieve(question: str) -> list[dict]:
    """Return mock chunks for lab purposes."""
    q_lower = question.lower()
    for key, content in MOCK_DOCS.items():
        if key in q_lower:
            return [{"content": content, "doc_uri": f"policy/{key.replace(' ', '-')}.pdf"}]
    return [{"content": "No relevant policy found.", "doc_uri": "policy/general.pdf"}]

# --- Prompt template ---
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an HR policy assistant. Answer ONLY using the context below.\n\nContext:\n{context}"),
    ("human", "{question}"),
])

# --- LLM (replace endpoint name with your Foundation Model endpoint) ---
llm = ChatDatabricks(
    endpoint="databricks-meta-llama-3-1-70b-instruct",
    max_tokens=256,
)

# --- Chain assembly ---
def format_context(docs: list[dict]) -> str:
    return "\n\n".join(d["content"] for d in docs)

rag_chain = (
    RunnablePassthrough.assign(
        context=RunnableLambda(lambda x: format_context(mock_retrieve(x["question"])))
    )
    | rag_prompt
    | llm
    | StrOutputParser()
)

# Quick smoke test (should return policy text)
test_response = rag_chain.invoke({"question": "What is the parental leave policy?"})
print(f"Smoke test response: {test_response}")
```

**Expected output:**
```
Smoke test response: Employees receive 12 weeks of paid parental leave.
```

---

### Step 2 — Log the Chain to MLflow

Log the chain with full dependency capture and register it in Unity Catalog.

```python
# Step 2: Log the LangChain RAG chain to MLflow with dependency tracking
# Goal: create a versioned, reproducible artifact in Unity Catalog so
# any team member can load and serve the exact same chain later.

input_example = {"question": "What is the parental leave policy?"}

with mlflow.start_run(run_name="lab09-rag-chain-v1") as run:
    model_info = mlflow.langchain.log_model(
        lc_model=rag_chain,                          # live chain object (OK for LangChain)
        artifact_path="rag_chain",                   # sub-dir in run artifact store
        registered_model_name="hr_catalog.agents.lab09_hr_rag",  # UC registration
        input_example=input_example,                 # stored for UI testing
        pip_requirements=[
            "langchain>=0.3.0",
            "databricks-langchain>=0.3.0",
        ],
    )
    print(f"Run ID: {run.info.run_id}")
    print(f"Model URI: {model_info.model_uri}")
```

**Expected output:**
```
Run ID: <run-uuid>
Model URI: runs:/<run-uuid>/rag_chain
```

Navigate to the MLflow Experiment in the Databricks UI. Click the run. Under **Artifacts**, expand `rag_chain/` and confirm you see `MLmodel`, `requirements.txt`, `conda.yaml`, and `input_example.json`.

---

### Step 3 — Add Manual Tracing with `@mlflow.trace`

Enable autolog for the LangChain framework and add a manual span around the retriever to capture per-retrieval latency and chunk data.

```python
# Step 3: Instrument the chain with MLflow tracing
# Goal: see span-level visibility into retrieval vs. generation latency separately,
# so we can isolate slowness or retrieval failures without adding print statements.

# Enable automatic tracing for all LangChain calls (LLM, prompt, output parser)
mlflow.langchain.autolog()

# Add a manual span around the retriever to capture chunk metadata as span attributes
@mlflow.trace(span_type=SpanType.RETRIEVER, name="mock_retrieve")
def traced_retrieve(question: str) -> list[dict]:
    """Retrieve chunks with span-level tracing for latency and chunk content."""
    return mock_retrieve(question)

# Rebuild chain with traced retriever
rag_chain_traced = (
    RunnablePassthrough.assign(
        context=RunnableLambda(lambda x: format_context(traced_retrieve(x["question"])))
    )
    | rag_prompt
    | llm
    | StrOutputParser()
)

# Invoke to generate a trace
with mlflow.start_run(run_name="lab09-traced-invocation"):
    response = rag_chain_traced.invoke({"question": "How many vacation days do new employees get?"})
    print(f"Response: {response}")
```

**Expected output:**
```
Response: New employees receive 15 vacation days in their first year.
```

Open the **Traces** tab in your MLflow Experiment. You should see one trace entry. Click it to expand:

- Root span: the LangChain `RunnableSequence` (CHAIN)
  - `mock_retrieve` span (RETRIEVER) — shows the question input and returned chunks
  - Prompt formatting span
  - LLM call span (LLM) — shows the prompt sent and the model's response
  - Output parser span

---

### Step 4 — Build the Eval Dataset

Construct a 5-question eval DataFrame with the columns required by the `databricks-agent` judge.

```python
# Step 4: Build a structured eval dataset for LLM-judge evaluation
# Goal: structure the eval data correctly so all three judges (groundedness,
# correctness, chunk relevance) can run — each requires different columns.

eval_df = pd.DataFrame({
    "request": [
        "What is the parental leave policy?",
        "How many vacation days do new employees get?",
        "What is the process for requesting FMLA?",
        "Is remote work available for all roles?",
        "What is the policy on expense reimbursement?",
    ],
    "expected_response": [
        "Employees receive 12 weeks of paid parental leave.",
        "New employees receive 15 vacation days in their first year.",
        "FMLA requests must be submitted to HR at least 30 days in advance.",
        "Remote work eligibility depends on role classification.",
        "Business expenses must be submitted within 30 days with receipts.",
    ],
    "retrieved_context": [
        [{"doc_uri": "policy/parental-leave.pdf",
          "content": "Employees receive 12 weeks of paid parental leave."}],
        [{"doc_uri": "policy/pto.pdf",
          "content": "New employees receive 15 vacation days in their first year."}],
        [{"doc_uri": "policy/fmla.pdf",
          "content": "FMLA requests must be submitted to HR at least 30 days in advance."}],
        [{"doc_uri": "policy/remote.pdf",
          "content": "Remote work eligibility depends on role classification."}],
        [{"doc_uri": "policy/expenses.pdf",
          "content": "Business expenses must be submitted within 30 days with receipts."}],
    ],
})

print(f"Eval dataset shape: {eval_df.shape}")
print(eval_df[["request", "expected_response"]].to_string())
```

**Expected output:**
```
Eval dataset shape: (5, 3)
   request                                        expected_response
0  What is the parental leave policy?             Employees receive 12 weeks...
...
```

---

### Step 5 — Run `mlflow.evaluate()` and Inspect Results

Run LLM-judge evaluation and interpret per-row scores and aggregate metrics.

```python
# Step 5: Run mlflow.evaluate() with databricks-agent model_type
# Goal: measure groundedness and correctness before deciding to promote
# this model version to the @champion alias.

with mlflow.start_run(run_name="lab09-eval-gate-check") as eval_run:
    result = mlflow.evaluate(
        model=model_info.model_uri,  # logged model generates responses on the fly
        data=eval_df,
        model_type="databricks-agent",   # activates Databricks LLM judges
        evaluator_config={
            "databricks-agent": {
                "global_guidelines": {
                    "english": ["The response must be in English"],
                }
            }
        },
    )

print("\n=== AGGREGATE METRICS ===")
for k, v in result.metrics.items():
    print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

print("\n=== PER-ROW RESULTS (first 3 rows) ===")
display(result.tables["eval_results"].head(3))
```

**Expected output (values will vary slightly based on LLM judge):**
```
=== AGGREGATE METRICS ===
  agent/percent_grounded: 1.000
  agent/percent_correct: 0.800
  ...

=== PER-ROW RESULTS (first 3 rows) ===
[table showing request, response, groundedness/yes, correctness/yes, rationale columns]
```

---

### Step 6 — Promote to `@champion` if Quality Gate Passes

Conditionally set the Unity Catalog alias based on evaluation results.

```python
# Step 6: Conditional promotion using UC model alias
# Goal: automate the champion promotion decision based on the eval gate,
# eliminating manual review for routine releases that pass the threshold.

from mlflow import MlflowClient

GROUNDEDNESS_GATE = 0.80
CORRECTNESS_GATE  = 0.70

grounded_pct = result.metrics.get("agent/percent_grounded", 0.0)
correct_pct  = result.metrics.get("agent/percent_correct", 0.0)

print(f"Groundedness: {grounded_pct:.1%}  (gate: {GROUNDEDNESS_GATE:.0%})")
print(f"Correctness:  {correct_pct:.1%}  (gate: {CORRECTNESS_GATE:.0%})")

if grounded_pct >= GROUNDEDNESS_GATE and correct_pct >= CORRECTNESS_GATE:
    client = MlflowClient()
    # Get the registered version number from the log step
    versions = client.search_model_versions("name='hr_catalog.agents.lab09_hr_rag'")
    latest_version = max(int(v.version) for v in versions)
    
    client.set_registered_model_alias(
        "hr_catalog.agents.lab09_hr_rag",
        "champion",
        version=latest_version,
    )
    print(f"\nPROMOTED: version {latest_version} → @champion")
else:
    print("\nNOT PROMOTED: quality gate not met. Review eval_results before promoting.")
```

**Expected output (if eval passes):**
```
Groundedness: 100.0%  (gate: 80%)
Correctness:  80.0%   (gate: 70%)

PROMOTED: version 1 → @champion
```

---

## Validation

Verify all lab steps succeeded with these checks:

```python
# Validation: confirm all expected outcomes are present

from mlflow import MlflowClient

client = MlflowClient()

# 1. Confirm model is registered in UC
model = client.get_registered_model("hr_catalog.agents.lab09_hr_rag")
print(f"[PASS] Registered model found: {model.name}")

# 2. Confirm @champion alias is set
aliases = {a.alias: a.version for a in client.get_registered_model("hr_catalog.agents.lab09_hr_rag").aliases}
assert "champion" in aliases, "ERROR: @champion alias not set"
print(f"[PASS] @champion alias → version {aliases['champion']}")

# 3. Confirm eval metrics were logged
runs = mlflow.search_runs(
    experiment_names=["/Shared/lab-09-mlflow-rag-agent"],
    filter_string="tags.mlflow.runName = 'lab09-eval-gate-check'",
)
assert len(runs) > 0, "ERROR: eval run not found"
metrics = runs.iloc[0].to_dict()
print(f"[PASS] Eval run found. Metrics logged: {[k for k in metrics if 'agent/' in k]}")

# 4. Confirm traces were generated (requires Traces UI check)
print("[CHECK] Open MLflow Experiment → Traces tab and confirm ≥1 trace with mock_retrieve span")
```

**Expected output:**
```
[PASS] Registered model found: hr_catalog.agents.lab09_hr_rag
[PASS] @champion alias → version 1
[PASS] Eval run found. Metrics logged: ['agent/percent_grounded', 'agent/percent_correct', ...]
[CHECK] Open MLflow Experiment → Traces tab and confirm ≥1 trace with mock_retrieve span
```

---

## What to Look for in the Traces Tab

Navigate to your MLflow Experiment → **Traces** tab. For each invocation from Step 3:

1. **Trace list row:** Shows `Request` (the question), `Response` (the answer), and total `Latency`. Scan for any with `Status: ERROR` — these indicate an exception was captured inside a span.

2. **Span tree (click a trace row):**
   - The root span should be labeled `RunnableSequence` or your chain name (CHAIN type).
   - Expand to find the `mock_retrieve` span (RETRIEVER type). Check its `Inputs` tab — the question should appear. Check `Outputs` — the chunk list should appear.
   - Find the LLM span (LLM type). Its `Inputs` tab shows the full formatted prompt sent to the model. Its `Outputs` shows the raw model response. `Attributes` shows token counts if available.

3. **Latency waterfall:** Look for which span took the most time. In real deployments, the LLM span dominates. If the RETRIEVER span is slow, that signals a vector search latency problem.

4. **Filter by span type:** Use the filter bar to show only `RETRIEVER` spans across all traces to quickly spot retrieval failures (zero chunks returned) or outlier latencies.

---

## Teardown

```python
# Teardown: clean up lab resources
# Note: the registered model is retained for forward reference from future labs.
# Only clean up if you will not be doing subsequent agent labs.

from mlflow import MlflowClient

client = MlflowClient()

# Optional: delete all versions and the registered model
# client.delete_registered_model("hr_catalog.agents.lab09_hr_rag")
# print("Registered model deleted.")

# Always safe: delete the @champion alias to free the pointer without deleting the model
# client.delete_registered_model_alias("hr_catalog.agents.lab09_hr_rag", "champion")
# print("@champion alias removed.")

print("Teardown complete. Registered model retained for subsequent labs.")
```

---

## Reflection Questions

1. In Step 3, the `mock_retrieve` span showed the chunks that were retrieved. What would you change about the retriever to improve groundedness if the chunks were consistently incorrect for a given category of questions?

2. The lab used a live chain object (`lc_model=rag_chain`) for `log_model()`. When would you switch to the `lc_model="./agent.py"` file path approach, and what problem does it solve?

3. In Step 5, `mlflow.evaluate()` was passed `model=model_info.model_uri`, which caused it to call the logged model to generate responses. What is the alternative, and when would you use it instead of having `mlflow.evaluate()` call the model?
