# LAB-11: PyFunc Chains with Pre- and Post-Processing

**Lab:** LAB-11 | **Section:** 05-Assembling and Deploying Applications | **Module:** 01-Chains and RAG Assembly | **Chapter:** 01-PyFunc Chains Pre/Post Processing | **Est. time:** 45–60 min

---

## Objective

Build, log, register, and locally validate a complete `mlflow.pyfunc.PythonModel` that encapsulates a LangChain-based LLM chain with explicit pre-processing (input validation and prompt construction) and post-processing (structured JSON extraction), using a `ModelSignature` enforced at serving time.

---

## Prerequisites

- Databricks Runtime 14.3 LTS ML or later (includes MLflow ≥ 2.12, LangChain pre-installed)
- A Databricks workspace with Unity Catalog enabled
- A Unity Catalog catalog and schema where you have `CREATE MODEL` privilege (e.g., `main.lab_11`)
- An OpenAI API key (or substitute a Databricks-hosted DBRX/LLAMA endpoint via `langchain_community.chat_models.ChatDatabricks`)
- Access to a Databricks cluster with internet egress (for OpenAI API calls) or a Databricks Model Serving endpoint for the LLM

---

## Setup

```bash
# Run in a Databricks notebook cell — installs any packages not already present
# in the Databricks Runtime ML environment
%pip install langchain==0.3.7 langchain-openai==0.2.9 langchain-community==0.3.7 mlflow>=2.14 --quiet
dbutils.library.restartPython()
```

```python
# Cell 2: Configure environment
import os
import mlflow

# Set your OpenAI API key (use Databricks Secrets in production)
os.environ["OPENAI_API_KEY"] = dbutils.secrets.get(scope="my-scope", key="openai-api-key")

# Set Unity Catalog as the MLflow registry
mlflow.set_registry_uri("databricks-uc")

# Configure the MLflow experiment
mlflow.set_experiment("/Users/<your-user>/lab-11-pyfunc-chain")

# Verify MLflow version
print(f"MLflow version: {mlflow.__version__}")
```

---

## Steps

### Step 1 — Define and verify the `ModelSignature`

Define the input/output schema before writing any chain logic. This forces you to treat the schema as the spec.

```python
import pandas as pd
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, ColSpec, DataType
import json

# Input: a question string and an optional context string
input_schema = Schema([
    ColSpec(DataType.string, "question"),
    ColSpec(DataType.string, "context_hint"),  # optional additional context
])

# Output: structured answer and extracted key_points
output_schema = Schema([
    ColSpec(DataType.string, "answer"),
    ColSpec(DataType.string, "key_points"),   # JSON-encoded list of strings
])

signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Verify the signature serializes correctly
sig_dict = signature.to_dict()
print("Input schema:")
print(json.dumps(json.loads(sig_dict["inputs"]), indent=2))
print("\nOutput schema:")
print(json.dumps(json.loads(sig_dict["outputs"]), indent=2))
```

**Expected output:** Two JSON objects showing `name` and `type` fields for each column, with `type: "string"` for all four columns.

---

### Step 2 — Implement the `PythonModel` subclass

Write the chain class with `load_context()` for one-time initialization and `predict()` for request logic.

```python
import re
import mlflow
import mlflow.pyfunc
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


class SummaryChain(mlflow.pyfunc.PythonModel):
    """
    LAB-11 PyFunc chain:
      PRE-PROCESSING:  validate inputs, inject context_hint into prompt
      LLM CALL:        LangChain ChatPromptTemplate + ChatOpenAI
      POST-PROCESSING: parse JSON block from response, extract key_points list
    """

    def load_context(self, context):
        """
        Runs ONCE when the model loads. Initialize the LLM client here.
        context.model_config carries deployment-time knobs.
        """
        llm_model = (context.model_config or {}).get("llm_model", "gpt-4o-mini")
        temperature = float((context.model_config or {}).get("temperature", "0"))

        self.llm = ChatOpenAI(model=llm_model, temperature=temperature)
        self.prompt = ChatPromptTemplate.from_template(
            "You are a helpful assistant.\n"
            "Additional context: {context_hint}\n\n"
            "Question: {question}\n\n"
            "Respond in this exact JSON format (no markdown fencing):\n"
            '{{"answer": "<your answer>", "key_points": ["<point1>", "<point2>"]}}'
        )
        print(f"[load_context] LLM initialized: model={llm_model}, temp={temperature}")

    def _pre_process(self, row):
        """Validate and extract fields from a single DataFrame row."""
        question = str(row.get("question", "")).strip()
        context_hint = str(row.get("context_hint", "")).strip()

        if not question:
            raise ValueError(
                f"'question' must be a non-empty string; got: {repr(row.get('question'))}"
            )
        if len(question) > 2000:
            raise ValueError(f"'question' exceeds 2000 characters (got {len(question)})")

        return question, context_hint or "No additional context provided."

    def _post_process(self, raw_response: str) -> dict:
        """
        Parse JSON from the LLM response.
        Handles cases where the model wraps output in markdown code fences.
        """
        # Strip markdown code fences if present
        cleaned = re.sub(r"^```[a-z]*\n?|```$", "", raw_response.strip(), flags=re.MULTILINE)
        cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            # Graceful fallback: treat entire response as answer, empty key_points
            parsed = {"answer": cleaned, "key_points": []}

        answer = str(parsed.get("answer", ""))
        key_points = parsed.get("key_points", [])
        # Ensure key_points is always a JSON string (output schema: DataType.string)
        return {"answer": answer, "key_points": json.dumps(key_points)}

    def predict(self, context, model_input, params=None):
        """
        Entry point for all inference requests.
        model_input: pandas DataFrame with columns 'question', 'context_hint'
        Returns: pandas DataFrame with columns 'answer', 'key_points'
        """
        results = []
        for _, row in model_input.iterrows():
            question, context_hint = self._pre_process(row)

            # LLM call via LangChain chain
            chain = self.prompt | self.llm
            raw = chain.invoke({"question": question, "context_hint": context_hint})
            raw_text = raw.content if hasattr(raw, "content") else str(raw)

            # Post-process to structured output
            result = self._post_process(raw_text)
            results.append(result)

        return pd.DataFrame(results)
```

---

### Step 3 — Log the model with `log_model()`

Log the model inside an MLflow run with the signature, an input example, pip requirements, and Unity Catalog registration.

```python
import json

# Input example for the endpoint's "Try it" UI and smoke tests
input_example = pd.DataFrame([{
    "question": "What are the main benefits of using MLflow PyFunc for LLM deployment?",
    "context_hint": "Focus on production deployment patterns.",
}])

UC_MODEL_NAME = "main.lab_11.summary_chain"   # Change catalog/schema as needed

with mlflow.start_run(run_name="lab-11-pyfunc-chain") as run:
    model_info = mlflow.pyfunc.log_model(
        name="summary_chain",
        python_model=SummaryChain(),
        signature=signature,
        input_example=input_example,
        pip_requirements=[
            "langchain==0.3.7",
            "langchain-openai==0.2.9",
            "langchain-community==0.3.7",
            "openai>=1.40",
            "pandas>=2.0",
        ],
        registered_model_name=UC_MODEL_NAME,
        model_config={"llm_model": "gpt-4o-mini", "temperature": "0"},
    )
    print(f"Run ID:        {run.info.run_id}")
    print(f"Model URI:     {model_info.model_uri}")
    print(f"Registered at: {UC_MODEL_NAME}")
```

**Expected output:** Two print lines with a valid run ID and a `runs:/…/summary_chain` URI.

---

### Step 4 — Load and validate locally

Load the logged model and run a prediction locally before considering endpoint deployment.

```python
# Load the logged model (reconstructs the PythonModel and calls load_context())
loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

# Test case 1: Valid input
test_df = pd.DataFrame([
    {
        "question": "Explain the difference between load_context and predict in MLflow PyFunc.",
        "context_hint": "Focus on lifecycle and performance implications.",
    }
])

result = loaded_model.predict(test_df)
print("=== Prediction Result ===")
print(result)
print("\nAnswer:")
print(result.iloc[0]["answer"])
print("\nKey Points (JSON):")
print(result.iloc[0]["key_points"])

# Verify output schema compliance
assert "answer" in result.columns, "Missing 'answer' column"
assert "key_points" in result.columns, "Missing 'key_points' column"
assert result["answer"].dtype == object, "Expected string dtype for 'answer'"
parsed_kp = json.loads(result.iloc[0]["key_points"])
assert isinstance(parsed_kp, list), "key_points must be a JSON list"
print("\n[PASS] Output schema validation passed.")
```

---

### Step 5 — Test input validation (pre-processing guardrail)

Verify that the pre-processing validation raises a clear error on bad input.

```python
import traceback

# Test case 2: Empty question should raise ValueError
bad_input = pd.DataFrame([{"question": "", "context_hint": "irrelevant"}])
try:
    loaded_model.predict(bad_input)
    print("[FAIL] Should have raised ValueError for empty question")
except Exception as e:
    print(f"[PASS] Correctly rejected empty question: {type(e).__name__}: {e}")

# Test case 3: Oversized question should raise ValueError
oversized = pd.DataFrame([{"question": "x" * 2001, "context_hint": ""}])
try:
    loaded_model.predict(oversized)
    print("[FAIL] Should have raised ValueError for oversized question")
except Exception as e:
    print(f"[PASS] Correctly rejected oversized question: {type(e).__name__}: {e}")
```

**Expected output:** Two `[PASS]` lines showing `ValueError` was raised with the correct messages.

---

### Step 6 — Inspect the logged model metadata

Inspect the `MLmodel` file and confirm signature, artifacts, and flavor are correct.

```python
import mlflow.pyfunc

# Access the model metadata
raw_model = mlflow.pyfunc.load_model(model_info.model_uri)
meta = raw_model.metadata

print("=== Model Metadata ===")
print(f"Model UUID:     {meta.model_uuid}")
print(f"MLflow version: {meta.mlflow_version}")
print(f"Flavors:        {list(meta.flavors.keys())}")
print(f"\nSignature inputs: {meta.signature.inputs.to_dict()}")
print(f"Signature outputs: {meta.signature.outputs.to_dict()}")
```

**Expected output:** `Flavors` shows `['python_function']`. Signature inputs and outputs match the `ColSpec` definitions from Step 1.

---

## Validation

Run the following checklist. Every item must be true before you consider this lab complete:

| Check | How to verify |
|---|---|
| `log_model()` completed without error | Run ID printed in Step 3 output |
| Model registered in Unity Catalog | Navigate to `main.lab_11.summary_chain` in Catalog Explorer; version 1 should appear |
| `load_model()` returns without error and calls `load_context()` | `[load_context]` print line appears when loading the model in Step 4 |
| `predict()` returns a DataFrame with columns `answer` and `key_points` | `[PASS] Output schema validation passed.` printed in Step 4 |
| `key_points` column contains a valid JSON list string | `isinstance(parsed_kp, list)` assertion passes in Step 4 |
| Empty question raises `ValueError` (not a silent empty response) | `[PASS]` line in Step 5, Test case 2 |
| Oversized question raises `ValueError` | `[PASS]` line in Step 5, Test case 3 |
| Signature shows `DataType.string` for all four columns | Signature print in Step 6 shows no `object` or `any` types |

---

## Teardown

```python
# Optional: delete the registered model version if you want to clean up the lab
from mlflow.tracking import MlflowClient

client = MlflowClient(registry_uri="databricks-uc")

# List versions
versions = client.search_model_versions(f"name='{UC_MODEL_NAME}'")
for v in versions:
    print(f"Version {v.version} — status: {v.status}")

# To delete a specific version:
# client.delete_model_version(name=UC_MODEL_NAME, version="1")

# To delete the registered model entirely (removes all versions):
# client.delete_registered_model(name=UC_MODEL_NAME)

# Terminate the cluster if running on an interactive cluster
# (no command needed — cluster auto-terminates per workspace policy)
print("Lab teardown: review and run delete commands above as needed.")
```

---

## Reflection Questions

1. If you had to deploy this same `SummaryChain` to both a development endpoint (using `gpt-4o-mini`) and a production endpoint (using `gpt-4o`), what changes would you make to the `log_model()` call and the `SummaryChain` class — and what would you leave unchanged?

2. The current `_post_process()` method has a silent fallback: if JSON parsing fails, it returns the raw LLM response as the answer with an empty `key_points` list. In a production system, what are the tradeoffs between this silent fallback approach and raising a hard error? Under what business conditions would each be appropriate?

3. Suppose a downstream team's data pipeline begins consuming this endpoint and they hard-code their code to expect exactly two columns. Three months later, you add a `confidence` column to the output schema. Walk through the impact on the downstream consumer and describe what you would do — both technically and organizationally — to manage this schema evolution.
