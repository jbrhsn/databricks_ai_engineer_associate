# MLflow PyFunc Chains: Pre- and Post-Processing

**Section:** 05-Assembling and Deploying Applications | **Module:** 01-Chains and RAG Assembly | **Est. time:** 2.5 hrs | **Exam mapping:** Application Development (30%) — assembling LLM chains; Assembling & Deploying (22%) — Model Serving deployment pipeline

---

## TL;DR

`mlflow.pyfunc.PythonModel` is the universal escape hatch for packaging any Python inference logic — including multi-step LLM chains with custom pre-processing and post-processing — into a single deployable MLflow artifact. You subclass `PythonModel`, implement `predict()`, optionally implement `load_context()` for heavy initialization, attach a `ModelSignature` to declare your I/O contract, then log and register the artifact for Databricks Model Serving to serve behind a REST endpoint. The pattern decouples chain assembly (how inputs are transformed and outputs are shaped) from deployment concerns (scaling, versioning, routing). **The one thing to remember: `predict()` is the only entry point Model Serving calls — all pre-processing and post-processing logic must live inside it or be invoked from it.**

---

## ELI5 — Explain It Like I'm 5

Imagine a sandwich shop where every customer order arrives as a plain note saying "I want a sandwich." Before the cook can do anything, a prep worker has to read the note, figure out what kind of bread and fillings are needed, and lay everything out (pre-processing). The cook then makes the sandwich (the LLM call). Then a finisher wraps it, cuts it, and puts it in a bag with a receipt (post-processing). The shop gives the customer one packaged item, not raw ingredients. MLflow PyFunc is the contract that says: "no matter what happens inside the shop, the customer always hands in a note and gets back a packaged item." The most common misconception is that pre- and post-processing are separate deployed services — they are not; they all live inside the same `predict()` method, bundled into one artifact that the serving endpoint calls as a single unit.

---

## Learning Objectives

By the end of this chapter you will be able to:
- [ ] Implement `mlflow.pyfunc.PythonModel` with `load_context()` and `predict()` to encapsulate a complete LLM chain including pre- and post-processing steps
- [ ] Define and attach a `ModelSignature` using `ColSpec` and `Schema` to enforce input/output contracts at serving time
- [ ] Log a PyFunc model with `mlflow.pyfunc.log_model()`, providing `artifacts`, `pip_requirements`, and a `signature`
- [ ] Load a logged PyFunc model with `mlflow.pyfunc.load_model()` and invoke it for local validation before deployment
- [ ] Explain how a PyFunc model flows through Databricks Model Serving registration, endpoint creation, and REST invocation

---

## Visual Overview

### PyFunc Chain: End-to-End Data Flow

```
Raw REST Request (JSON)
        │
        ▼
┌───────────────────────────────────────────────────┐
│                 predict(context, model_input)      │
│                                                   │
│  ┌─────────────────┐   ┌──────────────────────┐   │
│  │  PRE-PROCESSING │   │    LLM / Chain Call  │   │
│  │                 │   │                      │   │
│  │ • input validate│──►│ • prompt template    │   │
│  │ • prompt build  │   │ • LangChain / Graph  │   │
│  │ • context inject│   │ • LLM API call       │   │
│  └─────────────────┘   └──────────┬───────────┘   │
│                                   │               │
│                    ┌──────────────▼────────────┐  │
│                    │    POST-PROCESSING        │  │
│                    │ • output parsing          │  │
│                    │ • response filtering      │  │
│                    │ • structured extraction   │  │
│                    └──────────────┬────────────┘  │
└───────────────────────────────────┼───────────────┘
                                    ▼
                         Structured JSON Response
```

### ModelSignature Enforcement Flow

```
log_model(signature=sig)
        │
        ▼
  ┌─────────────┐      Schema mismatch?
  │  MLmodel    │──────────────────────► MlflowException at serve time
  │  (stores    │
  │  signature) │      Schema valid?
  └──────┬──────┘──────────────────────► predict() called with
         │                               validated input
         ▼
  Model Serving reads
  signature from MLmodel
  file on endpoint start
```

### Deployment Pipeline: PyFunc → Model Serving Endpoint

```
Notebook / CI
    │
    ▼
mlflow.pyfunc.log_model()
    │
    ▼
MLflow Run Artifact (runs:/<run_id>/model)
    │
    ▼
mlflow.register_model()  OR  registered_model_name= param
    │
    ▼
Unity Catalog Model Registry
(models:/<catalog>.<schema>.<name>/<version>)
    │
    ▼
Databricks Model Serving
Create / Update Endpoint
    │
    ▼
REST Endpoint  ──►  POST /invocations  ──►  predict()
```

### PythonModel Class Structure

```
class MyChain(mlflow.pyfunc.PythonModel)
│
├── load_context(context)          ← runs ONCE at endpoint startup
│     ├── loads heavy artifacts    (tokenizers, vector stores, graphs)
│     └── initializes clients      (LLM API clients, DB connections)
│
└── predict(context, model_input, params=None)   ← called per request
      ├── pre_process(model_input)
      ├── llm_call(prompt)
      └── post_process(raw_output)
```

---

## Key Concepts

### `mlflow.pyfunc.PythonModel` — The Base Class

**What is it?** `PythonModel` is the abstract base class you subclass to package any Python inference logic — including multi-step LLM chains, retrieval-augmented pipelines, and arbitrary pre/post transforms — as an MLflow model artifact.

**How does it work under the hood?** When you call `mlflow.pyfunc.log_model(python_model=MyModel())`, MLflow serializes your instance using cloudpickle and writes it alongside a generated `MLmodel` YAML file that declares `python_function` as the flavor. At load time (`mlflow.pyfunc.load_model()`), MLflow deserializes the pickle back into a `PyFuncModel` wrapper, calls `load_context()` once to initialize heavy state (e.g., loading artifact files from disk), and then exposes `.predict()` for inference. The `context` argument passed to both methods is a `PyfuncContext` object providing `context.artifacts` (a dict mapping names to local filesystem paths) and `context.model_config` (a dict of config knobs).

**Where does it appear in Databricks/MLflow?** You implement `PythonModel` in a notebook or Python file, log it inside an `mlflow.start_run()` block, register it to Unity Catalog, and Databricks Model Serving instantiates it as the serving handler behind your endpoint's `/invocations` REST route.

---

### `predict()` Interface

**What is it?** `predict(self, context, model_input, params=None)` is the single method Databricks Model Serving calls for every inference request. It is the mandatory entry point and must return data in a format compatible with the declared output schema.

**How does it work under the hood?** Model Serving deserializes the incoming JSON payload according to the endpoint's input schema (or falls back to a pandas DataFrame if no signature is set) and passes it as `model_input`. Your implementation runs sequentially: validate inputs, construct a prompt, invoke the LLM or LangChain/LangGraph chain, parse and filter the response, and return the result. Because `predict()` is called on a per-request basis, any state loaded in `load_context()` is available via `self`. The `params` argument (optional) carries request-time inference parameters (e.g., `temperature`, `max_tokens`) without modifying the model signature's input schema.

**Where does it appear in Databricks/MLflow?** When you POST to `https://<workspace>/serving-endpoints/<endpoint>/invocations`, the serving container calls `pyfunc_model.predict(data)`. The `PyFuncModel.predict()` wrapper enforces the input schema before delegating to your implementation's `predict()`.

---

### Input/Output Schema Enforcement (`ModelSignature`, `ColSpec`, `Schema`)

**What is it?** A `ModelSignature` is a declarative contract attached at log time that specifies the exact column names and data types expected at input and output. It is stored inside the `MLmodel` file and read by Model Serving to validate requests before they reach `predict()`.

**How does it work under the hood?** `ModelSignature` wraps two `Schema` objects (inputs and outputs), each built from a list of `ColSpec` (column-based) or `TensorSpec` (tensor-based) type descriptors. `ColSpec(type=DataType.string, name="question")` declares a named string column. At serving time, Model Serving reads the signature from the `MLmodel` file and rejects requests whose payload does not conform — returning a 400 error — before your `predict()` code ever runs. You can also use `mlflow.models.infer_signature(input_df, output_df)` to auto-derive the signature from example data.

**Where does it appear in Databricks/MLflow?** Signatures are required for models registered to Unity Catalog. You pass `signature=` to `mlflow.pyfunc.log_model()`. The Databricks Model Serving UI displays the inferred schema and uses it to generate an example curl command for the endpoint.

---

### Pre-Processing Steps

**What is it?** Pre-processing is any transformation applied to `model_input` before the LLM or chain call: input validation, prompt template rendering, context injection (retrieved documents, conversation history, user metadata), and schema normalization.

**How does it work under the hood?** Inside `predict()`, you extract fields from the input DataFrame or dict, run assertions or type coercions, interpolate values into a prompt string (commonly via LangChain `PromptTemplate` or a f-string), and optionally augment the prompt with context retrieved from a vector store loaded during `load_context()`. Because all this runs inside the same Python process as `load_context()`, loaded artifacts (e.g., a FAISS index or a LangGraph graph object) are available as `self` attributes without a network round-trip.

**Where does it appear in Databricks/MLflow?** In practice, the pre-processing block is the first 5–20 lines of `predict()`. LangChain's `ChatPromptTemplate.format_messages()` or LangGraph's `.invoke(state)` are typical calls here for the primary framework stack.

---

### Post-Processing Steps

**What is it?** Post-processing transforms the raw LLM output — typically an unstructured string or a framework-specific response object — into a structured, clean result that matches your declared output schema. Common tasks: JSON extraction, PII redaction, hallucination filter, confidence score annotation, and schema-conformant serialization.

**How does it work under the hood?** After the LLM/chain call returns, your `predict()` implementation parses the raw response (e.g., extracts a JSON block from a markdown-wrapped output using `re.search` or `json.loads`), filters or rewrites content that fails business rules, and constructs the return value — usually a `pandas.DataFrame`, a `list`, or a `dict` — that matches the output schema declared in the signature. If the output schema declares `ColSpec(type=DataType.string, name="answer")`, returning a DataFrame with a column `answer` of string type satisfies it.

**Where does it appear in Databricks/MLflow?** The return statement of `predict()`. For structured extraction in LangChain, `.with_structured_output()` on the LLM is a common upstream mechanism, but the serialization of that output to the declared schema still happens in your `predict()` return block.

---

### `mlflow.pyfunc.log_model()` and Loading for Inference

**What is it?** `mlflow.pyfunc.log_model()` records your `PythonModel` instance (or a code path) as an artifact in the active MLflow run, captures dependencies, and optionally registers it to a model registry. `mlflow.pyfunc.load_model(model_uri)` retrieves it and reconstructs a `PyFuncModel` object ready for local validation or deployment.

**How does it work under the hood?** `log_model()` serializes the model with cloudpickle (or as a code file for "models from code" pattern), collects pip requirements either explicitly from `pip_requirements=` or by inspecting the current environment, downloads any artifact URIs specified in `artifacts=` to local paths, and writes the whole bundle to the MLflow artifact store. The `registered_model_name=` parameter creates or increments a version in the registry atomically. `load_model()` reverses this: it downloads the artifact bundle, reconstructs the conda/virtualenv environment if needed, deserializes the model, calls `load_context()`, and returns the `PyFuncModel`.

**Where does it appear in Databricks/MLflow?** Logging happens in your training/authoring notebook. Loading for validation happens in a second notebook or CI job. In Databricks Model Serving, the platform calls `load_model()` internally when an endpoint starts or updates.

---

## Key Parameters / Configuration Knobs

| Parameter | What it controls | Decision rule |
|---|---|---|
| `python_model` | The `PythonModel` instance or callable or file path to serialize | Pass an instance for interactive notebooks; pass a file path (`"chain.py"`) for "models from code" pattern to avoid cloudpickle limitations in CI/CD |
| `artifacts` | Dict mapping logical names to URIs of files/dirs to package alongside the model | Include any file that `load_context()` reads (e.g., prompt templates, vector index) — anything not listed here is unavailable at serving time |
| `signature` | `ModelSignature` describing input/output schema; stored in MLmodel file | Always provide explicitly for Unity Catalog registration; use `infer_signature()` only when you have representative data; hand-craft when I/O is a single string column |
| `pip_requirements` | List of pip packages or path to `requirements.txt`; defines the serving container's package environment | Prefer explicit pinned versions (`langchain==0.3.x`) over unpinned to prevent serving environment drift |
| `registered_model_name` | UC or workspace registry path; triggers registration atomically on log | Set to `<catalog>.<schema>.<model_name>` format for Unity Catalog; omit to log-only without registration |
| `model_config` | Dict passed to `context.model_config` in `load_context()` and `predict()` | Use for deployment-time knobs (e.g., LLM endpoint name, retrieval top-k) so the same model artifact can be reused across environments without re-logging |
| `input_example` | Sample input to embed in the artifact; used to infer signature if not provided | Provide always; it doubles as the endpoint's built-in "Try it" test input in the Databricks UI |
| `code_paths` | List of local Python files prepended to `sys.path` at load time | Required when `predict()` imports helper modules defined outside the notebook `__main__` scope |

---

## Worked Example: Requirement → Decision

**Given:** A team is building a customer-support Q&A system. Users submit free-text questions via a Slack bot. The bot calls a Databricks Model Serving endpoint. The endpoint must (1) validate that the question is non-empty, (2) inject the user's account tier from the request metadata, (3) retrieve the top-3 relevant KB articles from a pre-built vector store, (4) call an LLM to generate an answer, and (5) return a JSON-serializable dict with `answer` and `sources` fields. The model must be registerable to Unity Catalog and auditable.

**Step 1 — Identify the goal:** Package the entire pipeline as a single deployable artifact that satisfies Unity Catalog registration and produces a consistent, schema-validated REST response.

**Step 2 — Define inputs:** A pandas DataFrame with columns `question` (string) and `account_tier` (string). These map to two `ColSpec` entries in the input `Schema`.

**Step 3 — Define outputs:** A pandas DataFrame with columns `answer` (string) and `sources` (string, JSON-encoded list). These map to two `ColSpec` entries in the output `Schema`.

**Step 4 — Apply constraints:**
- Unity Catalog registration requires an explicit `ModelSignature`.
- The vector store (FAISS index) is large; it must be loaded once at startup, not per-request.
- The LLM endpoint name may differ between dev and prod environments.
- cloudpickle cannot serialize live database connections — the vector store client must be constructed in `load_context()`, not at class definition time.

**Step 5 — Select the approach:** Implement a `PythonModel` subclass where `load_context()` reads a FAISS index from `context.artifacts["vector_store_path"]` and stores it as `self.retriever`, and `predict()` orchestrates validation → retrieval → LangGraph chain call → structured output. Log with `mlflow.pyfunc.log_model()` providing `artifacts={"vector_store_path": faiss_dir}`, an explicit `ModelSignature`, `pip_requirements`, and `registered_model_name` pointing to Unity Catalog.

Rationale vs alternatives: Using a raw LangChain `log_model` flavor would not support the custom input validation or the mixed-schema response format. Using a plain function-based pyfunc model would not support `load_context()`, forcing the FAISS index to reload on every request. The class-based `PythonModel` is the only pattern that satisfies all five constraints simultaneously.

---

## Implementation

```python
# Scenario: Package a RAG customer-support chain with validation, retrieval,
# LLM call, and structured output as a single deployable PyFunc artifact
# registered to Unity Catalog — so the Slack bot endpoint has a single
# schema-validated REST target with no per-request cold-start on vector store load.

import json
import re
import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow.models import ModelSignature, infer_signature
from mlflow.types.schema import Schema, ColSpec, DataType
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI  # or use Databricks DBRX endpoint


class SupportChainModel(mlflow.pyfunc.PythonModel):
    """
    Encapsulates: input validation → retrieval → LangGraph prompt assembly → LLM → JSON output.
    """

    def load_context(self, context):
        """
        Called ONCE when the serving endpoint starts. Load heavy artifacts here.
        context.artifacts maps the logical name to an absolute local path.
        """
        import faiss  # noqa: F401 — ensures FAISS is available
        from langchain_community.vectorstores import FAISS as FAISSStore
        from langchain_openai import OpenAIEmbeddings

        # Load pre-built FAISS vector store from packaged artifact
        self.vector_store = FAISSStore.load_local(
            context.artifacts["vector_store_path"],
            OpenAIEmbeddings(),
            allow_dangerous_deserialization=True,
        )
        # Read LLM endpoint name from model_config so dev/prod can differ
        llm_endpoint = (context.model_config or {}).get("llm_endpoint", "gpt-4o-mini")
        self.llm = ChatOpenAI(model=llm_endpoint, temperature=0)
        self.prompt = ChatPromptTemplate.from_template(
            "You are a support agent for account tier: {tier}.\n"
            "Context:\n{context}\n\nQuestion: {question}\n\nAnswer concisely."
        )

    def predict(self, context, model_input, params=None):
        """
        Called per request. model_input is a pandas DataFrame.
        Returns a pandas DataFrame with columns: answer, sources.
        """
        results = []
        for _, row in model_input.iterrows():
            question = str(row.get("question", "")).strip()
            tier = str(row.get("account_tier", "standard"))

            # --- PRE-PROCESSING ---
            if not question:
                raise ValueError("'question' field must be non-empty.")

            docs = self.vector_store.similarity_search(question, k=3)
            context_text = "\n".join(d.page_content for d in docs)
            sources = json.dumps([d.metadata.get("source", "") for d in docs])

            # --- LLM CALL ---
            chain = self.prompt | self.llm
            raw = chain.invoke({"tier": tier, "question": question, "context": context_text})
            answer = raw.content if hasattr(raw, "content") else str(raw)

            # --- POST-PROCESSING ---
            # Strip any markdown fencing the LLM might wrap the answer in
            answer = re.sub(r"^```[a-z]*\n?|```$", "", answer, flags=re.MULTILINE).strip()

            results.append({"answer": answer, "sources": sources})

        return pd.DataFrame(results)


# --- Signature definition ---
input_schema = Schema([
    ColSpec(DataType.string, "question"),
    ColSpec(DataType.string, "account_tier"),
])
output_schema = Schema([
    ColSpec(DataType.string, "answer"),
    ColSpec(DataType.string, "sources"),
])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

input_example = pd.DataFrame([{
    "question": "How do I reset my password?",
    "account_tier": "premium",
}])

# --- Log and register ---
FAISS_DIR = "/dbfs/tmp/support_kb_faiss"  # pre-built index path

with mlflow.start_run():
    model_info = mlflow.pyfunc.log_model(
        name="support_rag_chain",
        python_model=SupportChainModel(),
        artifacts={"vector_store_path": FAISS_DIR},
        signature=signature,
        input_example=input_example,
        pip_requirements=[
            "langchain==0.3.7",
            "langchain-openai==0.2.9",
            "langchain-community==0.3.7",
            "faiss-cpu==1.8.0",
            "pandas>=2.0",
        ],
        registered_model_name="main.support.rag_chain",  # Unity Catalog path
        model_config={"llm_endpoint": "gpt-4o-mini"},
    )
    print(f"Logged model URI: {model_info.model_uri}")

# --- Local validation before deploying ---
loaded = mlflow.pyfunc.load_model(model_info.model_uri)
test_input = pd.DataFrame([{"question": "What is my billing cycle?", "account_tier": "standard"}])
print(loaded.predict(test_input))
```

---

```python
# Anti-pattern: Initializing the LLM client and loading the vector store
# inside predict() on every call. This causes multi-second cold starts on
# EVERY request and exhausts memory on high-traffic endpoints.

class BadChainModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        # WRONG: re-loads FAISS index from disk on every single request
        from langchain_community.vectorstores import FAISS
        from langchain_openai import OpenAIEmbeddings, ChatOpenAI
        vector_store = FAISS.load_local(
            "/dbfs/tmp/support_kb_faiss",
            OpenAIEmbeddings(),
            allow_dangerous_deserialization=True,
        )
        llm = ChatOpenAI(model="gpt-4o-mini")  # re-instantiates HTTP client each time
        # ... rest of chain ...
        return pd.DataFrame([{"answer": "...", "sources": "[]"}])

# Correct approach: Move ALL one-time initialization into load_context().
# load_context() runs exactly once when the serving endpoint cold-starts.
# Subsequent requests share the already-loaded self.vector_store and self.llm,
# reducing per-request latency from seconds to milliseconds.

class GoodChainModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from langchain_community.vectorstores import FAISS
        from langchain_openai import OpenAIEmbeddings, ChatOpenAI
        self.vector_store = FAISS.load_local(
            context.artifacts["vector_store_path"],
            OpenAIEmbeddings(),
            allow_dangerous_deserialization=True,
        )
        self.llm = ChatOpenAI(model="gpt-4o-mini")

    def predict(self, context, model_input, params=None):
        # self.vector_store and self.llm are already warm
        question = model_input.iloc[0]["question"]
        docs = self.vector_store.similarity_search(question, k=3)
        raw = self.llm.invoke(question)
        return pd.DataFrame([{"answer": raw.content, "sources": "[]"}])
```

---

```python
# Scenario: Enforce strict output schema so downstream consumers
# never receive unexpected fields or types — preventing silent breakage
# when the LLM changes its response format.

from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, ColSpec, DataType

# Explicit schema — never rely on infer_signature() alone for LLM output
# because LLM output is a string and infer_signature maps it to object dtype.
output_schema = Schema([
    ColSpec(DataType.string, "answer"),
    ColSpec(DataType.string, "sources"),   # JSON-encoded list as string
    ColSpec(DataType.double, "confidence"), # Structured extraction field
])
input_schema = Schema([
    ColSpec(DataType.string, "question"),
])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Verify the signature serializes correctly before logging
import json
print(json.dumps(signature.to_dict(), indent=2))
# {
#   "inputs": "[{\"name\": \"question\", \"type\": \"string\", ...}]",
#   "outputs": "[{\"name\": \"answer\", ...}, ...]"
# }
```

---

## Common Pitfalls & Misconceptions

- **Putting initialization inside `predict()`** — Beginners reason that `predict()` is "where the model runs" and so naturally put all setup there. The correct mental model is that `load_context()` is a constructor that runs once per endpoint lifecycle; anything placed in `predict()` runs once per request, multiplying its cost by request throughput.

- **Omitting the `ModelSignature` and relying on input_example inference** — New users assume MLflow will infer a good-enough schema from `input_example`. For LLM chains, `infer_signature()` on a string output often produces `DataType.object` instead of `DataType.string`, causing silent coercion bugs in downstream consumers and blocking Unity Catalog registration in some runtime versions.

- **Logging artifacts outside the `artifacts=` dict** — Developers reference local paths like `/tmp/my_index` directly in `predict()` instead of registering them in `artifacts=`. At serving time, the container has no `/tmp/my_index`; only paths in `context.artifacts` are guaranteed to be downloaded and available. Any hardcoded path that works on the notebook driver will silently fail in the serving container.

- **Expecting `predict()` to receive Python dicts directly** — The Model Serving layer deserializes JSON to a pandas DataFrame before passing it to `predict()` (when using the default `dataframe_split` or `dataframe_records` input format). Treating `model_input` as a dict causes `AttributeError: 'dict' object has no attribute 'iterrows'`. Use `model_input.iloc[0]["field"]` or iterate rows.

- **Importing heavy libraries at module level instead of inside `load_context()`** — cloudpickle serializes the class at log time, but module-level imports (like `import faiss`) run in the training environment. If the serving container's environment differs (or if the import triggers side effects), this causes deployment failures. Deferred imports inside `load_context()` are safer and more explicit.

- **Confusing `mlflow.pyfunc.log_model()` name vs artifact_path** — The `artifact_path` parameter is deprecated in recent MLflow versions; the replacement is `name=`. Using the old parameter still works but emits a deprecation warning and may break in future MLflow versions.

---

## Key Definitions

| Term | Definition |
|---|---|
| `PythonModel` | Abstract base class in `mlflow.pyfunc` that you subclass to define custom inference logic; must implement `predict()`, optionally `load_context()` |
| `PyFuncModel` | The wrapper object returned by `mlflow.pyfunc.load_model()`; exposes `.predict()` but not custom methods on your class |
| `ModelSignature` | An MLflow object storing input and output `Schema` declarations; persisted inside `MLmodel` and read by Model Serving for request validation |
| `Schema` | A typed list of `ColSpec` or `TensorSpec` descriptors defining the expected columns/tensors and their data types |
| `ColSpec` | A column-level type descriptor: `ColSpec(type=DataType.string, name="question")` declares a named column of string type |
| `TensorSpec` | A tensor-level type descriptor for numpy-array inputs/outputs with shape and dtype; used for deep learning models |
| `load_context()` | Lifecycle hook called once per endpoint cold-start; used to load heavy artifacts and initialize clients before requests begin |
| `context.artifacts` | Dict inside `PyfuncContext` mapping logical artifact names (from `artifacts=` param at log time) to absolute local file paths available in the serving container |
| `context.model_config` | Dict inside `PyfuncContext` carrying deployment-time configuration values passed via `model_config=` at log or load time |
| `artifacts` param | Argument to `log_model()` mapping logical names to URIs; MLflow downloads these files and packages them with the model artifact |
| `pip_requirements` | Explicit list or file of pip packages to include in the serving container's Python environment |
| `infer_signature()` | MLflow helper that auto-derives a `ModelSignature` from example input/output data; use with caution for LLM outputs |

---

## Summary / Quick Recall

- Subclass `mlflow.pyfunc.PythonModel`; put one-time init in `load_context()`, all request logic in `predict()`.
- `predict(context, model_input, params=None)` is the only entry point Model Serving ever calls — pre-processing and post-processing both live inside it.
- `ModelSignature` = `ModelSignature(inputs=Schema([ColSpec(...)]), outputs=Schema([ColSpec(...)]))` — required for Unity Catalog registration.
- Pass artifact files via `artifacts={"name": "uri"}` in `log_model()`; access them as `context.artifacts["name"]` — never hardcode serving-container paths.
- `mlflow.pyfunc.load_model("models:/catalog.schema.model/1")` reconstructs the full `PyFuncModel`; test locally before deploying to an endpoint.
- cloudpickle serializes the class instance — avoid module-level imports of heavy packages; defer them into `load_context()`.
- `model_config=` lets you pass deployment-specific knobs (LLM endpoint name, top-k, temperature) without re-logging the model.

---

## Self-Check Questions

1. What is the purpose of `load_context()` in a `PythonModel` subclass, and when is it called relative to `predict()`?

   <details><summary>Answer</summary>

   `load_context()` is an optional lifecycle hook that MLflow calls **once** when the model is loaded — either locally via `mlflow.pyfunc.load_model()` or when a Databricks Model Serving endpoint cold-starts. It receives a `PyfuncContext` object providing `context.artifacts` (resolved local paths) and `context.model_config`. Its purpose is to perform expensive, one-time initialization — loading vector indices, instantiating LLM clients, building LangGraph graphs — so that `predict()` finds these objects already available on `self`.

   **Why the distractor fails:** A common wrong answer is "it validates inputs before `predict()` runs." Input validation is not the purpose of `load_context()`; schema validation against `ModelSignature` happens in the MLflow serving wrapper layer, not in your `load_context()`. The lifecycle distinction — once per load vs. once per request — is the key concept.

   </details>

2. A team logs a PyFunc model without providing a `signature=` argument to `mlflow.pyfunc.log_model()`. They then try to register it to a Unity Catalog path `main.ml.my_model`. What happens and how should they fix it?

   <details><summary>Answer</summary>

   Unity Catalog model registration requires a `ModelSignature` (in most Databricks Runtime versions, missing signatures trigger a validation error or warning that blocks registration or causes serving-time schema mismatches). The fix is to construct an explicit `ModelSignature` using `Schema` and `ColSpec` — or call `mlflow.models.infer_signature(input_example, predictions)` with representative data — and pass it as `signature=` to `log_model()`.

   **Why "just use input_example" fails:** Providing only `input_example` without an explicit `signature=` causes MLflow to attempt inference, but for LLM string outputs the inferred type is often `DataType.object` rather than `DataType.string`, producing a signature that will either fail UC validation or cause type coercion surprises at serving time. Explicit is always safer for LLM chains.

   </details>

3. **Which TWO** of the following are valid reasons to use `artifacts=` in `mlflow.pyfunc.log_model()` rather than hardcoding a DBFS path inside `predict()`?
   - A. `artifacts=` allows MLflow to track artifact lineage in the MLflow run UI.
   - B. Hardcoded DBFS paths are unavailable in the Model Serving container because it is isolated from DBFS.
   - C. `artifacts=` automatically compresses large files to reduce model size.
   - D. Using `context.artifacts["name"]` guarantees the file is present at a valid local path in any environment where the model loads.
   - E. `artifacts=` is required by the `ModelSignature` specification.

   <details><summary>Answer</summary>

   **Correct answers: A and D.**

   **A** is correct: artifact URIs logged via `artifacts=` are recorded as part of the MLflow run artifact lineage, viewable in the Experiments UI.

   **D** is correct: `context.artifacts["name"]` gives an absolute local filesystem path that MLflow has verified exists — the file was downloaded from the artifact store when the model was loaded. This is the fundamental contract that makes PyFunc models portable across notebook drivers, local machines, and serving containers.

   **Why B is a distractor:** While serving containers do not mount DBFS by default, the more precise reason to use `artifacts=` is portability and contract — the model artifact becomes self-contained. B describes a real limitation but is not the canonical reason `artifacts=` exists.

   **Why C is wrong:** `artifacts=` performs no compression; it copies or references files as-is.

   **Why E is wrong:** `ModelSignature` describes I/O schema for inference data, not artifact dependencies. The two are independent.

   </details>

4. You deploy a PyFunc model to a Databricks Model Serving endpoint. After 50 requests, the endpoint starts returning 400 errors with the message "Input validation failed: expected column 'account_tier' of type string." The model worked fine in local testing. What is the most likely root cause?

   <details><summary>Answer</summary>

   The serving endpoint enforces the `ModelSignature` logged with the model. The 400 error means incoming requests from production callers are not including the `account_tier` field, or are passing it as a non-string type (e.g., integer `1` instead of string `"premium"`). Local testing passed because the test DataFrame had the correct column; production callers (e.g., the Slack bot) are sending a different payload format.

   **Diagnosis path:** (1) Check the endpoint's registered signature in Unity Catalog UI or via `mlflow.pyfunc.load_model(uri).metadata.signature`. (2) Inspect the raw request payload using endpoint access logs. (3) Fix either the caller payload or, if `account_tier` is truly optional, make it optional with a default value in `predict()` and update the signature.

   **Why "the model has a bug in predict()" is a distractor:** A bug in `predict()` would surface as a 500 error (unhandled exception), not a 400 input validation error. The 400 category specifically means the input schema check failed before `predict()` ran.

   </details>

5. Your team has a working PyFunc chain model. A colleague suggests switching to `mlflow.langchain.log_model()` instead, arguing it is simpler and avoids the need to write `predict()`. Under what conditions is the PyFunc approach preferable, and when should you use the LangChain flavor instead?

   <details><summary>Answer</summary>

   **Use `mlflow.pyfunc.PythonModel` when:**
   - You need custom pre-processing logic (input validation, prompt construction from multiple fields, metadata injection) that the LangChain flavor cannot express as a pure chain.
   - You need custom post-processing (structured extraction, response filtering, PII redaction, multi-field output schema) that is not captured by the chain's final output node.
   - Your pipeline mixes frameworks (e.g., a LangGraph graph combined with a custom retriever not expressible as a standard LCEL chain).
   - You need `load_context()` to initialize non-LangChain components (custom databases, proprietary SDKs).

   **Use `mlflow.langchain.log_model()` when:**
   - Your entire pipeline is a clean LCEL runnable or a single LangChain chain/agent with standard LangChain components.
   - The input is a plain dict (e.g., `{"input": "..."}`) and the output is the chain's direct return, with no additional transformation.
   - You want automatic schema inference from the chain's type annotations without writing signature boilerplate.

   **The trade-off:** `mlflow.langchain.log_model()` is lower-boilerplate but gives you less control over the serving boundary. `mlflow.pyfunc.PythonModel` requires more code but is the correct choice whenever your chain has a non-trivial pre- or post-processing contract.

   </details>

---

## Further Reading

> ⚠️ Fast-evolving: MLflow PyFunc, Models from Code, and Databricks Model Serving APIs update frequently. Re-verify against current docs before any production deployment.

- [mlflow.pyfunc — Python API Reference](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html) — *verified 2026-07-11*
- [Creating Custom Pyfunc Models — MLflow Guide](https://mlflow.org/docs/latest/traditional-ml/creating-custom-pyfunc/index.html) — *verified 2026-07-11*
- [MLflow Models From Code Guide](https://mlflow.org/docs/latest/model/models-from-code.html) — *verified 2026-07-11*
- [mlflow.models — ModelSignature API](https://mlflow.org/docs/latest/python_api/mlflow.models.html#mlflow.models.ModelSignature) — *verified 2026-07-11*
- [Custom Models Overview — Databricks Model Serving](https://docs.databricks.com/en/machine-learning/model-serving/custom-models.html) — *verified 2026-07-11*
