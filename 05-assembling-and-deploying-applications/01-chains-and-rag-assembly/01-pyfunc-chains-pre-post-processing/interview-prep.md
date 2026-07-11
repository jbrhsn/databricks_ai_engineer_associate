# Interview Prep: MLflow PyFunc Chains with Pre- and Post-Processing

**Section:** 05-Assembling and Deploying | **Topic:** `mlflow.pyfunc.PythonModel`, `ModelSignature`, Databricks Model Serving | **Date:** 2026-07-11

---

## Core Conceptual Questions

| Question | Key points to cover | Common weak-answer trap |
|---|---|---|
| What is `mlflow.pyfunc.PythonModel` and why would you use it instead of a native MLflow flavor? | Base class for packaging arbitrary Python inference logic; survives framework lock-in; supports multi-step pipelines; `predict()` is the single serving entry point; class-based for stateful init vs function-based for simple cases | "It's just a wrapper around a model" — misses that it encapsulates the entire pipeline, not just the model call |
| Explain the difference between `load_context()` and `predict()` in terms of when they run and what belongs in each. | `load_context()` runs once per model load / endpoint cold-start; `predict()` runs per request; heavy artifacts (vector stores, LLM clients, graphs) belong in `load_context()`; request-scoped logic belongs in `predict()` | Saying both run per request, or saying `load_context()` is for validation |
| What is a `ModelSignature` and what problem does it solve? | Declarative I/O schema (`Schema` of `ColSpec`/`TensorSpec`) stored in `MLmodel`; enforces input types at serving time before `predict()` runs; required for Unity Catalog registration; machine-readable API contract | "It's optional metadata" — misses that it blocks bad requests at the gateway layer and is mandatory for UC |
| How does `artifacts=` in `log_model()` differ from hardcoding a file path inside `predict()`? | `artifacts=` downloads URIs and makes them available at `context.artifacts["name"]` (guaranteed local path); hardcoded paths fail in the serving container which has no access to the notebook driver's filesystem | "They both work" — ignores that hardcoded paths are inaccessible outside the training environment |
| What is the "models from code" pattern and when is it preferable to passing a class instance? | Pass a `.py` file path to `python_model=`; MLflow serializes the file, not a pickle; avoids cloudpickle limitations; makes chain logic reviewable in version control; preferred for CI/CD and credential safety | Confusing it with `code_paths=` (helper files) or thinking it requires a different `predict()` interface |

---

## Applied / Scenario Questions

**Q1: A team built a RAG pipeline. Their serving endpoint works locally but times out at 20 QPS. The logs show each request takes 8–10 seconds even though the LLM call alone takes 1 second. How do you diagnose and fix this?**

Strong answer framework:
- **Diagnose:** The 7–9 second gap between total latency and LLM latency points to per-request initialization. The first place to look is any `FAISS.load_local()`, `ChatOpenAI()` instantiation, or vector store client construction inside `predict()`.
- **Root cause:** The vector store or LLM client is being re-initialized on every request because the initialization code is inside `predict()` instead of `load_context()`.
- **Fix:** Move all one-time setup into `load_context()`. Assign loaded objects as `self.vector_store`, `self.llm`, etc. `predict()` then uses already-warm objects.
- **Verify:** Re-deploy and compare p99 latency on the endpoint metrics dashboard. Per-request latency should drop to LLM call time plus retrieval time only.
- **Bonus signal:** Mention that `load_context()` is called after `mlflow.pyfunc.load_model()` completes, giving you a local way to reproduce the timing before deploying.

---

**Q2: You need to register a PyFunc model to Unity Catalog path `main.ml.support_chain`. The model takes a DataFrame with columns `question` (string) and `tier` (string) and returns `answer` (string) and `sources` (string). Write the signature construction and log call skeleton.**

Strong answer framework:
```python
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, ColSpec, DataType

input_schema = Schema([
    ColSpec(DataType.string, "question"),
    ColSpec(DataType.string, "tier"),
])
output_schema = Schema([
    ColSpec(DataType.string, "answer"),
    ColSpec(DataType.string, "sources"),
])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

with mlflow.start_run():
    mlflow.pyfunc.log_model(
        name="support_chain",
        python_model=SupportChainModel(),
        artifacts={"vector_store_path": "/dbfs/tmp/kb_faiss"},
        signature=signature,
        input_example=pd.DataFrame([{"question": "...", "tier": "premium"}]),
        pip_requirements=["langchain==0.3.7", "faiss-cpu==1.8.0"],
        registered_model_name="main.ml.support_chain",
    )
```
- Emphasis on `registered_model_name` in `catalog.schema.model` format for UC.
- Explicit `signature=` not delegated to `infer_signature()` for LLM output.
- `input_example=` as a real DataFrame, not a dict.

---

**Q3: After updating the output schema of a PyFunc model (adding a `confidence` column), some downstream consumers start getting unexpected data. What broke and how do you prevent this?**

Strong answer framework:
- **What broke:** The downstream consumer was consuming the model's response without schema validation, treating the output as a fixed-shape object. Adding a column is backward-compatible for most consumers (they can ignore unknown fields), but if the consumer was iterating positionally over columns or using `df.columns[0]` directly, the new column shifts indices or changes shape.
- **Prevention:** (1) Communicate schema changes via the `ModelSignature` version history in UC. (2) Consumers should reference columns by name, never by position. (3) Use UC model aliases (e.g., `@champion`) so consumers can pin to a known-good version and only upgrade intentionally. (4) For breaking changes (removing or renaming a column), increment the major version and maintain the old version in "Archived" state during the transition window.

---

## System Design / Architecture Questions

**Q: Design the deployment pipeline for a customer support LLM chain that must support dev, staging, and production environments with different LLM endpoints and retrieval configurations, zero downtime on updates, and full audit traceability.**

**Approach:**

1. **Clarify requirements:**
   - How many environments? (dev / staging / prod)
   - What "zero downtime" means here: simultaneous traffic split or blue-green?
   - Audit: who needs to inspect what? (chain logic, input/output data, model version)

2. **Propose the architecture:**
   ```
   Authoring Notebook
         │
         ▼ mlflow.pyfunc.log_model(
                python_model="chain.py",          # models-from-code for auditability
                model_config={"llm_endpoint": "gpt-4o-mini-dev"},
                registered_model_name="main.ml.support_chain")
         │
         ▼
   Unity Catalog Model Registry
   (versions: 1=dev, 2=staging, 3=prod; aliases: @dev, @staging, @champion)
         │
         ▼ Databricks Model Serving Endpoint per environment
         │    dev-endpoint    → models:/main.ml.support_chain@dev
         │    staging-endpoint→ models:/main.ml.support_chain@staging
         │    prod-endpoint   → models:/main.ml.support_chain@champion
         │
         ▼ Zero-downtime updates: Databricks performs blue-green by default
           (old config serves until new config passes health checks)
   ```

3. **Justify tradeoffs:**
   - `model_config=` for env-specific LLM endpoints avoids logging separate artifacts per environment; same model binary, different runtime config.
   - UC aliases (`@champion`, `@staging`) decouple version numbers from environment references; CI/CD scripts update aliases, not hardcoded version numbers.
   - "Models from code" means `chain.py` is in git, making the chain logic auditable via git blame/diff without inspecting artifact store binaries.
   - Databricks Model Serving's zero-downtime update keeps the old endpoint warm until the new container passes health checks — no code needed to orchestrate this.

---

## Vocabulary That Signals Expertise

| Term / Phrase | When and why to use it |
|---|---|
| `load_context()` lifecycle hook | When explaining endpoint cold-start behavior or per-request latency issues — signals you understand the PyFunc lifecycle |
| `context.artifacts` | When discussing artifact portability — signals you know artifacts must be packaged, not referenced by ambient path |
| `ModelSignature` + `ColSpec` + `Schema` | When discussing UC registration or consumer contracts — signals familiarity with the type system beyond "infer_signature" |
| "models from code" pattern | When discussing CI/CD or cloudpickle limitations — signals awareness of the recommended modern pattern |
| `model_config=` | When discussing multi-environment deployments — signals you know how to avoid logging multiple artifact copies |
| UC model aliases (`@champion`, `@staging`) | When discussing zero-downtime updates or blue-green deployments — signals production MLOps maturity |
| "PyFuncModel wrapper" | When explaining what `load_model()` returns — signals you understand the wrapper vs the raw class |
| "input schema enforcement at the gateway" | When explaining the 400 vs 500 error distinction — signals understanding of where schema validation happens |

---

## Vocabulary That Signals Weakness

| Term / Phrase | Why it is a red flag |
|---|---|
| "I just use `infer_signature()` for everything" | Signals you have not encountered the LLM string-to-object dtype problem; interviewers probe with "what if the output is a JSON string?" |
| "I put the model loading in `predict()` for simplicity" | Signals unfamiliarity with `load_context()` — the most common production PyFunc mistake |
| "I hardcode the DBFS path in `predict()`" | Signals you have not shipped a PyFunc model to a real serving environment; containers do not have DBFS access |
| "Signatures are optional unless you need UC" | True in a narrow technical sense, but signals you treat the serving contract as an afterthought |
| "We just use `mlflow.langchain.log_model()` and it handles everything" | Signals you have not hit a use case requiring custom pre/post-processing — the interviewer will probe with a scenario that requires it |
| "cloudpickle serializes everything automatically" | Signals unawareness of cloudpickle limitations: live connections, module-level imports, lambda capture of non-picklable state |

---

## STAR Answer Frame

**Situation:** Our team deployed an LLM-powered document Q&A feature to Databricks Model Serving. After launch, we received reports that the endpoint was returning inconsistent response formats — sometimes a plain string, sometimes a JSON object — depending on which engineer had last updated the serving logic.

**Task:** I was responsible for standardizing the endpoint so that: (1) the response format was always a typed DataFrame with two named columns, (2) all environment-specific configuration was externalized, and (3) the chain logic was auditable in version control.

**Action:**
- Refactored the chain from a loose function into a `PythonModel` subclass with explicit `load_context()` / `predict()` separation.
- Defined an explicit `ModelSignature` with `Schema([ColSpec(DataType.string, "answer"), ColSpec(DataType.string, "sources")])` for the output.
- Migrated to the "models from code" pattern — the `chain.py` file went into git and `log_model(python_model="chain.py")` replaced the instance-based approach.
- Added `model_config={"llm_endpoint": "..."}` so dev, staging, and prod could use different LLM endpoints from the same artifact.
- Updated the CI pipeline to run `mlflow.pyfunc.load_model(uri).predict(input_example)` as a smoke test before promoting a version to `@champion`.

**Result:** Response format consistency went from "depends on who deployed last" to 100% enforced by the `ModelSignature` at the serving gateway. The first time a colleague accidentally changed the output column name, the UC registration step failed immediately rather than silently breaking downstream consumers. Deployment confidence increased enough that we moved to weekly model updates from monthly.

---

## Red Flags Interviewers Watch For

- **Confusing `PythonModel.predict()` signature with PyFuncModel.predict() signature.** Your class's `predict` takes `(self, context, model_input, params=None)`; the wrapper's `.predict()` takes `(data, params=None)`. Mixing these up during a whiteboard exercise signals surface-level familiarity.

- **Not knowing what `context` is.** If asked "what is the `context` argument in `predict()`?" and you say "it's the MLflow run context," that's wrong. It is a `PyfuncContext` object with `.artifacts` and `.model_config` — both populated from what was logged with `artifacts=` and `model_config=`.

- **Describing Model Serving as calling your `PythonModel` directly.** Model Serving calls `PyFuncModel.predict()` (the wrapper), which in turn calls your class's `predict()`. This matters when debugging: the schema enforcement and type coercion happen in the wrapper layer, not in your code.

- **Assuming `pip_requirements` is automatically captured without specification.** MLflow does attempt inference, but for complex environments (especially with LangChain's rapidly evolving dependency graph), inference often misses transitive dependencies. Explicit pinning is the safe default.

- **Treating `infer_signature()` as equivalent to writing a schema.** For standard ML models, inference is reliable. For LLM chains, string outputs infer as `object` and structured outputs vary. Knowing when `infer_signature` is sufficient vs when to hand-craft `ColSpec` entries is a signal of real deployment experience.
