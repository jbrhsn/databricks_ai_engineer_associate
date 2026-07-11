# MLflow Agent Framework — Logging, Tracing, and Deploying Agents

**Section:** 04 Application Development | **Module:** 03 Agents, MLflow & Genie | **Est. time:** 3 hrs | **Exam mapping:** Application Development 30% — log, trace, evaluate, and deploy AI agents with MLflow

---

## TL;DR

MLflow provides the end-to-end lifecycle toolchain for AI agents on Databricks: `mlflow.langchain.log_model()` packages a LangChain or LangGraph agent as a versioned, reproducible artifact; the MLflow Model Registry (backed by Unity Catalog) governs promotion from development to production; `@mlflow.trace` and `mlflow.langchain.autolog()` give you span-level observability into every node and LLM call; and `mlflow.evaluate()` with `model_type="databricks-agent"` runs LLM-judge scoring on groundedness, correctness, and chunk relevance before you ever push to serving. The Databricks Agent Framework wraps all of this with the `ResponsesAgent` interface so any framework—LangGraph, LangChain, LlamaIndex, OpenAI SDK—plugs in identically. **The one thing to remember: log the model with MLflow first; everything else (tracing, evaluation, serving, monitoring) flows from that single registered artifact.**

---

## ELI5 — Explain It Like I'm 5

Imagine you are a scientist who runs experiments in a lab notebook. Every time you try a new experiment, you write down exactly which chemicals you used, in what amounts, and what the result was—so that six months later you (or anyone else) can repeat it exactly. MLflow is that lab notebook for AI agents. When you call `mlflow.langchain.log_model()`, you are writing the complete recipe for your agent into the notebook: the code, the library versions, the input and output format, all of it. The Model Registry is the filing cabinet where you store all your recipes and decide which one is the "official" one currently in use. The most common misconception is thinking you can just save a Python pickle file and get the same result—you cannot, because pickle stores the object state but not the environment it lives in, so it breaks silently when library versions change. MLflow stores both the object and the environment together, reproducibly.

---

## Learning Objectives

By the end of this chapter you will be able to:

- [ ] Log a LangChain or LangGraph agent to MLflow using `mlflow.langchain.log_model()` with correct `artifact_path`, `registered_model_name`, and `input_example` arguments
- [ ] Instrument an agent with `@mlflow.trace` and `mlflow.langchain.autolog()` and identify the resulting span hierarchy in the MLflow UI Traces tab
- [ ] Execute `mlflow.evaluate()` with `model_type="databricks-agent"` against a structured eval dataset and interpret the LLM-judge metrics (groundedness, correctness, chunk relevance)
- [ ] Explain the Unity Catalog-backed Model Registry stage workflow and the champion/challenger pattern for safe promotion to production
- [ ] Describe what the `ResponsesAgent` interface provides and why Databricks recommends it over direct `ChatModel` wrapping

---

## Visual Overview

### MLflow Agent Lifecycle (Develop → Deploy)

```
┌─────────────────────────────────────────────────────────────────────┐
│                      DEVELOPMENT NOTEBOOK                           │
│                                                                     │
│   Build agent (LangGraph / LangChain / OpenAI SDK)                  │
│          │                                                          │
│          ▼                                                          │
│   mlflow.start_run()                                                │
│   mlflow.langchain.log_model(lc_model=..., name="rag-agent")       │
│          │                                                          │
│          ▼                                                          │
│   mlflow.evaluate(model_type="databricks-agent", data=eval_df)     │
│          │                                                          │
│          ▼                                                          │
│   mlflow.register_model(model_uri, "catalog.schema.rag_agent")     │
└────────────────────┬────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    UNITY CATALOG MODEL REGISTRY                     │
│                                                                     │
│   Version 1  ──►  Alias: @champion  ──►  Model Serving Endpoint    │
│   Version 2  ──►  Alias: @challenger ──► Shadow / A-B test         │
│                                                                     │
│   Promote challenger → champion when eval metrics pass gate         │
└─────────────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PRODUCTION SERVING ENDPOINT                      │
│                                                                     │
│   Databricks Model Serving ──► REST API ──► Client applications     │
│   MLflow Tracing (async)   ──► Traces tab ──► Online monitoring     │
└─────────────────────────────────────────────────────────────────────┘
```

### MLflow Tracing Span Hierarchy for a RAG Agent

```
Trace: user_question = "What is RAG?"
│
├── [ROOT] rag_agent  (span_type=CHAIN)
│     inputs: {"messages": [{"role": "user", "content": "..."}]}
│     outputs: {"messages": [{"role": "assistant", "content": "..."}]}
│     latency: 1840 ms
│
│   ├── retrieve_context  (span_type=RETRIEVER)
│   │     inputs: {"query": "What is RAG?"}
│   │     outputs: [{"doc_uri": "...", "content": "..."}]  (top-3 chunks)
│   │     latency: 320 ms
│   │
│   └── generate_response  (span_type=LLM)
│         inputs: {"messages": [...], "context": [...]}
│         outputs: {"content": "RAG stands for..."}
│         latency: 1510 ms
│         token_count: 412
```

### Model Registry Stage Transition

```
             ┌──────────────────────────────────────────────────────┐
             │              UNITY CATALOG MODEL REGISTRY             │
             │                                                      │
  log_model  │  Version N   →  @candidate alias                    │
  ──────────►│     ↓                                               │
             │  Run mlflow.evaluate() on hold-out eval set         │
             │     ↓                                               │
             │  Pass gate?                                         │
             │  ├── Yes ──►  client.set_registered_model_alias(    │
             │  │             "catalog.schema.agent",              │
             │  │             "champion", version=N)               │
             │  │             Old champion → @archived              │
             │  └── No  ──►  Keep @champion unchanged               │
             └──────────────────────────────────────────────────────┘
```

---

## Key Concepts

### `mlflow.langchain.log_model()` — Packaging Agents as Reproducible Artifacts

**What is it?** `mlflow.langchain.log_model()` serializes a LangChain chain, LangGraph compiled graph, or any `Runnable` as an MLflow model artifact, capturing the code, dependencies, and input/output schema in a single addressable location.

**How does it work under the hood?** MLflow uses the "models from code" approach for LangGraph (since MLflow 2.12.2), where instead of pickling the live object, it records a reference to the Python file that defines the agent; when the model is loaded, that file is re-executed to reconstruct the object. For simpler chains, it uses cloudpickle serialization. In both cases, MLflow records a `conda.yaml` and `requirements.txt` capturing the exact Python environment, and an `MLmodel` manifest file that describes the model flavor, input signature, and output signature. Optional `input_example` is stored as a JSON artifact and shown in the UI for easy testing.

**Where does it appear?** Called inside a `mlflow.start_run()` context block in a Databricks notebook or CI script. The logged model appears under the **Artifacts** tab of the MLflow Experiment run. When `registered_model_name` is supplied (as `catalog.schema.model_name` for Unity Catalog), the model is simultaneously registered in the Unity Catalog Model Registry and is immediately queryable via `models:/catalog.schema.model_name/version`.

### MLflow Model Registry (Unity Catalog-backed) — Versioned Governance

**What is it?** The MLflow Model Registry is a centralized versioned model store that tracks every logged model version, stores aliases (named pointers like `@champion` and `@challenger`), records lineage back to the originating run, and enforces governance through Unity Catalog permissions.

**How does it work under the hood?** Every call to `mlflow.register_model()` or `log_model(registered_model_name=...)` creates a new numbered version in the Registry. Aliases are mutable pointers that can be atomically swapped: `client.set_registered_model_alias("catalog.schema.agent", "champion", version=N)` updates the pointer without downtime. The Unity Catalog integration means the model inherits all UC three-part namespace governance: catalog-level isolation, schema-level organization, and table/model-level `GRANT` statements for access control. Champion/challenger testing works by assigning a new version the `@challenger` alias, routing a fraction of traffic to it via Model Serving, evaluating the metrics, and then swapping the `@champion` alias if quality improves.

**Where does it appear?** In the Databricks UI under **Machine Learning → Models** (Unity Catalog view). Programmatically via `MlflowClient().get_registered_model("catalog.schema.model_name")` and `client.set_registered_model_alias(...)`. Model Serving endpoints reference models via `models:/catalog.schema.model_name/@champion` URIs.

### MLflow Tracing for Agents — Span-Level Observability

**What is it?** MLflow Tracing captures the inputs, outputs, timing, and metadata of every step in an agent execution as a hierarchical tree of spans, providing complete causal visibility into how an agent reached its answer.

**How does it work under the hood?** Tracing has two modes. **Automatic**: `mlflow.langchain.autolog()` registers callbacks into the LangChain/LangGraph callback system; every node execution, LLM call, and retriever call emits a span automatically without code changes. **Manual**: the `@mlflow.trace` decorator wraps a Python function—when the function is called, MLflow opens a span, records the function name, arguments, return value, and wall-clock time, then closes the span. Nested `@mlflow.trace` decorators produce parent-child span relationships. `mlflow.start_span()` as a context manager offers finer-grained control for arbitrary code blocks. Each span carries a `span_type` (CHAIN, LLM, RETRIEVER, TOOL, FUNC) and custom attributes. Spans are grouped into a **Trace**, linked to the MLflow Experiment run.

**Where does it appear?** In the Databricks UI under the **Traces** tab of the MLflow Experiment. Each trace shows the full span tree, latency waterfall, token counts, and raw inputs/outputs per span. In production, traces from Model Serving endpoints are automatically logged asynchronously and appear in the same Traces tab, enabling offline evaluation on real traffic.

### `mlflow.evaluate()` with `model_type="databricks-agent"` — LLM-Judge Evaluation

**What is it?** `mlflow.evaluate()` with `model_type="databricks-agent"` is an LLM-as-a-judge evaluation harness that runs a structured set of LLM-graded quality metrics—groundedness, answer correctness, and chunk relevance—over a tabular evaluation dataset, producing per-row scores and aggregate metrics.

**How does it work under the hood?** The function accepts a Pandas DataFrame with columns `request` (the user question), `response` (the agent answer), optional `expected_response` (ground truth), and optional `retrieved_context` (the chunks the RAG retrieved). For each row, Databricks-hosted LLM judges analyze the response against the ground-truth and retrieved context and output a `yes/no` binary score plus a written rationale. The judges are called in parallel via Databricks Model Serving and results are written back as new columns on the DataFrame. Aggregate percentages (e.g., `percent_correct`, `percent_grounded`) are logged as metrics on the MLflow run, enabling comparison across experiment runs. In MLflow 3, `mlflow.genai.evaluate()` supersedes the older API but the `databricks-agent` model_type path remains supported for MLflow 2 compatibility.

**Where does it appear?** Results appear under the **Metrics** and **Tables** sections of the MLflow Experiment run. `result.tables["eval_results"]` returns the per-row scored DataFrame directly in the notebook. The Traces tab also shows individual inference traces generated during evaluation.

### Databricks Agent Framework — `ResponsesAgent` and Unity Catalog Integration

**What is it?** The Databricks Agent Framework is Databricks's opinionated wrapper layer that standardizes how agents are packaged, deployed, and monitored on the platform, independent of which authoring library was used. Its primary interface is MLflow's `ResponsesAgent` abstract class.

**How does it work under the hood?** A `ResponsesAgent` subclass implements two methods: `predict()` for synchronous responses and optionally `predict_stream()` for streaming. The agent is wrapped with `mlflow.pyfunc.ResponsesAgent` and logged with `mlflow.pyfunc.log_model()`, which captures an OpenAI-compatible Responses API schema as the model signature. This schema is what allows the agent to be queried from the AI Playground, wired into the Review App for human feedback, and monitored by the same LLM judges used during development—without any code changes. The framework is framework-agnostic: LangGraph, LangChain, LlamaIndex, and OpenAI SDK agents can all be wrapped identically. Unity Catalog integration means the deployed endpoint inherits UC access controls, lineage, and audit logging.

**Where does it appear?** In Databricks, deployed as a Model Serving endpoint. The `ResponsesAgent` interface is defined in `mlflow.pyfunc`. Deployed agents are callable via `POST <endpoint>/invocations` (Model Serving) or `POST <app-url>/responses` (Databricks Apps). Tracing and evaluation flow through the same MLflow Experiment linked at logging time.

---


---

## Key Parameters / Configuration Knobs

| Parameter | What it controls | Decision rule |
|---|---|---|
| `mlflow.langchain.log_model(lc_model=...)` | Path to the Python file defining the agent (models-from-code) or the live chain object | Use a file path (string) for LangGraph agents to avoid pickle serialization failures; pass the live object only for simple LangChain chains that fully support cloudpickle |
| `mlflow.langchain.log_model(artifact_path=...)` | Sub-directory name inside the MLflow run's artifact store where the model is saved | Use a short, descriptive name (`"agent"`, `"rag-chain"`); this becomes part of the `model_uri` |
| `mlflow.langchain.log_model(registered_model_name=...)` | Whether and where to register the model in the Unity Catalog Model Registry | Supply as `"catalog.schema.model_name"` when you want immediate registration; omit when iterating locally and registering separately |
| `mlflow.langchain.log_model(input_example=...)` | A sample input stored with the model for UI testing and signature inference | Always provide; without it, Model Serving cannot validate inputs and the AI Playground shows no default request |
| `mlflow.evaluate(model_type=...)` | Which set of LLM judges and scoring logic to activate | Set `"databricks-agent"` for RAG/agent evaluation; omit or use `"classifier"` / `"regressor"` for traditional ML |
| `mlflow.evaluate(data=...)` | The eval dataset as a Pandas DataFrame or MLflow Dataset | Must contain at minimum a `request` column; add `expected_response` and `retrieved_context` to unlock correctness and groundedness judges respectively |
| `@mlflow.trace(span_type=...)` | The semantic type tag on the span, used for filtering in the Traces UI | Set `SpanType.LLM` for LLM calls, `SpanType.RETRIEVER` for vector search, `SpanType.CHAIN` for orchestration logic, `SpanType.TOOL` for function calls |
| `mlflow.start_run(experiment_id=...)` | Which MLflow Experiment receives the run's metrics, params, and artifacts | Set to the experiment created for the project; on Databricks, `mlflow.set_experiment("/path/to/experiment")` is idiomatic |

---


---

## Worked Example: Requirement → Decision

**Given:** A team has built a LangGraph RAG agent that answers HR policy questions. It retrieves from a Unity Catalog Vector Search index, calls a Databricks-hosted LLM, and returns answers. The team needs to (1) package it for reproducible deployment, (2) measure quality before releasing, and (3) observe failures in production.

**Step 1 — Identify the goal:** Log the agent as a versioned MLflow model with full dependency capture, evaluate quality against a 10-question eval set, register in Unity Catalog, deploy to Model Serving, and enable production tracing.

**Step 2 — Define inputs:** A compiled LangGraph graph object (`app`), a Pandas DataFrame `eval_df` with `request`, `expected_response`, and `retrieved_context` columns, and a Unity Catalog namespace `hr_catalog.agents.hr_rag_v1`.

**Step 3 — Define outputs:** A registered model version at `hr_catalog.agents.hr_rag_v1` with the `@champion` alias, an MLflow run with `percent_grounded` and `percent_correct` metrics, and a Model Serving endpoint callable by downstream applications.

**Step 4 — Apply constraints:** LangGraph graphs use stateful checkpointers that cannot be pickled reliably; the `ResponsesAgent` interface is required for AI Playground compatibility; eval must pass 80% groundedness before promotion.

**Step 5 — Select the approach:** Log with `mlflow.langchain.log_model(lc_model="./agent.py", ...)` (models-from-code to avoid pickle issues), evaluate with `mlflow.evaluate(model_type="databricks-agent")`, register with `registered_model_name="hr_catalog.agents.hr_rag_v1"`, and promote with `client.set_registered_model_alias(...)` only if `percent_grounded >= 0.80`. Alternative of pickling directly fails because LangGraph checkpointers hold open database connections that are not serializable; alternative of manual evaluation with `print()` statements loses repeatability and cannot compare across versions.

---

## Implementation

```python
# Scenario: Package a LangGraph RAG agent for reproducible deployment so that
# any team member can load and run the exact same version without environment drift.

import mlflow
from mlflow.models import infer_signature

# --- agent.py (separate file, required for LangGraph models-from-code logging) ---
# from langgraph.prebuilt import create_react_agent
# from databricks_langchain import ChatDatabricks
# ...
# mlflow.models.set_model(compiled_graph)  # marks the object to be loaded

# --- logging_notebook.py ---
mlflow.set_experiment("/Shared/hr-rag-agent-dev")

input_example = {
    "messages": [{"role": "user", "content": "What is the parental leave policy?"}]
}

with mlflow.start_run(run_name="v1-langgraph-rag") as run:
    # Log the LangGraph agent from code (avoids pickle serialization issues)
    model_info = mlflow.langchain.log_model(
        lc_model="./agent.py",           # path to file containing set_model()
        artifact_path="agent",            # sub-dir in the run's artifact store
        registered_model_name="hr_catalog.agents.hr_rag_v1",  # UC registration
        input_example=input_example,      # stored for UI testing + sig inference
        pip_requirements=[
            "langchain>=0.3.0",
            "langgraph>=0.2.0",
            "databricks-langchain>=0.3.0",
        ],
    )
    print(f"Model URI: {model_info.model_uri}")
    # → models:/hr_catalog/agents/hr_rag_v1/1
```

```python
# Scenario: Instrument an existing RAG agent with manual tracing to get
# span-level visibility into retrieval and generation steps independently,
# so we can measure retriever latency vs. LLM latency separately.

import mlflow
from mlflow.entities import SpanType

mlflow.langchain.autolog()  # auto-traces LangChain/LangGraph calls

# Additional manual spans for custom business logic
@mlflow.trace(span_type=SpanType.RETRIEVER, name="vector_search_retrieve")
def retrieve_context(query: str, index_name: str) -> list[dict]:
    """Fetch top-k documents from Unity Catalog Vector Search."""
    from databricks.vector_search.client import VectorSearchClient
    client = VectorSearchClient()
    index = client.get_index(index_name=index_name)
    results = index.similarity_search(
        query_text=query,
        columns=["chunk_text", "doc_uri"],
        num_results=3,
    )
    return [
        {"content": row[0], "doc_uri": row[1]}
        for row in results.get("result", {}).get("data_array", [])
    ]

@mlflow.trace(span_type=SpanType.CHAIN, name="hr_rag_agent")
def run_agent(question: str) -> str:
    """Orchestrate retrieval + generation with full span tracing."""
    context = retrieve_context(question, "hr_catalog.agents.hr_policy_index")
    # LangChain/LangGraph call is auto-traced by autolog
    response = rag_chain.invoke({
        "question": question,
        "context": context,
    })
    return response
```

```python
# Anti-pattern: Saving a LangGraph agent as a plain pickle file to DBFS.
# This approach loses dependency tracking, breaks evaluation, and silently
# fails when the Python environment changes even slightly.

import pickle

# WRONG — Do not do this
with open("/dbfs/models/hr_rag_agent.pkl", "wb") as f:
    pickle.dump(compiled_graph, f)
# Problems:
# 1. No dependency record — reloading on a different cluster version silently breaks
# 2. LangGraph checkpointers (DB connections, thread state) are not picklable
# 3. No MLflow run → no metrics, no eval results, no lineage, no Model Serving support
# 4. Cannot compare this version against future versions

# CORRECT — Use MLflow models-from-code logging (shown in snippet 1 above)
# mlflow.langchain.log_model(lc_model="./agent.py", artifact_path="agent", ...)
```

```python
# Scenario: Run LLM-judge evaluation against a 5-question eval set to gate
# production promotion — must achieve >=80% groundedness before pushing to serving.

import mlflow
import pandas as pd

eval_data = pd.DataFrame({
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
        [{"doc_uri": "policy/parental-leave.pdf", "content": "12 weeks paid leave..."}],
        [{"doc_uri": "policy/pto.pdf", "content": "15 days first year..."}],
        [{"doc_uri": "policy/fmla.pdf", "content": "Submit 30 days in advance..."}],
        [{"doc_uri": "policy/remote.pdf", "content": "Role-dependent eligibility..."}],
        [{"doc_uri": "policy/expenses.pdf", "content": "Submit within 30 days..."}],
    ],
})

with mlflow.start_run(run_name="eval-v1-gate-check"):
    result = mlflow.evaluate(
        model="models:/hr_catalog/agents/hr_rag_v1/1",  # logged model URI
        data=eval_data,
        model_type="databricks-agent",  # activates Databricks LLM judges
        evaluator_config={
            "databricks-agent": {
                "global_guidelines": {
                    "english": ["Response must be in English"],
                    "concise": ["Response must be under 200 words"],
                }
            }
        },
    )

# Inspect per-row results
display(result.tables["eval_results"])

# Gate check: promote only if grounded >=80%
grounded_pct = result.metrics.get("agent/percent_grounded", 0)
print(f"Groundedness: {grounded_pct:.1%}")

if grounded_pct >= 0.80:
    from mlflow import MlflowClient
    client = MlflowClient()
    client.set_registered_model_alias(
        "hr_catalog.agents.hr_rag_v1", "champion", version=1
    )
    print("Promoted to @champion")
else:
    print("Did NOT meet quality gate. Investigate eval_results before promoting.")
```

---

## Common Pitfalls & Misconceptions

- **Pickling LangGraph agents** — Beginners reach for `pickle` or `cloudpickle` because they work for simple Python objects and they assume an agent is just a Python object. LangGraph compiled graphs contain stateful checkpointers that hold live database connections and thread-local state; these cannot be serialized, causing `TypeError` at save time or silent corruption at load time. Use `mlflow.langchain.log_model(lc_model="./agent.py")` (models-from-code) so MLflow stores the definition file, not the live object.

- **Calling `mlflow.evaluate()` without `retrieved_context`** — Engineers often include only `request` and `expected_response`, assuming that is enough for groundedness scoring. Groundedness measures whether the response is supported by the retrieved context; without supplying `retrieved_context`, the judge has nothing to check against and the groundedness metric is skipped or returns null. Always include `retrieved_context` in the eval DataFrame if your agent does retrieval.

- **Confusing Unity Catalog model aliases with deprecated MLflow stage strings** — Older MLflow 1.x documentation uses lifecycle stages (`Staging`, `Production`, `Archived`) set via `transition_model_version_stage()`. The Unity Catalog-backed Model Registry replaces stages with arbitrary-string aliases (`@champion`, `@challenger`, `@latest`) set via `set_registered_model_alias()`. Using the deprecated stage API on a UC-backed model raises an error because UC does not support lifecycle stages.

- **Thinking `mlflow.langchain.autolog()` replaces manual `@mlflow.trace`** — Autolog instruments the LangChain/LangGraph callback system and traces all framework-level calls. But custom Python functions outside the framework (your own retrieval preprocessing, post-processing logic, tool functions) are invisible to autolog. Use `@mlflow.trace` to add manual spans around custom code, which then nest naturally inside the autolog spans.

- **Forgetting `mlflow.set_experiment()` before `start_run()`** — If no experiment is set, MLflow silently logs to the default experiment (`/Default`), making it impossible to find the run later. On Databricks, always call `mlflow.set_experiment("/Shared/project-name/experiment-name")` at the top of the notebook before any `start_run()` call.

---

## Key Definitions

| Term | Definition |
|---|---|
| MLflow Experiment | A named collection of runs that groups related model training or evaluation iterations; on Databricks, backed by a workspace path |
| MLflow Run | A single execution context (started with `mlflow.start_run()`) that records parameters, metrics, tags, and artifacts; the atomic unit of lineage |
| Span | A single timed operation within a trace, carrying a name, type, inputs, outputs, and start/end timestamps; the atomic unit of tracing |
| Trace | A hierarchical tree of spans representing the full execution path of one agent invocation, from the root user request to the final response |
| LLM Judge | A hosted LLM called by `mlflow.evaluate()` to score agent responses on qualitative criteria (groundedness, correctness, relevance) and return a binary yes/no with written rationale |
| Groundedness | An LLM-judge metric that scores whether the agent's response is supported by the retrieved context (yes = response is grounded in retrieved docs) |
| Model Registry | A versioned catalog of logged MLflow models with lineage, aliases, and governance; on Databricks, backed by Unity Catalog |
| Registered Model Alias | A mutable named pointer (e.g., `@champion`, `@challenger`) to a specific version in the Model Registry, used by Model Serving endpoints for zero-downtime promotion |
| `ResponsesAgent` | An MLflow abstract class that defines the standard predict/predict_stream interface for agents, providing automatic compatibility with AI Playground, Agent Evaluation, and Model Monitoring |
| Models-from-Code | An MLflow logging approach where the model is represented as a Python file path rather than a serialized object, eliminating pickle serialization issues for complex agents |

---

## Summary / Quick Recall

- `mlflow.langchain.log_model(lc_model="./agent.py", ...)` is the correct way to log LangGraph agents; avoid pickle.
- Supply `registered_model_name="catalog.schema.model_name"` to log and register in one step; omit only when iterating.
- Unity Catalog Model Registry uses **aliases** (`@champion`, `@challenger`) — not the deprecated stage strings (`Staging`, `Production`).
- `mlflow.langchain.autolog()` traces all LangChain/LangGraph framework calls; add `@mlflow.trace` for custom Python code outside the framework.
- `mlflow.evaluate(model_type="databricks-agent")` requires `request` column; add `retrieved_context` to unlock groundedness scoring, `expected_response` for correctness.
- Eval results land in `result.tables["eval_results"]` (per-row) and `result.metrics` (aggregates on the MLflow run).
- The `ResponsesAgent` interface makes any framework-based agent compatible with AI Playground, Review App, and production monitoring without code changes.

---

## Self-Check Questions

1. **[Recall]** What is the primary reason Databricks recommends using `mlflow.langchain.log_model(lc_model="./agent.py")` (models-from-code) over passing the live compiled LangGraph graph object directly?

   <details><summary>Answer</summary>

   **Correct: models-from-code avoids pickle serialization failures** inherent in LangGraph compiled graphs, which contain stateful checkpointers (live database connections, thread-local state) that cannot be serialized with cloudpickle. By storing the Python file that defines and sets the agent, MLflow re-executes the file at load time, reconstructing the object in whatever environment it is loaded into.

   **Why the distractor fails:** "It makes the model smaller" is wrong — file-based logging does not necessarily reduce artifact size. "It enables distributed training" is also wrong — model logging has nothing to do with training parallelism.

   </details>

2. **[Application]** Your team has logged version 3 of a RAG agent to the Unity Catalog Model Registry at `prod_catalog.agents.hr_bot`. The current `@champion` alias points to version 2. After running `mlflow.evaluate()` on version 3 and confirming it meets the quality gate, which Python call correctly promotes version 3 to champion?

   ```python
   from mlflow import MlflowClient
   client = MlflowClient()
   # Which of these is correct?
   # A. client.transition_model_version_stage("prod_catalog.agents.hr_bot", 3, "Production")
   # B. client.set_registered_model_alias("prod_catalog.agents.hr_bot", "champion", version=3)
   # C. client.update_model_version("prod_catalog.agents.hr_bot", 3, stage="Production")
   # D. mlflow.register_model("models:/prod_catalog/agents/hr_bot/3", stage="champion")
   ```

   <details><summary>Answer</summary>

   **Correct: B — `client.set_registered_model_alias("prod_catalog.agents.hr_bot", "champion", version=3)`**

   Unity Catalog-backed Model Registry uses **aliases** (arbitrary string pointers) instead of lifecycle stages. `set_registered_model_alias()` atomically moves the named pointer to the specified version — no downtime, instantly reflected in any endpoint that references `@champion`.

   **Why A and C fail:** `transition_model_version_stage()` and `stage=` parameters belong to the legacy workspace Model Registry (MLflow 1.x). They raise errors or are silently ignored on Unity Catalog-backed registries because UC does not implement lifecycle stages.

   **Why D fails:** `mlflow.register_model()` creates a new version registration; it does not set aliases. The `stage=` kwarg is also not a valid parameter for this function.

   </details>

3. **Which TWO** of the following columns in the eval DataFrame are required to enable the **groundedness** LLM-judge metric when calling `mlflow.evaluate(model_type="databricks-agent")`?

   - A. `request`
   - B. `expected_response`
   - C. `retrieved_context`
   - D. `trace`
   - E. `response`

   <details><summary>Answer</summary>

   **Correct: A (`request`) and C (`retrieved_context`)**

   Groundedness measures whether the agent's response is supported by the documents the retriever fetched. The judge needs the original user query (`request`) to understand the intent, and the `retrieved_context` (list of `{doc_uri, content}` dicts) to check whether the answer is factually grounded in those documents.

   **Why B is wrong:** `expected_response` is needed for the *answer correctness* judge, not groundedness. You can score groundedness without knowing what the correct answer "should" be.

   **Why D is wrong:** `trace` is optional metadata that Agent Evaluation can extract intermediate outputs from; it is not required for groundedness scoring if `retrieved_context` is supplied directly.

   **Why E is wrong:** `response` is needed (the judge scores the agent's answer), but it is generated by calling the model during evaluation — you don't supply it separately when passing `model=` to `mlflow.evaluate()`.

   </details>

4. **[Analysis]** A data scientist enables `mlflow.langchain.autolog()` for a LangGraph RAG agent. After reviewing the Traces tab, they notice that the vector search preprocessing step (a custom Python function they wrote) shows no span and its latency is invisible. What is the root cause and the correct fix?

   <details><summary>Answer</summary>

   **Root cause:** `mlflow.langchain.autolog()` instruments the LangChain/LangGraph callback system. It traces framework-defined operations (graph nodes that are `Runnable` steps, LLM calls, built-in retriever calls). A custom Python function defined outside the framework's `Runnable` interface is not visible to the callback system, so autolog never opens a span for it.

   **Correct fix:** Decorate the custom function with `@mlflow.trace(span_type=SpanType.RETRIEVER)`. Because `@mlflow.trace` is compatible with autolog, the new span will automatically nest inside the parent graph span in the trace tree — no conflict or duplicate spans are created.

   **Why simply "enabling more verbose logging" fails:** Increasing autolog verbosity only affects framework-level calls; it cannot trace arbitrary Python functions that are outside the framework abstraction.

   </details>

5. **[Trade-off]** Your team is debating two approaches for evaluating a production agent: (A) run `mlflow.evaluate()` offline on a fixed 50-question eval set before every release, or (B) collect production traces and run LLM judges asynchronously on live traffic after deployment. What does each approach catch that the other misses, and what is the recommended combined strategy?

   <details><summary>Answer</summary>

   **Approach A (offline eval set) catches:** regressions on known question patterns before users are exposed; provides a deterministic quality gate that blocks bad releases. Misses: distribution shift (real users ask questions not in your eval set), edge cases, and latency/cost behavior under real concurrency.

   **Approach B (production monitoring) catches:** live distribution shift, novel failure modes, and real-world latency/cost. Misses: pre-release regressions — by the time you detect a problem in production, users have already experienced degraded quality.

   **Recommended combined strategy:** Use offline evaluation (`mlflow.evaluate()`) as a pre-release quality gate with a minimum pass threshold (e.g., 80% groundedness). Use the same judge configuration in MLflow 3 production monitoring (`mlflow.genai.evaluate()` on production traces) as continuous post-release monitoring. Databricks specifically designs the same scorer/judge configuration to work in both modes so the quality definition is consistent across the lifecycle.

   </details>

---

## Further Reading

- [Use agents on Databricks — Agent Framework overview](https://docs.databricks.com/aws/en/agents/agent-framework/build-agents) — *verified 2026-07-11* — Index of all Databricks agent building, deploying, and evaluation tools
- [MLflow Tracing — GenAI observability (Databricks)](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/) — *verified 2026-07-11* — Overview of MLflow Tracing for end-to-end agent observability
- [Function decorators for manual tracing](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/app-instrumentation/manual-tracing/function-decorator) — *verified 2026-07-11* — `@mlflow.trace` decorator reference with `span_type`, `attributes`, and multi-decorator ordering
- [Agent Evaluation (MLflow 2)](https://docs.databricks.com/aws/en/agents/agent-evaluation/) — *verified 2026-07-11* — `mlflow.evaluate()` input schema, LLM judge metrics, and evaluation set design
- [Evaluate and monitor AI agents (MLflow 3)](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/) — *verified 2026-07-11* — MLflow 3 evaluation and production monitoring overview
- [MLflow LangChain Flavor](https://mlflow.org/docs/latest/genai/flavors/langchain/) — *verified 2026-07-11* — `mlflow.langchain.log_model()` and models-from-code logging for LangGraph
- [MLflow Agent Packaging and Deployment](https://mlflow.org/docs/latest/genai/flavors/) — *verified 2026-07-11* — Overview of all MLflow model flavors including `ResponsesAgent`
