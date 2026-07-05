# PyFunc, MLflow, and Unity Catalog Registration

**Section:** Assembling and Deploying Apps | **Module:** Packaging and Registration | **Est. time:** 2.5 hrs
**Exam mapping:** Databricks GenAI Engineer Associate — Assembling and Deploying Applications domain (~20%); MLflow model logging, the models-from-code pattern, and Unity Catalog registration are directly tested

## Learning Objectives

By the end of this chapter, you will be able to:

- Log a LangGraph `ResponsesAgent` as an MLflow model using the **models-from-code** pattern
- Explain why agents are logged as a *file path*, not a pickled Python object
- Declare `resources` at log time so a served agent gets automatic authentication passthrough
- Register a logged model to Unity Catalog's three-level namespace
- Manage deployment status with **aliases** instead of the deprecated **stages**
- Describe what changed in MLflow 3 vs MLflow 2 (`LoggedModel`, `name=`, `models:/<id>` URIs)

## Core Concepts

### Why "packaging" an agent is different from packaging a trained model

A traditional ML model has learned weights — you train it, and the result is a binary object
(a fitted scikit-learn estimator, a PyTorch state dict) that you serialize with pickle and reload
later. An LLM agent has **no trained weights of its own**. It is *code*: a graph of nodes that call
a foundation-model endpoint, retrieve from a vector index, and run tools. There is nothing meaningful
to pickle.

This is why Databricks and MLflow 3 use the **models-from-code** approach for agents. Instead of
serializing an object, you point MLflow at the **Python file that defines the agent**, and MLflow
stores that file as the model artifact. At serving time, MLflow re-runs the file to reconstruct the
agent. Your agent's "weights" are its source code plus the endpoints and indexes it talks to.

> **Version note (2026):** models-from-code has been available since MLflow 2.12.2 and is the
> Databricks-recommended way to log GenAI agents. MLflow 3 is the current major version.

### The models-from-code logging pattern

Three things make an agent script loggable:

1. The script defines the agent (a `ResponsesAgent` subclass, or any framework agent).
2. The script calls `mlflow.models.set_model(agent)` to tell MLflow which object is the model.
3. You call `mlflow.pyfunc.log_model(python_model="agent.py", ...)` — passing the **file path
   string**, not the object.

Here is a minimal agent file (`agent.py`) — this is the LangGraph `ResponsesAgent` you built in
Section 03, saved as its own file:

```python
# agent.py
import mlflow
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse
from langgraph.prebuilt import create_react_agent
from databricks_langchain import ChatDatabricks, VectorSearchRetrieverTool

# Build the underlying LangGraph agent
llm = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct")
retriever_tool = VectorSearchRetrieverTool(
    index_name="prod.agents.databricks_docs_index",
    tool_name="databricks_docs_retriever",
    tool_description="Retrieves relevant Databricks documentation.",
)
_graph = create_react_agent(llm, tools=[retriever_tool])


class DocsAgent(ResponsesAgent):
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        messages = [{"role": m.role, "content": m.content} for m in request.input]
        result = _graph.invoke({"messages": messages})
        text = result["messages"][-1].content
        return ResponsesAgentResponse(
            output=[self.create_text_output_item(text=text, id="msg-1")]
        )


# CRITICAL: this line tells MLflow which object is the servable model
mlflow.models.set_model(DocsAgent())
```

And the logging call (run this from a driver notebook, not from inside `agent.py`):

```python
import mlflow
from mlflow.models.resources import (
    DatabricksServingEndpoint,
    DatabricksVectorSearchIndex,
)

input_example = {"input": [{"role": "user", "content": "What is AI Search?"}]}

with mlflow.start_run():
    logged_agent = mlflow.pyfunc.log_model(
        python_model="agent.py",          # a PATH string, not DocsAgent()
        name="agent",                     # MLflow 3: use name=, not artifact_path=
        input_example=input_example,
        resources=[
            DatabricksServingEndpoint(endpoint_name="databricks-meta-llama-3-3-70b-instruct"),
            DatabricksVectorSearchIndex(index_name="prod.agents.databricks_docs_index"),
        ],
    )

print(logged_agent.model_uri)   # models:/<model_id>
```

Key parameters:

| Parameter | What it does |
|---|---|
| `python_model="agent.py"` | The file path to the agent code (models-from-code). MLflow **executes** this file at log time to validate it, and again at serving time to load the agent. |
| `name="agent"` | MLflow 3 replaces `artifact_path=` with `name=`. |
| `input_example` | Used to infer the model signature (auto for `ResponsesAgent` — see below). |
| `code_paths=[...]` | Additional local `.py` files or directories your agent imports (non-PyPI dependencies). |
| `pip_requirements` / `extra_pip_requirements` | Manually override / add pip dependencies. MLflow infers them from top-level imports otherwise. |
| `resources=[...]` | **The Databricks resources the agent needs** (see next section). |

### `resources` and automatic authentication passthrough

This is the most important Databricks-specific concept in the chapter, and one the exam reliably
tests. When your agent runs on a serving endpoint, it needs to call *other* Databricks resources —
a foundation-model endpoint for the LLM, an AI Search index for retrieval, maybe a UC function or a
Genie space. Those calls require credentials.

By declaring the resources **at log time** via the `resources=` list, you enable **automatic
authentication passthrough**: when the endpoint is deployed, Databricks provisions short-lived,
least-privilege credentials scoped exactly to those declared resources. The endpoint owner's
permissions on those resources are verified first.

Resource classes live in `mlflow.models.resources`:

- `DatabricksServingEndpoint(endpoint_name="...")` — a foundation-model or custom endpoint
- `DatabricksVectorSearchIndex(index_name="...")` — an AI Search index (three-level UC name)
- `DatabricksFunction(function_name="...")` — a Unity Catalog function tool
- `DatabricksGenieSpace(genie_space_id="...")` — a Genie space
- `DatabricksApp(app_name="...")` — for custom MCP servers hosted as Databricks Apps

**If you forget to declare a resource, the deployed endpoint will fail at runtime** with an
authentication error — even though the same code worked fine in the notebook (where it ran as you).
This "works locally, fails on the endpoint" symptom is a classic exam scenario.

The alternative auth model is **on-behalf-of-user (OBO)** with `ModelServingUserCredentials()`, where
the agent acts with the *calling user's* permissions instead of a service credential. This is used
when per-user data access enforcement matters.

### `ResponsesAgent` and automatic signature inference

Because you wrapped the agent with `mlflow.pyfunc.ResponsesAgent` in Section 03, MLflow **automatically
infers the model signature** from the standard `ResponsesAgentRequest` / `ResponsesAgentResponse`
schemas. You do not supply a signature, and you don't even strictly need an `input_example`. MLflow
also auto-appends the metadata `{"task": "agent/v1/responses"}`.

Contrast with a plain `PythonModel`, where you *must* provide an `input_example` (so MLflow can infer
a signature) or define the signature manually — a serving endpoint refuses a model with no signature.

> The Databricks log-agent docs (updated 2026-06-30) state plainly: *"If using ResponsesAgent, you
> can skip [the Model Signature] section; MLflow automatically infers a valid signature."*

### Registering to Unity Catalog

Logging produces a model in the tracking store. **Registering** promotes it to Unity Catalog, where
UC governs who can read it and what resources it may access.

```python
import mlflow

mlflow.set_registry_uri("databricks-uc")   # default in MLflow 3, shown for clarity

registered = mlflow.register_model(
    model_uri=logged_agent.model_uri,
    name="prod.agents.docs_agent",          # three-level: catalog.schema.model
)
print(registered.version)                   # 1, 2, 3, ...
```

Two things are non-negotiable:

1. **The name is a three-level namespace:** `catalog.schema.model`. This is not optional in UC.
2. **The model version must have a signature.** `ResponsesAgent` gives you this automatically;
   otherwise the `input_example` inference or a manual signature is required.

**Permissions to create a registered model:** `CREATE MODEL` + `USE SCHEMA` + `USE CATALOG`. To add a
new version: model ownership or `CREATE MODEL VERSION`. (Registered models are a subtype of the
`FUNCTION` securable in UC, so grants use `GRANT ... ON FUNCTION`.)

### Aliases replace stages (the big one)

In the legacy Workspace Model Registry, you moved model versions through **stages**: `None` →
`Staging` → `Production` → `Archived`. **Unity Catalog does not support stages.** If an exam answer
says "transition the model to the Production stage," it is wrong for UC.

Instead, UC uses two orthogonal mechanisms:

- **Environment = the catalog.** `dev.agents.docs_agent`, `staging.agents.docs_agent`,
  `prod.agents.docs_agent` are separate registered models in separate catalogs.
- **Deployment status = an alias.** An alias is a mutable, named pointer to a specific version, e.g.
  `@champion`, `@production`, `@challenger`.

```python
from mlflow import MlflowClient
client = MlflowClient(registry_uri="databricks-uc")

# Point the "champion" alias at version 3
client.set_registered_model_alias("prod.agents.docs_agent", "champion", 3)

# Resolve and load by alias
loaded = mlflow.pyfunc.load_model("models:/prod.agents.docs_agent@champion")

# Or load a specific version
loaded_v3 = mlflow.pyfunc.load_model("models:/prod.agents.docs_agent/3")

# Remove an alias (the version itself is untouched)
client.delete_registered_model_alias("prod.agents.docs_agent", "champion")
```

Aliases decouple "which version is live" from your application code — flipping the `@production`
alias to a new version reroutes traffic without a code change, and rolling back is just pointing the
alias at the old version again.

## Deep Dive / Advanced Topics

### MLflow 3 vs MLflow 2 — what actually changed

The exam targets MLflow 3, and several renames trip people up:

| Concept | MLflow 2 | MLflow 3 |
|---|---|---|
| Model as an entity | A run artifact (`runs:/<run_id>/<artifact_path>`) | A **first-class `LoggedModel`** (`models:/<model_id>`) |
| Log parameter | `artifact_path="agent"` | **`name="agent"`** (`artifact_path` deprecated but works) |
| Run required to log? | Yes | **No** — you can log without `mlflow.start_run()` |
| Default registry | Workspace Model Registry (`databricks`) | **Unity Catalog** (`databricks-uc`) |
| GenAI evaluation | `mlflow.evaluate` | **`mlflow.genai.evaluate`** |
| Promotion lifecycle | Stages | Aliases + **deployment jobs** |

Other MLflow 3 facts worth knowing:
- The **`LoggedModel`** entity tracks metrics, params, and traces across training → eval and
  dev → staging → prod, surfaced on the UC model-version page.
- **Tracing** is first-class and OpenTelemetry-compatible, with auto-instrumentation for 20+
  frameworks (LangChain/LangGraph, OpenAI, Anthropic, LlamaIndex).
- **Forward-compat is one-directional:** an MLflow 3 client can read 2.x artifacts, but a 2.x client
  generally **cannot** read 3.x models/traces.

### `ResponsesAgent` vs `ChatModel` vs `PythonModel`

MLflow offers several pyfunc interfaces. Choosing the right one matters for what features you unlock:

| Interface | Use for | Signature |
|---|---|---|
| `PythonModel` | General-purpose; no agent features | Must provide `input_example` or manual signature |
| `ChatModel` | Simple stateless single-turn chat | ChatCompletion schema |
| `ResponsesAgent` | **Full agents:** tool calls, streaming, multi-turn, tracing | Auto-inferred |

`ResponsesAgent` is the current recommendation — it supersedes the older `ChatAgent` and supports
multiple output messages (including intermediate tool-calling steps), streaming, and OpenAI Responses
API compatibility. Its methods are:

```python
def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse: ...
def predict_stream(self, request: ResponsesAgentRequest) -> Generator[ResponsesAgentStreamEvent, None, None]: ...
```

Helper factories (`create_text_output_item`, `create_function_call_item`, `create_text_delta`, etc.)
build the response items so you don't hand-construct the schema.

### Validating before you deploy

Registering does not guarantee the model *loads*. Use `mlflow.models.predict()` to load the logged
model in a subprocess with its declared environment and run the `input_example` through it — this
catches missing dependencies or import errors *before* you burn 15 minutes on a failed deployment:

```python
mlflow.models.predict(
    model_uri=logged_agent.model_uri,
    input_data=input_example,
    env_manager="uv",   # rebuild the model's environment to validate deps
)
```

## Worked Examples & Practice

### End-to-end: log, validate, register, alias

You've built the `DocsAgent` from Section 03 and saved it as `agent.py`. Ship it through the full
packaging pipeline:

```python
import mlflow
from mlflow import MlflowClient
from mlflow.models.resources import (
    DatabricksServingEndpoint,
    DatabricksVectorSearchIndex,
)

mlflow.set_registry_uri("databricks-uc")
input_example = {"input": [{"role": "user", "content": "What is AI Search?"}]}

# 1. Log using models-from-code, declaring resources for auth passthrough
with mlflow.start_run(run_name="docs-agent-v1"):
    logged = mlflow.pyfunc.log_model(
        python_model="agent.py",
        name="agent",
        input_example=input_example,
        resources=[
            DatabricksServingEndpoint(endpoint_name="databricks-meta-llama-3-3-70b-instruct"),
            DatabricksVectorSearchIndex(index_name="prod.agents.databricks_docs_index"),
        ],
    )

# 2. Validate it actually loads in its own environment BEFORE registering
mlflow.models.predict(
    model_uri=logged.model_uri,
    input_data=input_example,
    env_manager="uv",
)

# 3. Register to Unity Catalog (three-level name)
registered = mlflow.register_model(
    model_uri=logged.model_uri,
    name="prod.agents.docs_agent",
)

# 4. Set the champion alias so downstream deploys/apps can reference @champion
client = MlflowClient(registry_uri="databricks-uc")
client.set_registered_model_alias("prod.agents.docs_agent", "champion", registered.version)

print(f"Registered version {registered.version}, aliased @champion")
```

**Failure mode to observe:** delete the `DatabricksVectorSearchIndex` line from `resources`, log,
register, and deploy (Module 02). The notebook test passes because the notebook runs as *you* and you
have access to the index. But once served, the endpoint has no credential for the index, and the
first query that triggers retrieval fails with a permission/auth error. The lesson: notebook success
does not prove deployment success — declared `resources` are what make retrieval work on the endpoint.

## Common Pitfalls & Misconceptions

- **Pitfall:** Passing the agent *object* to `python_model` (`python_model=DocsAgent()`) → **Why it happens:** it mirrors the old object-based logging habit → **Fix:** for models-from-code, pass the **file path string** (`python_model="agent.py"`). Passing an object either fails or silently pickles state you didn't intend to serialize.

- **Pitfall:** Forgetting to declare `resources` at log time → **Why it happens:** the agent works perfectly in the notebook, so it "looks done" → **Fix:** every serving endpoint, vector index, UC function, and Genie space the agent calls must be listed in `resources=[...]`. Otherwise auth passthrough has nothing to provision and the endpoint fails at runtime.

- **Pitfall:** Trying to "promote the model to the Production stage" in Unity Catalog → **Why it happens:** stages were the mechanism in the legacy Workspace Model Registry → **Fix:** UC has no stages. Use environment **catalogs** (dev/staging/prod) plus **aliases** (`@champion`, `@production`) to express deployment status.

- **Pitfall:** Using a two-level or one-level model name when registering to UC → **Why it happens:** habits from the legacy registry where names were flat → **Fix:** UC requires a three-level `catalog.schema.model` name (or a bare model name only if the workspace default catalog is a UC catalog).

- **Pitfall:** Hardcoding a PAT or API key inside `agent.py` → **Why it happens:** it's the quickest way to make an external call work → **Fix:** models-from-code stores your file **in plain text** in the model artifact. Read secrets from environment variables / Databricks Secrets, and prefer resource-based auth passthrough over embedded credentials.

## Key Definitions

| Term | Definition |
|---|---|
| Models-from-code | The MLflow pattern of logging a model by its source **file path** (`python_model="agent.py"`) rather than serializing an object; MLflow re-runs the file to load the model |
| `mlflow.models.set_model()` | The call inside the model file that designates which object MLflow should treat as the servable model |
| `resources` | The list of Databricks resources (serving endpoints, vector indexes, UC functions, Genie spaces) declared at log time to enable automatic authentication passthrough on the serving endpoint |
| Authentication passthrough | Databricks provisioning short-lived, least-privilege credentials to a deployed endpoint for exactly the resources declared at log time |
| `LoggedModel` | MLflow 3's first-class model entity, referenced by `models:/<model_id>`, tracking metrics/params/traces across the lifecycle |
| Three-level namespace | The Unity Catalog naming scheme `catalog.schema.model` required for registered models |
| Alias | A mutable, named pointer to a specific registered model version (e.g., `@champion`, `@production`); replaces the deprecated stage concept |
| `ResponsesAgent` | The recommended MLflow pyfunc interface for agents; auto-infers a signature and supports tool calls, streaming, and multi-turn output |

## Summary / Quick Recall

- Agents are logged with **models-from-code**: `python_model="agent.py"` is a **file path**, not an object; the file calls `mlflow.models.set_model(agent)`
- Declare every endpoint/index/function the agent uses in **`resources=[...]`** for **auth passthrough** — the top cause of "works locally, fails on the endpoint"
- `ResponsesAgent` **auto-infers the signature**; a plain `PythonModel` needs `input_example` or a manual signature
- Register to UC with a **three-level name** `catalog.schema.model` via `mlflow.register_model`
- **UC has no stages** — use environment **catalogs** + **aliases** (`@champion`, `@production`); load with `models:/name@alias`
- MLflow 3: **`name=`** replaces `artifact_path=`, model URI is **`models:/<model_id>`**, registry defaults to **`databricks-uc`**, no run required to log
- Validate with `mlflow.models.predict()` before deploying to catch dependency/load failures early

## Self-Check Questions

1. Your teammate logs a LangGraph agent with `mlflow.pyfunc.log_model(python_model=my_agent_instance, name="agent")` and it fails or behaves oddly when served. What's wrong?

<details>
<summary>Answer</summary>
They passed the agent *object* instead of the file path. The models-from-code pattern requires `python_model="agent.py"` — a path to the file that defines the agent and calls `mlflow.models.set_model()`. Agents have no meaningful weights to serialize, so MLflow needs the source file to reconstruct the agent at serving time, not a pickled instance.
</details>

2. An agent works flawlessly in the notebook but every query that triggers retrieval fails once the agent is deployed to a serving endpoint. What is the most likely cause and fix?

<details>
<summary>Answer</summary>
The AI Search index (and/or the LLM serving endpoint) was not declared in `resources` at log time, so authentication passthrough provisioned no credential for it. In the notebook the code ran as the user, who had access; on the endpoint there is no such credential. Fix: add `DatabricksVectorSearchIndex(index_name=...)` (and `DatabricksServingEndpoint(...)`) to the `resources=[...]` list, then re-log, re-register, and re-deploy.
</details>

3. A colleague says, "I'll promote the new agent version by moving it to the Production stage in Unity Catalog." Why is this incorrect, and what should they do instead?

<details>
<summary>Answer</summary>
Unity Catalog does not support model stages — stages were a Workspace Model Registry concept. Deployment status in UC is expressed with **aliases**. They should set an alias (`client.set_registered_model_alias("prod.agents.docs_agent", "production", <version>)`) and load via `models:/prod.agents.docs_agent@production`. Environment separation (dev/staging/prod) is handled by separate **catalogs**, not stages.
</details>

4. Why does a `ResponsesAgent` not require you to provide an `input_example` or signature, while a plain `PythonModel` does — and why does the signature matter at all?

<details>
<summary>Answer</summary>
`ResponsesAgent` has standardized request/response schemas (`ResponsesAgentRequest`/`ResponsesAgentResponse`), so MLflow can auto-infer the signature and even supplies a default input example. A plain `PythonModel` has arbitrary I/O, so MLflow needs an `input_example` (to infer) or a manual signature. The signature matters because Model Serving refuses to serve a model version with no signature — it needs to know the expected input/output shape to validate requests.
</details>

5. You logged an agent and registered it, but you want to catch dependency errors before spending 15 minutes on a deployment that might fail. What do you do, and why isn't a successful notebook run enough?

<details>
<summary>Answer</summary>
Run `mlflow.models.predict(model_uri=..., input_data=input_example, env_manager="uv")`. This loads the logged model in a fresh subprocess using the *captured* environment (inferred pip requirements), which is what the serving endpoint will use — not your notebook's live environment. A notebook run succeeds using whatever packages happen to be installed in the session, so it can hide missing or mis-inferred dependencies that only surface when the model is loaded from its own recorded environment.
</details>

## Further Reading

- [Databricks — Log and register AI agents](https://docs.databricks.com/aws/en/agents/agent-framework/log-agent) — models-from-code logging, `resources`, signature inference, UC registration (updated 2026-06-30)
- [MLflow — Models From Code](https://mlflow.org/docs/latest/ml/model/models-from-code/) — the file-path logging pattern and `set_model()`
- [MLflow — ResponsesAgent introduction](https://mlflow.org/docs/latest/genai/flavors/responses-agent-intro/) — interface spec, request/response types, helper factories
- [Databricks — Manage model lifecycle in Unity Catalog](https://docs.databricks.com/aws/en/machine-learning/manage-model-lifecycle/) — three-level names, aliases vs stages, permissions (updated 2026-06-23)
- [Databricks — Get started with MLflow 3](https://docs.databricks.com/aws/en/mlflow/mlflow-3-install) — `LoggedModel`, `name=` vs `artifact_path=`, `models:/<id>` URIs (updated 2026-06-23)
