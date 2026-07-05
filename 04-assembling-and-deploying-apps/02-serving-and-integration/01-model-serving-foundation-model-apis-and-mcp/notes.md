# Model Serving, Foundation Model APIs, and MCP

**Section:** Assembling and Deploying Apps | **Module:** Serving and Integration | **Est. time:** 2.5 hrs
**Exam mapping:** Databricks GenAI Engineer Associate — Assembling and Deploying Applications domain (~20%); deploying agents, Foundation Model API modes, querying endpoints, and MCP are directly tested

## Learning Objectives

By the end of this chapter, you will be able to:

- Deploy a registered agent with `agents.deploy()` and explain what it automatically provides
- Distinguish the agent deployment path from the plain-model deployment path
- Configure a serving endpoint: scale-to-zero, workload size, provisioned concurrency, CPU/GPU, traffic splitting
- Distinguish the three Foundation Model API access modes and choose one for a scenario
- Query a serving endpoint via the OpenAI client, MLflow Deployments SDK, REST, and `ai_query()`
- Explain what MCP is, why Databricks adopted it, and connect an agent to a managed MCP server

## Core Concepts

### Model Serving in one sentence

**Model Serving** is Databricks' unified, serverless way to deploy any model or agent as an
auto-scaling REST endpoint. Each served entity becomes an HTTP API; the platform handles scaling,
load balancing, and a unified query interface (REST, the MLflow Deployment API, and the `ai_query()`
SQL function). It serves three kinds of things: **custom models**, **agents** (a special case of
custom model), and **foundation models** (Databricks-hosted or external).

### Deploying an agent: `agents.deploy()`

For an *agent* specifically, you don't hand-build the endpoint config. You use **`agents.deploy()`**
from the `databricks-agents` SDK, passing the UC model name and version:

```python
from databricks import agents

deployment = agents.deploy("prod.agents.docs_agent", version=3)
print(deployment.query_endpoint)   # the URL to query the agent
```

This one call does a lot for you automatically:

- **Creates the serving endpoint** with autoscaling and load balancing
- **Provisions short-lived auth credentials** for the resources you declared at log time
  (the `resources=[...]` from the previous module) — this is where auth passthrough pays off
- **Enables the Review App** — a web UI where stakeholders chat with the agent and leave feedback
- **Enables real-time MLflow tracing** of every request
- **Creates inference tables** (via AI Gateway) that log requests and responses for monitoring

You can pass extra kwargs, e.g. `scale_to_zero_enabled=True` or a custom `endpoint_name`.
Prerequisites: `mlflow>=3.1.3` and `databricks-agents>=1.1.0` (the MLflow 3 path); deploying from
outside a notebook requires `databricks-agents>=1.1.0`. Deployment can take **up to 15 minutes**.

**Zero-downtime updates:** call `agents.deploy()` again with the **same UC model name** but a new
version. Databricks adds the new version to the existing endpoint and shifts traffic once it's ready;
in-flight requests keep hitting the old version. Only change the *version* (not the endpoint or model
name) to guarantee zero downtime.

Management APIs: `agents.list_deployments()`, `agents.get_deployments(model_name=...)`,
`agents.delete_deployment(model_name=..., model_version=...)`.

> **2026 direction:** Databricks now *recommends deploying new agents on Databricks Apps* (covered in
> the next chapter) for fuller control over code and server config. `agents.deploy()` to Model Serving
> is still fully supported and remains the exam-relevant path; there is a migration guide between them.

### Deploying a plain model: the other path

A plain custom model (a scikit-learn/XGBoost/PyTorch model, or any non-agent pyfunc) does **not** use
`agents.deploy()`. It uses one of:

- **MLflow Deployments SDK:** `mlflow.deployments.get_deploy_client("databricks")` → `create_endpoint(...)`
- **Databricks WorkspaceClient SDK:** `WorkspaceClient().serving_endpoints.create(...)`
- **REST:** `POST /api/2.0/serving-endpoints`
- **The Serving UI**

The distinction the exam tests: the **agent path** (`agents.deploy()`) layers on the Review App,
tracing, inference tables, and auth passthrough scaffolding; the **plain-model path** just stands up
the endpoint.

### Endpoint configuration

Whichever path you use, endpoints expose the same knobs (via `served_entities`):

- **`scale_to_zero_enabled`** — when true, the endpoint scales to zero replicas when idle, saving cost
  but adding **cold-start latency** on the next request. **Not recommended for production** because
  capacity isn't guaranteed. Only *custom-model* endpoints can also be manually **stopped/started**.
- **`workload_size`** — request-concurrency band: `Small` (0–4 concurrent), `Medium` (8–16),
  `Large` (16–64). Rule of thumb: concurrency ≈ QPS × model run time.
- **`min_provisioned_concurrency` / `max_provisioned_concurrency`** — an alternative to `workload_size`
  for finer control; **must be multiples of 4**. On GPU endpoints, replicas = concurrency ÷ 4.
- **`workload_type`** — `CPU` (default), `CPU_MEDIUM/LARGE`, or GPU: `GPU_SMALL` (T4), `GPU_MEDIUM`
  (A10G), `MULTIGPU_MEDIUM` (4×A10G), etc.
- **Environment variables / secrets** — set in advanced config to reach non-Databricks resources.
- **Traffic splitting (A/B)** — put multiple served entities (e.g., two model versions) on one
  endpoint and route by percentage via `traffic_config.routes[]`; percentages must sum to 100.

```python
from mlflow.deployments import get_deploy_client
client = get_deploy_client("databricks")

client.create_endpoint(name="fraud-model-endpoint", config={
    "served_entities": [{
        "name": "fraud-v3",
        "entity_name": "prod.risk.fraud_model",   # UC three-level name
        "entity_version": "3",
        "workload_size": "Small",
        "scale_to_zero_enabled": False,
    }]
})
```

**Config gotchas:** endpoint names **cannot start with `databricks-`** (reserved for preconfigured FM
endpoints); the **creator identity is immutable** and used for UC access — Databricks recommends a
long-lived **service principal** as the creator, not a personal account.

### Foundation Model APIs: the three access modes

Databricks gives you three ways to consume LLMs, and choosing correctly is a classic exam item:

| Mode | What it is | When to use |
|---|---|---|
| **Pay-per-token** | Preconfigured `databricks-*` endpoints in every workspace; billed per token | Getting started, prototyping, low/variable throughput |
| **Provisioned throughput** | Dedicated capacity with performance/throughput guarantees; supports fine-tuned & custom weights; HIPAA-eligible | Production, high throughput, custom/fine-tuned models |
| **External models** | A governed proxy to third-party providers (OpenAI, Anthropic, Cohere, Bedrock, Vertex, etc.) | Centralizing/governing access to non-Databricks LLMs |

**Pay-per-token endpoint names** carry the `databricks-` prefix, e.g. (2026): chat models like
`databricks-meta-llama-3-3-70b-instruct`, `databricks-claude-sonnet-4-5`, `databricks-gpt-oss-120b`,
`databricks-gemini-2-5-flash`; embeddings `databricks-gte-large-en`, `databricks-bge-large-en`,
`databricks-qwen3-embedding-0-6b`. (Specific model names churn constantly — know the *categories*,
not a memorized list.)

**External models** are configured with an `external_model` block naming the `provider`, the `task`
(`llm/v1/chat`, `llm/v1/completions`, `llm/v1/embeddings`), and the provider API key (from Databricks
Secrets):

```python
client.create_endpoint(name="anthropic-chat", config={
    "served_entities": [{
        "name": "claude",
        "external_model": {
            "name": "claude-3-5-sonnet",
            "provider": "anthropic",
            "task": "llm/v1/chat",
            "anthropic_config": {
                "anthropic_api_key": "{{secrets/my_scope/anthropic_key}}"
            },
        },
    }]
})
```

### Querying a serving endpoint

The query API is **OpenAI-compatible** regardless of the underlying model. Four ways:

```python
# 1. OpenAI client (pip install databricks-openai) — model = endpoint name
from openai import OpenAI
client = OpenAI(api_key=TOKEN, base_url="https://<host>/serving-endpoints")
resp = client.chat.completions.create(
    model="databricks-meta-llama-3-3-70b-instruct",
    messages=[{"role": "user", "content": "What is AI Search?"}],
)

# 2. MLflow Deployments SDK
from mlflow.deployments import get_deploy_client
deploy_client = get_deploy_client("databricks")
resp = deploy_client.predict(
    endpoint="databricks-meta-llama-3-3-70b-instruct",
    inputs={"messages": [{"role": "user", "content": "What is AI Search?"}]},
)

# 3. REST — POST /serving-endpoints/{name}/invocations

# 4. ai_query() SQL — batch/real-time inference over a table
```

```sql
-- Batch inference across a whole column with ai_query()
SELECT ticket_id,
       ai_query('databricks-meta-llama-3-3-70b-instruct',
                'Classify the sentiment of: ' || ticket_text) AS sentiment
FROM support.tickets
LIMIT 100;
```

`ai_query()` requires **Databricks Runtime 18.2+** and **serverless** compute (or a serverless
notebook/workflow) — it does **not** run on Pro/Classic SQL warehouses. It's the tool of choice for
batch inference over Delta tables.

### MCP (Model Context Protocol)

**MCP is an open standard** for connecting AI agents to tools, data, and prompts through a *uniform
interface*. Instead of writing a bespoke integration for every tool, an agent speaks MCP to any MCP
server. Databricks adopted MCP so agents get one consistent way to reach Databricks capabilities, all
governed by Unity Catalog permissions and monitored through the Unity AI Gateway.

There are **three kinds of MCP server**, distinguished by URL pattern:

| Type | URL pattern |
|---|---|
| **Managed** (Databricks-hosted) | `https://<host>/api/2.0/mcp/<service>/<path>` |
| **External** (MCP Service via AI Gateway) | `https://<host>/ai-gateway/mcp-services/<catalog>.<schema>.<service>` |
| **Custom** (hosted as a Databricks App) | `https://<app-url>/mcp` |

**Managed MCP servers** (Public Preview) cover the main Databricks tool sources:

- **AI Search:** `/api/2.0/mcp/ai-search/{catalog}/{schema}/{index}` — retrieval over an index (the
  current recommended retrieval path for agents; the old `/mcp/vector-search/` prefix still works)
- **Unity Catalog functions:** `/api/2.0/mcp/functions/{catalog}/{schema}/{function}`
- **Genie space:** `/api/2.0/mcp/genie/{genie_space_id}` — natural-language SQL over a space
- **Databricks SQL:** `/api/2.0/mcp/sql` — AI-generated SQL

Agents connect using the `databricks-mcp` package (`DatabricksMCPClient`) or via framework
integrations (LangGraph's `DatabricksMCPServer`, the OpenAI Agents SDK's MCP support):

```python
from databricks_mcp import DatabricksMCPClient
from databricks.sdk import WorkspaceClient

ws = WorkspaceClient()
url = f"{ws.config.host}/api/2.0/mcp/functions/system/ai"
mcp = DatabricksMCPClient(server_url=url, workspace_client=ws)
mcp.list_tools()
mcp.call_tool("system__ai__python_exec", {"code": "print('hi')"})
```

**Custom MCP servers** are hosted as Databricks Apps. When you log an agent that uses MCP tools,
declare the needed resources — `DatabricksMCPClient(...).get_databricks_resources(server_url)` returns
what a managed server needs, and a custom server needs its `DatabricksApp` declared as a resource.

## Deep Dive / Advanced Topics

### Auth passthrough vs on-behalf-of-user, revisited at serving time

The `resources` you declared at log time (previous module) are what make **automatic authentication
passthrough** work: `agents.deploy()` provisions least-privilege, short-lived credentials for exactly
those resources, after verifying the endpoint owner's permissions. The alternative is **on-behalf-of-
user (OBO)** with `ModelServingUserCredentials()`, where the agent runs with the *calling user's*
permissions — necessary when different users must see different data (row-level security carried into
the agent). For MCP, OBO means the MCP tool call is authorized as the end user.

### Inference tables and the deprecated feedback model

`agents.deploy()` auto-creates **inference tables** (AI Gateway inference tables for agents) that log
every request/response — the raw material for the monitoring and evaluation you'll do in Section 05.
Note two deprecations to be aware of: the old **feedback model** is deprecated in favor of MLflow 3
`log_feedback` / trace labeling, and the old **request/assessment logs** are superseded by MLflow 3
real-time tracing. A subtle gotcha: if the agent notebook lives in a **Git folder**, real-time tracing
won't work by default — set a non-Git experiment via `mlflow.set_experiment()` before deploying.

### `ai_query()` power features

`ai_query()` is more than a single-string call. It supports:
- `returnType` and `failOnError=false` (returns a `STRUCT{response, errorMessage}` so one bad row
  doesn't kill a batch job)
- `modelParameters` (e.g. `named_struct('max_tokens', 100, 'temperature', 0.7)`)
- `responseFormat` for **structured outputs** (JSON schema / DDL, on DBR 15.4 LTS+)
- multimodal `files => content` for image inputs

This makes it a legitimate production tool for large-scale batch classification, extraction, and
summarization over Delta tables — not just a demo helper.

## Worked Examples & Practice

### Deploy the docs agent and smoke-test it

Continuing from the previous module, `prod.agents.docs_agent` version 3 is registered and aliased
`@champion`. Deploy and query it.

```python
from databricks import agents
from mlflow.deployments import get_deploy_client

# 1. Deploy the registered agent version (creates endpoint + Review App + tracing + inference tables)
deployment = agents.deploy("prod.agents.docs_agent", version=3, scale_to_zero_enabled=False)
print("Query URL:", deployment.query_endpoint)

# 2. Smoke-test through the MLflow Deployments SDK
client = get_deploy_client("databricks")
resp = client.predict(
    endpoint=deployment.endpoint_name,
    inputs={"input": [{"role": "user", "content": "How do I enable Change Data Feed?"}]},
)
print(resp)

# 3. Ship a new version later with zero downtime — same model name, new version
agents.deploy("prod.agents.docs_agent", version=4)   # traffic shifts once v4 is ready
```

**Failure mode to observe:** deploy with `scale_to_zero_enabled=True` for a low-traffic internal tool.
The first query after an idle period takes several seconds (cold start) while a replica spins up.
Users perceive the agent as "slow" or "broken" intermittently. For a latency-sensitive production
agent, disable scale-to-zero so a replica is always warm — trading cost for consistent latency.

## Common Pitfalls & Misconceptions

- **Pitfall:** Using `mlflow.deployments.create_endpoint()` (the plain-model path) to deploy an agent and then wondering where the Review App / tracing went → **Why it happens:** both are "deploying a model" → **Fix:** deploy agents with `agents.deploy()`. The agent path adds the Review App, real-time tracing, inference tables, and auth passthrough that the plain path doesn't set up.

- **Pitfall:** Leaving `scale_to_zero_enabled=True` on a latency-sensitive production endpoint → **Why it happens:** it saves money and "works" in testing → **Fix:** scale-to-zero adds cold-start latency and doesn't guarantee capacity; disable it for production agents that need consistent response times.

- **Pitfall:** Naming a custom endpoint with a `databricks-` prefix → **Why it happens:** it looks like a natural convention → **Fix:** the `databricks-` prefix is reserved for Databricks' preconfigured Foundation Model endpoints; choose a different name.

- **Pitfall:** Confusing pay-per-token with provisioned throughput for a high-volume production workload → **Why it happens:** pay-per-token is the easiest to start with → **Fix:** pay-per-token is great for prototyping and variable/low volume but has no throughput guarantees; production high-throughput or fine-tuned/custom models need **provisioned throughput**.

- **Pitfall:** Expecting `ai_query()` to run on a Pro or Classic SQL warehouse → **Why it happens:** it's a SQL function, so it feels like it should run anywhere SQL does → **Fix:** `ai_query()` requires DBR 18.2+ and serverless compute; it does not run on Pro/Classic SQL warehouses.

## Key Definitions

| Term | Definition |
|---|---|
| Model Serving | Databricks' serverless platform for deploying models and agents as auto-scaling REST endpoints |
| `agents.deploy()` | The `databricks-agents` SDK call that deploys a registered agent, auto-provisioning the endpoint, auth passthrough, Review App, tracing, and inference tables |
| Scale-to-zero | An endpoint setting that drops to zero replicas when idle to save cost, at the price of cold-start latency; discouraged for production |
| Workload size | The request-concurrency band of an endpoint (Small/Medium/Large) |
| Pay-per-token | A Foundation Model API mode using preconfigured `databricks-*` endpoints billed per token; for prototyping and low/variable volume |
| Provisioned throughput | A Foundation Model API mode with dedicated capacity and performance guarantees; for production, high throughput, and fine-tuned/custom models |
| External models | A Foundation Model API mode that proxies to third-party LLM providers under Databricks governance |
| `ai_query()` | A SQL function for batch/real-time inference against a serving endpoint; requires DBR 18.2+ and serverless |
| MCP (Model Context Protocol) | An open standard giving agents a uniform interface to tools/data; on Databricks, managed/external/custom servers governed by Unity Catalog and the AI Gateway |
| Inference tables | Auto-created tables logging endpoint requests/responses for monitoring and evaluation |

## Summary / Quick Recall

- **`agents.deploy()`** deploys agents and auto-adds the **Review App, tracing, inference tables, and auth passthrough**; plain models use `mlflow.deployments` / WorkspaceClient / REST / UI
- Auth passthrough works because you declared **`resources`** at log time — that's the payoff
- **Zero-downtime updates:** re-deploy the same model name with a new **version**
- **Scale-to-zero** saves cost but adds cold-start latency — avoid for latency-sensitive production
- **`workload_size`** (Small/Medium/Large) or **provisioned concurrency** (multiples of 4) sizes an endpoint; **traffic splitting** enables A/B across versions
- Three FM API modes: **pay-per-token** (prototype/low volume), **provisioned throughput** (production/fine-tuned), **external models** (governed third-party proxy)
- Endpoint queries are **OpenAI-compatible**: OpenAI client, MLflow Deployments SDK, REST `/invocations`, or **`ai_query()`** (DBR 18.2+, serverless) for batch
- **MCP** = uniform agent-to-tool interface; managed servers for **AI Search, UC functions, Genie, SQL** at `/api/2.0/mcp/...`; custom servers hosted as Databricks Apps
- Custom endpoint names can't start with **`databricks-`**; endpoint creator identity is immutable — use a service principal

## Self-Check Questions

1. You deployed an agent with `mlflow.deployments.get_deploy_client("databricks").create_endpoint(...)`. It serves fine, but there's no Review App and MLflow tracing isn't capturing requests. What happened?

<details>
<summary>Answer</summary>
You used the plain-model deployment path instead of the agent path. `agents.deploy()` (from `databricks-agents`) is what auto-enables the Review App, real-time tracing, and inference tables for an agent. Re-deploy the registered agent with `agents.deploy("catalog.schema.model", version=N)` to get the full agent scaffolding.
</details>

2. A team runs a customer-facing agent that must respond consistently in under a second, but they enabled scale-to-zero to cut costs. Users complain the agent is intermittently slow. Diagnose and fix.

<details>
<summary>Answer</summary>
Scale-to-zero drops the endpoint to zero replicas when idle; the first request after idle triggers a cold start of several seconds. For a latency-sensitive, customer-facing agent, disable scale-to-zero (`scale_to_zero_enabled=False`) so a replica stays warm. This trades some idle cost for consistent latency — the right trade-off for production UX.
</details>

3. When should you use provisioned throughput instead of pay-per-token Foundation Model APIs?

<details>
<summary>Answer</summary>
Use provisioned throughput for production workloads that need guaranteed throughput/latency, for high request volumes, or when serving fine-tuned or custom-weight models (and for HIPAA-eligible workloads). Pay-per-token is best for prototyping and low or variable volume — it's easy and shared but offers no performance guarantees.
</details>

4. You need to classify sentiment for 2 million support tickets stored in a Delta table, using an LLM, as a scheduled batch job. What's the most appropriate Databricks mechanism, and what compute requirement must you meet?

<details>
<summary>Answer</summary>
Use the `ai_query()` SQL function against a Foundation Model endpoint (e.g., `ai_query('databricks-meta-llama-3-3-70b-instruct', 'Classify sentiment: ' || ticket_text)`), optionally with `failOnError => false` so bad rows don't fail the whole job. It requires Databricks Runtime 18.2+ on serverless compute (a serverless notebook or workflow) — it will not run on Pro/Classic SQL warehouses.
</details>

5. What problem does MCP solve for agents on Databricks, and what are the three server types with their URL patterns?

<details>
<summary>Answer</summary>
MCP gives agents a single, uniform interface to reach many different tools/data sources instead of bespoke per-tool integrations, all governed by Unity Catalog and monitored via the AI Gateway. The three types: **managed** (`https://<host>/api/2.0/mcp/<service>/...` — e.g. AI Search, UC functions, Genie, SQL), **external MCP services** (`https://<host>/ai-gateway/mcp-services/<catalog>.<schema>.<service>`), and **custom** servers hosted as Databricks Apps (`https://<app-url>/mcp`).
</details>

## Further Reading

- [Databricks — Model Serving overview](https://docs.databricks.com/aws/en/machine-learning/model-serving/) — serverless serving of models and agents (updated 2026-06-30)
- [Databricks — Deploy an agent](https://docs.databricks.com/aws/en/agents/agent-framework/deploy-agent) — `agents.deploy()`, Review App, tracing, zero-downtime updates (updated 2026-07-01)
- [Databricks — Foundation Model APIs](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/) — pay-per-token, provisioned throughput, external models
- [Databricks — Foundation Model APIs supported models](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/supported-models) — current endpoint names (updated 2026-07-02)
- [Databricks — Model Context Protocol (MCP)](https://docs.databricks.com/aws/en/agents/mcp/) — managed/external/custom servers and URL patterns (updated 2026-06-30)
- [Databricks — `ai_query` function](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_query) — batch inference over Delta tables (updated 2026-06-30)
