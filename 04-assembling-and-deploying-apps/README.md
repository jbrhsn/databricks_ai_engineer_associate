# Section 04 — Assembling and Deploying Apps

**Estimated time:** 10 hours
**Prerequisites:** Section 02 (Data Preparation) — Delta tables, embeddings, AI Search concepts; Section 03 (Application Development) — LangGraph agents, `ResponsesAgent`, `ChatDatabricks`
**Exam mapping:** Databricks GenAI Engineer Associate — Assembling and Deploying Applications domain (~20% of exam)

## Overview

Section 03 built the agent; this section ships it. You will learn how to **package** a LangGraph
agent as an MLflow model (using the models-from-code pattern), **register** it to Unity Catalog,
build and configure the **AI Search (Vector Search)** index it retrieves from, **serve** it as a
real-time endpoint, and **operationalize** the whole thing with prompt versioning, CI/CD via
Declarative Automation Bundles, and a Databricks Apps chat UI.

Key Databricks-specific context from live docs (2026-06 to 2026-07):
- **MLflow 3** is the current version. Models are logged with the **models-from-code** approach
  (`python_model="agent.py"` — a file path, not a pickled object), and registered to Unity Catalog.
  **Stages are gone** — use environment catalogs + **aliases** (`@champion`, `@production`).
- **Model Serving** deploys registered agents as serverless REST endpoints. For agents specifically,
  use **`agents.deploy()`** from `databricks-agents` (bundles auth passthrough, Review App, tracing,
  inference tables) rather than the plain-model deployment path.
- **Foundation Model APIs** offer three access modes: **pay-per-token**, **provisioned throughput**,
  and **external models** (a governed proxy to OpenAI/Anthropic/etc.).
- **MCP (Model Context Protocol)** is Databricks' standard interface for connecting agents to tools —
  managed MCP servers exist for AI Search, Unity Catalog functions, Genie, and SQL.
- **Product renames to know (2026):** "Databricks Vector Search" is now **"Databricks AI Search"**;
  "Databricks Asset Bundles" are now **"Declarative Automation Bundles"** (same `databricks.yml`,
  same `databricks bundle` CLI). The exam may use either name.

## Learning Outcomes

By completing this section, you will be able to:

- Log a LangGraph agent as an MLflow 3 model using the models-from-code pattern
- Declare `resources` at log time so a served agent gets automatic authentication passthrough
- Register a model to Unity Catalog's three-level namespace and manage aliases instead of stages
- Create AI Search endpoints and Delta Sync / Direct Access indexes with the correct sync mode
- Choose STANDARD vs STORAGE_OPTIMIZED endpoints and TRIGGERED vs CONTINUOUS sync for a workload
- Deploy an agent with `agents.deploy()` and configure scale-to-zero, workload size, and concurrency
- Distinguish pay-per-token, provisioned throughput, and external model serving modes
- Query serving endpoints via the OpenAI client, MLflow Deployments SDK, REST, and `ai_query()`
- Explain MCP and connect an agent to a managed MCP server
- Version prompts in the MLflow Prompt Registry and reference them by alias
- Build a CI/CD pipeline for a GenAI app using Declarative Automation Bundles and deployment jobs
- Deploy a chat UI for the agent using Databricks Apps

## Topics Covered

**Module 01 — Packaging and Registration**
- MLflow 3 models-from-code logging (`python_model="agent.py"`, `code_paths`, `resources`)
- `mlflow.pyfunc.ResponsesAgent` signature auto-inference
- Unity Catalog registration: three-level names, aliases vs deprecated stages, permissions
- MLflow 3 vs MLflow 2: `LoggedModel`, `name=` vs `artifact_path=`, `models:/<id>` URIs
- AI Search endpoints (STANDARD vs STORAGE_OPTIMIZED) and indexes (Delta Sync vs Direct Access)
- Managed vs self-managed embeddings; TRIGGERED vs CONTINUOUS sync; Change Data Feed prereq
- Querying: ANN / full-text / hybrid (RRF), filters, reranking, embedding-model constraints

**Module 02 — Serving and Integration**
- Model Serving: serverless endpoints, `agents.deploy()` vs plain-model deployment
- Endpoint config: scale-to-zero, workload size, provisioned concurrency, CPU/GPU, traffic splitting
- Foundation Model APIs: pay-per-token, provisioned throughput, external models
- Querying endpoints: OpenAI client, MLflow Deployments SDK, REST, `ai_query()` SQL
- MCP: managed servers (AI Search, UC functions, Genie, SQL), custom servers, URL patterns
- MLflow Prompt Registry: versions, aliases, `{{variable}}` templates, UC governance
- CI/CD: Declarative Automation Bundles, targets, deployment jobs, deploy-code vs deploy-model
- Databricks Apps: chat UI templates, `app.yaml`, binding serving endpoints, authentication

## How This Section Fits

Section 02 built the data pipeline and vector index; Section 03 built the agent that queries it.
This section is the bridge from a working notebook prototype to a governed, monitored production
service. It sets up Section 05 (Governance, Evaluation, and Monitoring), which instruments the
deployed endpoints with tracing, evaluation, and the Unity AI Gateway you enable here.

## Study Tips

- **Know the deployment stack top to bottom:** log (models-from-code) → register (UC, three-level
  name) → deploy (`agents.deploy()`) → serve (endpoint) → surface (Databricks Apps). The exam loves
  questions that test whether you know which step a given API belongs to.
- **`resources=[...]` at log time is the single most tested "gotcha."** If you forget to declare the
  serving endpoints and vector search indexes your agent uses, the deployed endpoint fails auth at
  runtime. Memorize `DatabricksServingEndpoint` and `DatabricksVectorSearchIndex`.
- **Stages are gone in Unity Catalog.** If an answer says "promote the model to the Production
  stage," it is wrong for UC. The correct mechanisms are environment catalogs + aliases.
- **TRIGGERED vs CONTINUOUS and STANDARD vs STORAGE_OPTIMIZED** are frequent index questions. Know
  that storage-optimized endpoints support TRIGGERED sync only.
- **`agents.deploy()` is agent-specific.** Plain custom models use `mlflow.deployments` / the
  WorkspaceClient / REST / UI. The agent path adds the Review App, tracing, and inference tables.
- **Watch the renames.** AI Search (formerly Vector Search) and Declarative Automation Bundles
  (formerly Asset Bundles) may appear under either name — the underlying APIs still say
  `vector-search` and `databricks bundle`.

## Chapters / Modules

- [ ] [01 Packaging and Registration](./01-packaging-and-registration/) — 5 hrs
- [ ] [02 Serving and Integration](./02-serving-and-integration/) — 5 hrs
