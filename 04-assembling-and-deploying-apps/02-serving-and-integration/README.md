# Serving and Integration

**Part of section:** Assembling and Deploying Apps
**Estimated time:** 5 hours
**Prerequisites:** Module 01 (Packaging and Registration) — a registered UC agent model and an AI Search index
**Exam mapping:** Assembling and Deploying Applications domain (~20%) — Model Serving, Foundation Model APIs, MCP, CI/CD, and Databricks Apps

## Overview

With the agent packaged and the index built, this module takes it live. You will deploy the agent
to a serverless Model Serving endpoint, understand the three Foundation Model API access modes,
query endpoints in every supported way, connect agents to tools via MCP, version prompts, wire up
CI/CD with Declarative Automation Bundles, and surface the agent through a Databricks Apps chat UI.

## Learning Outcomes

By completing this module, you will be able to:

- Deploy a registered agent with `agents.deploy()` and configure the serving endpoint
- Distinguish pay-per-token, provisioned throughput, and external model serving modes
- Query serving endpoints via OpenAI client, MLflow Deployments SDK, REST, and `ai_query()`
- Connect an agent to managed MCP servers and explain why Databricks adopted MCP
- Version prompts in the MLflow Prompt Registry and promote code through a CI/CD pipeline with bundles
- Deploy a chat UI for the agent using Databricks Apps

## Topics Covered

- Model Serving endpoints; `agents.deploy()` vs `mlflow.deployments`/WorkspaceClient/REST/UI
- Endpoint config: scale-to-zero, workload size, provisioned concurrency, CPU/GPU, traffic splitting
- Foundation Model APIs: pay-per-token, provisioned throughput, external models; current endpoint names
- Querying endpoints and `ai_query()` for batch inference
- MCP: managed servers (AI Search, UC functions, Genie, SQL), custom servers, URL patterns, auth modes
- MLflow Prompt Registry: versions, aliases, `{{variable}}` templates, UC governance
- Declarative Automation Bundles: `databricks.yml`, targets, deploy/run, deployment jobs, deploy-code pattern
- Databricks Apps: chat UI templates, `app.yaml`, binding serving endpoints, service-principal vs user auth

## How This Module Fits

This module consumes the artifacts from Module 01 (the registered agent and the index) and produces
a running, user-facing application. It also introduces the operational glue — prompt versioning,
CI/CD, and app hosting — that Section 05 (Governance, Evaluation, and Monitoring) builds on to add
tracing, evaluation gates, and the Unity AI Gateway.

## Study Tips

- **`agents.deploy()` vs the plain-model path** is a recurring exam distinction — know what the agent
  path adds (Review App, tracing, inference tables, auth passthrough).
- Memorize the **three Foundation Model API modes** and when to use each: pay-per-token (quick start,
  shared), provisioned throughput (production, guarantees, fine-tuned), external (governed third-party proxy).
- For MCP, know the **three server types and their URL patterns** and that AI Search's MCP server is
  the current recommended retrieval path.
- For CI/CD, remember **deploy-code is the recommended pattern**, environments map to **UC catalogs**,
  and promotion flips **model aliases**. Prompt versions are immutable; aliases are mutable.
- For Apps, remember `bundle deploy` uploads code but **does not restart the app** — you must
  `bundle run` and poll for `RUNNING`.

## Chapters

- [ ] [01 Model Serving, Foundation Model APIs, and MCP](./01-model-serving-foundation-model-apis-and-mcp/notes.md) — 2.5 hrs
- [ ] [02 Prompt Registry, CI/CD, and Databricks Apps UIs](./02-prompt-registry-cicd-and-databricks-apps-uis/notes.md) — 2.5 hrs
