# Section 03 — Application Development

**Estimated time:** 12 hours
**Prerequisites:** Section 01 (Foundations) — LLM core concepts, prompt engineering; Section 02 (Data Preparation) — AI Search index already built and queryable
**Exam mapping:** Databricks GenAI Engineer Associate — Application Development domain (~30% of exam — the highest-weighted domain)

## Overview

This is the largest exam domain and the core engineering section of the repo. You will build
complete LLM-powered applications: RAG chains, tool-calling agents, multi-agent systems, and
Databricks-native agent deployments. The primary orchestration framework throughout this section
is **LangGraph** — a stateful, graph-based agent runtime that Databricks explicitly supports and
recommends for custom agent development (as of 2026-07-01 docs).

Key Databricks-specific context from live docs (updated 2026-07-01):
- **Custom agents** can be written with any framework: LangGraph, OpenAI Agents SDK, LlamaIndex
- The recommended production interface is **`mlflow.pyfunc.ResponsesAgent`** — wrap any agent in it to get automatic MLflow tracing, Playground compatibility, and deployment support
- **Knowledge Assistant** (Agent Bricks) is the no-code/low-code RAG agent builder for domain-specific Q&A
- **Unity AI Gateway** (Beta) governs LLM traffic with guardrails, rate limits, spend caps, and PII policies
- **MCP (Model Context Protocol)** is how Databricks agents connect to external tools and data sources

## Learning Outcomes

By completing this section, you will be able to:

- Build a RAG chain using LangGraph nodes and edges that queries a Databricks AI Search index
- Manage multi-turn conversation memory using LangGraph's checkpointer persistence
- Define and register tools (Unity Catalog functions, Python tools, MCP servers) for agent use
- Build a ReAct-style tool-calling agent with `create_react_agent` and LangGraph `StateGraph`
- Design a multi-agent system with a supervisor node routing to specialized sub-agents
- Implement human-in-the-loop approval workflows using LangGraph `interrupt()`
- Create a Knowledge Assistant agent (no-code) and expose it as an endpoint
- Wrap any agent with `mlflow.pyfunc.ResponsesAgent` for Databricks deployment compatibility
- Implement input and output guardrails using Unity AI Gateway service policies
- Explain the role of `databricks-langchain` (`ChatDatabricks`) in connecting LangGraph to Databricks Foundation Model APIs

## Topics Covered

**Module 01 — Chains, Agents, and Frameworks**
- LangGraph fundamentals: StateGraph, nodes, edges, conditional edges, TypedDict state
- LangGraph persistence: checkpointers (InMemorySaver, PostgresSaver), thread IDs
- Building a RAG chain in LangGraph: retrieve → augment → generate
- LangGraph `interrupt()` for human-in-the-loop workflows
- Tool definition: `@tool` decorator, Unity Catalog functions, MCP servers
- `create_react_agent` prebuilt for quick tool-calling agents
- Multi-agent design: supervisor patterns, subgraphs
- Databricks-native agents: Knowledge Assistant, Supervisor Agent
- `mlflow.pyfunc.ResponsesAgent` wrapper for Databricks compatibility
- OpenAI Agents SDK (alternative to LangGraph — brief overview)

**Module 02 — Embeddings, Guardrails, and Model Selection**
- Embedding models on Databricks: `databricks-bge-large-en`, `databricks-gte-large-en`, `databricks-qwen3-embedding-0-6b`
- `DatabricksEmbeddings` integration with LangGraph retrieval nodes
- Input guardrails: PII detection, prompt injection defense, toxicity filtering
- Output guardrails: schema validation, content moderation, fallback responses
- Unity AI Gateway service policies for guardrails (2026 Beta)
- Model selection for application roles: retriever, generator, router, evaluator

## How This Section Fits

Section 02 built the data pipeline and vector index. This section builds the application that
queries it. Section 04 (Assembling and Deploying) covers packaging your LangGraph agent with
MLflow PyFunc, registering in Unity Catalog, and serving it via Model Serving endpoints. Section
05 (Governance) covers evaluation, monitoring, and MLflow tracing — all of which instrument the
agents built here.

## Study Tips

- **LangGraph is the exam-expected orchestration framework** — Databricks docs explicitly show
  LangGraph examples for custom agents. Know `StateGraph`, `add_node`, `add_edge`,
  `add_conditional_edges`, `compile`, and `invoke`. Know `create_react_agent` for quick ReAct agents.
- **`ResponsesAgent` is the deployment wrapper** — any agent written in any framework is wrapped
  with `ResponsesAgent` to integrate with Databricks Playground, tracing, and serving.
- **Knowledge Assistant is the no-code path** — for simple domain Q&A, it is the recommended
  starting point; for complex agent logic, use custom LangGraph agents.
- **Unity AI Gateway is in Beta (2026)** — know what it does (guardrails, rate limits, spend
  caps) even if the exam tests it lightly; it is the direction Databricks is heading for governance.
- **MCP is increasingly important** — Databricks uses MCP as the standard interface for agent
  tool connections. Know what it is and why it's used.
- **The `ChatDatabricks` class** from `databricks-langchain` is how you connect LangGraph nodes
  to Databricks Foundation Model API endpoints. This is the Databricks-specific LLM integration.

## Chapters / Modules

- [ ] [01 Chains, Agents, and Frameworks](./01-chains-agents-and-frameworks/) — 7 hrs
- [ ] [02 Embeddings, Guardrails, and Model Selection](./02-embeddings-guardrails-and-model-selection/) — 5 hrs
