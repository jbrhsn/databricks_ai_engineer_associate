# Module 01 — Chains, Agents, and Frameworks

**Part of section:** 03 Application Development
**Estimated time:** 7 hours
**Prerequisites:** Section 01 (prompt engineering, model selection); Section 02 (AI Search index built and queryable)
**Exam mapping:** Application Development domain (~30%); LangGraph patterns, RAG chains, tool-calling agents, and multi-agent design are all directly tested

## Overview

This module builds the core of a production LLM application: starting from a simple
LangGraph RAG chain, adding persistence and multi-turn memory, extending to tool-calling
agents, building multi-agent systems, and finally deploying via Databricks-native tooling.

The primary framework throughout is **LangGraph** (updated 2026) — the low-level
orchestration runtime Databricks explicitly recommends for custom agents. LangGraph is
framework-agnostic at the LLM layer; it uses `ChatDatabricks` (from `databricks-langchain`)
to call Databricks Foundation Model API endpoints.

Key architectural insight: LangGraph is an orchestration runtime, not an LLM library. Its
job is to manage graph state, route execution, persist checkpoints, and handle interrupts.
The LLM calls happen inside nodes.

## Learning Outcomes

By completing this module, you will be able to:

- Define a `TypedDict` state and build a `StateGraph` with nodes and edges
- Build a complete RAG chain in LangGraph: retrieve → augment → generate
- Add multi-turn memory to a graph using LangGraph checkpointers and `thread_id`
- Use LangGraph `interrupt()` for human-in-the-loop approval workflows
- Define Python tools with `@tool` and build a ReAct agent with `create_react_agent`
- Add conditional edges to route execution based on tool calls or other logic
- Design a multi-agent supervisor pattern where a router dispatches to specialist agents
- Use subgraphs to encapsulate reusable agent logic
- Create a Knowledge Assistant (no-code) and retrieve its endpoint
- Wrap a LangGraph agent with `mlflow.pyfunc.ResponsesAgent` for Databricks deployment

## Topics Covered

- LangGraph core primitives: `StateGraph`, `TypedDict` state, nodes, edges, `START`/`END`
- Conditional edges: `add_conditional_edges` for routing
- Compilation: `builder.compile(checkpointer=...)` and invocation
- RAG graph: retriever node → prompt-building node → generation node
- LangGraph persistence: `InMemorySaver` (dev) vs. `PostgresSaver` (prod); `thread_id` config
- `interrupt()` function: pausing execution, resuming with `Command(resume=...)`
- Tool definition: `@tool` decorator, docstring as tool description
- `create_react_agent` prebuilt: fast ReAct agent from model + tools list
- Custom ReAct loop in `StateGraph`: agent node → tool node → conditional routing
- Multi-agent supervisor: `StateGraph` router node dispatching to subgraph agents
- Databricks-native agents: Knowledge Assistant (Agent Bricks), Supervisor Agent
- `mlflow.pyfunc.ResponsesAgent` interface for Databricks deployment compatibility
- MCP (Model Context Protocol): overview and when to use it

## How This Module Fits

Chapter 01 covers LangGraph fundamentals + RAG chains. Chapter 02 builds on that with
tool-calling agents and multi-agent systems. Chapter 03 covers Databricks-native deployment
options (Knowledge Assistant, Supervisor Agent, ResponsesAgent). Chapter 03's content bridges
directly to Section 04 (Deploying), which covers MLflow packaging and serving.

## Study Tips

- **Read the LangGraph concepts before looking at code.** The state → node → edge mental model
  is the foundation for everything else. Understand it before trying to memorize API calls.
- **`create_react_agent` vs. custom `StateGraph`:** `create_react_agent` is the fast path for
  simple tool-calling agents. Custom `StateGraph` is required whenever you need conditional
  routing, multi-agent patterns, human-in-the-loop, or custom state. Know both.
- **Checkpointer = memory.** Without a checkpointer, every `invoke` call starts fresh — there
  is no conversation history. With a checkpointer + `thread_id`, the graph resumes from where
  it left off. This is the entire memory mechanism in LangGraph.
- **`interrupt()` re-runs the node from the beginning on resume.** Any code before the
  `interrupt()` call runs again. Keep pre-interrupt code idempotent. This is a common exam trap.
- **`ResponsesAgent` is required for Databricks deployment.** Databricks Playground, Agent
  Evaluation, and Model Serving all expect the `ResponsesAgent` interface. It is the deployment
  wrapper, not the agent itself.

## Chapters

- [ ] [01 Chains, LangGraph Patterns, and Memory](./01-chains-langgraph-patterns-and-memory/notes.md) — 2.5 hrs
- [ ] [02 Agents, Tools, and Multi-Agent Design](./02-agents-tools-and-multi-agent-design/notes.md) — 2.5 hrs
- [ ] [03 Agent Bricks and Databricks-Native Agents](./03-agent-bricks-and-databricks-native-agents/notes.md) — 2 hrs
