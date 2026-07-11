# Multi-Agent Systems and Genie Agents

**Section:** 04 — Application Development | **Module:** 03 — Agents, MLflow & Genie | **Est. time:** 3 hrs | **Exam mapping:** Application Development (30%) — multi-agent orchestration, tool-calling, state management

> ⚠️ **Fast-evolving:** Genie Spaces was renamed to **Genie Agents** in July 2026. The Conversation API endpoint prefix remains `/api/2.0/genie/spaces/`. Always verify against [Databricks Genie Agents docs](https://docs.databricks.com/aws/en/genie/) before relying on UI details or product names.

---

## TL;DR

Multi-agent systems decompose a single complex task into specialised sub-agents, each with a narrow responsibility, orchestrated by a supervisor that routes work via conditional edges in a shared state graph. The Databricks Agent Framework is framework-agnostic — LangGraph, LangChain, LlamaIndex, and OpenAI agents all integrate natively. Genie Agents (formerly Genie Spaces) are Databricks' built-in natural-language analytics interface that translates business questions to SQL over Unity Catalog data, and they can be embedded as a "data analyst" sub-agent via the Conversation API.

**The one thing to remember: a supervisor node in LangGraph reads the shared state, decides which sub-agent should act next, and returns a `Command` object containing the `goto` target — this single pattern scales from two agents to twenty.**

---

## ELI5 — Explain It Like I'm 5

Imagine a busy restaurant kitchen. The head chef (supervisor) reads the ticket, then shouts "cold apps to the garde manger, hot entrées to the sauté cook." Each cook (sub-agent) handles only their station, returns the finished dish to the pass, and the head chef plates everything together. No cook has to know how every other station works — they just do their part and hand it off. The most important misconception to fix: people assume the supervisor must run everything sequentially, one agent waiting for the previous to completely finish. In reality the supervisor re-evaluates state after each sub-agent returns, so routing is dynamic — if the garde manger signals "out of arugula," the supervisor can reroute the ticket to a different station mid-flight. The state graph is the pass — every agent reads from it and writes back to it, so nothing is lost between handoffs.

---

## Learning Objectives

By the end of this chapter you will be able to:
- [ ] Explain the supervisor pattern in LangGraph and implement it using `Command` routing and `add_conditional_edges`
- [ ] Distinguish the supervisor topology from the agent-as-tool pattern and select the right one for a given requirement
- [ ] Design a shared `StateGraph` schema that allows multiple agents to exchange messages without coupling their internals
- [ ] Integrate a Genie Agent into a multi-agent system using the Genie Conversation API as a tool
- [ ] Compare LangGraph's graph-based multi-agent approach with CrewAI's role-based crew model

---

## Visual Overview

### Supervisor Multi-Agent Graph Topology

```
                        ┌──────────────────────┐
                        │    User / Caller     │
                        └──────────┬───────────┘
                                   │ invoke(state)
                                   ▼
                        ┌──────────────────────┐
                        │    Supervisor Node   │ ◄─────────────────────┐
                        │  (router LLM call)   │                       │
                        └──────────┬───────────┘                       │
                                   │ Command(goto=...)                  │
              ┌────────────────────┼────────────────────┐              │
              ▼                    ▼                    ▼              │
   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
   │ Retrieval Agent  │  │ Analytics Agent  │  │ Synthesis Agent  │   │
   │  (RAG / vector   │  │  (Genie Agent    │  │  (LLM summary +  │   │
   │   search tool)   │  │   Conversation   │  │   citation)      │   │
   └────────┬─────────┘  │   API tool)      │  └────────┬─────────┘   │
            │            └────────┬─────────┘           │             │
            │                    │                       │             │
            └────────────────────┴───────────────────────┘             │
                         Write to shared state                          │
                         messages channel ──────────────────────────────┘
                                                    loop until FINISH
```

### Agent-as-Tool Invocation Pattern

```
   Parent Agent
   ┌─────────────────────────────────────────────┐
   │  LLM decides to call tool: "search_agent"   │
   │  ┌───────────────────────────────────────┐  │
   │  │  tool_call: {name: "search_agent",    │  │
   │  │             args: {query: "..."}}     │  │
   │  └───────────────┬───────────────────────┘  │
   └─────────────────┬┘                          │
                     │  @tool invocation          │
                     ▼                            │
           ┌─────────────────────┐               │
           │   Sub-Agent         │               │
           │  (search_agent fn)  │               │
           │  runs its own graph │               │
           └────────┬────────────┘               │
                    │ returns str / dict          │
                    └────────────────────────────►│
                         tool result appended     │
                         to messages              │
```

### Genie Agent Architecture in a Multi-Agent System

```
   Supervisor
   ┌──────────────────────────────────────────────────┐
   │  "question needs structured data analysis"       │
   │  → route to genie_analytics_tool                 │
   └─────────────────┬────────────────────────────────┘
                     │ tool call
                     ▼
   ┌──────────────────────────────────────────────────┐
   │  genie_analytics_tool (@tool wrapper)            │
   │                                                  │
   │  POST /api/2.0/genie/spaces/{id}/start-conversation │
   │       {"content": natural_language_question}     │
   │                                                  │
   │  Poll GET ...messages/{msg_id} → status=COMPLETED│
   │                                                  │
   │  Extract SQL + result table from attachments     │
   └─────────────────┬────────────────────────────────┘
                     │ returns structured answer
                     ▼
   ┌──────────────────────────────────────────────────┐
   │  Synthesis Agent                                 │
   │  Combines retrieval results + Genie SQL results  │
   │  → final grounded answer with citations          │
   └──────────────────────────────────────────────────┘

   Unity Catalog governs all data access behind Genie Agent.
   Row-level security enforced per requesting user identity.
```

---

## Key Concepts

### Multi-Agent Architectures

**What it is:** A multi-agent system is a collection of two or more specialised AI agents that collaborate to solve a task that is too broad, too complex, or too varied for a single monolithic agent to handle reliably.

**How it works:** Each agent owns a bounded responsibility — retrieving documents, querying structured data, writing code, synthesising results. A shared state graph (in LangGraph, a `StateGraph`) serves as the communication medium: agents read from the state, perform their work, and write updated messages or intermediate results back. The orchestrating logic (supervisor or swarm-style handoff) determines which agent runs next based on the current state contents. Because agents are nodes in a graph, control can cycle back through a supervisor repeatedly until a terminal condition is met.

**Where it appears:** Databricks Agent Framework at `https://docs.databricks.com/aws/en/agents/` is framework-agnostic — LangGraph, LangChain, LlamaIndex, and OpenAI Agents SDK are all supported. The `ResponsesAgent` interface in MLflow wraps any multi-agent graph for deployment, evaluation, and tracing on Databricks.

---

### Supervisor Pattern in LangGraph

**What it is:** The supervisor pattern places a single "router" node at the centre of the graph. That node's only job is to read the current state and decide which sub-agent (or `FINISH`) should run next.

**How it works:** The supervisor invokes the LLM with structured output (e.g. a Pydantic model with a `next` field that must be one of the registered agent names or `"FINISH"`). The node then returns a `Command(goto=state["next"])` object — LangGraph's first-class mechanism for dynamic edge routing introduced after conditional edges. The graph calls `add_conditional_edges(supervisor_node, routing_fn, {"retrieval": retrieval_node, "analytics": analytics_node, FINISH: END})` to wire the possible transitions at build time. At runtime the supervisor's decision determines which edge is traversed; the chosen sub-agent writes its results to the shared messages channel and returns control to the supervisor for the next routing decision.

**Where it appears:** `StateGraph.add_conditional_edges()` and `Command` objects are defined in `langgraph.graph`. The pattern appears in the LangGraph multi-agent concepts documentation and is the primary recommended topology for Databricks-hosted multi-agent systems.

---

### Agent-as-Tool Pattern

**What it is:** The agent-as-tool pattern exposes an entire sub-agent as a callable `@tool` function that a parent agent can invoke through normal tool-calling mechanics — no shared state graph required.

**How it works:** The sub-agent (which may itself be a full LangGraph graph) is wrapped in a Python function decorated with `@tool`. That function's docstring becomes the tool description the parent LLM reads when deciding to call it. When the parent agent's tool executor calls the function, the sub-agent runs, and its final output (a string or dict) is returned as the tool result and appended to the parent's messages list. This pattern is simpler to wire up than a shared state graph but sacrifices shared intermediate state visibility — the parent only sees the final return value, not the sub-agent's reasoning steps.

**Where it appears:** `@tool` decorator from `langchain_core.tools`. Used inside LangGraph node functions via `ToolNode` or manual invocation. Databricks Agent Framework supports this pattern through the `agent-tool` documentation at `/aws/en/agents/agent-framework/agent-tool`.

---

### Shared State and Message Passing

**What it is:** The shared state is a typed Python `TypedDict` schema that all nodes in a LangGraph `StateGraph` read from and write to — it is the single source of truth for in-flight work.

**How it works:** Each key in the state schema has an associated reducer that governs how new values are merged with existing ones. For the `messages` channel, LangGraph provides the `add_messages` reducer: when a node returns `{"messages": [new_msg]}`, `add_messages` appends the new message to the existing list rather than replacing it. This gives all agents a shared, ordered conversation history. The state object is passed as the sole argument to every node function; nodes return a dict containing only the keys they wish to update. LangGraph applies all reducers after each node returns before passing the updated state to the next node.

**Where it appears:** `from langgraph.graph.message import add_messages` and `Annotated[list, add_messages]` in the state schema definition. The `StateGraph(AgentState)` constructor registers the state class at graph build time.

---

### Genie Agents (formerly Genie Spaces)

> ⚠️ **Fast-evolving:** Genie Spaces was renamed to Genie Agents as of July 8, 2026. The underlying Conversation API (endpoint prefix `/api/2.0/genie/spaces/`) is unchanged but UI names, billing, and feature surface are actively evolving. Verify current details at [https://docs.databricks.com/aws/en/genie/](https://docs.databricks.com/aws/en/genie/) before authoring production integrations.

**What it is:** A Genie Agent is a domain-specific natural-language chat interface in Databricks. Data practitioners configure it with Unity Catalog datasets, example SQL queries, business-semantic SQL expressions (measures, filters, join specs), and plain-text instructions. End users ask questions in natural language and receive generated SQL, result tables, and visualisations.

**How it works:** Genie uses a compound AI system — not a single LLM pass — to translate questions to SQL. It intelligently selects relevant table and column metadata, example SQL queries, and knowledge-store context (synonyms, measure definitions, join specs) to construct a prompt that generates a read-only SQL query. The query runs on a configured SQL warehouse; Unity Catalog row-level security and column masks are enforced per requesting user identity. For complex, multi-step questions, **Agent mode** (formerly Research Agent) breaks the question into hypotheses, runs multiple SQL queries, and synthesises a structured report with citations and visualisations. Agent mode is accessible only from the Databricks UI (not the Conversation API as of July 2026).

**Where it appears:** Created and managed in the Databricks UI under the Genie section. Programmatically integrated via the Genie Conversation API: `POST /api/2.0/genie/spaces/{space_id}/start-conversation` starts a conversation; `GET .../messages/{message_id}` polls for status and retrieves the generated SQL and result in the `attachments` field. The Management API (`POST /api/2.0/genie/spaces`) creates agents programmatically for CI/CD workflows.

---

### CrewAI Comparison

**What it is:** CrewAI is a Python framework that models multi-agent collaboration through "crews" of role-based agents. Each agent has a `role`, `goal`, and `backstory` (all plain text); a `Task` object defines what work is done and which agent does it. Crews can operate sequentially or hierarchically, where a manager LLM delegates tasks to worker agents.

**How it works:** CrewAI abstracts away the graph topology entirely — you define agents and tasks declaratively, and the framework handles the execution loop. There is no explicit state graph, message reducer, or conditional edge; inter-agent communication is mediated by a shared "kickoff" context that is appended to each task's result. This makes CrewAI faster to prototype but harder to reason about for complex routing or precise state management.

**Where it appears:** `from crewai import Agent, Task, Crew`. CrewAI is positioned in this repo as a breadth/comparison framework. For production Databricks deployments, LangGraph is preferred because its explicit state graph gives full control over routing, allows precise `interrupt_before` breakpoints for human-in-the-loop, and integrates natively with MLflow tracing.

| Dimension | LangGraph | CrewAI |
|---|---|---|
| State model | Explicit `TypedDict` + reducers | Implicit task context string |
| Routing | Conditional edges + `Command` | Sequential or LLM-managed hierarchy |
| Debugging | Per-node MLflow trace spans | Task-level logs |
| Production fit | High (precise control, HITL support) | Moderate (rapid prototyping) |
| Databricks native | `ResponsesAgent` integration | Via custom wrapper |

---

## Key Parameters / Configuration Knobs

| Parameter | What it controls | Decision rule |
|---|---|---|
| `add_messages` reducer | How new messages are merged into the `messages` state key | Always use `add_messages` (not `operator.add`) for the messages channel; `operator.add` duplicates earlier messages on every update because it concatenates the full list |
| `Command(goto=...)` | The next node to route to after the supervisor node returns | Use `Command` instead of returning a plain dict when the routing target must be determined at runtime from within the node itself; use `add_conditional_edges` with a static map for routing logic that lives outside the node |
| `recursion_limit` | Maximum number of node invocations before LangGraph raises `GraphRecursionError` | Set to `> 2 × (number_of_agents + 1)` to allow a supervisor to visit each agent at least once; default is 25 — lower to catch infinite-loop bugs in development |
| `interrupt_before` | Nodes at which the graph pauses and waits for human input before proceeding | Set `interrupt_before=["supervisor"]` to review every routing decision; set `interrupt_before=["synthesis_agent"]` to approve the final answer before delivery |
| `space_id` (Genie API) | Identifies the specific Genie Agent to query | Copy from the Genie Agent's **Settings** tab in the Databricks UI, or retrieve via `GET /api/2.0/genie/spaces` |
| Genie polling interval | How frequently to call `GET .../messages/{msg_id}` | Poll every 1–5 seconds; apply exponential backoff up to 60 s; stop polling after 10 minutes if status is not `COMPLETED`/`FAILED`/`CANCELLED` |

---

## Worked Example: Requirement → Decision

**Given:** A financial analytics team needs an AI assistant that can answer complex business questions like "Why did EMEA revenue drop in Q3 and what customers were most affected?" The question requires both unstructured document retrieval (sales call notes in a vector store) and structured data analysis (revenue tables in Unity Catalog). The team uses Databricks and needs full observability.

**Step 1 — Identify the goal:** Build a multi-agent system that routes sub-tasks to a retrieval agent (for call notes) and a Genie analytics agent (for revenue data), then synthesises a coherent answer.

**Step 2 — Define inputs:** User question (string), Unity Catalog revenue tables (accessible via Genie Agent), vector store with indexed call notes (retrieval tool), MLflow experiment for tracing.

**Step 3 — Define outputs:** A synthesised answer with citations to both retrieved documents and SQL query results, logged as an MLflow trace for observability.

**Step 4 — Apply constraints:**
- Revenue data is governed by Unity Catalog row-level security — must use Genie API (not raw SQL) to enforce per-user access.
- The system must avoid infinite loops — set `recursion_limit=15` (3 agents × 4 round-trips + 3 margin).
- Human-in-the-loop approval required before the final synthesis is returned — use `interrupt_before=["synthesis_agent"]`.
- Must integrate with Databricks Apps for deployment.

**Step 5 — Select the approach:** Use the **supervisor pattern** in LangGraph with three nodes (`retrieval_agent`, `genie_agent`, `synthesis_agent`) connected via `add_conditional_edges` from a central `supervisor` node. The supervisor uses structured output with a `Next` enum (`RETRIEVAL | ANALYTICS | SYNTHESIS | FINISH`). The Genie API is wrapped as a `@tool` called `query_revenue_data` and passed to the `genie_agent` node. This is preferred over CrewAI because Unity Catalog security enforcement requires controlled, per-request API calls with proper authentication — LangGraph's explicit state graph makes the call sequence auditable and the `interrupt_before` hook enables the required human approval step before synthesis.

---

## Implementation

```python
# Scenario: Route a complex business question to two specialist sub-agents using a
# supervisor node, then synthesise the results — without coupling the agents to each other.

from typing import Annotated, Literal
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.types import Command
from langchain_databricks import ChatDatabricks

# --- Shared state schema -------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    next: str          # supervisor writes the routing target here

# --- Sub-agent: retrieval -----------------------------------------------

retrieval_llm = ChatDatabricks(endpoint="databricks-claude-sonnet-4-5")

def retrieval_agent(state: AgentState) -> dict:
    """Retrieves relevant documents from the vector store."""
    last_msg = state["messages"][-1].content
    # In production: call a vector search tool here
    result = retrieval_llm.invoke([
        HumanMessage(content=f"Search knowledge base for: {last_msg}")
    ])
    return {"messages": [result]}

# --- Sub-agent: analytics via Genie ------------------------------------

@tool
def query_genie_agent(question: str, space_id: str = "YOUR_SPACE_ID") -> str:
    """Queries a Genie Agent to run natural-language SQL analytics."""
    import requests, time, os
    token = os.environ["DATABRICKS_TOKEN"]
    host  = os.environ["DATABRICKS_HOST"]
    headers = {"Authorization": f"Bearer {token}"}

    # Start conversation
    resp = requests.post(
        f"{host}/api/2.0/genie/spaces/{space_id}/start-conversation",
        headers=headers,
        json={"content": question},
    ).json()
    conv_id = resp["conversation"]["id"]
    msg_id  = resp["message"]["id"]

    # Poll for completion
    for _ in range(120):        # up to 10 min at 5-s intervals
        time.sleep(5)
        msg = requests.get(
            f"{host}/api/2.0/genie/spaces/{space_id}"
            f"/conversations/{conv_id}/messages/{msg_id}",
            headers=headers,
        ).json()
        if msg.get("status") in ("COMPLETED", "FAILED", "CANCELLED"):
            break

    # Extract SQL text and summary from attachments
    attachments = msg.get("attachments") or []
    parts = [a.get("text", {}).get("content", "") for a in attachments if a.get("text")]
    return "\n".join(parts) or "No result from Genie."

analytics_llm = ChatDatabricks(endpoint="databricks-claude-sonnet-4-5")

def analytics_agent(state: AgentState) -> dict:
    """Delegates structured data questions to Genie via tool call."""
    last_msg = state["messages"][-1].content
    result = analytics_llm.bind_tools([query_genie_agent]).invoke([
        HumanMessage(content=last_msg)
    ])
    return {"messages": [result]}

# --- Sub-agent: synthesis -----------------------------------------------

synthesis_llm = ChatDatabricks(endpoint="databricks-claude-sonnet-4-5")

def synthesis_agent(state: AgentState) -> dict:
    """Combines all prior messages into a final grounded answer."""
    result = synthesis_llm.invoke(
        state["messages"] + [HumanMessage(content="Synthesise a final answer.")]
    )
    return {"messages": [result]}

# --- Supervisor ---------------------------------------------------------

from pydantic import BaseModel

class RouteDecision(BaseModel):
    next: Literal["retrieval_agent", "analytics_agent", "synthesis_agent", "FINISH"]
    reasoning: str

supervisor_llm = ChatDatabricks(endpoint="databricks-claude-sonnet-4-5")

def supervisor(state: AgentState) -> Command:
    """Reads state and routes to the next agent or terminates."""
    decision = supervisor_llm.with_structured_output(RouteDecision).invoke(
        state["messages"] + [HumanMessage(
            content="Which agent should act next? Choose one of: "
                    "retrieval_agent, analytics_agent, synthesis_agent, FINISH."
        )]
    )
    goto = END if decision.next == "FINISH" else decision.next
    return Command(goto=goto, update={"next": decision.next})

# --- Graph assembly -----------------------------------------------------

graph = StateGraph(AgentState)
graph.add_node("supervisor",       supervisor)
graph.add_node("retrieval_agent",  retrieval_agent)
graph.add_node("analytics_agent",  analytics_agent)
graph.add_node("synthesis_agent",  synthesis_agent)

graph.set_entry_point("supervisor")

# Sub-agents always return to supervisor after completing
for node in ("retrieval_agent", "analytics_agent", "synthesis_agent"):
    graph.add_edge(node, "supervisor")

# Supervisor routes conditionally
graph.add_conditional_edges(
    "supervisor",
    lambda state: state["next"],
    {
        "retrieval_agent":  "retrieval_agent",
        "analytics_agent":  "analytics_agent",
        "synthesis_agent":  "synthesis_agent",
        "FINISH":           END,
    },
)

app = graph.compile(recursion_limit=15, interrupt_before=["synthesis_agent"])
```

---

```python
# Anti-pattern: Chaining agent invocations sequentially with .invoke() calls
# instead of a shared state graph — breaks state sharing, hides intermediate
# results from the supervisor, and makes tracing impossible.

# WRONG — do not do this
def bad_multi_agent(question: str) -> str:
    retrieval_result = retrieval_agent_fn(question)        # no shared state
    analytics_result = analytics_agent_fn(question)        # cannot see retrieval_result
    final = synthesis_agent_fn(retrieval_result + analytics_result)  # manual concat
    return final
# Problems:
# 1. Each agent is blind to what the others found — no shared message history.
# 2. The supervisor has no control — routing is hardcoded, not dynamic.
# 3. MLflow tracing sees three unrelated traces, not one coherent run.
# 4. Human-in-the-loop is impossible without threading the state through manually.

# CORRECT — use a shared StateGraph so all agents share the messages channel
# See the supervisor pattern implementation above.
```

---

```python
# Scenario: Expose a complete sub-agent graph as a @tool so a parent agent
# can invoke it through standard tool-calling without sharing a state graph.

from langchain_core.tools import tool
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage

def build_search_sub_agent():
    """Returns a compiled LangGraph that performs document search."""
    from typing import Annotated
    from langgraph.graph.message import add_messages
    from typing_extensions import TypedDict

    class SearchState(TypedDict):
        messages: Annotated[list, add_messages]

    search_llm = ChatDatabricks(endpoint="databricks-claude-sonnet-4-5")

    def search_node(state: SearchState) -> dict:
        result = search_llm.invoke(state["messages"])
        return {"messages": [result]}

    g = StateGraph(SearchState)
    g.add_node("search", search_node)
    g.set_entry_point("search")
    g.set_finish_point("search")
    return g.compile()

_search_agent = build_search_sub_agent()

@tool
def search_documents(query: str) -> str:
    """Search internal documents for information relevant to the query.
    Use when the question requires unstructured knowledge from document stores."""
    result = _search_agent.invoke({"messages": [HumanMessage(content=query)]})
    return result["messages"][-1].content

# Parent agent uses search_documents as a regular tool
parent_llm = ChatDatabricks(endpoint="databricks-claude-sonnet-4-5")
parent_with_tools = parent_llm.bind_tools([search_documents])
```

---

## Common Pitfalls & Misconceptions

- **Replacing the message list instead of appending** — Beginners return `{"messages": new_message}` (a single message object) or forget the `add_messages` annotation and accidentally overwrite the entire history with each node update. The correct model is: always annotate the `messages` key with `Annotated[list, add_messages]` and return a list even for single-message updates: `{"messages": [new_message]}`.

- **Treating the supervisor as stateless** — Newcomers call the supervisor LLM without passing the full state history, causing it to re-route to agents that have already completed their work, producing an infinite loop. The correct model is: always pass `state["messages"]` to the supervisor so it can reason about what has already been done and terminate with `FINISH`.

- **Forgetting `recursion_limit` until production** — LangGraph's default `recursion_limit=25` silently caps execution, which causes confusing partial results when a supervisor cycles more than expected. The correct model is: always set an explicit `recursion_limit` at compile time and add a `GraphRecursionError` catch in the caller so the failure mode is observable.

- **Assuming Genie Agent mode is API-accessible** — Developers try to trigger the Genie Agent multi-step "Agent mode" via the Conversation API but receive only standard single-query responses. As of July 2026, Agent mode is only accessible from the Databricks UI — not the API. The correct model is: use the Conversation API for programmatic single-question tool calls in multi-agent systems; reserve Agent mode for interactive analyst workflows in the UI.

- **Confusing agent-as-tool with shared-state multi-agent** — Developers use `@tool`-wrapped sub-agents expecting the parent to see the sub-agent's intermediate reasoning steps, then wonder why traces are incomplete. The correct model is: `@tool` wrapping hides internal reasoning — the parent sees only the final return string. For visibility into sub-agent reasoning, use a shared `StateGraph` instead.

---

## Key Definitions

| Term | Definition |
|---|---|
| Supervisor pattern | A multi-agent topology where a central router node reads shared state and dispatches work to specialist sub-agents using conditional edges or `Command` routing |
| Agent-as-tool | A design pattern where an entire sub-agent (possibly a compiled LangGraph graph) is wrapped in a `@tool` function and invoked by a parent agent through tool-calling mechanics |
| `add_messages` reducer | A LangGraph reducer for the `messages` channel that appends new messages to the existing list rather than replacing it, preserving conversation history across nodes |
| `Command` (LangGraph) | A return type from a node that simultaneously updates state and specifies the next node to route to, enabling runtime routing decisions inside the node itself |
| `recursion_limit` | The maximum number of node invocations in a single `graph.invoke()` call before LangGraph raises `GraphRecursionError` |
| `interrupt_before` | A compile-time option on a LangGraph graph that pauses execution before the named node(s) and waits for external input, enabling human-in-the-loop workflows |
| Genie Agent | A Databricks compound-AI system (formerly Genie Spaces) that translates natural-language questions to read-only SQL queries over Unity Catalog data, enforcing per-user row-level security |
| Genie Conversation API | REST API (`/api/2.0/genie/spaces/{space_id}/start-conversation`) that enables programmatic integration of a Genie Agent into applications and multi-agent systems |
| Knowledge store | Agent-level semantic metadata in a Genie Agent: column descriptions, synonyms, SQL measures, filters, and join specs that improve NL-to-SQL accuracy |
| Compound AI system | An AI application that combines multiple interacting components (retrieval, reasoning, SQL execution) rather than relying on a single LLM inference call |

---

## Summary / Quick Recall

- A supervisor node in LangGraph returns `Command(goto=agent_name)` to route dynamically; `add_conditional_edges` maps possible targets at build time.
- The `messages` channel uses the `add_messages` reducer — always return `{"messages": [msg]}` (a list), never a bare message object.
- Agent-as-tool hides sub-agent reasoning from the parent; use shared `StateGraph` when you need visibility into intermediate steps.
- Set `recursion_limit` explicitly at compile time; default 25 will silently cap complex workflows.
- Genie Agents (formerly Genie Spaces) expose Unity Catalog data to multi-agent systems via the Conversation API; data access is governed by per-user Unity Catalog row security.
- `interrupt_before=["node_name"]` enables human-in-the-loop approval gates without restructuring the graph.
- CrewAI suits rapid prototyping with role abstractions; LangGraph is preferred for production Databricks deployments requiring precise state control and observability.

---

## Self-Check Questions

1. In LangGraph's supervisor pattern, what does the supervisor node return to dynamically route to the next sub-agent?

   <details><summary>Answer</summary>

   The supervisor returns a `Command(goto=agent_name)` object. `Command` is LangGraph's first-class mechanism for combining a state update with a routing decision inside a single node return. The distractor answer is "a plain dict with a `next` key" — while updating state with `{"next": agent_name}` is necessary, that alone does not instruct the graph to traverse an edge; the routing logic in `add_conditional_edges` must read that value, or `Command(goto=...)` must be returned directly.

   </details>

2. A multi-agent system's supervisor keeps re-routing to the retrieval agent after it has already run twice, never reaching synthesis. What is the most likely cause?

   <details><summary>Answer</summary>

   The supervisor LLM is not receiving the full message history in its prompt, so it cannot determine that retrieval has already completed. Each call to the supervisor should pass `state["messages"]` — all prior agent messages — so the routing LLM can reason about what work is done. A secondary cause is that the retrieval agent is not writing its results back to the shared `messages` channel with the `add_messages` reducer, making its output invisible to the supervisor.

   </details>

3. **Which TWO** of the following are correct reasons to prefer LangGraph's shared `StateGraph` over wrapping sub-agents as `@tool` functions in a production Databricks multi-agent system?
   - A. `@tool` wrapping is not supported by LangChain or LangGraph
   - B. A shared `StateGraph` allows the supervisor to observe each agent's intermediate messages, enabling more accurate routing decisions
   - C. `@tool`-wrapped agents cannot access Unity Catalog data
   - D. MLflow tracing captures the full message history across all nodes in a `StateGraph`, giving end-to-end observability
   - E. `@tool` wrapping always increases latency by at least 2×

   <details><summary>Answer</summary>

   **B and D.** B is correct because in a shared `StateGraph`, every agent's output is appended to the `messages` channel, so the supervisor sees the full context when making subsequent routing decisions — `@tool` wrapping only surfaces the sub-agent's final return string. D is correct because MLflow Tracing automatically spans across nodes in a LangGraph `StateGraph`, giving a single coherent trace for the entire multi-agent run; `@tool`-based sub-agents generate separate, disconnected traces.

   A is wrong — `@tool` wrapping is fully supported by LangChain and LangGraph. C is wrong — `@tool`-wrapped agents can freely call any Python code including Unity Catalog-backed APIs. E is wrong — latency difference depends on the workload, not the wrapping mechanism.

   </details>

4. Your team wants to query a Genie Agent programmatically as a tool in a LangGraph multi-agent system. The Genie Agent's `space_id` is known. Describe the API interaction sequence required.

   <details><summary>Answer</summary>

   1. `POST /api/2.0/genie/spaces/{space_id}/start-conversation` with `{"content": question}`. This returns a `conversation_id` and `message_id` with status `IN_PROGRESS`.
   2. Poll `GET /api/2.0/genie/spaces/{space_id}/conversations/{conversation_id}/messages/{message_id}` every 1–5 seconds.
   3. Continue polling until `status` is `COMPLETED`, `FAILED`, or `CANCELLED` (or a 10-minute timeout).
   4. Extract the answer from the `attachments` array — look for `text.content` for the narrative summary and `query.statement` for the generated SQL.
   5. Return the extracted content as the tool result string.

   The wrong approach is to treat the Genie API as synchronous and read the result immediately after the POST — `status` will be `IN_PROGRESS` and `attachments` will be null.

   </details>

5. A solutions architect proposes using CrewAI instead of LangGraph for a new enterprise multi-agent system on Databricks. The system must enforce per-user Unity Catalog row-level security, support human-in-the-loop approval before final responses, and emit MLflow traces for every agent step. Evaluate the trade-offs.

   <details><summary>Answer</summary>

   **LangGraph is the stronger choice for these specific requirements.** For per-user Unity Catalog security: both frameworks can call the Genie API, but LangGraph's explicit state graph makes the call sequence auditable; CrewAI's implicit context passing makes it harder to guarantee that the per-user token is propagated correctly to every data access call. For human-in-the-loop: LangGraph's `interrupt_before` is a first-class compile-time feature that pauses the graph and resumes from checkpointed state — CrewAI does not have an equivalent primitive as of 2026. For MLflow tracing: LangGraph integrates with MLflow through the `ResponsesAgent` interface and auto-spans each node; CrewAI requires manual `mlflow.log_*` instrumentation.

   CrewAI would be justified for a rapid prototype or a simpler workflow without per-user security constraints or approval gates — its declarative role/task model is faster to set up. But for the stated production requirements, the added control LangGraph provides outweighs its higher initial setup cost.

   </details>

---

## Further Reading

- [Genie Agents concepts — Databricks on AWS](https://docs.databricks.com/aws/en/genie/concepts) — *verified 2026-07-11* — Covers how Genie Agents generate responses, the knowledge store, and data access governance
- [Use the Genie Agents API — Databricks on AWS](https://docs.databricks.com/aws/en/genie/conversation-api) — *verified 2026-07-11* — Full Conversation API reference including request/response schemas, polling patterns, and Management API
- [Agent mode in Genie Agents — Databricks on AWS](https://docs.databricks.com/aws/en/genie/agent-mode) — *verified 2026-07-11* — Multi-step reasoning mode for complex exploratory questions
- [Use agents on Databricks — Databricks on AWS](https://docs.databricks.com/aws/en/agents/agent-framework/build-agents) — *verified 2026-07-11* — Framework-agnostic overview of building, deploying, and evaluating agents on Databricks
- [Author an AI agent and deploy it on Databricks Apps](https://docs.databricks.com/aws/en/agents/agent-framework/author-agent) — *verified 2026-07-11* — LangGraph and OpenAI SDK agent authoring patterns, `ResponsesAgent` interface, and deployment workflow
