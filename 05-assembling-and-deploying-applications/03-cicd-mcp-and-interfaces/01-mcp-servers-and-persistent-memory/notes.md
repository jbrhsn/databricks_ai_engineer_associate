# MCP Servers and Persistent Memory

**Section:** 05 — Assembling and Deploying Applications | **Module:** 03 — CI/CD, MCP, and Interfaces | **Est. time:** 3 hrs | **Exam mapping:** Assembling & Deploying (22%) — Integrate external tools and manage agent state

> ⚠️ **Fast-evolving:** MCP and Databricks managed MCP servers are in Public Preview as of July 2026. API endpoints, OAuth scopes, and the `databricks-mcp` library are subject to change. Verify against official Databricks docs before authoring production code.

---

## TL;DR

The Model Context Protocol (MCP) is an open JSON-RPC 2.0 standard that lets AI agents discover and call tools, resources, and prompt templates hosted on external servers — eliminating the need to hard-code integrations for every data source or API. Databricks surfaces MCP through Unity AI Gateway as both managed servers (AI Search, Genie, SQL, UC functions) and custom servers deployed as Databricks Apps, all governed by Unity Catalog permissions. Persistent memory for agents spans short-term in-context state (LangGraph `TypedDict` + checkpointer) and long-term external storage (Delta tables, Vector Search for semantic recall, Redis for session state). **The one thing to remember: MCP gives an agent a standardised plug to reach any tool; persistent memory gives that agent the ability to remember across turns and sessions.**

---

## ELI5 — Explain It Like I'm 5

Imagine you are building with LEGO and your pieces are scattered across five different boxes in different rooms. Every time you want a piece you have to run to the right room, find the right box, and bring it back. MCP is like putting a universal docking station in your room: all the boxes plug into it, and you just ask the docking station for a piece by name — you never run to different rooms again. The most common misconception is that MCP makes the boxes disappear or merges them together; they stay exactly where they are, but the docking station handles all the routing so your LEGO-building (the agent) never has to know which room each piece came from. Persistent memory is your notebook on the desk: short-term memory is the scrap of paper you are using right now (gone after you finish), and long-term memory is the notebook you put on the shelf so that the next time you sit down, you can read what you wrote last week.

---

## Learning Objectives

By the end of this chapter you will be able to:
- [ ] Describe the MCP client ↔ server ↔ primitive model and explain the difference between stdio and Streamable HTTP transport mechanisms
- [ ] Register and call a Databricks managed MCP server from LangGraph agent code using `DatabricksMultiServerMCPClient`
- [ ] Distinguish the three MCP primitive types (tools, resources, prompts) and explain when each is appropriate
- [ ] Implement short-term in-context memory with a LangGraph `MemorySaver` checkpointer and long-term memory with a Delta table read/write node
- [ ] Articulate the trade-offs between stateless and stateful agent deployment on Databricks Model Serving and Databricks Apps

---

## Visual Overview

### MCP Architecture: Host, Client, Server

```
┌────────────────────────────────────────────────────────────┐
│  MCP Host  (AI application — e.g. LangGraph agent)         │
│                                                            │
│  ┌────────────────┐   ┌────────────────┐                   │
│  │  MCP Client 1  │   │  MCP Client 2  │  ...              │
│  └───────┬────────┘   └───────┬────────┘                   │
└──────────┼────────────────────┼───────────────────────────┘
           │ dedicated           │ dedicated
           │ connection          │ connection
           ▼                     ▼
┌──────────────────┐   ┌───────────────────────────────────┐
│  MCP Server A    │   │  MCP Server B (Databricks managed) │
│  (local, stdio)  │   │  (remote, Streamable HTTP / SSE)   │
│                  │   │                                    │
│  ├── tools/list  │   │  ├── tools/list                    │
│  ├── tools/call  │   │  ├── tools/call                    │
│  └── resources/  │   │  └── resources/                    │
│      list+get    │   │      list+get                      │
└──────────────────┘   └────────────────────────────────────┘
```

### MCP Transport Comparison

```
stdio (local process)                 Streamable HTTP (remote)
─────────────────────                 ────────────────────────
Client                                Client
  │  spawn subprocess                   │  HTTP POST  ──► Server
  ▼                                     │  (JSON-RPC body)
Server Process                          │
  │  stdin  ◄── JSON-RPC request        │  ◄── HTTP 200 application/json
  │  stdout ──► JSON-RPC response       │  or
  │  stderr ──► logs (optional)         │  ◄── text/event-stream (SSE)
  │                                     │       for streaming responses
  ▼                                     │
  terminated when client closes stdin   │  Mcp-Session-Id header for
                                        │  stateful session tracking
```

### MCP Primitives Decision Tree

```
What does the agent need from the MCP server?
│
├── Perform an action / call an API / run a function?
│       └──► Tool  (tools/list → tools/call)
│
├── Read context data (file contents, DB schema, docs)?
│       └──► Resource  (resources/list → resources/read)
│
└── Structured prompt template / few-shot examples?
        └──► Prompt  (prompts/list → prompts/get)
```

### Agent Memory Patterns

```
Turn N                            Turn N+1 (same session)     Future session
─────────────────────             ─────────────────────────   ───────────────
LangGraph state (TypedDict)       Checkpointer rehydrates      Delta / Vector
  messages: [...]        ──►      state from thread_id   ──►  table loaded
  context: "..."                  short-term memory still     at session start
                                  in context                  long-term facts
```

---

## Key Concepts

### MCP Architecture: Host, Client, and Server

MCP uses a three-role architecture: the **MCP Host** is the AI application (e.g. a LangGraph agent) that orchestrates one or more **MCP Clients**; each client maintains a dedicated connection to exactly one **MCP Server**. Clients are created programmatically when the host application initialises; they perform a capability-negotiation handshake (JSON-RPC `initialize` request) with their server, then expose the server's capabilities to the host. The host application uses the aggregated set of capabilities — tools, resources, prompts — when deciding what to hand to the language model. In Databricks, the `DatabricksMCPClient` and `DatabricksMultiServerMCPClient` classes in the `databricks-langchain` / `databricks-mcp` libraries implement this role; they handle OAuth authentication automatically against Unity AI Gateway.

### MCP Server Types: stdio vs. Streamable HTTP Transport

**stdio transport** launches the MCP server as a child subprocess of the host; the client writes JSON-RPC messages to the process's `stdin` and reads responses from `stdout`. This approach carries zero network overhead and is the simplest deployment model — the server lives and dies with the client process, making it well suited for local tools or developer workstations. **Streamable HTTP transport** (which replaces the deprecated HTTP+SSE transport from protocol version 2024-11-05) runs the server as an independent long-lived process that accepts HTTP POST requests from one or many clients; it can optionally return `text/event-stream` responses for streaming. This is the transport used by all Databricks managed MCP servers — the `Mcp-Session-Id` header enables optional stateful session tracking across requests. Choosing between them is a deployment decision: use stdio for local single-user tools and Streamable HTTP for shared, governed, remotely hosted servers.

### MCP Primitives: Tools, Resources, and Prompts

MCP servers expose three primitive types that clients discover dynamically at runtime. **Tools** are executable functions the agent can invoke (e.g. run a SQL query, call an API); the client discovers them with `tools/list` and invokes them with `tools/call`, passing a JSON object validated against the tool's `inputSchema`. **Resources** are read-only context data sources (e.g. a database schema, a file listing); the client discovers them with `resources/list` and fetches them with `resources/read`. **Prompts** are reusable prompt templates or few-shot examples; the client fetches them with `prompts/get`. The language model never calls MCP primitives directly — the client intercepts tool-call outputs from the model and routes them through the appropriate MCP server, then returns the result as a tool message in the conversation.

### Registering MCP Servers on Databricks

Databricks provides three MCP server categories, all accessible via Unity AI Gateway:

1. **Managed servers** — pre-configured, no setup needed. URL pattern: `https://<workspace>/api/2.0/mcp/<service>/<path>`. Includes AI Search, Genie (natural-language analytics), Databricks SQL (read/write SQL), and Unity Catalog functions.
2. **External MCP Services** — external servers (e.g. Slack, GitHub) registered as Unity Catalog securables via `ai-gateway/mcp-services/`. Unity Catalog governs access and credential management.
3. **Custom servers** — developer-built MCP servers deployed as Databricks Apps; URL: `https://<app-url>/mcp`.

All three types are visible in the workspace UI under **AI Gateway → MCPs**, and all use Unity Catalog permissions for access control.

### Short-Term In-Context Memory (LangGraph Checkpointer)

Short-term memory in a LangGraph agent is the `TypedDict` state object that accumulates `messages` across nodes within a single conversation thread. The state persists across turns because every node returns only its partial update and a **checkpointer** (e.g. `MemorySaver`, `SqliteSaver`, `AsyncPostgresSaver`) snapshots the full state at each super-step. The host identifies a conversation thread with a `thread_id` in the `config` dict: `{"configurable": {"thread_id": "user-123"}}`. When the agent is called again with the same `thread_id`, the checkpointer rehydrates state from the last snapshot, giving the LLM access to prior conversation turns without any external database call. This memory exists only for the duration of the session (or until the checkpointer store is cleared), and the entire history is injected into the model's context window at each call — making it unsuitable for unbounded conversation histories.

### Long-Term External Memory (Delta, Vector Search, Redis)

Long-term memory persists beyond a single session and is retrieved selectively rather than injecting the entire history. Three common backends in Databricks:

- **Delta tables** store structured facts (user preferences, completed tasks, entities) as rows; a dedicated `memory_read` node queries the table at session start and a `memory_write` node upserts new facts at session end.
- **Vector Search (AI Search)** stores embeddings of past interactions or documents; a retrieval node performs ANN search to surface semantically relevant memories from potentially millions of entries without filling the context window.
- **Redis** stores lightweight session state (current task, last tool called) as key-value pairs with TTL; useful for agent replicas that must share state in a horizontally scaled deployment.

In LangGraph, these backends are accessed from regular Python nodes: the node calls the Delta table / Vector Search index / Redis client, reads the result into the state dict, and subsequent nodes see the retrieved memory as part of state.

### Stateless vs. Stateful Agent Deployment

A **stateless agent** holds no memory between API calls — each invocation starts from a blank context, making it trivially horizontally scalable, easy to test, and safe to cache. Databricks Model Serving endpoints are stateless by default. A **stateful agent** maintains conversation thread state (via a checkpointer) and must route requests for the same `thread_id` to a consistent store; Databricks Apps is the recommended deployment target because the app manages the server process lifecycle. The architectural decision hinges on the use case: customer-service chatbots need stateful multi-turn memory; batch inference pipelines prefer stateless independence. A hybrid pattern uses stateless Model Serving with an external checkpointer (Delta or Redis) to achieve both scalability and persistence.

---

## Key Parameters / Configuration Knobs

| Parameter | What it controls | Decision rule |
|---|---|---|
| `transport` (MCP server type) | Whether the server communicates via `stdio` (subprocess) or Streamable HTTP (independent process) | Use `stdio` for local single-user dev tools; use Streamable HTTP for shared, remotely hosted, multi-client servers |
| `Mcp-Session-Id` header | Enables stateful session tracking on Streamable HTTP servers | Include on every request after initialization only when the server assigns it; omit for truly stateless request-response servers |
| `thread_id` in LangGraph config | Identifies the conversation thread; determines which checkpointer snapshot to rehydrate | Always set to a stable user- or session-scoped ID in production; use `uuid4()` per test run in unit tests to avoid state bleed |
| Checkpointer backend | Where state snapshots are persisted (`MemorySaver` = in-process RAM; `SqliteSaver` = local file; `AsyncPostgresSaver` / Delta = durable external store) | Use `MemorySaver` for local development only; use a durable external store for any multi-process or multi-replica production deployment |
| `MCP-Protocol-Version` header | Tells the server which MCP spec version the client is using; server uses it to format responses correctly | Set to the version negotiated during `initialize`; omit only for servers that cannot receive it (legacy compatibility) |
| `databricks_mcp` server URL | Selects which managed, external, or custom MCP server to connect to | Use managed server URLs (`/api/2.0/mcp/...`) for zero-setup Databricks-native tools; use MCP Service URLs for governed external SaaS integrations |
| `credentials_strategy` in `WorkspaceClient` | Determines how the agent authenticates to Databricks MCP servers | Use `ModelServingUserCredentials` for on-behalf-of-user auth in Model Serving; use service principal OAuth for background agents |
| TTL on Redis keys | How long session state survives inactivity before automatic eviction | Set TTL ≤ your expected session timeout (e.g. 30 min for chat UI); set no TTL only for permanent cross-session state that belongs in Delta |

---

## Worked Example: Requirement → Decision

**Given:** A customer-support agent needs to answer billing questions (structured data in a Delta table via Genie), search support documentation (unstructured text in an AI Search index), and remember the customer's open ticket number across turns in a single chat session. The agent is deployed for concurrent use by 50 support agents at once.

**Step 1 — Identify the goal:** The agent needs multi-tool access with session-scoped memory, discoverable via a standard interface, without hard-coding each integration.

**Step 2 — Define inputs:** Customer question (string), customer ID (for memory lookup), two Databricks data assets: a Genie Agent space ID (for billing queries) and an AI Search index (for documentation search).

**Step 3 — Define outputs:** A natural-language answer with sources cited; the active ticket number persisted in the conversation state across follow-up questions.

**Step 4 — Apply constraints:**
- 50 concurrent users → deployment must handle concurrent connections; a local `MemorySaver` (in-process) would lose state across replicas.
- Governance requirement: only support staff with `CAN_RUN` on the Genie space and `SELECT` on the AI Search index should invoke those tools.
- Memory must survive page refresh within a session but is not needed across sessions.

**Step 5 — Select the approach:** Use `DatabricksMultiServerMCPClient` with two servers: the managed Genie Agent MCP server and the managed AI Search MCP server. Use `create_react_agent` (LangGraph) with both servers' tools. Attach a Redis-backed or `AsyncPostgresSaver` checkpointer keyed by `thread_id = session_id` to persist conversation state across replicas. Deploy on Databricks Apps (not Model Serving) so the app service principal is granted `CAN_RUN` on the Genie space. Rationale vs alternatives: hard-coded LangChain tool wrappers would require custom auth logic per integration and bypass Unity AI Gateway governance; a purely stateless deployment would lose the ticket number on every follow-up turn.

---

## Implementation

```python
# Scenario: Build a LangGraph ReAct agent that calls two Databricks MCP servers
# (Genie for billing data and AI Search for documentation), governed by Unity Catalog.
# Constraint: must work in a multi-replica Databricks Apps deployment.

import asyncio
from databricks.sdk import WorkspaceClient
from databricks_langchain import (
    ChatDatabricks,
    DatabricksMCPServer,
    DatabricksMultiServerMCPClient,
)
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver  # swap for AsyncPostgresSaver in prod

workspace_client = WorkspaceClient()
host = workspace_client.config.host
GENIE_SPACE_ID = "01ef..."  # your Genie space ID

async def build_support_agent():
    mcp_client = DatabricksMultiServerMCPClient(
        [
            DatabricksMCPServer(
                name="genie-billing",
                url=f"{host}/api/2.0/mcp/genie/{GENIE_SPACE_ID}",
                workspace_client=workspace_client,
            ),
            DatabricksMCPServer(
                name="ai-search-docs",
                url=f"{host}/api/2.0/mcp/ai-search/prod/support/docs_index",
                workspace_client=workspace_client,
            ),
        ]
    )

    async with mcp_client:
        tools = await mcp_client.get_tools()
        llm = ChatDatabricks(endpoint="databricks-claude-sonnet-4-5")
        # MemorySaver keeps state in-process; replace with AsyncPostgresSaver for
        # multi-replica deployments (Databricks Apps with external Postgres or Delta).
        checkpointer = MemorySaver()
        agent = create_react_agent(llm, tools, checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "customer-session-abc123"}}

        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "What is my current billing amount?"}]},
            config=config,
        )
        print(result["messages"][-1].content)

        # Follow-up turn — checkpointer rehydrates prior messages automatically
        result2 = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "And what is my open support ticket number?"}]},
            config=config,
        )
        print(result2["messages"][-1].content)

asyncio.run(build_support_agent())
```

```python
# Scenario: Add a long-term memory read node to a LangGraph agent so it recalls
# user preferences from a Delta table at the start of every session.

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, SystemMessage
from databricks.sdk import WorkspaceClient
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_preferences: str  # loaded from Delta at session start

def memory_read_node(state: AgentState) -> dict:
    """Read user preferences from Delta long-term memory store."""
    # In production, derive user_id from auth context or session config
    user_id = "user-42"
    df = spark.sql(
        f"SELECT preferences FROM main.default.agent_memory WHERE user_id = '{user_id}'"
    )
    rows = df.collect()
    prefs = rows[0]["preferences"] if rows else "No prior preferences found."
    # Inject preferences as a system message so the LLM sees them
    system_msg = SystemMessage(content=f"User preferences: {prefs}")
    return {"messages": [system_msg], "user_preferences": prefs}

def memory_write_node(state: AgentState) -> dict:
    """Persist extracted preferences back to Delta after the conversation."""
    # In a real implementation, extract structured facts from the conversation
    # and upsert them; this is a simplified placeholder.
    spark.sql(
        f"MERGE INTO main.default.agent_memory AS t "
        f"USING (SELECT 'user-42' AS user_id, 'prefers concise answers' AS preferences) AS s "
        f"ON t.user_id = s.user_id "
        f"WHEN MATCHED THEN UPDATE SET preferences = s.preferences "
        f"WHEN NOT MATCHED THEN INSERT *"
    )
    return {}

# Wire the memory nodes around the agent node in the graph
builder = StateGraph(AgentState)
builder.add_node("memory_read", memory_read_node)
builder.add_node("agent", lambda state: state)  # placeholder for LLM call node
builder.add_node("memory_write", memory_write_node)
builder.add_edge(START, "memory_read")
builder.add_edge("memory_read", "agent")
builder.add_edge("agent", "memory_write")
builder.add_edge("memory_write", END)
graph = builder.compile()
```

```python
# Anti-pattern: Sharing a MemorySaver instance across multiple agent replicas
# by storing it as a module-level singleton in a Databricks Apps deployment.

# BAD — each replica has its own in-process MemorySaver; thread_id lookups
# on replica 2 will not find state written by replica 1, silently losing memory.
from langgraph.checkpoint.memory import MemorySaver
_GLOBAL_CHECKPOINTER = MemorySaver()  # ← module-level singleton

def get_agent():
    return create_react_agent(llm, tools, checkpointer=_GLOBAL_CHECKPOINTER)

# Correct approach: use an external durable checkpointer shared across all replicas.
# AsyncPostgresSaver (langgraph-checkpoint-postgres) or a custom Delta-backed
# checkpointer stores state outside the process so any replica can read it.

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
import os

async def get_agent_production():
    # DATABASE_URL comes from a Databricks secret scope
    checkpointer = await AsyncPostgresSaver.from_conn_string(
        os.environ["CHECKPOINT_DB_URL"]
    )
    await checkpointer.setup()  # creates tables if not present
    return create_react_agent(llm, tools, checkpointer=checkpointer)

# What breaks with the anti-pattern: under round-robin load balancing, a user's
# second message hits a different replica and gets an empty context window,
# so the agent behaves as if the conversation never started. No error is raised —
# the failure is silent and the user just sees a confused agent.
```

---

## Common Pitfalls & Misconceptions

- **MCP replaces all tool wrappers** — Beginners assume that adopting MCP means rewriting every existing LangChain tool as an MCP server. In reality, MCP is a transport and discovery protocol: your existing Python function tools continue to work inside LangGraph nodes unchanged; MCP is added when you want to integrate *external* servers (databases, SaaS APIs, Databricks managed services) through a governed, standardised interface without embedding credentials in the agent.

- **stdio is simpler, so it must be worse for production** — Beginners equate "simpler" with "less capable" and assume Streamable HTTP is always the right choice. The deciding factor is deployment topology, not capability: if the server and client run on the same machine and you do not need multi-tenant access, stdio is fully production-ready and avoids network latency and auth overhead entirely.

- **MemorySaver is sufficient for production** — Tutorials frequently use `MemorySaver` for simplicity, leading beginners to deploy it without modification. `MemorySaver` stores snapshots in Python heap memory; it is lost on process restart, cannot be shared across replicas, and grows unboundedly with conversation length. Any multi-replica or long-lived production deployment needs an external durable checkpointer.

- **Long-term memory = dumping the full history into context** — Beginners conflate checkpointer state (which does accumulate messages in context) with long-term memory. Injecting months of conversation history into every LLM call blows through the context window and increases latency and cost linearly. Long-term memory must be *retrieved selectively* — use Vector Search similarity search to surface only the most relevant past facts, or summarise and compress old turns before storing.

- **All three MCP primitive types call the LLM directly** — Some learners assume that fetching a Resource or Prompt also triggers an LLM call. Resources and Prompts are purely data-fetch operations; only Tool calls trigger the agent to route the result back to the LLM as a tool message. Confusing them leads to over-engineering: wrapping a simple file read as a Tool when a Resource would suffice.

- **Thread ID is optional in development** — Beginners omit `thread_id` during testing because the agent still responds. Without `thread_id`, the checkpointer stores state under a null key; every invocation overwrites the previous snapshot, and multi-turn memory silently fails. Always set `thread_id` in the config dict even in local tests.

---

## Key Definitions

| Term | Definition |
|---|---|
| **MCP (Model Context Protocol)** | An open JSON-RPC 2.0 standard defining how AI host applications connect to external servers that expose tools, resources, and prompts |
| **MCP Host** | The AI application (e.g. a LangGraph agent) that creates and manages one or more MCP client connections |
| **MCP Client** | A component within the host that maintains a single dedicated connection to one MCP server and surfaces that server's capabilities to the host |
| **MCP Server** | A program (local or remote) that implements the MCP protocol and exposes primitives (tools, resources, prompts) to connected clients |
| **stdio transport** | An MCP transport mechanism where the client spawns the server as a subprocess and exchanges JSON-RPC messages over stdin/stdout |
| **Streamable HTTP transport** | An MCP transport mechanism where the server runs as an independent HTTP process; clients POST JSON-RPC requests and receive responses as JSON or SSE streams |
| **Tool (MCP primitive)** | An executable function on an MCP server that the agent can invoke via `tools/call`; defined by a JSON Schema `inputSchema` |
| **Resource (MCP primitive)** | A read-only data source on an MCP server that provides context to the agent via `resources/read`; not executable |
| **Prompt (MCP primitive)** | A reusable prompt template or few-shot example fetched from an MCP server via `prompts/get` |
| **Checkpointer** | A LangGraph component that snapshots agent state at every super-step, enabling multi-turn memory and fault-tolerant resumption |
| **thread_id** | A string identifier passed in LangGraph's `config` dict that scopes a checkpointer snapshot to a specific conversation thread |
| **Short-term memory** | Conversation history held in the agent's context window within a session; managed by the LangGraph state and checkpointer |
| **Long-term memory** | Facts or interaction summaries persisted in an external store (Delta, Vector Search, Redis) and retrieved selectively across sessions |
| **Managed MCP server** | A pre-configured MCP server hosted by Databricks (Genie, AI Search, SQL, UC functions) requiring no setup and governed by Unity Catalog |
| **Unity AI Gateway** | The Databricks enterprise control plane that governs access, monitors activity, and manages MCP servers across a workspace |

---

## Summary / Quick Recall

- MCP is a JSON-RPC 2.0 standard with three roles — Host (application), Client (connection manager), Server (tool/resource/prompt provider).
- Two transports: **stdio** (local subprocess, zero network overhead) vs **Streamable HTTP** (independent remote process, supports multi-client, used by Databricks managed servers).
- Three primitive types: **Tools** (executable, `tools/call`), **Resources** (read-only context, `resources/read`), **Prompts** (templates, `prompts/get`).
- Databricks surfaces MCP via Unity AI Gateway; managed servers include Genie, AI Search, Databricks SQL, and UC functions — all governed by Unity Catalog permissions.
- Use `DatabricksMultiServerMCPClient` + `create_react_agent` to wire Databricks MCP servers into a LangGraph agent with a single async context manager.
- Short-term memory = LangGraph `TypedDict` state + checkpointer + `thread_id`; survives across turns in a session, not across process restarts.
- Long-term memory = Delta (structured facts), Vector Search (semantic recall), Redis (session KV state); retrieved selectively via dedicated LangGraph nodes.
- `MemorySaver` is in-process only — replace with `AsyncPostgresSaver` or a Delta-backed checkpointer for multi-replica production deployments.

---

## Self-Check Questions

1. What is the role of an MCP Client in the Model Context Protocol architecture?

   <details><summary>Answer</summary>

   An MCP Client is a component created by the MCP Host application that maintains a single dedicated connection to exactly one MCP Server. It performs the capability-negotiation handshake (`initialize` request), discovers the server's primitives (tools, resources, prompts), and routes the host application's requests to the server. The distractor is confusing the Client with the Host: the Host is the overall AI application that manages one or more Clients; the Client is a subordinate connection object that handles the protocol with a specific server. There is one Client per Server, not one Client per application.

   </details>

2. A LangGraph agent deployed as a Databricks App serves multiple concurrent users. After deploying with `MemorySaver`, support engineers report that some users' follow-up questions receive answers as if the previous turn never happened. What is the most likely cause?

   <details><summary>Answer</summary>

   `MemorySaver` stores state snapshots in the process's heap memory. When multiple replicas of the Databricks App are running (or when the process restarts), a user's second request may be routed to a different replica or a fresh process that has no snapshot for their `thread_id`. The fix is to replace `MemorySaver` with a durable external checkpointer (`AsyncPostgresSaver`, a Redis-backed checkpointer, or a custom Delta-backed implementation) so all replicas read from and write to the same store. The distractor — "the `thread_id` is missing from the config" — would cause every invocation to overwrite the same null-key snapshot, producing different symptoms (state bleed between users, not missing state).

   </details>

3. **Which TWO** of the following are MCP primitive types that a server can expose to clients?
   - A. Sampling
   - B. Tool
   - C. Elicitation
   - D. Resource
   - E. Checkpoint

   <details><summary>Answer</summary>

   **B (Tool) and D (Resource)** are correct. Tools are executable functions invoked via `tools/call`; Resources are read-only data sources fetched via `resources/read`. Both are primitives that *servers* expose to clients. A (Sampling) and C (Elicitation) are primitives that *clients* expose to servers (they allow servers to request LLM completions or additional user input), which is the reverse direction — a common distractor. E (Checkpoint) is a LangGraph concept, not an MCP primitive.

   </details>

4. An agent must answer questions about a company's sales data using natural language. The data lives in Databricks and the team wants Unity Catalog to govern who can query it. Which Databricks managed MCP server is the best fit, and why?

   <details><summary>Answer</summary>

   The **Genie Agent MCP server** (`/api/2.0/mcp/genie/{genie_space_id}`) is the best fit. Genie accepts natural-language questions, translates them into SQL, and returns results from the underlying Delta tables — all within Unity Catalog's permission model, so access is automatically gated by the calling user's grants. The Databricks SQL MCP server is read/write and designed for AI-generated SQL from coding tools, not natural-language analytics. AI Search is for unstructured document retrieval, not structured tabular data queries. UC functions MCP server runs predefined SQL functions and would require pre-authoring every possible query as a function — not suitable for open-ended natural-language questions.

   </details>

5. A team is debating whether to deploy their customer-service agent on Databricks Model Serving (stateless endpoint) or Databricks Apps (stateful process). The agent needs to remember the customer's name and open ticket number across a 10-message conversation, but each conversation is independent — no cross-session memory is needed. Which deployment is more appropriate, and what is the key trade-off?

   <details><summary>Answer</summary>

   **Databricks Apps** is more appropriate for this use case. Because the agent needs per-session conversational memory (customer name, ticket number across 10 turns), it needs a checkpointer that survives multiple API calls within the same session. Databricks Apps manages the server process lifecycle and allows attaching a durable checkpointer. The key trade-off: Databricks Apps carries higher operational complexity (process management, app lifecycle) and is priced by app compute, while Model Serving is a fully managed, auto-scaling stateless endpoint priced per token/request. A hybrid pattern — Model Serving + an external Redis or Postgres checkpointer — can achieve both scalability and per-session memory, at the cost of additional infrastructure to maintain. For a team without that infrastructure, Databricks Apps is the simpler path to per-session memory.

   </details>

---

## Further Reading

- [Model Context Protocol — Introduction](https://modelcontextprotocol.io/introduction) — *verified 2026-07-11* — Official MCP overview, primitives, and ecosystem
- [MCP Architecture Overview](https://modelcontextprotocol.io/docs/concepts/architecture) — *verified 2026-07-11* — Host/Client/Server model, data layer, transport layer, and JSON-RPC lifecycle
- [MCP Transports Specification](https://modelcontextprotocol.io/docs/concepts/transports) — *verified 2026-07-11* — stdio and Streamable HTTP transport details, session management, backwards compatibility
- [Model Context Protocol (MCP) on Databricks](https://docs.databricks.com/aws/en/generative-ai/agent-framework/mcp.html) — *verified 2026-07-11* — Databricks MCP overview, Unity AI Gateway, managed vs external vs custom servers
- [Use MCP servers in agents — Databricks](https://docs.databricks.com/aws/en/agents/mcp/use-mcp-in-agents.html) — *verified 2026-07-11* — `databricks-mcp` library, LangGraph and OpenAI Agents SDK integration code, deployment patterns
- [Databricks Managed MCP Servers](https://docs.databricks.com/aws/en/agents/mcp/managed-mcp.html) — *verified 2026-07-11* — Available managed servers (Genie, AI Search, SQL, UC functions), OAuth scopes, URL patterns
