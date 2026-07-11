# LangGraph Fundamentals for Stateful Agent Development

**Section:** 04 — Application Development | **Module:** 01 — Frameworks and Prompting | **Est. time:** 3 hrs | **Exam mapping:** Application Development (30%) — Build agents using LangGraph

---

## TL;DR

LangGraph models agent workflows as directed graphs where nodes are Python functions that read and write a shared `TypedDict` state object, and edges determine routing between nodes — including conditional loops back to earlier nodes until a terminal condition is met. A checkpointer persists the state snapshot at every super-step, enabling multi-turn conversation memory, fault-tolerant resumption, and human-in-the-loop workflows across stateless API calls. **The one thing to remember: state is a shared dict that every node can read and mutate; edges route based on that state; a checkpointer makes the whole sequence durable.**

---

## ELI5 — Explain It Like I'm 5

Imagine a customer service call centre with a paper form that gets passed from desk to desk. The form starts blank, and each person at each desk reads what is already written, adds their part, and passes it to the next desk or back to an earlier desk if more work is needed. When the form is finally done, it gets filed in a cabinet (the checkpointer) so the next time that customer calls, you can pull the form out and pick up exactly where you left off. That is LangGraph: the form is the `TypedDict` state, the desks are nodes, the routing rules are edges, and the filing cabinet is the checkpointer. The most common misconception is that nodes communicate by calling each other like functions — they do not; every node reads from and writes to the shared form, and the graph runtime handles passing the form along.

---

## Learning Objectives

By the end of this chapter you will be able to:
- [ ] Define a `TypedDict` state schema and wire it into a `StateGraph`, including reducers for accumulating values
- [ ] Add nodes and static or conditional edges using `add_node`, `add_edge`, and `add_conditional_edges`
- [ ] Attach a `MemorySaver` or `SqliteSaver` checkpointer to a compiled graph and invoke it with a `thread_id` for cross-turn memory
- [ ] Stream graph execution with `.stream()` and distinguish `updates` mode from `values` mode
- [ ] Diagnose and fix the three most common LangGraph pitfalls: missing `thread_id`, unbounded recursion, and global variable state

---

## Visual Overview

### StateGraph Execution Flow

```
User input
    │
    ▼
┌─────────────────────────────────────────────┐
│  START (virtual entry node)                 │
└──────────────────────┬──────────────────────┘
                       │  initial state dict
                       ▼
              ┌────────────────┐
              │   Node A       │  reads state, returns partial update
              │  (e.g. call    │
              │   LLM / tools) │
              └────────┬───────┘
                       │  updated state
                       ▼
              ┌────────────────┐
              │   Node B       │  reads state, returns partial update
              └────────┬───────┘
                       │
                       ▼
              ┌─────────────────┐
              │  END (terminal) │
              └─────────────────┘
                       │
                       ▼
              Final state returned to caller

Key: Every node receives the FULL current state.
     Every node returns ONLY the keys it changes.
     The graph merges the update into state before passing it forward.
```

### Conditional Edge Routing

```
                 ┌──────────────────────┐
                 │     agent node       │
                 │  (LLM decides to     │
                 │   call a tool or     │
                 │   produce a final    │
                 │   answer)            │
                 └──────────┬───────────┘
                            │
                state["tool_calls"] present?
                            │
               ┌────────────┴──────────┐
         Yes ──►                        ◄── No
               │                        │
               ▼                        ▼
       ┌───────────────┐        ┌──────────────┐
       │  tool_node    │        │     END      │
       │  (executes    │        │  (return     │
       │   tool calls, │        │   final ans) │
       │   appends     │        └──────────────┘
       │   results)    │
       └───────┬───────┘
               │
               └──────────────────► agent node
                              (loop back for next turn)
```

### Checkpointing Thread Model

```
Thread "user-42"                     Checkpointer (InMemorySaver / SqliteSaver)
─────────────────────                ─────────────────────────────────────────
Turn 1: invoke({"messages":[m1]})
   └──► super-step 1 ─────────────────► checkpoint_ns="" id="abc1" state={msgs:[m1,a1]}
   └──► super-step 2 ─────────────────► checkpoint_ns="" id="abc2" state={msgs:[m1,a1,t1,a2]}
   └──► END

Turn 2: invoke({"messages":[m2]})
   └── graph loads checkpoint abc2 ◄── get_tuple(thread_id="user-42")
   └──► super-step 3 ─────────────────► checkpoint_ns="" id="abc3" state={msgs:[...,m2,a3]}
   └──► END

Turn 3: invoke({"messages":[m3]})  ←  same pattern; state grows with conversation

On process restart: checkpointer (SqliteSaver) still has abc1,abc2,abc3 on disk
InMemorySaver: lost on process restart — use SqliteSaver for dev persistence
```

---

## Key Concepts

### StateGraph and TypedDict State

**What is it?**
`StateGraph` is LangGraph's primary graph class, parameterized by a user-defined state schema — typically a `TypedDict` or `Pydantic` model. The state is a single shared dictionary that all nodes in the graph read from and write partial updates back to. Reducers defined on state fields (via `Annotated`) control how updates are merged rather than simply overwritten.

**How does it work under the hood?**
When you call `StateGraph(MyState)`, LangGraph registers a channel for each key in `MyState`. As nodes execute, each returns a partial dict (only the keys it modified). LangGraph calls each key's reducer — by default a simple overwrite, but with `Annotated[list, operator.add]` it becomes an append. After all nodes in a super-step complete, the merged state becomes the input to the next super-step. This is the Pregel-style message-passing model: nodes communicate only through the shared state, never by directly calling each other.

**Where does it appear in the Databricks/framework ecosystem?**
In Databricks agent code, `StateGraph` is the foundation of every custom LangGraph agent. The Databricks `create_react_agent` helper from `langgraph.prebuilt` wraps a `StateGraph` internally. When registering an agent with MLflow for deployment on Databricks, the compiled graph is wrapped in an MLflow `ResponsesAgent` — but the graph itself is a plain `StateGraph` under the hood. API: `from langgraph.graph import StateGraph, MessagesState, START, END`.

---

### Nodes and Edges

**What is it?**
A **node** is a Python function (sync or async) that accepts the current state and returns a partial state update dict. An **edge** is a directed connection from one node to another. A **conditional edge** is an edge whose destination is determined at runtime by a routing function that inspects the current state and returns a node name (or `END`).

**How does it work under the hood?**
Nodes are registered with `graph.add_node("name", fn)`. LangGraph wraps each function in a `RunnableLambda`, adding batch and async support automatically. Static edges (`add_edge`) create a fixed arc in the graph's adjacency list — always followed. Conditional edges (`add_conditional_edges`) call a router function after the source node finishes; the return value is looked up in an optional mapping dict to resolve to a final node name. If the router returns a list of names, all listed nodes execute in parallel as part of the same super-step. The `START` and `END` constants are virtual nodes that represent the entry and exit points of the graph.

**Where does it appear in the Databricks/framework ecosystem?**
- `graph.add_node("agent", agent_fn)` — registers a node
- `graph.add_edge(START, "agent")` — entry point
- `graph.add_edge("tools", "agent")` — static arc (always loop back)
- `graph.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})` — conditional routing
- In the `create_react_agent` prebuilt, a conditional edge named `should_continue` decides whether to call tools or terminate based on whether the last message contains tool calls.

---

### Checkpointing and Memory

**What is it?**
A checkpointer is a persistence backend attached to a compiled graph that saves a full snapshot of the graph state (a `StateSnapshot`) after every super-step. This enables multi-turn conversations (the graph can reload prior state by `thread_id`), fault-tolerant execution (resume from the last successful checkpoint), human-in-the-loop workflows (inspect and modify state mid-execution), and time-travel debugging.

**How does it work under the hood?**
LangGraph provides three built-in checkpointer classes: `InMemorySaver` (in-RAM, lost on restart), `SqliteSaver` (file-backed SQLite, for local dev persistence), and `PostgresSaver` (production-grade). Checkpointers implement the `BaseCheckpointSaver` interface with `.put`, `.put_writes`, `.get_tuple`, and `.list` methods. When you invoke the graph with a config containing `{"configurable": {"thread_id": "..."}}`, the runtime calls `get_tuple` to load the latest checkpoint for that thread, resumes from it, and calls `put` after each super-step to persist the new snapshot. Without a `thread_id`, the graph treats each invocation as a fresh conversation with no prior state.

**Where does it appear in the Databricks/framework ecosystem?**
- `from langgraph.checkpoint.memory import InMemorySaver` — in-process dev testing
- `from langgraph.checkpoint.sqlite import SqliteSaver` — local file persistence
- `graph = builder.compile(checkpointer=InMemorySaver())` — attach at compile time
- `graph.invoke(inputs, config={"configurable": {"thread_id": "user-42"}})` — resume/create thread
- On Databricks: when deploying via MLflow `ResponsesAgent`, the Agent Server handles checkpointing infrastructure automatically; custom backends are used for on-prem or custom workspaces.

---

### The `START` and `END` Nodes

**What is it?**
`START` and `END` are virtual node constants imported from `langgraph.graph`. They are not real Python functions — they are sentinel values used as edge endpoints to mark the entry and exit points of the graph.

**How does it work under the hood?**
When you call `graph.add_edge(START, "first_node")`, LangGraph records that `first_node` should receive the initial user input when `invoke` or `stream` is called. When a conditional edge router returns `END` (or a node has a static edge to `END`), the graph marks that execution path as complete. A super-step in which all active nodes have no further destinations causes the graph to halt. If a node has no outgoing edge and is not `END`, the graph raises a validation error at compile time.

**Where does it appear in the Databricks/framework ecosystem?**
- `from langgraph.graph import START, END` — universal import
- Entry point: `builder.add_edge(START, "retrieve")` — where the graph begins
- Terminal condition: `builder.add_edge("validate", END)` — where it stops
- Conditional exit: `{"end": END}` — mapping value in `add_conditional_edges`

---

### Streaming Graph Execution

**What is it?**
`.stream()` (sync) and `.astream()` (async) are methods on a compiled graph that yield intermediate state information after each super-step instead of blocking until the entire graph finishes. This is essential for observability: you can see which node ran, what it returned, and what the accumulated state looks like at each step.

**How does it work under the hood?**
LangGraph supports several stream modes selectable via the `stream_mode` parameter. The two most important for agent observability are: `"updates"` (yields only the dict keys changed by each node — low bandwidth) and `"values"` (yields the full state after each node — easier to debug). With `version="v2"` (LangGraph ≥ 1.1), every chunk is a `StreamPart` dict with `{"type": ..., "ns": ..., "data": ...}`, enabling type-safe filtering. The `"messages"` stream mode yields `(token, metadata)` tuples for token-by-token LLM output — critical for streaming chat UIs.

**Where does it appear in the Databricks/framework ecosystem?**
- `for chunk in graph.stream(inputs, stream_mode="updates"):` — local dev
- `for chunk in graph.stream(inputs, config=cfg, stream_mode="messages", version="v2"):` — streaming tokens with thread memory
- Databricks `ResponsesAgent` streaming: agent emits `output_text.delta` events using the same underlying graph `.astream()` mechanism, surfaced to the Databricks AI Playground and chat UI
- `graph.get_state(config)` — inspect latest checkpoint state outside a stream

---

## Key Parameters / Configuration Knobs

| Parameter | What it controls | Decision rule |
|---|---|---|
| `StateGraph(state_schema)` | The schema of all state channels in the graph | Use `TypedDict` for most cases; use `MessagesState` (prebuilt subclass) when your only state is a `messages` list; use `Pydantic BaseModel` when you need runtime field validation at the cost of ~10–20% extra overhead. |
| `checkpointer` (compile arg) | Persistence backend for state snapshots | Use `InMemorySaver` for unit tests and demos; use `SqliteSaver` for local dev where you need cross-session persistence; use `PostgresSaver` for production. Never use `InMemorySaver` if the process might restart between user turns. |
| `thread_id` in `config["configurable"]` | Identifies which conversation thread to load/save | Always set a stable, per-conversation UUID; if absent, the graph runs stateless (no prior turns loaded). Use `str(uuid.uuid4())` at session start and store it client-side. |
| `recursion_limit` in `config` | Maximum number of super-steps before `GraphRecursionError` | Default is 1000. For agentic loops (agent ↔ tools), set to 25–50 to prevent runaway loops during development. For production long-running agents, use `RemainingSteps` managed value for graceful degradation instead of relying on the hard limit. |
| `stream_mode` (invoke/stream arg) | What data each streaming chunk contains | Use `"updates"` in production (low bandwidth, per-node diffs); use `"values"` during debugging (full state snapshot every step); use `"messages"` for real-time token streaming to a chat UI. |

---

## Worked Example: Requirement → Decision

**Given:**
You are building a customer support agent for an e-commerce platform. The agent must: (1) look up an order by order ID, (2) decide whether the issue can be auto-resolved (e.g., refund under $50) or must be escalated to a human, and (3) either send a resolution message or create an escalation ticket — then loop back if the LLM needs to call another tool. Conversations must persist so a customer can resume a chat after closing their browser.

**Step 1 — Identify the goal:**
Build a stateful, multi-turn LangGraph agent that handles both auto-resolution and escalation paths, with full conversation memory across browser sessions.

**Step 2 — Define inputs:**
- Customer messages (list of LangChain `BaseMessage` objects)
- Order lookup tool (returns order details given an order ID)
- Escalation tool (creates a ticket and returns a ticket ID)
- Refund tool (processes a refund and returns confirmation)

**Step 3 — Define outputs:**
- Updated conversation history with assistant response
- Order resolution status (resolved / escalated) stored in state
- Persisted state so the next turn can continue where this one left off

**Step 4 — Apply constraints:**
- **Memory:** must survive process restarts → `SqliteSaver`, not `InMemorySaver`
- **Looping:** agent may call tools multiple times before responding → need a conditional edge back to the agent node after tools execute
- **Termination:** agent must stop looping when it emits a final message with no tool calls → `END` when no tool calls present
- **Recursion guard:** set `recursion_limit=30` to prevent runaway tool loops in dev

**Step 5 — Select the approach with rationale vs alternatives:**
**Chosen: `StateGraph` with `MessagesState` + `SqliteSaver` + conditional edge routing.** Using `MessagesState` (which has the `add_messages` reducer built in) avoids manually defining the messages channel. The conditional edge after the `agent` node checks `state["messages"][-1].tool_calls` — if non-empty, route to `tools`; if empty, route to `END`. This is identical to the `create_react_agent` prebuilt pattern but explicit, which makes the routing logic auditable.

**Why not alternatives:**
- **`create_react_agent` prebuilt without customization:** Insufficient because we need custom state fields (`order_status`, `ticket_id`) that `MessagesState` alone does not expose.
- **Sequential chain (no graph):** Cannot implement the tool-call loop without graph-level conditional edges; a chain always executes the same steps in the same order.

---

## Implementation

### Snippet 1: Complete minimal StateGraph with checkpointing

```python
# Scenario: Build a stateful customer support agent that persists conversation
# history across sessions using SqliteSaver, so customers can resume chats
# after closing their browser without losing context.

import sqlite3
import operator
from typing import Annotated, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

# ── 1. Define state schema ──────────────────────────────────────────────────
class SupportState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]  # accumulate messages
    order_status: str   # "pending" | "resolved" | "escalated"
    ticket_id: str      # filled when escalated

# ── 2. Define tools ─────────────────────────────────────────────────────────
@tool
def lookup_order(order_id: str) -> str:
    """Look up order details by order ID."""
    # In production: query Databricks SQL warehouse or Unity Catalog
    return f"Order {order_id}: shipped 2026-07-01, total $35.00, status=delivered"

@tool
def process_refund(order_id: str, amount: float) -> str:
    """Process a refund for an order."""
    return f"Refund of ${amount} processed for order {order_id}. Confirmation: REF-999"

@tool
def escalate_to_human(reason: str) -> str:
    """Create a support ticket for human review."""
    return f"Ticket TKT-1234 created: {reason}"

tools = [lookup_order, process_refund, escalate_to_human]

# ── 3. Define nodes ─────────────────────────────────────────────────────────
from langchain_core.language_models import BaseChatModel

def make_agent_node(model: BaseChatModel):
    """Factory: bind tools to the model and return a node function."""
    model_with_tools = model.bind_tools(tools)

    def agent_node(state: SupportState) -> dict:
        response = model_with_tools.invoke(state["messages"])
        return {"messages": [response]}  # appended via operator.add reducer

    return agent_node

def tool_node(state: SupportState) -> dict:
    """Execute all tool calls in the last AIMessage."""
    last_msg = state["messages"][-1]
    tool_results = []
    tool_map = {t.name: t for t in tools}

    for tc in last_msg.tool_calls:
        result = tool_map[tc["name"]].invoke(tc["args"])
        tool_results.append(ToolMessage(content=result, tool_call_id=tc["id"]))

    return {"messages": tool_results}

def should_continue(state: SupportState) -> Literal["tools", "end"]:
    """Route: if last message has tool calls, go to tool_node; else terminate."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"

# ── 4. Build and compile the graph ──────────────────────────────────────────
def build_support_agent(model: BaseChatModel) -> object:
    builder = StateGraph(SupportState)

    builder.add_node("agent", make_agent_node(model))
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END}   # mapping: router output → node name
    )
    builder.add_edge("tools", "agent")   # always loop back after tool execution

    # Attach SqliteSaver for cross-session persistence
    conn = sqlite3.connect("support_agent.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    return builder.compile(checkpointer=checkpointer)

# ── 5. Invoke with thread_id for persistent memory ──────────────────────────
# from langchain_anthropic import ChatAnthropic
# model = ChatAnthropic(model="claude-sonnet-4-6")
# graph = build_support_agent(model)

config = {"configurable": {"thread_id": "customer-uuid-1234"}}

# Turn 1
# result = graph.invoke(
#     {"messages": [HumanMessage(content="My order ORD-789 hasn't arrived")],
#      "order_status": "pending", "ticket_id": ""},
#     config=config,
#     recursion_limit=30
# )

# Turn 2 (same thread_id — state resumes automatically from Turn 1)
# result = graph.invoke(
#     {"messages": [HumanMessage(content="Can I get a refund?")]},
#     config=config
# )
```

---

### Snippet 2: Streaming invocation for real-time observability

```python
# Scenario: Stream a graph's execution step-by-step to log which node ran
# and what state it produced — essential for debugging agentic loops in
# staging and for streaming responses to a chat UI in production.

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END

# (Re-use the graph from Snippet 1 or build a simple demo graph here)

def stream_agent_turn(graph, user_message: str, thread_id: str):
    """Stream a single agent turn, printing each node's output as it arrives."""
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": [{"role": "user", "content": user_message}]}

    print(f"\n=== Turn: {user_message[:50]}... ===")

    # stream_mode="updates" → yields {node_name: partial_state_update}
    for chunk in graph.stream(inputs, config=config, stream_mode="updates"):
        for node_name, state_update in chunk.items():
            if node_name == "__end__":
                continue
            messages = state_update.get("messages", [])
            if messages:
                last = messages[-1]
                content = getattr(last, "content", str(last))
                print(f"[{node_name}] {content[:120]}")

    # Inspect full state after the turn
    snapshot = graph.get_state(config)
    print(f"\nMessages in thread: {len(snapshot.values.get('messages', []))}")
    print(f"Next nodes: {snapshot.next}")  # () means the graph finished cleanly

# Token-by-token streaming for chat UI (messages mode)
def stream_tokens(graph, user_message: str, thread_id: str):
    """Stream LLM tokens as they are generated — for real-time chat UI."""
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": [{"role": "user", "content": user_message}]}

    for chunk in graph.stream(
        inputs, config=config,
        stream_mode="messages",   # yields (token_chunk, metadata) tuples
        version="v2"
    ):
        if chunk["type"] == "messages":
            msg_chunk, metadata = chunk["data"]
            if msg_chunk.content:
                print(msg_chunk.content, end="", flush=True)
    print()  # newline after streaming
```

---

### Snippet 3 (Anti-pattern): Using global variables for state

```python
# Anti-pattern: Storing agent state in a global Python variable instead of
# in the LangGraph TypedDict state. This breaks under concurrent requests,
# loses state across process restarts, and makes testing impossible.

# ── WRONG: Global variable "state" ──────────────────────────────────────────
_global_conversation = []  # Anti-pattern: shared mutable global

def agent_node_wrong(state):  # ignores the actual LangGraph state
    _global_conversation.append(state["messages"][-1])  # mutation of global
    # Problems:
    # 1. Two concurrent users share the same _global_conversation → data corruption
    # 2. On process restart, _global_conversation is empty → lost memory
    # 3. Unit tests cannot inject a clean state → test pollution
    # 4. Checkpointer cannot persist or restore a global variable
    response = AIMessage(content="Hello from broken agent")
    _global_conversation.append(response)
    return {"messages": [response]}

# ── CORRECT: TypedDict state passed through the graph ──────────────────────
# The correct approach is already shown in Snippet 1. The key mental model:
# - Each invocation receives a FRESH copy of the state for that thread
# - Nodes return PARTIAL dicts; LangGraph merges them via reducers
# - The checkpointer saves and restores the FULL state dict per thread_id
# - There is ZERO shared mutable state between concurrent graph invocations

def agent_node_correct(state: SupportState) -> dict:
    # ✓ Reads from state (thread-safe, isolated per thread_id)
    last_user_msg = state["messages"][-1].content
    response = AIMessage(content=f"I see your message: {last_user_msg}")
    return {"messages": [response]}  # ✓ Returns partial update; graph merges it

# What breaks with global variables:
# - Race condition: request A writes to _global_conversation at the same time as request B → interleaved messages
# - Stale data: user 2 sees user 1's conversation history
# - Checkpointing fails silently: InMemorySaver saves graph state, not global variables
# - Testing: you must reset _global_conversation between tests manually — easy to forget
```

---

## Common Pitfalls & Misconceptions

- **"I can skip `thread_id` since I only have one user"** — Beginners pass no config to `graph.invoke()` because they are testing locally with a single user. The graph runs without error, but every invocation starts a fresh conversation with no prior messages loaded, even if a checkpointer is attached. The correct mental model: a checkpointer without a `thread_id` is inert — it saves state but has no key to retrieve it by; always pass `config={"configurable": {"thread_id": "some-stable-id"}}`.

- **"Nodes communicate by calling each other"** — Coming from sequential chain patterns, beginners write nodes that directly invoke other node functions. This bypasses the graph routing logic and the checkpointer entirely, producing unbounded recursion and no persisted state. The correct mental model: nodes are isolated functions; they only read and write `state`; the graph runtime is the only entity that calls nodes.

- **"InMemorySaver is fine for production"** — Beginners use `InMemorySaver` because it requires no setup and the demo works perfectly. But `InMemorySaver` stores checkpoints in RAM — they vanish on any process restart, container redeploy, or serverless cold start. The correct mental model: use `InMemorySaver` only in unit tests and one-shot demos; use `SqliteSaver` for local development with cross-session persistence; use `PostgresSaver` for production.

- **"I need to return the full state dict from each node"** — Beginners return the entire state object from every node, worried that unreturned keys will be lost. This overwrites all existing state with the node's values, destroying data written by previous nodes. The correct mental model: return only the keys you changed; LangGraph calls the appropriate reducer to merge your partial update into the existing state.

- **"The graph will stop automatically when the agent says it is done"** — Beginners write an agent that produces a final answer but forget to add a conditional edge that routes to `END`. The graph loops infinitely (until `GraphRecursionError`) because there is always an edge back to the agent or tool node. The correct mental model: the graph terminates only when a node has a static edge to `END` or a conditional router returns `END`; you must explicitly design the termination condition.

---

## Key Definitions

| Term | Definition |
|---|---|
| **StateGraph** | The primary LangGraph class for building stateful agent workflows. Parameterized by a `TypedDict` or Pydantic model that defines all state channels. Nodes and edges are registered on a `StateGraph` instance before calling `.compile()`. |
| **State channel** | A single key in the `TypedDict` state schema. Each channel has an associated reducer function (default: overwrite). Channels are the only mechanism by which nodes share information. |
| **Reducer** | A binary function `(current_value, update) → new_value` that merges a node's partial state update into the existing channel value. Default reducer: overwrite. Common custom reducer: `operator.add` for list accumulation. |
| **Node** | A Python function registered with `add_node` that accepts the current state and returns a partial state update dict. Can be sync or async. |
| **Edge** | A directed connection between two nodes. Static edges (`add_edge`) always fire; conditional edges (`add_conditional_edges`) call a routing function to determine the destination at runtime. |
| **Checkpointer** | A persistence backend (`InMemorySaver`, `SqliteSaver`, `PostgresSaver`) attached at compile time. Saves a full state snapshot after every super-step. Enables multi-turn memory and fault-tolerant resumption via `thread_id`. |
| **Thread** | A named conversation context identified by a `thread_id` string. The checkpointer stores all checkpoints for a thread keyed by this ID. Different `thread_id` values are completely isolated conversations. |
| **Super-step** | One "tick" of graph execution: all nodes scheduled for that step run (potentially in parallel), then their state updates are merged. The checkpointer saves one checkpoint per super-step. |
| **START** | Virtual entry node constant (`from langgraph.graph import START`). Add an edge from `START` to the first real node to define the graph's entry point. |
| **END** | Virtual terminal node constant (`from langgraph.graph import END`). Routing to `END` halts that execution path. When all active paths reach `END`, the graph finishes. |
| **recursion_limit** | A config key (not inside `configurable`) that caps the number of super-steps. Raises `GraphRecursionError` when exceeded. Default: 1000. |

---

## Summary / Quick Recall

1. **`StateGraph` + `TypedDict` is the core pattern** — define all agent state in one schema, attach reducers via `Annotated` for fields that accumulate values, and every node reads/writes that same schema.
2. **Nodes return partial dicts, not full state** — only return the keys you changed; LangGraph merges them with the existing state via reducers.
3. **Conditional edges power agentic loops** — `add_conditional_edges("agent", router_fn, {...})` is how you implement the "call tools then loop back" pattern; always include `END` as one possible routing destination.
4. **Checkpointer + `thread_id` = multi-turn memory** — compiling with a checkpointer and passing a stable `thread_id` per conversation is all that is needed for cross-session persistence; no manual state management required.
5. **Use `SqliteSaver` for local dev, `PostgresSaver` for production** — `InMemorySaver` data disappears on restart; plan for this from the first day of development.
6. **`.stream(stream_mode="updates")` for production observability** — yields per-node state diffs with minimal bandwidth; upgrade to `"messages"` mode only for streaming tokens to a chat UI.
7. **`recursion_limit` is a safety valve, not a design strategy** — design an explicit `END` termination condition; use `RemainingSteps` for graceful degradation when approaching the limit.

---

## Self-Check Questions

1. What is the default reducer behaviour for a `TypedDict` state field that has no `Annotated` type hint?

   <details><summary>Answer</summary>

   The default reducer is **overwrite**: when a node returns a new value for that key, it completely replaces the previous value. For example, if `state["order_status"] = "pending"` and a node returns `{"order_status": "resolved"}`, the new state has `order_status == "resolved"` — the old value is gone.

   **Why the most tempting wrong answer fails:** Many candidates think the default is to append or merge values. Appending only happens when you explicitly annotate the field with a reducer such as `Annotated[list, operator.add]`. Without the annotation, every write to that key is a full replacement.

   </details>

2. You have a LangGraph agent compiled with `InMemorySaver`. After restarting your Python process and re-creating the graph, you invoke it with the same `thread_id` as a previous session. What happens?

   <details><summary>Answer</summary>

   The graph starts a **completely fresh conversation** with no prior messages — it behaves as if the `thread_id` is new. `InMemorySaver` stores checkpoints in the process's RAM, which is wiped on restart. The `thread_id` in the new process's `InMemorySaver` has no checkpoints, so `get_tuple` returns `None` and the graph initialises from scratch.

   **Why the most tempting wrong answer fails:** Candidates assume that re-using the same `thread_id` string is sufficient to restore memory. The `thread_id` is just a lookup key — if the storage backend (RAM) no longer contains the data, the key resolves to nothing. For cross-restart persistence, use `SqliteSaver` (writes to disk) or `PostgresSaver`.

   </details>

3. **Which TWO** of the following are correct ways to implement a conditional loop in a LangGraph agent that calls tools and returns to the agent node after each tool execution?

   - A. Return the name of the previous node from the routing function passed to `add_conditional_edges`
   - B. Call `add_edge("tools", "agent")` to create a static arc from the tool node back to the agent node
   - C. In the routing function, return `"agent"` when tool calls are present; add `add_conditional_edges("agent", router, {"agent": "tools", "end": END})`
   - D. Set `recursion_limit=1` so the graph automatically loops back to the agent after each step
   - E. Call `graph.compile(loop=True)` to enable looping behaviour

   <details><summary>Answer</summary>

   **Correct: B and C.**

   **B is correct** because a static edge from `"tools"` to `"agent"` (`add_edge("tools", "agent")`) unconditionally routes execution back to the agent node after every tool call. This is the standard pattern in the `create_react_agent` prebuilt: after tools execute, always return control to the agent.

   **C is correct** because using `add_conditional_edges` on the agent node with a router that returns `"agent"` (or maps to `"tools"`) when tool calls are present, combined with an `END` exit, creates the full agent–tools loop. In practice, the routing function returns the name of the *next* node (e.g., `"tools"` not `"agent"`), then the static `add_edge("tools", "agent")` completes the loop.

   **Why A is wrong:** The routing function in `add_conditional_edges` specifies the *next* node to visit, not the previous one. Returning the source node name from the router would create an edge directly from the agent node to itself, skipping the tool node entirely.

   **Why D is wrong:** `recursion_limit` is a safety cap on super-steps, not a looping mechanism. Setting it to 1 would cause `GraphRecursionError` after the first step.

   **Why E is wrong:** There is no `loop=True` parameter on `compile()`. Looping is achieved through graph structure (edges back to earlier nodes), not a compile flag.

   </details>

4. Your production LangGraph agent is raising `GraphRecursionError` intermittently. You examine the logs and see the graph bouncing between the `agent` node and the `tools` node hundreds of times before failing. What is the most likely root cause, and what is the correct fix?

   <details><summary>Answer</summary>

   **Root cause:** The conditional router after the `agent` node is never returning `END` (or routing to a terminal node). This happens when: (a) the LLM is stuck in a tool-calling loop because the tool results do not satisfy the stopping condition the LLM was prompted with, or (b) the router function checks for a condition (e.g., `state["resolved"] == True`) that the agent never sets because the LLM did not produce the expected output.

   **Correct fix (two-part):**
   1. **Prompt engineering**: Update the system prompt to clearly instruct the LLM to stop calling tools and emit a final text response when it has enough information, rather than continuing to call tools for marginal information.
   2. **Defensive routing**: Use `RemainingSteps` (a managed LangGraph value) in the state and add a check in the router: if `state["remaining_steps"] <= 2`, route to `END` or a `fallback_node` that generates a graceful "I couldn't fully resolve this" response. This prevents a hard `GraphRecursionError` and gives users a meaningful response.

   **Why just raising `recursion_limit` is wrong:** Increasing the limit is a band-aid — it delays the error but does not fix the underlying loop. The graph will still fail; it just takes longer.

   **Why just catching `GraphRecursionError` externally is suboptimal:** External catching terminates the graph abruptly, losing all intermediate state and giving the user no response. The `RemainingSteps` approach allows the graph to complete gracefully with a partial result.

   </details>

5. You are comparing two approaches for a Databricks production agent: (A) `InMemorySaver` with a UUID `thread_id` per request (stateless, fresh UUID each call), versus (B) `SqliteSaver` with a stable per-user UUID `thread_id`. For a customer support chatbot where users ask follow-up questions across multiple browser sessions, which approach is correct and what is the key trade-off?

   <details><summary>Answer</summary>

   **Correct approach: B — `SqliteSaver` with a stable per-user UUID.**

   **Why B is correct:** A customer support chatbot requires cross-session memory — when a user asks "What about my refund from yesterday?" in a new session, the agent must have access to the previous conversation. `SqliteSaver` persists checkpoints to a file on disk, so the state survives process restarts and new browser sessions. The stable per-user UUID ensures the same conversation thread is resumed.

   **Why A fails for this use case:** Using a fresh UUID per request (approach A) makes every call a brand-new conversation with no prior context. Even with `InMemorySaver`, using a new `thread_id` each time means the checkpointer never has anything to load — the "memory" is structurally impossible. This pattern is correct only for stateless single-turn agents (e.g., a one-shot document classifier).

   **Key trade-off to articulate:** `SqliteSaver` introduces disk I/O on every super-step (`.put` write per checkpoint), adding ~1–5ms latency per step compared to in-memory. For a customer support agent with 3–10 turns per session, this overhead is negligible relative to LLM call latency (hundreds of ms). For production at scale with high concurrency, `PostgresSaver` is preferred because SQLite does not handle concurrent writes well.

   </details>

---

## Further Reading

- [LangGraph Overview — LangChain Documentation](https://docs.langchain.com/oss/python/langgraph/overview) — *verified 2026-07-11* — Authoritative overview of LangGraph's role as an orchestration runtime; covers persistence, streaming, and human-in-the-loop.
- [LangGraph Graph API Overview](https://docs.langchain.com/oss/python/langgraph/graph-api) — *verified 2026-07-11* — Complete reference for `StateGraph`, nodes, edges, reducers, `START`/`END`, `Command`, and `Send`.
- [LangGraph Persistence](https://docs.langchain.com/oss/python/langgraph/persistence) — *verified 2026-07-11* — Full documentation on checkpointers, thread model, `InMemorySaver`, `SqliteSaver`, and `PostgresSaver`.
- [LangGraph Streaming](https://docs.langchain.com/oss/python/langgraph/streaming) — *verified 2026-07-11* — Comprehensive guide to stream modes (`updates`, `values`, `messages`, `custom`), v2 format, and token streaming.
- [LangGraph Checkpointers Reference](https://docs.langchain.com/oss/python/langgraph/checkpointers) — *verified 2026-07-11* — Detailed reference for checkpointer libraries, `BaseCheckpointSaver` interface, durability modes, and building custom backends.
- [Use Agents on Databricks](https://docs.databricks.com/aws/en/agents/agent-framework/build-agents) — *verified 2026-07-11* — Official Databricks guide to building and deploying agents with LangGraph, LangChain, OpenAI, and LlamaIndex on Databricks Apps.
- [Author an AI Agent on Databricks Apps](https://docs.databricks.com/en/agents/agent-framework/author-agent.html) — *verified 2026-07-11* — Step-by-step guide to wrapping LangGraph agents in `ResponsesAgent`, deploying with Databricks Apps, and integrating Unity AI Gateway.
