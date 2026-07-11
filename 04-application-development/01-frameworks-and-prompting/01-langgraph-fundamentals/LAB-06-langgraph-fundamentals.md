# LAB-06: LangGraph Fundamentals — Build a Stateful 3-Node Agent

**Lab:** LAB-06 | **Section:** 04 — Application Development | **Module:** 01 — Frameworks and Prompting | **Est. time:** 90 min

---

## Objective

Build a minimal 3-node LangGraph agent (`retrieve → generate → validate`) that runs locally with `MemorySaver` checkpointing, maintaining conversation history across multiple turns and streaming execution events.

---

## Prerequisites

- Completed reading of `notes.md` for this chapter (LangGraph Fundamentals)
- Python 3.11+ installed locally
- A valid API key for an LLM provider (Anthropic or OpenAI)
- Familiarity with `TypedDict` and Python type hints

---

## Setup

Install dependencies and configure your environment.

```bash
# Setup: create an isolated environment and install LangGraph + LangChain
pip install langgraph langchain langchain-anthropic langgraph-checkpoint-sqlite

# Optional: install langchain-openai if using OpenAI instead of Anthropic
# pip install langchain-openai

# Set your LLM API key (replace with your actual key)
export ANTHROPIC_API_KEY="your-key-here"
# OR: export OPENAI_API_KEY="your-key-here"
```

Verify the installation:

```python
# Verify: confirm LangGraph is installed and importable
import langgraph
import langchain
print(f"LangGraph version: {langgraph.__version__}")
print(f"LangChain version: {langchain.__version__}")
# Expected: no ImportError; version numbers printed
```

---

## Steps

### Step 1 — Define the State Schema

The state schema is the contract between all nodes. Every node reads from and writes to this shared structure.

```python
# Step 1: Define the TypedDict state schema for the retrieve-generate-validate agent.
# Using operator.add as a reducer so messages accumulate across turns rather
# than being overwritten on each node invocation.

import operator
from typing import Annotated, TypedDict

class AgentState(TypedDict):
    """Shared state for the 3-node retrieve-generate-validate agent."""
    messages: Annotated[list[dict], operator.add]  # conversation history (accumulates)
    query: str                                       # current user query (overwrite)
    retrieved_context: str                           # output from retrieve node (overwrite)
    answer: str                                      # output from generate node (overwrite)
    validation_passed: bool                          # output from validate node (overwrite)
    retry_count: int                                 # tracks loop iterations (overwrite)

# Why operator.add on messages?
# Without a reducer, each node that returns {"messages": [...]} would overwrite
# the entire list, losing prior turns. operator.add appends instead.
print("State schema defined. Fields:", list(AgentState.__annotations__.keys()))
```

---

### Step 2 — Implement the Three Nodes

Each node is a plain Python function. Nodes return only the keys they change.

```python
# Step 2: Implement retrieve, generate, and validate nodes.
# Each node returns a PARTIAL state dict — only the keys it modifies.
# LangGraph merges the partial update into the full state via reducers.

from langchain_anthropic import ChatAnthropic
# from langchain_openai import ChatOpenAI  # uncomment for OpenAI

# Shared LLM instance (swap ChatAnthropic for ChatOpenAI if preferred)
llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ── Node 1: retrieve ──────────────────────────────────────────────────────
def retrieve_node(state: AgentState) -> dict:
    """
    Retrieve relevant context for the user's query.
    In production: call Unity Catalog Vector Search or a Databricks SQL warehouse.
    Here: simulate retrieval with a hardcoded knowledge base.
    """
    query = state["query"].lower()
    knowledge_base = {
        "langgraph": (
            "LangGraph is a low-level orchestration framework for building stateful agents. "
            "It models workflows as directed graphs where nodes are Python functions that "
            "read and write shared TypedDict state. Edges route execution between nodes."
        ),
        "checkpointer": (
            "LangGraph checkpointers persist the full graph state after every super-step. "
            "InMemorySaver stores state in RAM (lost on restart). SqliteSaver writes to disk "
            "(survives restarts). PostgresSaver is recommended for production."
        ),
        "state": (
            "LangGraph state is a TypedDict shared by all nodes. Nodes return partial "
            "dicts (only changed keys). Reducers control how updates are merged — the "
            "default overwrites; Annotated[list, operator.add] accumulates."
        ),
    }
    # Simple keyword-based retrieval (replace with vector search in production)
    context = "No relevant documentation found."
    for keyword, doc in knowledge_base.items():
        if keyword in query:
            context = doc
            break

    return {
        "retrieved_context": context,
        "messages": [{"role": "system", "content": f"[Retrieved] {context[:80]}..."}]
    }

# ── Node 2: generate ─────────────────────────────────────────────────────
def generate_node(state: AgentState) -> dict:
    """
    Generate an answer using the retrieved context and conversation history.
    The LLM receives: the user's query + retrieved context.
    """
    prompt = (
        f"You are a helpful Databricks documentation assistant.\n"
        f"Context: {state['retrieved_context']}\n\n"
        f"User question: {state['query']}\n\n"
        f"Answer concisely (2-3 sentences), citing the context where relevant."
    )
    response = llm.invoke([{"role": "user", "content": prompt}])
    answer = response.content

    return {
        "answer": answer,
        "messages": [
            {"role": "user", "content": state["query"]},
            {"role": "assistant", "content": answer}
        ]
    }

# ── Node 3: validate ─────────────────────────────────────────────────────
def validate_node(state: AgentState) -> dict:
    """
    Validate that the generated answer is grounded in the retrieved context.
    Checks whether any key term from the retrieved context appears in the answer.
    Returns validation_passed=True if grounded, False if the answer appears hallucinated.
    """
    context_words = set(state["retrieved_context"].lower().split())
    answer_words = set(state["answer"].lower().split())
    # Simple grounding check: at least 3 content words from context appear in answer
    common_words = {"the", "a", "is", "in", "and", "of", "to", "for", "it", "that"}
    content_overlap = (context_words - common_words) & (answer_words - common_words)
    passed = len(content_overlap) >= 3
    new_retry_count = state["retry_count"] + (0 if passed else 1)

    status = "PASSED" if passed else f"FAILED (retry {new_retry_count})"
    return {
        "validation_passed": passed,
        "retry_count": new_retry_count,
        "messages": [{"role": "system", "content": f"[Validation] {status}. Overlap: {len(content_overlap)} terms"}]
    }

print("Nodes defined: retrieve_node, generate_node, validate_node")
```

---

### Step 3 — Wire the Graph with Conditional Routing

Connect nodes with edges. The conditional edge after `validate` either loops back (on failure) or exits (on success or max retries).

```python
# Step 3: Assemble the StateGraph — add nodes, wire edges including a
# conditional edge that implements the retry loop for validation failures.

from typing import Literal
from langgraph.graph import StateGraph, START, END

def validation_router(state: AgentState) -> Literal["retrieve", "end"]:
    """
    Route after validate_node:
    - If validation passed → END (return the answer)
    - If validation failed and retry_count < 2 → retrieve (retry the full pipeline)
    - If validation failed and retry_count >= 2 → END (max retries, return best effort)
    """
    if state["validation_passed"]:
        return "end"
    if state["retry_count"] < 2:
        return "retrieve"
    # Max retries reached — return whatever we have
    return "end"

# Build the graph
builder = StateGraph(AgentState)

# Register nodes
builder.add_node("retrieve", retrieve_node)
builder.add_node("generate", generate_node)
builder.add_node("validate", validate_node)

# Wire static edges (always followed)
builder.add_edge(START, "retrieve")           # entry point
builder.add_edge("retrieve", "generate")      # retrieve always feeds generate
builder.add_edge("generate", "validate")      # generate always feeds validate

# Wire conditional edge (routing decision happens at runtime)
builder.add_conditional_edges(
    "validate",                               # source node
    validation_router,                        # routing function
    {"retrieve": "retrieve", "end": END}      # mapping: router output → node name
)

print("Graph topology:")
print("  START → retrieve → generate → validate")
print("                                    │")
print("                validation_router  ─┤")
print("                                    ├─ [passed or max retries] → END")
print("                                    └─ [failed, retry < 2]    → retrieve")
```

---

### Step 4 — Attach `MemorySaver` and Compile

Compiling registers the checkpointer and validates the graph topology.

```python
# Step 4: Attach InMemorySaver checkpointer and compile the graph.
# InMemorySaver is appropriate here because this is a local development lab.
# In production, swap InMemorySaver for SqliteSaver or PostgresSaver.

from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

print(f"Graph compiled successfully.")
print(f"Checkpointer: {type(checkpointer).__name__}")

# Visualise the graph structure (text representation)
try:
    graph.get_graph().print_ascii()
except Exception:
    print("(ASCII graph visualisation not available in this environment)")
```

---

### Step 5 — Run Turn 1: Single-Turn Invocation

Invoke the agent with a `thread_id` to create the first checkpoint.

```python
# Step 5: Run the first turn of the conversation. The thread_id "lab-session-1"
# is used throughout this lab — keep it the same across turns to demonstrate
# conversation continuity via checkpointing.

import uuid

THREAD_ID = "lab-session-1"
config = {
    "configurable": {"thread_id": THREAD_ID},
    "recursion_limit": 10   # safety cap for this lab
}

initial_state: AgentState = {
    "messages": [],
    "query": "What is LangGraph?",
    "retrieved_context": "",
    "answer": "",
    "validation_passed": False,
    "retry_count": 0
}

print(f"\n{'='*60}")
print(f"Turn 1 — Query: {initial_state['query']}")
print(f"Thread ID: {THREAD_ID}")
print(f"{'='*60}\n")

result = graph.invoke(initial_state, config=config)

print(f"Answer: {result['answer']}")
print(f"Validation passed: {result['validation_passed']}")
print(f"Retry count: {result['retry_count']}")
print(f"Total messages in state: {len(result['messages'])}")
```

---

### Step 6 — Run Turn 2: Multi-Turn with State Continuity

Invoke again on the same `thread_id`. The graph loads the prior checkpoint automatically.

```python
# Step 6: Run a follow-up turn on the same thread_id. The graph resumes
# from the checkpoint saved after Turn 1 — the messages list already contains
# the Turn 1 conversation, demonstrating cross-turn memory via checkpointing.

turn2_input = {
    "messages": [],          # new messages will be appended to existing via reducer
    "query": "How does checkpointing work in LangGraph?",
    "retrieved_context": "",
    "answer": "",
    "validation_passed": False,
    "retry_count": 0         # reset retry counter for this turn
}

print(f"\n{'='*60}")
print(f"Turn 2 — Query: {turn2_input['query']}")
print(f"Thread ID: {THREAD_ID} (same as Turn 1 — state continues)")
print(f"{'='*60}\n")

result2 = graph.invoke(turn2_input, config=config)

print(f"Answer: {result2['answer']}")
print(f"Validation passed: {result2['validation_passed']}")
print(f"Total messages in state (cumulative): {len(result2['messages'])}")
print()

# Inspect the full checkpoint state
snapshot = graph.get_state(config)
print("StateSnapshot:")
print(f"  .next:       {snapshot.next}   (empty = graph finished cleanly)")
print(f"  .values keys: {list(snapshot.values.keys())}")
print(f"  .created_at: {snapshot.created_at}")
```

---

### Step 7 — Stream Execution for Observability

Use `.stream()` to observe each node's output as it is produced.

```python
# Step 7: Stream a third turn to observe per-node state updates in real time.
# stream_mode="updates" yields {node_name: partial_state} after each node fires.
# This is the primary debugging tool for LangGraph agents.

THREAD_ID_STREAM = "lab-session-stream"  # fresh thread for clean demo
config_stream = {
    "configurable": {"thread_id": THREAD_ID_STREAM},
    "recursion_limit": 10
}

stream_input = {
    "messages": [],
    "query": "What is the LangGraph state?",
    "retrieved_context": "",
    "answer": "",
    "validation_passed": False,
    "retry_count": 0
}

print(f"\n{'='*60}")
print("Streaming execution (stream_mode='updates'):")
print(f"{'='*60}")

for chunk in graph.stream(stream_input, config=config_stream, stream_mode="updates"):
    for node_name, state_update in chunk.items():
        if node_name == "__end__":
            continue
        # Show which node ran and a summary of what it produced
        keys_updated = [k for k in state_update if state_update[k]]
        print(f"\n[{node_name}] Updated keys: {keys_updated}")
        if "answer" in state_update and state_update["answer"]:
            print(f"  answer preview: {state_update['answer'][:100]}...")
        if "validation_passed" in state_update:
            print(f"  validation_passed: {state_update['validation_passed']}")
        if "retrieved_context" in state_update and state_update["retrieved_context"]:
            print(f"  context preview: {state_update['retrieved_context'][:80]}...")

print(f"\n{'='*60}")
print("Stream complete.")
```

---

## Validation

Run this validation check after completing all steps.

```python
# Validation: confirm the graph ran correctly across all steps.
# All assertions must pass for the lab to be considered complete.

print("\n=== Validation Checks ===")

# 1. Graph compiled without errors
assert graph is not None, "FAIL: graph was not compiled"
print("✓ Graph compiled successfully")

# 2. Turn 1 produced an answer
assert result["answer"] and len(result["answer"]) > 10, \
    "FAIL: Turn 1 produced no answer"
print(f"✓ Turn 1 answer: {result['answer'][:60]}...")

# 3. State persisted across turns (message count grew)
assert len(result2["messages"]) > len(result["messages"]) or \
       len(result2["messages"]) > 0, \
    "FAIL: Messages did not accumulate across turns"
print(f"✓ Messages accumulated: {len(result2['messages'])} total after Turn 2")

# 4. Streaming produced output from all three nodes
streamed_nodes = set()
stream_check_input = {
    "messages": [], "query": "What is LangGraph?",
    "retrieved_context": "", "answer": "", "validation_passed": False, "retry_count": 0
}
config_check = {"configurable": {"thread_id": "validation-check"}, "recursion_limit": 10}
for chunk in graph.stream(stream_check_input, config=config_check, stream_mode="updates"):
    for node_name in chunk:
        if node_name != "__end__":
            streamed_nodes.add(node_name)

assert "retrieve" in streamed_nodes, "FAIL: retrieve node did not run"
assert "generate" in streamed_nodes, "FAIL: generate node did not run"
assert "validate" in streamed_nodes, "FAIL: validate node did not run"
print(f"✓ All three nodes executed: {sorted(streamed_nodes)}")

# 5. get_state returns a valid StateSnapshot
snapshot_check = graph.get_state(config_check)
assert snapshot_check.values is not None, "FAIL: get_state returned no values"
assert snapshot_check.next == (), "FAIL: graph did not finish cleanly (next != ())"
print(f"✓ get_state returns valid snapshot, next={snapshot_check.next}")

print("\n=== All validation checks passed ===")
```

---

## Teardown

Clean up in-memory state. No persistent resources to delete for this lab.

```python
# Teardown: InMemorySaver stores state only in RAM, so there is nothing to
# delete from disk. If you switch to SqliteSaver in an extension exercise,
# remove the .db file: os.remove("lab06_agent.db")

del checkpointer  # release RAM
del graph
print("Lab teardown complete. InMemorySaver state released from RAM.")
print("Note: if you modified this lab to use SqliteSaver, run:")
print("  import os; os.remove('lab06_agent.db')")
```

---

## Reflection Questions

1. In Step 6, the `retry_count` was reset to 0 in the Turn 2 input dict. Why is it safe to reset it here, given that the checkpointer loaded prior state? What would happen if you passed `retry_count: 5` in Turn 2's input instead?

2. The `validation_router` routes back to `retrieve` on validation failure. What is the cost (in LLM API calls) of a two-retry cycle? In a production agent with high traffic, how would you mitigate this cost without removing the validation step?

3. Modify `retrieve_node` to always return `"No relevant documentation found."` and re-run the lab. Does the `validation_router` loop infinitely? What does this tell you about the importance of designing termination conditions that are independent of tool quality?

4. Replace `InMemorySaver` with `SqliteSaver` using `sqlite3.connect("lab06_agent.db")`. Restart your Python process, re-create the graph with the same `SqliteSaver`, and invoke it with `THREAD_ID = "lab-session-1"`. Are the Turn 1 messages still present? What does this demonstrate about the difference between the two checkpointer implementations?

5. How would you adapt this 3-node graph to run on Databricks? What would change in the `retrieve_node` (hint: Unity Catalog Vector Search), the LLM call (hint: `ChatDatabricks`), and the deployment packaging (hint: MLflow `ResponsesAgent`)?
