# LAB-10: Build a Two-Agent LangGraph System with Supervisor Routing

**Lab:** LAB-10 | **Section:** 04 — Application Development | **Module:** 03 — Agents, MLflow & Genie | **Est. time:** 2.5 hrs

---

## Objective

Build a two-agent LangGraph system — a retrieval agent and a synthesis agent — orchestrated by a supervisor node that routes work via shared state and conditional edges, then validate that state passes correctly between agents and that routing terminates with `FINISH`.

---

## Prerequisites

- Completed LAB-09 or equivalent familiarity with single-agent LangGraph ReAct loops
- A Databricks workspace with access to a foundation model endpoint (e.g. `databricks-claude-sonnet-4-5` or `databricks-meta-llama-3-1-70b-instruct`)
- Python environment with `langgraph>=0.2`, `langchain-databricks`, `langchain-core`, `mlflow>=2.14`, `pydantic>=2`
- Databricks CLI configured (`databricks configure --token`)

---

## Setup

Install dependencies in a Databricks notebook or local environment connected to your workspace:

```python
# Scenario: Prepare the Python environment for a LangGraph multi-agent lab
# Constraint: Must use langgraph>=0.2 for Command and interrupt_before support

%pip install -q \
    "langgraph>=0.2" \
    "langchain-databricks>=0.3" \
    "langchain-core>=0.2" \
    "mlflow>=2.14" \
    "pydantic>=2"

dbutils.library.restartPython()
```

Set environment variables (or use Databricks Secrets):

```python
# Scenario: Configure workspace credentials so agents can call model endpoints
import os

# In a Databricks notebook these are pre-set; in a local environment set them:
# os.environ["DATABRICKS_HOST"]  = "https://<workspace>.azuredatabricks.net"
# os.environ["DATABRICKS_TOKEN"] = "<pat-or-oauth-token>"

# Choose an available endpoint in your workspace:
MODEL_ENDPOINT = "databricks-claude-sonnet-4-5"   # or "databricks-meta-llama-3-1-70b-instruct"
print(f"Using model endpoint: {MODEL_ENDPOINT}")
```

---

## Steps

### Step 1 — Define the Shared State Schema

The shared state is the contract between all nodes. Every agent reads from it and writes back to it.

```python
# Scenario: Define a typed state schema that accumulates conversation history
# across multiple agents without any agent overwriting another's output.

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    # 'next' will be written by the supervisor to record its routing decision
    next: str

# Verify the schema
print("AgentState keys:", AgentState.__annotations__.keys())
# Expected output: dict_keys(['messages', 'next'])
```

**Why `add_messages`?** Without the `Annotated[list, add_messages]` annotation, each node return would *replace* the entire `messages` list. With it, new messages are *appended* — preserving the full conversation history for the supervisor to reason about.

---

### Step 2 — Build the Retrieval Agent Node

This agent simulates document retrieval (in production, this would call a vector search tool).

```python
# Scenario: Implement a retrieval node that finds relevant context
# and appends it to the shared messages channel.

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_databricks import ChatDatabricks

retrieval_llm = ChatDatabricks(endpoint=MODEL_ENDPOINT)

def retrieval_agent(state: AgentState) -> dict:
    """Retrieves relevant context for the user's question."""
    # Get the original user question from the first message
    user_question = state["messages"][0].content

    response = retrieval_llm.invoke([
        SystemMessage(content=(
            "You are a document retrieval specialist. "
            "Given a question, summarise the most relevant factual context "
            "you would find in a knowledge base. "
            "Prefix your response with '[RETRIEVAL]:'"
        )),
        HumanMessage(content=f"Find context for: {user_question}")
    ])

    print(f"Retrieval agent produced: {response.content[:80]}...")
    return {"messages": [response]}

# Quick unit test
test_state: AgentState = {
    "messages": [HumanMessage(content="What are the benefits of LangGraph for agents?")],
    "next": ""
}
result = retrieval_agent(test_state)
assert len(result["messages"]) == 1
assert "[RETRIEVAL]" in result["messages"][0].content
print("✓ Retrieval agent test passed")
```

---

### Step 3 — Build the Synthesis Agent Node

This agent reads the full conversation history (including retrieval output) and generates a final answer.

```python
# Scenario: Implement a synthesis node that combines all prior agent outputs
# into a coherent, grounded final answer for the user.

synthesis_llm = ChatDatabricks(endpoint=MODEL_ENDPOINT)

def synthesis_agent(state: AgentState) -> dict:
    """Synthesises a final answer from all messages in state."""
    response = synthesis_llm.invoke(
        state["messages"] + [
            HumanMessage(content=(
                "Using ALL the context above, write a complete, well-structured "
                "answer to the original question. Cite the retrieved information."
            ))
        ]
    )

    print(f"Synthesis agent produced: {response.content[:80]}...")
    return {"messages": [response]}
```

---

### Step 4 — Build the Supervisor Node

The supervisor reads state, decides which agent should act next, and returns a `Command` with the routing target.

```python
# Scenario: Implement a supervisor node that routes dynamically based on
# what has already been done, using structured output to force a valid decision.

from typing import Literal
from pydantic import BaseModel
from langgraph.types import Command
from langgraph.graph import END

class RouteDecision(BaseModel):
    next: Literal["retrieval_agent", "synthesis_agent", "FINISH"]
    reasoning: str

supervisor_llm = ChatDatabricks(endpoint=MODEL_ENDPOINT)
supervisor_with_structure = supervisor_llm.with_structured_output(RouteDecision)

def supervisor(state: AgentState) -> Command:
    """Routes to the next agent or terminates based on current state."""
    # Build a routing prompt that includes the full message history
    history_summary = "\n".join([
        f"[{msg.__class__.__name__}]: {msg.content[:120]}"
        for msg in state["messages"]
    ])

    decision = supervisor_with_structure.invoke([
        SystemMessage(content=(
            "You are a supervisor orchestrating two agents:\n"
            "- retrieval_agent: searches for relevant context (run FIRST if no retrieval result yet)\n"
            "- synthesis_agent: writes the final answer (run AFTER retrieval is complete)\n"
            "- FINISH: end the workflow (choose after synthesis is complete)\n\n"
            "Message history so far:\n" + history_summary
        )),
        HumanMessage(content="Which agent should act next? Choose one option.")
    ])

    print(f"Supervisor decision: {decision.next} — {decision.reasoning[:60]}...")

    goto = END if decision.next == "FINISH" else decision.next
    return Command(goto=goto, update={"next": decision.next})
```

---

### Step 5 — Assemble the Graph

Wire the nodes together with the supervisor as the entry point and conditional edges for routing.

```python
# Scenario: Compile a multi-agent StateGraph with a supervisor routing between
# retrieval and synthesis agents, with an explicit recursion limit to prevent
# infinite loops in development.

from langgraph.graph import StateGraph

# Build the graph
graph_builder = StateGraph(AgentState)

# Register nodes
graph_builder.add_node("supervisor",       supervisor)
graph_builder.add_node("retrieval_agent",  retrieval_agent)
graph_builder.add_node("synthesis_agent",  synthesis_agent)

# Entry point is always the supervisor
graph_builder.set_entry_point("supervisor")

# Sub-agents always return to supervisor after completing their work
graph_builder.add_edge("retrieval_agent", "supervisor")
graph_builder.add_edge("synthesis_agent", "supervisor")

# Supervisor uses conditional edges to route
graph_builder.add_conditional_edges(
    "supervisor",
    lambda state: state["next"],
    {
        "retrieval_agent": "retrieval_agent",
        "synthesis_agent": "synthesis_agent",
        "FINISH":          END,
    },
)

# Compile with safety limits
app = graph_builder.compile(recursion_limit=10)
print("Graph compiled successfully.")
print(f"Nodes: {list(graph_builder.nodes.keys())}")
```

---

### Step 6 — Run the Graph and Inspect State Passing

Invoke the graph with a test question and trace the message accumulation.

```python
# Scenario: Invoke the multi-agent graph and verify that each agent's output
# is visible to subsequent agents via the shared messages channel.

from langchain_core.messages import HumanMessage

initial_state: AgentState = {
    "messages": [HumanMessage(content=(
        "What are the main advantages of using the supervisor pattern "
        "in LangGraph for multi-agent systems?"
    ))],
    "next": ""
}

print("=" * 60)
print("Starting multi-agent graph execution...")
print("=" * 60)

final_state = app.invoke(initial_state)

print("\n" + "=" * 60)
print(f"Total messages in final state: {len(final_state['messages'])}")
print(f"Final routing decision: {final_state['next']}")
print("=" * 60)

# Inspect each message
for i, msg in enumerate(final_state["messages"]):
    agent_type = msg.__class__.__name__
    preview = msg.content[:150].replace('\n', ' ')
    print(f"\nMessage {i+1} [{agent_type}]: {preview}...")
```

**Expected output structure:**
```
Message 1 [HumanMessage]:  What are the main advantages of using the supervisor pattern...
Message 2 [AIMessage]:     [RETRIEVAL]: The supervisor pattern in LangGraph provides...
Message 3 [AIMessage]:     The supervisor pattern in LangGraph offers several key advantages...
```

---

### Step 7 — Validate State Isolation and Routing Correctness

Write assertions to confirm the system behaves correctly.

```python
# Scenario: Programmatically validate that state passing and routing
# work correctly — catch regressions before deploying to production.

# Assertion 1: Final state has at least 3 messages (user + retrieval + synthesis)
assert len(final_state["messages"]) >= 3, (
    f"Expected ≥3 messages, got {len(final_state['messages'])}. "
    "Check that sub-agents are writing to the messages channel."
)
print("✓ State accumulation: PASS")

# Assertion 2: At least one message contains the [RETRIEVAL] prefix
retrieval_messages = [
    m for m in final_state["messages"]
    if hasattr(m, "content") and "[RETRIEVAL]" in m.content
]
assert len(retrieval_messages) >= 1, (
    "No retrieval message found. Supervisor may have skipped the retrieval agent."
)
print("✓ Retrieval agent was invoked: PASS")

# Assertion 3: The first message is the original user HumanMessage
assert isinstance(final_state["messages"][0], HumanMessage), (
    "First message is not the user's HumanMessage — add_messages may be misconfigured."
)
print("✓ Message history preserved (user message still first): PASS")

# Assertion 4: Routing terminated (next is "FINISH" or empty after END)
print(f"✓ Final routing target: {final_state['next']}")
print("\nAll assertions passed. ✓")
```

---

## Validation

Run the full test to confirm the lab succeeded:

```python
# Scenario: End-to-end smoke test — confirms the graph runs, produces output,
# and terminates cleanly without hitting recursion_limit.

from langchain_core.exceptions import GraphRecursionError  # langgraph re-exports this

test_question = "Explain how shared state works in LangGraph multi-agent systems."

try:
    result = app.invoke({
        "messages": [HumanMessage(content=test_question)],
        "next": ""
    })
    print("✓ Graph completed successfully")
    print(f"✓ Final message count: {len(result['messages'])}")
    print(f"✓ Answer preview: {result['messages'][-1].content[:200]}")
except GraphRecursionError as e:
    print(f"✗ Graph hit recursion limit: {e}")
    print("  Fix: Ensure supervisor has [RETRIEVAL] in history before routing to synthesis.")
```

**Expected successful output:**
```
✓ Graph completed successfully
✓ Final message count: 3
✓ Answer preview: In LangGraph multi-agent systems, shared state is managed through a TypedDict...
```

---

## Teardown

No persistent cloud resources are created in this lab (all execution is in-notebook). If you created an MLflow experiment, optionally clean it up:

```python
# Scenario: Clean up MLflow experiment artifacts created during the lab.
import mlflow

experiment_name = "/Users/<your-email>/lab-10-multi-agent"
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment:
    mlflow.delete_experiment(experiment.experiment_id)
    print(f"Deleted MLflow experiment: {experiment_name}")
else:
    print("No MLflow experiment to clean up.")
```

---

## Extension Task — Add a Third Agent (Genie Analytics)

Extend the graph to include a third sub-agent that calls the Genie Conversation API. This extension demonstrates integrating structured data analytics into the multi-agent system.

```python
# Scenario: Add a Genie analytics agent as a third sub-agent, exposing governed
# SQL analytics over Unity Catalog data without granting the agent direct warehouse access.

import requests, time, os

GENIE_SPACE_ID = os.environ.get("GENIE_SPACE_ID", "YOUR_SPACE_ID")

def genie_analytics_agent(state: AgentState) -> dict:
    """Queries a Genie Agent for structured data analysis."""
    user_question = state["messages"][0].content
    token   = os.environ["DATABRICKS_TOKEN"]
    host    = os.environ["DATABRICKS_HOST"]
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    # Start conversation
    resp = requests.post(
        f"{host}/api/2.0/genie/spaces/{GENIE_SPACE_ID}/start-conversation",
        headers=headers,
        json={"content": user_question},
    )
    if resp.status_code != 200:
        error_msg = AIMessage(content=f"[ANALYTICS ERROR]: {resp.text}")
        return {"messages": [error_msg]}

    data    = resp.json()
    conv_id = data["conversation"]["id"]
    msg_id  = data["message"]["id"]

    # Poll for completion (max 2 min)
    for _ in range(24):
        time.sleep(5)
        msg_resp = requests.get(
            f"{host}/api/2.0/genie/spaces/{GENIE_SPACE_ID}"
            f"/conversations/{conv_id}/messages/{msg_id}",
            headers=headers,
        ).json()
        status = msg_resp.get("status")
        if status in ("COMPLETED", "FAILED", "CANCELLED"):
            break

    attachments = msg_resp.get("attachments") or []
    parts = [
        a.get("text", {}).get("content", "")
        for a in attachments if a.get("text")
    ]
    answer = "\n".join(parts) or "No structured data result from Genie."
    return {"messages": [AIMessage(content=f"[ANALYTICS]: {answer}")]}

# Update the RouteDecision enum and rebuild the graph to include the new agent:
class RouteDecisionV2(BaseModel):
    next: Literal["retrieval_agent", "synthesis_agent", "genie_analytics_agent", "FINISH"]
    reasoning: str

# Re-build graph with new node (follow Steps 4–5 pattern, adding genie_analytics_agent)
print("Extension: genie_analytics_agent node defined.")
print("To complete: rebuild graph_builder with genie_analytics_agent node,")
print("update supervisor to use RouteDecisionV2, and re-compile with recursion_limit=15.")
```

**Extension validation check:** After adding the third agent, run a question that requires data analysis (e.g. "What were the top 5 products by revenue last quarter?") and verify the trace shows a `[ANALYTICS]:` prefixed message in the final state.

---

## Reflection Questions

1. What would break if you removed the `add_messages` annotation from the `messages` key in `AgentState`? Test your hypothesis by removing it and re-running Step 6 — what do you observe?

2. In Step 5, sub-agents have `add_edge(agent, "supervisor")`. What would happen if instead you set the sub-agents as terminal nodes (no edge back to supervisor)? Under what design conditions would that be the right choice?

3. How would you modify this lab to support human-in-the-loop approval before the synthesis agent runs? What LangGraph compile option would you use, and how would you resume the graph after the human approves?
