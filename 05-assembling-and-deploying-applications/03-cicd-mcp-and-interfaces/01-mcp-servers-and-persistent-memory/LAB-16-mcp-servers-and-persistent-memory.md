# LAB-16: MCP Servers and Persistent Memory

**Lab:** LAB-16 | **Section:** 05 — Assembling and Deploying Applications | **Module:** 03 — CI/CD, MCP, and Interfaces | **Est. time:** 2 hrs

> ⚠️ **Fast-evolving:** Databricks managed MCP servers and the `databricks-mcp` library are in Public Preview as of July 2026. Step URLs and package versions may change. Verify against [official Databricks docs](https://docs.databricks.com/aws/en/agents/mcp/use-mcp-in-agents.html) before running.

---

## Objective

Build a LangGraph ReAct agent that calls a Databricks managed MCP server, verify tool discovery and invocation, and then extend the agent with a two-layer memory architecture: a `MemorySaver` checkpointer for within-session memory and a Delta table read/write node for cross-session long-term memory.

---

## Prerequisites

- Completed LAB-15 (CI/CD for agent promotion) or equivalent familiarity with LangGraph `StateGraph` and `create_react_agent`
- Databricks workspace with Unity Catalog enabled and at least one UC schema you can write to (e.g. `main.default`)
- Serverless compute enabled in the workspace (required for managed `system.ai` MCP tools)
- Databricks CLI configured: `databricks auth login --host https://<workspace-hostname>`
- Python 3.12+ local environment or a Databricks notebook with access to PyPI

---

## Setup

Install the required packages in your local environment or a Databricks notebook:

```bash
pip install -U \
  "mcp>=1.9" \
  "databricks-sdk[openai]" \
  "databricks-mcp" \
  "databricks-langchain" \
  "langgraph>=0.2" \
  "mlflow>=3.1.0"
```

Create the Delta table that will serve as the long-term memory store:

```sql
-- Run in a Databricks SQL editor or notebook cell
CREATE TABLE IF NOT EXISTS main.default.lab16_agent_memory (
  user_id   STRING NOT NULL,
  facts     STRING,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
  CONSTRAINT pk_lab16_memory PRIMARY KEY (user_id)
)
USING DELTA
TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true');
```

Set your workspace hostname in a notebook environment variable or `.env` file:

```bash
# In a terminal or notebook magic cell
export DATABRICKS_HOST="https://<your-workspace>.azuredatabricks.net"
export DATABRICKS_CONFIG_PROFILE="DEFAULT"  # profile name from auth login
```

---

## Steps

### Step 1 — Connect to the Managed MCP Server and List Tools

Create a `DatabricksMCPClient` pointing at the `system.ai` managed UC functions server and print all available tools.

```python
# Step 1: Discover tools on the system.ai managed MCP server
# Goal: verify that the Databricks managed MCP server is reachable and
# returns a non-empty tool list before wiring into an agent.

import asyncio
from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksMCPClient

workspace_client = WorkspaceClient()
host = workspace_client.config.host

mcp_server_url = f"{host}/api/2.0/mcp/functions/system/ai"
mcp_client = DatabricksMCPClient(
    server_url=mcp_server_url,
    workspace_client=workspace_client,
)

tools = mcp_client.list_tools()
print(f"Found {len(tools)} tools on system.ai MCP server:")
for t in tools:
    print(f"  - {t.name}: {t.description[:80]}...")
```

Expected output: a list of tools including `system__ai__python_exec` and others. If the list is empty or you receive a 403, verify that serverless compute is enabled and your Unity Catalog principal has `EXECUTE` on `system.ai`.

### Step 2 — Call a Tool Directly Through the MCP Client

Invoke `system__ai__python_exec` to run a Python expression and verify the tool-call round trip.

```python
# Step 2: Direct MCP tool call (without an agent loop)
# Goal: confirm tool invocation works end-to-end before adding LLM orchestration.

result = mcp_client.call_tool(
    "system__ai__python_exec",
    {"code": "import math; print(math.factorial(10))"}
)

for content_item in result.content:
    print(content_item.text)
# Expected output: 3628800
```

### Step 3 — Build a LangGraph ReAct Agent with MCP Tools

Wire the MCP server tools into a LangGraph `create_react_agent` with a `MemorySaver` checkpointer for within-session memory.

```python
# Step 3: Build a LangGraph ReAct agent backed by the system.ai MCP server
# Goal: the agent should answer a multi-step question that requires a tool call,
# then remember the context in a follow-up question within the same thread.

import asyncio
from databricks.sdk import WorkspaceClient
from databricks_langchain import (
    ChatDatabricks,
    DatabricksMCPServer,
    DatabricksMultiServerMCPClient,
)
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

workspace_client = WorkspaceClient()
host = workspace_client.config.host

async def run_agent():
    mcp_client = DatabricksMultiServerMCPClient(
        [
            DatabricksMCPServer(
                name="system-ai",
                url=f"{host}/api/2.0/mcp/functions/system/ai",
                workspace_client=workspace_client,
            )
        ]
    )

    async with mcp_client:
        tools = await mcp_client.get_tools()
        print(f"Agent has access to {len(tools)} tools via MCP.")

        llm = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct")
        checkpointer = MemorySaver()
        agent = create_react_agent(llm, tools, checkpointer=checkpointer)

        # Use a stable thread_id to test multi-turn memory
        config = {"configurable": {"thread_id": "lab16-session-001"}}

        # Turn 1
        result1 = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "Calculate the 20th Fibonacci number using Python."}]},
            config=config,
        )
        print("Turn 1:", result1["messages"][-1].content)

        # Turn 2 — agent should remember it just calculated a Fibonacci number
        result2 = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "Now calculate the 25th Fibonacci number the same way."}]},
            config=config,
        )
        print("Turn 2:", result2["messages"][-1].content)

asyncio.run(run_agent())
```

### Step 4 — Add a Long-Term Memory Read Node

Extend the graph with a `memory_read` node that loads user facts from the Delta table into the agent state before the LLM runs.

```python
# Step 4: Add a long-term memory read/write layer using Delta
# Goal: the agent reads stored user facts from Delta at session start and
# writes new facts back at session end, enabling cross-session recall.

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from databricks_langchain import ChatDatabricks
from databricks.sdk import WorkspaceClient
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
workspace_client = WorkspaceClient()
llm = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct")

USER_ID = "lab16-user-001"
MEMORY_TABLE = "main.default.lab16_agent_memory"

class Lab16State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    long_term_facts: str

def memory_read_node(state: Lab16State) -> dict:
    """Load user facts from Delta at session start."""
    df = spark.sql(
        f"SELECT facts FROM {MEMORY_TABLE} WHERE user_id = '{USER_ID}'"
    )
    rows = df.collect()
    facts = rows[0]["facts"] if rows else "No stored facts yet."
    system_msg = SystemMessage(
        content=f"Remembered facts about this user: {facts}"
    )
    return {"messages": [system_msg], "long_term_facts": facts}

def llm_node(state: Lab16State) -> dict:
    """Call the LLM with the current message history."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def memory_write_node(state: Lab16State) -> dict:
    """Persist any new facts the conversation revealed back to Delta."""
    # In production, use an LLM-powered extraction step here.
    # For the lab, we write a static demonstration fact.
    new_fact = "User asked about Fibonacci numbers in this session."
    spark.sql(
        f"MERGE INTO {MEMORY_TABLE} AS t "
        f"USING (SELECT '{USER_ID}' AS user_id, '{new_fact}' AS facts, current_timestamp() AS updated_at) AS s "
        f"ON t.user_id = s.user_id "
        f"WHEN MATCHED THEN UPDATE SET t.facts = s.facts, t.updated_at = s.updated_at "
        f"WHEN NOT MATCHED THEN INSERT *"
    )
    print(f"Wrote facts to Delta for user {USER_ID}.")
    return {}

builder = StateGraph(Lab16State)
builder.add_node("memory_read", memory_read_node)
builder.add_node("llm", llm_node)
builder.add_node("memory_write", memory_write_node)
builder.add_edge(START, "memory_read")
builder.add_edge("memory_read", "llm")
builder.add_edge("llm", "memory_write")
builder.add_edge("memory_write", END)

graph = builder.compile(checkpointer=MemorySaver())

config = {"configurable": {"thread_id": "lab16-delta-session-001"}}
result = graph.invoke(
    {"messages": [HumanMessage(content="What do you remember about me?")]},
    config=config,
)
print("LLM response:", result["messages"][-1].content)

# Run again — this time memory_read should return the fact we wrote
result2 = graph.invoke(
    {"messages": [HumanMessage(content="Remind me what we discussed before.")]},
    config=config,
)
print("LLM response (second invocation):", result2["messages"][-1].content)
```

### Step 5 — Observe the Memory Lifecycle

Query the Delta table directly to confirm the write node stored the fact.

```sql
-- Step 5: Verify the long-term memory write in Delta
SELECT user_id, facts, updated_at
FROM main.default.lab16_agent_memory
WHERE user_id = 'lab16-user-001';
```

Expected output: one row with the fact written by `memory_write_node` and a recent `updated_at` timestamp.

---

## Validation

Run the following checklist to confirm each lab objective was achieved:

```python
# Validation: verify all three lab objectives programmatically
from databricks_mcp import DatabricksMCPClient
from databricks.sdk import WorkspaceClient
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
workspace_client = WorkspaceClient()
host = workspace_client.config.host

# Check 1: MCP server reachable and has tools
mcp_server_url = f"{host}/api/2.0/mcp/functions/system/ai"
client = DatabricksMCPClient(server_url=mcp_server_url, workspace_client=workspace_client)
tools = client.list_tools()
assert len(tools) > 0, "FAIL: No tools returned from MCP server"
print(f"PASS: MCP server returned {len(tools)} tools")

# Check 2: Direct tool call returns expected output
result = client.call_tool("system__ai__python_exec", {"code": "print(2 + 2)"})
assert "4" in result.content[0].text, "FAIL: Tool call did not return expected output"
print("PASS: Direct MCP tool call succeeded")

# Check 3: Delta long-term memory table has at least one row
rows = spark.sql(
    "SELECT COUNT(*) AS cnt FROM main.default.lab16_agent_memory"
).collect()
assert rows[0]["cnt"] > 0, "FAIL: Delta memory table is empty — Step 4 may not have run"
print(f"PASS: Delta memory table has {rows[0]['cnt']} row(s)")

print("\nAll validation checks passed.")
```

---

## Teardown

Clean up lab resources to avoid ongoing compute or storage costs:

```sql
-- Drop the Delta table created for this lab
DROP TABLE IF EXISTS main.default.lab16_agent_memory;
```

```python
# Optional: clear the MemorySaver checkpointer (in-process, cleared on restart)
# No explicit teardown needed for MemorySaver — process restart clears it.
# If you used AsyncPostgresSaver or a persistent backend, drop the checkpointer tables
# according to that backend's documentation.
print("Lab teardown complete. Delta table dropped.")
```

---

## Reflection Questions

1. In Step 3, the agent uses `MemorySaver` and a fixed `thread_id`. What would happen if you deployed this agent with two replicas and a user's second request was routed to the second replica? How would you fix this for a real multi-replica deployment, and which LangGraph checkpointer would you choose?

2. In Step 4, `memory_write_node` writes a hard-coded fact string. In a production agent, how would you extract meaningful, structured facts from an open-ended conversation? Describe the LLM-based extraction pattern you would use and where in the graph you would place it.

3. Consider the trade-off between short-term (checkpointer) memory and long-term (Delta) memory. For an agent that handles 100-message conversations spanning multiple sessions, when does the short-term checkpointer approach break down, and what architectural change would you make to prevent context-window overflow while preserving recall quality?
