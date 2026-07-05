# Agents, Tools, and Multi-Agent Design

**Section:** Application Development | **Module:** Chains, Agents, and Frameworks | **Est. time:** 2.5 hrs
**Exam mapping:** Databricks GenAI Engineer Associate — Application Development domain (~30%); tool-calling agents, ReAct pattern, and multi-agent design are all directly tested

## Learning Objectives

By the end of this chapter, you will be able to:

- Define Python tools using the `@tool` decorator and write effective tool docstrings
- Build a ReAct-style tool-calling agent quickly with `create_react_agent`
- Build a custom tool-calling agent loop from scratch using `StateGraph` with conditional routing
- Register Unity Catalog functions as agent tools
- Describe the MCP (Model Context Protocol) and when Databricks uses it
- Design a multi-agent supervisor system where a router node dispatches to specialist agents
- Use LangGraph subgraphs to encapsulate specialized agent logic

## Core Concepts

### What Is an Agent?

An **agent** is an LLM that, given a goal and a set of tools, decides autonomously what
actions to take and in what order to accomplish the goal. Unlike a chain (which follows a
fixed sequence of steps), an agent iterates:

```
User goal
  → LLM decides: "I need to call tool X with these args"
  → Tool X executes and returns result
  → LLM decides: "Now I need tool Y..." or "I have enough info to answer"
  → ... loop continues until LLM decides it's done
  → Final answer returned
```

The most common agent pattern is **ReAct** (Reasoning + Acting): the LLM alternates between
reasoning about what to do next and acting by calling a tool.

### Defining Tools

A **tool** is a Python function that an LLM can call. LangGraph (via `databricks-langchain` +
LangChain) uses the `@tool` decorator:

```python
from langchain_core.tools import tool

@tool
def search_databricks_docs(query: str) -> str:
    """Search the Databricks AI Search index for relevant documentation.

    Args:
        query: The search query in natural language.

    Returns:
        A formatted string of the top 3 most relevant documentation snippets.
    """
    results = index.similarity_search(
        query_text=query,
        columns=["chunk_text", "source_path"],
        num_results=3
    )
    snippets = []
    for row in results["result"]["data_array"]:
        snippets.append(f"[Source: {row[1]}]\n{row[0]}")
    return "\n\n".join(snippets)

@tool
def get_current_datetime() -> str:
    """Return the current date and time in ISO 8601 format."""
    from datetime import datetime
    return datetime.now().isoformat()

@tool
def calculate_token_budget(
    context_window: int,
    system_prompt_tokens: int,
    query_tokens: int,
    max_output_tokens: int
) -> dict:
    """Calculate how many tokens are available for retrieved context in a RAG pipeline.

    Args:
        context_window: Total context window size in tokens.
        system_prompt_tokens: Number of tokens used by the system prompt.
        query_tokens: Number of tokens in the user query.
        max_output_tokens: Maximum tokens reserved for the model's response.

    Returns:
        A dict with 'available_tokens' and 'recommended_chunks' (assuming 400 tokens/chunk).
    """
    available = context_window - system_prompt_tokens - query_tokens - max_output_tokens - 200
    return {
        "available_tokens": max(0, available),
        "recommended_chunks": max(0, available) // 400
    }
```

**What makes a good tool docstring:**
- The first sentence is what the LLM reads to decide *whether* to call this tool
- Be specific about what it returns — the LLM uses the return to reason about next steps
- Describe args clearly — the LLM must generate these values from natural language

**Tool name:** The function name becomes the tool name. Use descriptive snake_case names.

---

### `create_react_agent` — Fast Path for Tool-Calling Agents

LangGraph's `create_react_agent` builds a complete ReAct agent from a model + tools list.
It is the fast path when you do not need custom routing or state beyond messages:

```python
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from databricks_langchain import ChatDatabricks

llm = ChatDatabricks(endpoint="databricks-meta-llama-3-1-70b-instruct")

tools = [search_databricks_docs, get_current_datetime, calculate_token_budget]

# Create a ReAct agent — automatically handles the tool-call loop
agent = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=InMemorySaver(),  # optional: adds multi-turn memory
    prompt="You are a helpful Databricks expert. Use your tools to answer questions accurately."
)

# Invoke
config = {"configurable": {"thread_id": "agent-session-1"}}
result = agent.invoke(
    {"messages": [{"role": "user", "content": "How many chunks fit in DBRX Instruct's context window if my system prompt is 500 tokens and I expect 800 token responses?"}]},
    config=config
)

# The last AIMessage contains the final answer
print(result["messages"][-1].content)
```

`create_react_agent` internally builds a `StateGraph` with:
- An `agent` node: calls the LLM with tools bound
- A `tools` node: executes tool calls
- A conditional edge: if the LLM returned tool calls → `tools`; otherwise → `END`

---

### Custom ReAct Loop with `StateGraph`

When you need custom state, routing, or behavior beyond what `create_react_agent` provides,
build the ReAct loop explicitly:

```python
from typing import TypedDict, Annotated, Literal
import operator
import json
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AnyMessage, ToolMessage

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

# Bind tools to the model
llm_with_tools = ChatDatabricks(
    endpoint="databricks-meta-llama-3-1-70b-instruct"
).bind_tools(tools)

def agent_node(state: AgentState) -> dict:
    """Call the LLM. It may return a tool call or a final answer."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def tool_node(state: AgentState) -> dict:
    """Execute all tool calls in the last message."""
    last_message = state["messages"][-1]
    tool_results = []
    for tool_call in last_message.tool_calls:
        # Look up the tool by name
        tool_fn = {t.name: t for t in tools}[tool_call["name"]]
        result = tool_fn.invoke(tool_call["args"])
        tool_results.append(ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"]
        ))
    return {"messages": tool_results}

def route_after_agent(state: AgentState) -> Literal["tools", "__end__"]:
    """Route: if LLM made tool calls → execute them; else → done."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

# Build the graph
builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", route_after_agent, {
    "tools": "tools",
    END: END
})
builder.add_edge("tools", "agent")  # after tools, loop back to agent

agent_graph = builder.compile(checkpointer=InMemorySaver())
```

**The ReAct loop visualized:**
```
START → agent → [tool calls?] → YES → tools → agent → [tool calls?] → NO → END
```

---

### Unity Catalog Functions as Agent Tools

Databricks supports registering Unity Catalog SQL/Python functions and using them directly
as agent tools. This is how agents access governed data transformations:

```python
from databricks_langchain import UCFunctionToolkit

# Create a toolkit from UC functions in a schema
uc_toolkit = UCFunctionToolkit(
    function_names=[
        "main.rag_tools.search_knowledge_base",
        "main.rag_tools.get_product_details",
        "main.rag_tools.lookup_customer_record"
    ]
)

# Get the tools — these wrap UC functions as LangChain tools
uc_tools = uc_toolkit.get_tools()

# Use them in create_react_agent or bind_tools as usual
agent = create_react_agent(model=llm, tools=uc_tools)
```

**Why UC functions as tools:**
- UC access control governs who can call the function
- UC lineage tracks which data the agent accessed
- Functions can be shared across agents without code duplication

---

### MCP — Model Context Protocol

MCP is a standardized protocol for agents to connect to tools, data sources, and services.
Databricks uses MCP as the integration layer for agent tooling (updated 2026-07-01 docs):

> "Connect agents to tools, data, and workflows through standardized MCP servers."

**How it works:**
- An MCP server exposes a set of tools via a standard JSON-RPC interface
- An agent connects to the MCP server and can call its tools as if they were local tools
- Databricks provides managed MCP servers for common enterprise services (Google Drive, Jira, Slack, GitHub)

**When to use MCP vs. local Python tools:**
- Use **local tools** (`@tool` decorator) for logic that runs in the agent process: data transformations, calculations, API calls where the credentials are in Databricks secrets
- Use **MCP** for external services where the tool logic is maintained separately, or for managed Databricks connectors (UC functions exposed via MCP are the canonical example)

For the exam: know that MCP is the Databricks-preferred mechanism for agent-tool integration, especially for external systems.

---

### Multi-Agent Systems: Supervisor Pattern

A **multi-agent system** uses multiple specialized agents, coordinated by a supervisor (router)
that decides which agent should handle each task. This pattern is recommended when:
- A single agent is too complex to reliably handle all tasks
- Different tasks require different system prompts, tool sets, or models
- You want to independently test and improve each specialist agent

**Supervisor architecture:**

```
User query
  → Supervisor (router) LLM
      → "This is a data question" → Data Agent subgraph
      → "This is a code question" → Code Agent subgraph
      → "This is a docs question" → Docs RAG Agent subgraph
  → Consolidated answer back to user
```

**Implementation:**

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing import TypedDict, Annotated, Literal
import operator
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage

# ── Shared state ──────────────────────────────────────────────────────────────
class SupervisorState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    next_agent: str   # which specialist to route to

# ── Specialist agents (pre-built subgraphs) ───────────────────────────────────
# These would be fully defined create_react_agents or custom graphs
docs_agent = create_react_agent(
    model=llm,
    tools=[search_databricks_docs],
    prompt="You are a documentation specialist. Answer questions about Databricks docs."
)

code_agent = create_react_agent(
    model=llm,
    tools=[calculate_token_budget, get_current_datetime],
    prompt="You are a code and technical specialist. Help with calculations and technical details."
)

# ── Supervisor node ───────────────────────────────────────────────────────────
ROUTING_SYSTEM_PROMPT = """You are a routing supervisor. Given a user message, decide which
specialist should handle it.

Specialists:
- "docs_agent": answers questions about Databricks documentation and features
- "code_agent": handles technical calculations, code, and implementation questions
- "FINISH": the question has been answered

Respond with exactly one word: docs_agent, code_agent, or FINISH."""

def supervisor(state: SupervisorState) -> Command[Literal["docs_agent", "code_agent", "__end__"]]:
    """Route the conversation to the appropriate specialist."""
    response = llm.invoke([
        {"role": "system", "content": ROUTING_SYSTEM_PROMPT},
        # Pass only the last user message for routing decision
        state["messages"][-1]
    ])
    decision = response.content.strip().lower()

    if decision == "finish":
        return Command(goto=END)
    return Command(goto=decision)

# ── Wrapper nodes for subgraphs ────────────────────────────────────────────────
def run_docs_agent(state: SupervisorState) -> Command[Literal["supervisor"]]:
    result = docs_agent.invoke({"messages": state["messages"]})
    # Take the last message (the agent's response) and route back to supervisor
    return Command(
        goto="supervisor",
        update={"messages": [result["messages"][-1]]}
    )

def run_code_agent(state: SupervisorState) -> Command[Literal["supervisor"]]:
    result = code_agent.invoke({"messages": state["messages"]})
    return Command(
        goto="supervisor",
        update={"messages": [result["messages"][-1]]}
    )

# ── Supervisor graph ──────────────────────────────────────────────────────────
supervisor_builder = StateGraph(SupervisorState)
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("docs_agent", run_docs_agent)
supervisor_builder.add_node("code_agent", run_code_agent)
supervisor_builder.add_edge(START, "supervisor")
# Edges from specialist agents back to supervisor are handled by Command(goto=...)

multi_agent_system = supervisor_builder.compile(checkpointer=InMemorySaver())

# ── Test ──────────────────────────────────────────────────────────────────────
config = {"configurable": {"thread_id": "multi-agent-1"}}
result = multi_agent_system.invoke(
    {"messages": [HumanMessage(content="How many chunks fit in DBRX's context window?")]},
    config=config
)
print(result["messages"][-1].content)
```

**Key design decisions for multi-agent systems:**
1. **State sharing**: Use a shared state `TypedDict` or pass messages via `Command(update=...)`
2. **Loop prevention**: The supervisor must be able to route to `FINISH` — without this, the system loops infinitely
3. **Specialist isolation**: Each specialist agent should have its own system prompt and tool set
4. **Error handling**: If a specialist agent fails, the supervisor should be able to handle the error and route to a fallback

---

## Deep Dive / Advanced Topics

### `Command` for Dynamic Graph Navigation

`Command` is a more powerful alternative to conditional edges when a node needs to both
update state AND specify the next node dynamically:

```python
from langgraph.types import Command
from typing import Literal

def smart_router(state: AgentState) -> Command[Literal["expert_a", "expert_b", "__end__"]]:
    # This node can both update state and route in a single return
    decision = classify_query(state["messages"][-1].content)
    return Command(
        goto=decision,                          # specify next node
        update={"routing_metadata": decision}   # update state simultaneously
    )
```

Use `Command` when:
- You need to update state AND route in the same operation
- A node is resuming from an `interrupt()` and needs to choose the next step
- You are building a supervisor that updates shared state before dispatching

Use `add_conditional_edges` when:
- The routing logic is purely based on existing state (no new state to add)
- The routing function is simple and doesn't need to update state

### Parallel Execution (Fan-Out/Fan-In)

LangGraph supports parallel node execution by adding multiple edges from the same source:

```python
# Fan-out: retrieve from multiple sources in parallel
builder.add_edge("process_query", "search_ai_search")    # parallel
builder.add_edge("process_query", "search_sql_database") # parallel

# Fan-in: both parallel nodes must complete before merge
builder.add_edge("search_ai_search", "merge_results")
builder.add_edge("search_sql_database", "merge_results")

# The merge node receives state after BOTH parallel nodes finish
def merge_results(state: AgentState) -> dict:
    # state["retrieved_chunks"] contains results from BOTH parallel searches
    # because retrieved_chunks uses operator.add reducer
    return {"merged": True}
```

For parallel execution to work correctly, the shared state fields must use reducers (otherwise the second node's write overwrites the first's).

## Worked Examples & Practice

### End-to-End: Research Agent with Tool Use

A research agent that uses multiple tools to answer a complex question:

```python
@tool
def search_knowledge_base(query: str) -> str:
    """Search the internal knowledge base for relevant information.

    Args:
        query: Search query in natural language.

    Returns:
        Top 3 relevant document snippets with source references.
    """
    results = index.similarity_search(
        query_text=query, columns=["chunk_text", "title"], num_results=3
    )
    out = []
    for row in results["result"]["data_array"]:
        out.append(f"[{row[1]}]\n{row[0]}")
    return "\n\n".join(out) or "No relevant results found."

@tool
def run_sql_query(sql: str) -> str:
    """Run a SQL query against Unity Catalog tables to retrieve structured data.

    Args:
        sql: A SELECT SQL statement. Only SELECT is permitted.

    Returns:
        Query results as a formatted string, or an error message.
    """
    if not sql.strip().upper().startswith("SELECT"):
        return "Error: Only SELECT statements are permitted."
    try:
        df = spark.sql(sql)
        return df.limit(10).toPandas().to_string(index=False)
    except Exception as e:
        return f"Query failed: {e}"

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Args:
        expression: A Python math expression string (e.g. '32768 - 500 - 100 - 1000').

    Returns:
        The result as a string.
    """
    import ast
    try:
        # Only allow safe literals and basic operators
        tree = ast.parse(expression, mode='eval')
        result = eval(compile(tree, '<string>', 'eval'))
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"

# Build the research agent
research_agent = create_react_agent(
    model=ChatDatabricks(endpoint="databricks-meta-llama-3-1-70b-instruct"),
    tools=[search_knowledge_base, run_sql_query, calculate],
    checkpointer=InMemorySaver(),
    prompt="""You are a research assistant with access to a knowledge base, SQL query capability,
and a calculator. For each question:
1. First search the knowledge base for relevant context
2. If the question involves data, use SQL to retrieve specific numbers
3. Use the calculator for any arithmetic
4. Synthesize a clear, cited answer

Always cite your sources."""
)

config = {"configurable": {"thread_id": "research-1"}}
result = research_agent.invoke(
    {"messages": [{"role": "user", "content": "How many document chunks fit in DBRX's context window if each chunk is 400 tokens, system prompt is 800 tokens, and I want 1200 tokens of output space?"}]},
    config=config
)

# Print the full tool call + response trace
for msg in result["messages"]:
    role = type(msg).__name__
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        print(f"[{role}] Tool calls: {[tc['name'] for tc in msg.tool_calls]}")
    elif hasattr(msg, "content") and msg.content:
        print(f"[{role}] {msg.content[:300]}")
```

**Failure mode to observe:** Remove the calculator tool. Ask: "If the context window is 32,768 tokens, system prompt is 800 tokens, and I want 1,200 output tokens, how many 400-token chunks fit?" The agent will attempt to compute this in its head — and will frequently get the arithmetic wrong. Adding the calculator tool produces consistent, correct results. This demonstrates why giving agents precise computational tools matters.

## Common Pitfalls & Misconceptions

- **Pitfall:** Writing vague tool docstrings → **Why it happens:** Developers think the docstring is for humans → **Fix:** The docstring is the LLM's primary interface for understanding the tool. The first sentence determines whether the model calls the tool at all. Be specific: "Search the knowledge base for technical documentation" is better than "Useful for searching."

- **Pitfall:** Not handling tool errors gracefully → **Why it happens:** Tools are expected to always succeed in demos → **Fix:** Tools should return error messages as strings, not raise exceptions. The LLM can read an error string and adapt; an unhandled exception crashes the agent. Wrap tool implementations in `try/except` and return descriptive error messages.

- **Pitfall:** Building a supervisor with no `FINISH` route → **Why it happens:** Focus on the routing logic, forgetting the exit condition → **Fix:** Every supervisor must have a path to `END`. Without it, the system loops indefinitely. After a specialist answers, the supervisor must be able to determine "this has been answered, stop routing."

- **Pitfall:** Using `create_react_agent` for everything, including cases that need custom state → **Why it happens:** `create_react_agent` is simple → **Fix:** `create_react_agent` uses fixed `MessagesState` and a fixed ReAct loop. If you need to track custom state fields (retrieval score, user preferences, structured outputs from earlier steps), you must use a custom `StateGraph`.

- **Pitfall:** Binding tools directly to the model without `bind_tools()` in a custom loop → **Why it happens:** Forgetting the binding step → **Fix:** The model must know about the tools to generate tool call messages. Use `llm.bind_tools(tools)` before using the model in an agent node. Without this, the model generates free-form text even when a tool call would be appropriate.

## Key Definitions

| Term | Definition |
|---|---|
| Agent | An LLM that autonomously decides what actions (tool calls) to take, in what order, to accomplish a goal |
| ReAct | Reasoning + Acting — the most common agent pattern: LLM alternates between reasoning about next steps and calling tools |
| Tool | A Python function annotated with `@tool` that an LLM can call; the function's docstring is the LLM's interface for understanding when and how to call it |
| `bind_tools()` | A LangChain method that registers a list of tools with a model so it knows their schemas and can generate tool call messages |
| `create_react_agent` | A LangGraph prebuilt that creates a complete ReAct agent from a model + tools list; internally builds a `StateGraph` with agent and tools nodes |
| Tool node | A LangGraph node that executes the tool calls from the previous LLM message and returns `ToolMessage` results |
| Supervisor pattern | A multi-agent architecture where a router LLM dispatches incoming tasks to specialist sub-agents based on task type |
| `Command` | A LangGraph return type from a node that can both update state (`update=...`) and specify the next node (`goto=...`) in one operation |
| Unity Catalog function tool | A UC-registered SQL or Python function exposed as an agent tool via `UCFunctionToolkit`, subject to UC access controls and lineage tracking |
| MCP | Model Context Protocol — a standardized JSON-RPC interface for agents to connect to external tools and data sources; Databricks uses it for managed service integrations |

## Summary / Quick Recall

- Tools = `@tool`-decorated Python functions; the docstring is the LLM's decision guide
- `create_react_agent(model, tools, checkpointer=...)` — fast ReAct agent; uses `MessagesState`
- Custom ReAct loop: `agent_node` → conditional (`tool_calls?`) → `tools` node → back to `agent_node`
- `llm.bind_tools(tools)` is required for the model to generate tool call messages
- Multi-agent supervisor: router LLM → specialist subgraphs; must have a FINISH/END route
- `Command(goto=..., update=...)` — simultaneously route and update state from a node
- UC functions as tools: `UCFunctionToolkit` wraps UC-registered functions as LangChain tools with UC governance
- MCP: Databricks-standard protocol for connecting agents to external services

## Self-Check Questions

1. You build a ReAct agent with `create_react_agent`. It always generates text responses and never calls tools, even for questions that clearly require tool use. What is the most likely cause?

<details>
<summary>Answer</summary>
The tools were not bound to the model. In a custom `StateGraph`, you must call `llm.bind_tools(tools)` before using the model in the agent node. When using `create_react_agent`, the tools are passed directly: `create_react_agent(model=llm, tools=[tool1, tool2])`. If the tools list is empty or the model was not bound to the tools, the LLM generates free-form text without tool calls because it has no knowledge of what tools are available.
</details>

2. What is the difference between `create_react_agent` and building a custom `StateGraph` ReAct loop? When would you choose each?

<details>
<summary>Answer</summary>
`create_react_agent` is a prebuilt that creates a ReAct agent with fixed `MessagesState`, a standard agent node (LLM with tools), a tools execution node, and a conditional edge routing between them. Choose it when your agent only needs to maintain a conversation history and call tools — no custom state fields, no custom routing beyond "call tools or stop." Choose a custom `StateGraph` when you need: custom state fields (e.g., retrieval score, structured output), non-standard routing logic (e.g., routing to different tools based on query type), multi-agent patterns (supervisor/subgraph), human-in-the-loop interrupts with custom state, or any behavior that doesn't fit the standard ReAct loop.
</details>

3. Your multi-agent supervisor system has been running for 30 minutes on a single query and is consuming credits rapidly. What design flaw caused this and how do you fix it?

<details>
<summary>Answer</summary>
The supervisor has no FINISH/END route — every specialist agent response routes back to the supervisor, which then routes to another specialist, creating an infinite loop. Fix: the supervisor's routing function must include a "FINISH" option that routes to `END`. The supervisor should detect when the question has been sufficiently answered (e.g., the last message is a final answer from a specialist, or a maximum turn count has been reached) and terminate execution. Add `max_recursion` depth configuration or a `turns_elapsed` counter in state as a safety valve.
</details>

4. You define a tool called `query_database(sql: str) -> str`. An LLM uses it and generates `sql="DROP TABLE customer_data"`. How should the tool handle this?

<details>
<summary>Answer</summary>
The tool should validate the SQL and return an error string if it is not a SELECT statement: `if not sql.strip().upper().startswith("SELECT"): return "Error: Only SELECT statements are permitted."` The LLM can read this error and adjust its next action. Never raise an exception from a tool — that crashes the agent. Never execute DDL or DML statements from an agent tool without explicit authorization checks. Defense-in-depth: the tool validates input, UC-level permissions prevent the service principal from having DROP TABLE rights anyway.
</details>

5. How does the `Command` return type differ from using `add_conditional_edges` for routing? Give a scenario where `Command` is the better choice.

<details>
<summary>Answer</summary>
`add_conditional_edges` reads existing state to determine the next node but cannot update state simultaneously. `Command(goto=..., update=...)` both updates state and specifies the next node in one return. Choose `Command` when a node needs to write new state AND route based on something that isn't already in state — for example, a supervisor node that decides which specialist to call AND records that routing decision in state for audit/debugging: `return Command(goto=specialist_name, update={"routing_decision": specialist_name, "timestamp": datetime.now().isoformat()})`. Also prefer `Command` when resuming from `interrupt()` and the resume value determines both a state update and the next node.
</details>

## Further Reading

- [LangGraph `create_react_agent`](https://langchain-ai.github.io/langgraph/how-tos/create-react-agent/) — official prebuilt ReAct agent reference
- [Databricks — Connect agents to tools](https://docs.databricks.com/en/agents/agent-framework/agent-tool.html) — official guide to tool types: local Python, UC functions, MCP servers
- [Databricks — MCP on Databricks](https://docs.databricks.com/en/agents/mcp/index.html) — Model Context Protocol integration reference
- [Databricks — Build agents page](https://docs.databricks.com/en/agents/agent-framework/build-agents.html) — overview of Databricks agent development options (updated 2026-07-01)
- [LangGraph multi-agent patterns](https://docs.langchain.com/oss/python/langgraph/multi_agent) — supervisor, hierarchical, and collaborative multi-agent architectures
