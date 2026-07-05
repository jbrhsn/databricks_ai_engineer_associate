# Chains, LangGraph Patterns, and Memory

**Section:** Application Development | **Module:** Chains, Agents, and Frameworks | **Est. time:** 2.5 hrs
**Exam mapping:** Databricks GenAI Engineer Associate — Application Development domain (~30%); LangGraph graph construction, RAG chain patterns, and persistence are directly tested

## Learning Objectives

By the end of this chapter, you will be able to:

- Explain the LangGraph execution model: state, nodes, edges, and the graph compile/invoke cycle
- Define a typed state schema using `TypedDict` and `Annotated` reducers
- Build a linear LangGraph pipeline with `StateGraph`, `add_node`, `add_edge`, and `compile`
- Add conditional routing between nodes using `add_conditional_edges`
- Build a complete RAG chain in LangGraph: retriever node → prompt-building node → generation node
- Add multi-turn conversation memory using a checkpointer and `thread_id` configuration
- Use `ChatDatabricks` from `databricks-langchain` to call Databricks Foundation Model API endpoints inside LangGraph nodes

## Core Concepts

### The LangGraph Mental Model

LangGraph is described in its own documentation (2026) as a **low-level orchestration framework and runtime** for building stateful agents. Its key design principle: **LangGraph does not abstract prompts or LLM calls**. It only manages execution flow and state. The three core primitives:

| Primitive | What it is | Your responsibility |
|-----------|-----------|---------------------|
| **State** | A `TypedDict` that holds all data flowing through the graph | Define all fields; specify how updates are merged |
| **Node** | A Python function `(state) -> dict` that reads state, does work, returns a partial update | All LLM calls, retrievals, and business logic go here |
| **Edge** | A connection between nodes that determines execution order | Define the graph topology |

The complete lifecycle:
```
Define state schema
  → Build StateGraph
  → Add nodes (Python functions)
  → Add edges (execution flow)
  → Compile (optionally with checkpointer)
  → Invoke (pass input, get output)
```

### State: TypedDict with Reducers

State is a `TypedDict`. Each field holds data that flows through the graph.

**Simple state:**
```python
from typing import TypedDict

class RAGState(TypedDict):
    question: str       # user's question
    context: str        # retrieved chunks
    answer: str         # generated answer
```

**State with reducers** — when multiple nodes write to the same field, a reducer specifies how to merge updates. The most common is `operator.add` for message lists:

```python
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage

class AgentState(TypedDict):
    # operator.add appends new messages rather than replacing the list
    messages: Annotated[list[AnyMessage], operator.add]
    # Simple fields — last writer wins (default)
    context: str
```

**`MessagesState` shorthand** — LangGraph provides a built-in state that includes a `messages` field pre-configured with `operator.add`:
```python
from langgraph.graph import MessagesState
# Equivalent to: TypedDict with messages: Annotated[list[AnyMessage], operator.add]
```

### Nodes: Python Functions

A node is a Python function that accepts the full state and returns a dict of partial updates. Only the keys you return are updated in state; all other keys remain unchanged.

```python
from databricks_langchain import ChatDatabricks

llm = ChatDatabricks(endpoint="databricks-meta-llama-3-1-70b-instruct")

def generate_node(state: RAGState) -> dict:
    """Generates an answer using the retrieved context."""
    prompt = f"""Answer the question based only on the context below.

Context: {state['context']}

Question: {state['question']}

Answer:"""
    response = llm.invoke(prompt)
    return {"answer": response.content}
    # Only 'answer' is updated; 'question' and 'context' are preserved
```

**Key rule:** Return only the fields you want to update. Never return the full state — that replaces all other fields with None.

### Edges: Connecting Nodes

**Linear edge** — always goes from node A to node B:
```python
builder.add_edge("retrieve", "generate")  # retrieve always goes to generate
```

**Conditional edge** — a function inspects state and returns the name of the next node:
```python
def route_after_generation(state: RAGState) -> str:
    """Decide whether to return the answer or flag for human review."""
    if len(state["answer"]) < 50:  # very short answer — probably didn't find context
        return "flag_for_review"
    return END  # return to caller

builder.add_conditional_edges(
    "generate",             # source node
    route_after_generation, # routing function
    {                       # optional mapping: return value → node name
        "flag_for_review": "review_node",
        END: END
    }
)
```

**`START` and `END`** are sentinel nodes provided by LangGraph:
- `START` — the entry point; the first node receives the initial input
- `END` — the exit point; the graph finishes and returns state

### Building and Compiling a Graph

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(RAGState)

# Add nodes
builder.add_node("retrieve", retrieve_node)
builder.add_node("generate", generate_node)

# Add edges
builder.add_edge(START, "retrieve")  # entry point
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)    # exit point

# Compile — this validates the graph and returns an executable
graph = builder.compile()
```

### Invoking a Graph

```python
# Single invocation
result = graph.invoke({"question": "What is Delta Lake?"})
print(result["answer"])

# Streaming node outputs
for chunk in graph.stream({"question": "What is Delta Lake?"}):
    print(chunk)
# Each chunk is {node_name: state_update_dict}
```

---

### Building a RAG Chain in LangGraph

A RAG chain has three natural nodes:

```
START → [retrieve] → [build_prompt] → [generate] → END
```

**Full implementation:**

```python
%pip install langgraph databricks-langchain langchain
dbutils.library.restartPython()

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from databricks_langchain import ChatDatabricks, DatabricksEmbeddings
from databricks.vector_search.client import VectorSearchClient
from langchain_core.messages import HumanMessage, SystemMessage

# ── State ─────────────────────────────────────────────────────────────────────
class RAGState(TypedDict):
    question: str
    retrieved_chunks: list[str]
    answer: str

# ── Dependencies ──────────────────────────────────────────────────────────────
llm = ChatDatabricks(endpoint="databricks-meta-llama-3-1-70b-instruct")
vsc = VectorSearchClient()
index = vsc.get_index("rag_demo_endpoint", "main.rag_demo.document_chunks_gold_index")

# ── Nodes ─────────────────────────────────────────────────────────────────────
def retrieve(state: RAGState) -> dict:
    """Query AI Search for relevant chunks."""
    results = index.similarity_search(
        query_text=state["question"],
        columns=["chunk_text"],
        num_results=5
    )
    chunks = [row[0] for row in results["result"]["data_array"]]
    return {"retrieved_chunks": chunks}

def generate(state: RAGState) -> dict:
    """Generate an answer from retrieved context."""
    context = "\n\n---\n\n".join(state["retrieved_chunks"])
    messages = [
        SystemMessage(content="""You are a helpful assistant. Answer the question based only
on the provided context. If the context doesn't contain the answer, say "I don't know."
Cite which part of the context supports your answer."""),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {state['question']}")
    ]
    response = llm.invoke(messages)
    return {"answer": response.content}

# ── Graph ─────────────────────────────────────────────────────────────────────
builder = StateGraph(RAGState)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)

rag_graph = builder.compile()

# ── Test ──────────────────────────────────────────────────────────────────────
result = rag_graph.invoke({"question": "What is the token limit for bge-large-en?"})
print(result["answer"])
```

---

### LangGraph Persistence: Multi-Turn Memory

Without a checkpointer, every `invoke` call is stateless — the graph has no memory of previous turns. A **checkpointer** writes the graph state to durable storage after each node executes, keyed by a `thread_id`. Passing the same `thread_id` resumes the conversation from where it left off.

**Available checkpointers:**

| Checkpointer | Storage | Use case |
|-------------|---------|---------|
| `InMemorySaver` | RAM (lost on restart) | Development, testing |
| `SqliteSaver` | Local SQLite file | Local dev with persistence |
| `PostgresSaver` | PostgreSQL | Production (requires `langgraph-checkpoint-postgres`) |

**Adding memory to the RAG chain:**

```python
from langgraph.checkpoint.memory import InMemorySaver
from typing import Annotated
import operator
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage

# ── State with conversation history ──────────────────────────────────────────
class ConversationalRAGState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]  # full conversation history
    retrieved_chunks: list[str]

# ── Updated generate node ─────────────────────────────────────────────────────
def generate_with_history(state: ConversationalRAGState) -> dict:
    context = "\n\n---\n\n".join(state["retrieved_chunks"])
    # Build messages: system + conversation history + context injection
    system = SystemMessage(content=f"""You are a helpful assistant.
Answer based on this context:

{context}

If you can't find the answer in the context, say "I don't know."""")

    # Use the full message history (the last message is the current question)
    messages = [system] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [AIMessage(content=response.content)]}

def retrieve_from_last_message(state: ConversationalRAGState) -> dict:
    # Extract the latest user question from the message list
    last_human = next(
        m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)
    )
    results = index.similarity_search(
        query_text=last_human.content,
        columns=["chunk_text"],
        num_results=5
    )
    chunks = [row[0] for row in results["result"]["data_array"]]
    return {"retrieved_chunks": chunks}

# ── Graph with checkpointer ───────────────────────────────────────────────────
builder2 = StateGraph(ConversationalRAGState)
builder2.add_node("retrieve", retrieve_from_last_message)
builder2.add_node("generate", generate_with_history)
builder2.add_edge(START, "retrieve")
builder2.add_edge("retrieve", "generate")
builder2.add_edge("generate", END)

# InMemorySaver for dev; replace with PostgresSaver for production
checkpointer = InMemorySaver()
conversational_rag = builder2.compile(checkpointer=checkpointer)

# ── Multi-turn conversation ───────────────────────────────────────────────────
config = {"configurable": {"thread_id": "user-session-1"}}

# Turn 1
result1 = conversational_rag.invoke(
    {"messages": [HumanMessage(content="What is the token limit for bge-large-en?")]},
    config=config
)
print("Turn 1:", result1["messages"][-1].content)

# Turn 2 — the graph remembers the prior turn via the checkpointer
result2 = conversational_rag.invoke(
    {"messages": [HumanMessage(content="How does that compare to text-embedding-3-small?")]},
    config=config
)
print("Turn 2:", result2["messages"][-1].content)
```

**The `thread_id` is the conversation cursor.** Same `thread_id` = resume; different `thread_id` = new conversation.

---

### LangGraph `interrupt()` for Human-in-the-Loop

`interrupt()` pauses graph execution, surfaces a payload to the caller, and waits for a response. When resumed with `Command(resume=...)`, execution continues from the node where `interrupt()` was called (the node re-runs from the start, so pre-interrupt code runs again).

**RAG with human review gate:**

```python
from langgraph.types import interrupt, Command
from typing import Literal

class ReviewableRAGState(TypedDict):
    question: str
    retrieved_chunks: list[str]
    answer: str
    approved: bool

def generate_for_review(state: ReviewableRAGState) -> dict:
    """Generate answer, then pause for human review before returning."""
    context = "\n\n---\n\n".join(state["retrieved_chunks"])
    messages = [
        SystemMessage(content="Answer based only on the provided context."),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {state['question']}")
    ]
    response = llm.invoke(messages)
    draft_answer = response.content

    # Pause here — surface the draft to the caller
    approved = interrupt({
        "instruction": "Review this answer before it is returned to the user.",
        "draft_answer": draft_answer,
        "question": state["question"]
    })

    # Code below runs after resume
    return {"answer": draft_answer, "approved": approved}

def route_after_review(state: ReviewableRAGState) -> Literal["return_answer", "regenerate"]:
    return "return_answer" if state["approved"] else "regenerate"

# Build graph with interrupt support
builder3 = StateGraph(ReviewableRAGState)
builder3.add_node("retrieve", retrieve)
builder3.add_node("generate", generate_for_review)
builder3.add_conditional_edges("generate", route_after_review, {
    "return_answer": END,
    "regenerate": "generate"  # loop back and regenerate
})
builder3.add_edge(START, "retrieve")
builder3.add_edge("retrieve", "generate")

checkpointer3 = InMemorySaver()
review_graph = builder3.compile(checkpointer=checkpointer3)

config = {"configurable": {"thread_id": "review-thread-1"}}

# First invocation — hits the interrupt and pauses
stream = review_graph.stream_events(
    {"question": "What is MLflow?", "retrieved_chunks": [], "answer": "", "approved": False},
    config=config, version="v3"
)
_ = stream.output  # drive to completion
print("Interrupt payload:", stream.interrupts[0].value)

# Resume with approval decision
resumed = review_graph.stream_events(
    Command(resume=True),  # True = approved
    config=config, version="v3"
)
final = resumed.output
print("Final answer:", final["answer"])
```

**Critical rule (from LangGraph docs 2026):** The node that contains `interrupt()` **re-executes from the beginning** on resume. Any code before the `interrupt()` call runs again. Keep pre-interrupt code idempotent (or place side effects after the `interrupt()` call).

---

## Deep Dive / Advanced Topics

### State Reducers — Custom Merge Logic

The default state update is "last writer wins" — returning a value overwrites the previous value. For fields that should accumulate rather than overwrite, use `Annotated[T, reducer_fn]`:

```python
from typing import Annotated
import operator

class PipelineState(TypedDict):
    messages: Annotated[list, operator.add]     # accumulate messages
    retrieved_docs: Annotated[list, operator.add]  # accumulate across parallel retrieval
    final_answer: str                             # overwrite (default)
    error_count: Annotated[int, operator.add]     # accumulate error counts
```

Custom reducer function:
```python
def keep_latest_5(existing: list, new: list) -> list:
    """Keep only the 5 most recent items."""
    return (existing + new)[-5:]

class BoundedState(TypedDict):
    recent_queries: Annotated[list[str], keep_latest_5]
```

### Subgraphs for Reusable Logic

A compiled LangGraph graph can be used as a node inside another graph — this is called a **subgraph**. Subgraphs are the reuse mechanism in LangGraph:

```python
# Define a retrieval subgraph that can be reused
retrieval_builder = StateGraph(RetrievalState)
retrieval_builder.add_node("search_ai_search", search_node)
retrieval_builder.add_node("rerank", rerank_node)
retrieval_builder.add_edge(START, "search_ai_search")
retrieval_builder.add_edge("search_ai_search", "rerank")
retrieval_builder.add_edge("rerank", END)
retrieval_subgraph = retrieval_builder.compile()

# Use the subgraph as a node in a parent graph
parent_builder = StateGraph(ParentState)
parent_builder.add_node("retrieve", retrieval_subgraph)  # subgraph as node
parent_builder.add_node("generate", generate_node)
parent_builder.add_edge(START, "retrieve")
parent_builder.add_edge("retrieve", "generate")
parent_builder.add_edge("generate", END)
```

### `ChatDatabricks` — Connecting to Foundation Model APIs

`ChatDatabricks` from the `databricks-langchain` package is the LangChain/LangGraph integration for Databricks Foundation Model API endpoints:

```python
from databricks_langchain import ChatDatabricks

# Standard LLM node
llm = ChatDatabricks(
    endpoint="databricks-meta-llama-3-1-70b-instruct",
    temperature=0,
    max_tokens=1024
)

# With Unity AI Gateway
llm_governed = ChatDatabricks(
    endpoint="<ai-gateway-endpoint>",
    use_ai_gateway=True  # route through Unity AI Gateway for governance
)
```

`ChatDatabricks` returns a standard LangChain `BaseMessage` object, compatible with all LangGraph message state patterns.

## Worked Examples & Practice

### Complete Conversational RAG Pipeline — Production Pattern

This is the production-grade pattern combining all concepts: typed state, RAG retrieval, multi-turn memory, and conditional quality routing.

```python
%pip install langgraph databricks-langchain databricks-vectorsearch
dbutils.library.restartPython()

from typing import TypedDict, Annotated, Literal
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from databricks_langchain import ChatDatabricks
from databricks.vector_search.client import VectorSearchClient
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage

# ── Config ────────────────────────────────────────────────────────────────────
llm = ChatDatabricks(endpoint="databricks-meta-llama-3-1-70b-instruct", temperature=0)
vsc = VectorSearchClient()
index = vsc.get_index("rag_demo_endpoint", "main.rag_demo.document_chunks_gold_index")

# ── State ─────────────────────────────────────────────────────────────────────
class ConvRAGState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    context_chunks: list[str]
    retrieval_score: float   # quality signal for conditional routing

# ── Nodes ─────────────────────────────────────────────────────────────────────
def retrieve(state: ConvRAGState) -> dict:
    last_human = next(m for m in reversed(state["messages"]) if isinstance(m, HumanMessage))
    results = index.similarity_search(
        query_text=last_human.content,
        columns=["chunk_text"],
        num_results=5,
        query_type="HYBRID"  # vector + keyword search
    )
    chunks = [row[0] for row in results["result"]["data_array"]]
    scores = [row[-1] for row in results["result"]["data_array"]]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    return {"context_chunks": chunks, "retrieval_score": avg_score}

def generate(state: ConvRAGState) -> dict:
    context = "\n\n---\n\n".join(state["context_chunks"])
    system = SystemMessage(content=f"""You are a helpful assistant answering questions
about Databricks. Use only the provided context to answer.
If the context doesn't contain the answer, say so clearly.

Context:
{context}""")
    response = llm.invoke([system] + state["messages"])
    return {"messages": [AIMessage(content=response.content)]}

def route_on_retrieval(state: ConvRAGState) -> Literal["generate", "no_context_response"]:
    """Route based on retrieval quality."""
    return "generate" if state["retrieval_score"] > 0.3 else "no_context_response"

def no_context_response(state: ConvRAGState) -> dict:
    return {"messages": [AIMessage(
        content="I couldn't find relevant information to answer that question. "
                "Please try rephrasing or ask about a different topic."
    )]}

# ── Graph ─────────────────────────────────────────────────────────────────────
builder = StateGraph(ConvRAGState)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_node("no_context_response", no_context_response)

builder.add_edge(START, "retrieve")
builder.add_conditional_edges("retrieve", route_on_retrieval, {
    "generate": "generate",
    "no_context_response": "no_context_response"
})
builder.add_edge("generate", END)
builder.add_edge("no_context_response", END)

graph = builder.compile(checkpointer=InMemorySaver())

# ── Multi-turn session ────────────────────────────────────────────────────────
config = {"configurable": {"thread_id": "session-abc"}}

questions = [
    "What is Databricks AI Search?",
    "How does it handle hybrid search?",
    "What algorithm does it use internally?"
]

for q in questions:
    result = graph.invoke(
        {"messages": [HumanMessage(content=q)]},
        config=config
    )
    print(f"Q: {q}")
    print(f"A: {result['messages'][-1].content[:200]}...\n")
```

**Failure mode to observe:** Comment out the `checkpointer=InMemorySaver()` argument. Run the three-question sequence. Observe that question 2 ("How does it handle...") gets a poor answer because the graph has no memory of question 1 — it has lost the context that question 2 refers to AI Search. This demonstrates why checkpointers are essential for conversational agents.

## Common Pitfalls & Misconceptions

- **Pitfall:** Returning the full state from a node → **Why it happens:** Feels like the "safe" thing to do → **Fix:** Return only the fields you changed. Returning the full state can accidentally overwrite fields that other nodes updated, causing subtle state corruption in parallel graphs.

- **Pitfall:** Forgetting to add `START` and `END` edges → **Why it happens:** Easy to omit in the heat of building a graph → **Fix:** `compile()` will raise a validation error if the graph has unreachable nodes or no entry/exit points. Always add `builder.add_edge(START, "first_node")` and `builder.add_edge("last_node", END)`.

- **Pitfall:** Using `InMemorySaver` in production → **Why it happens:** It works perfectly in notebooks → **Fix:** `InMemorySaver` stores state in RAM; the state is lost when the process restarts. Use `PostgresSaver` or `SqliteSaver` for any production application that needs persistent conversation history.

- **Pitfall:** Placing non-idempotent side effects before `interrupt()` → **Why it happens:** The interrupt is placed at the end of a node for "convenience" → **Fix:** On resume, the node re-executes from the beginning — all code before `interrupt()` runs again. An API call that creates a record will create it twice. Either make pre-interrupt code idempotent (upsert not insert) or move side effects after the `interrupt()` call.

- **Pitfall:** Using the same `thread_id` for unrelated conversations → **Why it happens:** Hardcoded test thread ID copied between tests → **Fix:** Generate a unique UUID per conversation: `config = {"configurable": {"thread_id": str(uuid.uuid4())}}`. Reusing a `thread_id` always resumes the old conversation context.

- **Pitfall:** Wrapping `interrupt()` in a `try/except Exception` block → **Why it happens:** Defensive coding → **Fix:** `interrupt()` works by raising a special internal exception. A bare `except Exception` catches it, preventing the interrupt from working. Either use specific exception types or avoid try/except around `interrupt()`.

## Key Definitions

| Term | Definition |
|---|---|
| LangGraph | A low-level Python framework for building stateful, graph-based agent workflows; manages state, execution flow, persistence, and human-in-the-loop |
| `StateGraph` | The main LangGraph class for defining a computational graph; accepts a state `TypedDict` class |
| Node | A Python function `(state) -> dict` that reads graph state and returns a partial update |
| Edge | A directed connection between nodes specifying execution order |
| Conditional edge | An edge whose target node is determined at runtime by a routing function that inspects state |
| Reducer | A function that specifies how to merge new values into an existing state field when multiple sources write to the same field |
| Checkpointer | A persistence backend (memory, SQLite, PostgreSQL) that saves graph state after each node, enabling multi-turn memory and fault tolerance |
| `thread_id` | A string identifier passed in graph config that tells the checkpointer which state thread to load and write to; each unique `thread_id` is an independent conversation |
| `interrupt()` | A LangGraph function that pauses graph execution at the call site, saves state to the checkpointer, and surfaces a payload to the caller; resumes via `Command(resume=...)` |
| `Command` | A LangGraph type used to resume interrupted execution: `Command(resume=value)` passes a value back into the interrupted node |
| `ChatDatabricks` | A `databricks-langchain` class for calling Databricks Foundation Model API endpoints inside LangGraph nodes; returns standard LangChain message objects |

## Summary / Quick Recall

- LangGraph = state + nodes + edges; compile → invoke
- State is a `TypedDict`; use `Annotated[list, operator.add]` for accumulating fields
- Nodes return partial state updates — only the changed keys
- `add_conditional_edges(source, routing_fn, {return_value: target_node})` for branching
- `builder.compile(checkpointer=...)` enables memory; `thread_id` in config selects the conversation
- `InMemorySaver` for dev; `PostgresSaver` for production
- `interrupt(payload)` pauses the graph; `Command(resume=value)` resumes it
- The node containing `interrupt()` re-runs from the beginning on resume — keep pre-interrupt code idempotent
- `ChatDatabricks(endpoint=...)` connects LangGraph nodes to Databricks Foundation Model APIs

## Self-Check Questions

1. A colleague builds a RAG graph and notices that in multi-turn conversations, the second question always ignores the context established in the first turn. What is the most likely cause and fix?

<details>
<summary>Answer</summary>
The graph was compiled without a checkpointer (or is being invoked without a consistent `thread_id`). Without a checkpointer, every `invoke` call starts with a blank state — the graph has no memory of prior turns. Fix: compile with `checkpointer=InMemorySaver()` (dev) or `PostgresSaver` (prod) and pass the same `config = {"configurable": {"thread_id": "consistent-id"}}` on every call for the same conversation session.
</details>

2. You have a LangGraph node that creates an audit log entry in a database, then calls `interrupt()` to ask for human approval. After approval, the audit entry appears twice in the database. Why, and how do you fix it?

<details>
<summary>Answer</summary>
When `interrupt()` is called, the graph saves state and pauses. On resume, the node **re-executes from the beginning** — so the audit log creation code runs again, creating a duplicate entry. Fix: move the audit log creation to after the `interrupt()` call so it only executes once (after approval is received). Alternatively, make the database write idempotent (use upsert with a deterministic ID derived from the request, so writing it twice has no effect).
</details>

3. What is the difference between `add_edge` and `add_conditional_edges`? When do you use each?

<details>
<summary>Answer</summary>
`add_edge(source, target)` always routes from source to target — unconditional. `add_conditional_edges(source, routing_fn, mapping)` calls a routing function at runtime to determine the target based on the current state. Use `add_edge` for sequential pipelines where the next step is always the same. Use `add_conditional_edges` when the next step depends on the result of the current node — e.g., routing between "generate" and "no_context_response" based on retrieval score, or routing a ReAct agent between "call_tool" and "end" based on whether the LLM made a tool call.
</details>

4. Your RAG state has a `retrieved_chunks: list[str]` field. Multiple parallel retrieval nodes write to this field. Without a reducer, what happens? How do you fix it?

<details>
<summary>Answer</summary>
Without a reducer, the last parallel node to write wins — earlier writes are overwritten. You lose all but one node's results. Fix: annotate the field with `operator.add` as a reducer: `retrieved_chunks: Annotated[list[str], operator.add]`. This tells LangGraph to append new values to the list rather than replace it, so all parallel results are accumulated.
</details>

5. What does `ChatDatabricks(endpoint="databricks-meta-llama-3-1-70b-instruct")` return when you call `.invoke()` on it, and why does this matter for LangGraph?

<details>
<summary>Answer</summary>
It returns a LangChain `AIMessage` object — a standard LangChain message type with a `.content` attribute containing the model's text response. This matters for LangGraph because the `Annotated[list[AnyMessage], operator.add]` state pattern (and `MessagesState`) expects standard LangChain message objects. Using `ChatDatabricks` ensures full compatibility with LangGraph's message state, streaming, and history management patterns without any conversion step.
</details>

## Further Reading

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — official LangGraph documentation; the definitive reference for graph construction, state, and persistence
- [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence) — checkpointers, stores, thread IDs, and multi-turn memory
- [LangGraph interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts) — `interrupt()`, `Command(resume=...)`, and human-in-the-loop patterns (2026, the primary source for this chapter)
- [Databricks — Build custom agents](https://docs.databricks.com/en/agents/agent-framework/author-agent.html) — official Databricks guide to custom agent development with LangGraph (updated 2026-07-01)
- [databricks-langchain PyPI](https://pypi.org/project/databricks-langchain/) — `ChatDatabricks` and `DatabricksEmbeddings` integration package
