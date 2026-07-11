# LangChain Components for RAG and Agent Pipelines

**Section:** Application Development | **Module:** Frameworks and Prompting | **Est. time:** 3 hrs | **Exam mapping:** Domain 3 (Application Development) — 30% weight

---

## TL;DR

LangChain provides a composable set of building blocks — LCEL, ChatPromptTemplate, Retriever, RunnablePassthrough, output parsers, and ChatDatabricks — that wire together into production-grade RAG and agent pipelines using the `|` pipe operator. LCEL composition is not syntax sugar: it is the mechanism that enables streaming, async execution, and parallel branches to work automatically on any composed chain. **The one thing to remember: every component in a LangChain chain implements the Runnable interface, and composing them with `|` is what buys you streaming, batching, and async for free.**

---

## ELI5 — Explain It Like I'm 5

Think of a restaurant kitchen with a strict pass system: the prep cook hands a tray to the grill cook, who hands a tray to the plating station, who hands a tray to the server. Each station only accepts a specific tray format and only produces a specific tray format — no improvising. If a station decides to go rogue and work outside the pass system (passing plates directly to the server, skipping the grill), the whole kitchen loses the ability to track orders, add a new station, or speed up multiple orders at once. LangChain components work the same way: every component is a "station" that has a defined input tray and output tray. The LCEL `|` operator is the pass system — it guarantees that each station's output tray goes to the correct next station in the right format. A common misconception is that you could get the same result by calling each station yourself with a loop; in reality, the pass system is what allows the kitchen to fill five orders simultaneously and stream food out as each plate is ready, rather than waiting for the whole table's order to complete.

---

## Learning Objectives

By the end of this chapter you will be able to:
- [ ] Explain how LCEL composes Runnables with the `|` operator and why this enables streaming and async automatically
- [ ] Build a complete RAG chain using `ChatDatabricks`, `DatabricksVectorSearch.as_retriever()`, `ChatPromptTemplate`, and `StrOutputParser`
- [ ] Describe the purpose and mechanism of `RunnablePassthrough` and `RunnableLambda` for data injection and transformation within a chain
- [ ] Select the correct output parser (`StrOutputParser`, `JsonOutputParser`, `PydanticOutputParser`) given a downstream consumer's requirements
- [ ] Diagnose why sequential `.invoke()` calls break streaming and what the correct LCEL refactor looks like

---

## Visual Overview

### LCEL Runnable Execution Pipeline

```
chain = retriever | format_docs | prompt | llm | parser
          │              │          │       │       │
          ▼              ▼          ▼       ▼       ▼
    List[Document]   str (ctx)   ChatPrompt  AIMsg  str
                                 Value

invoke("query") ──► Each Runnable.invoke() called in sequence
stream("query") ──► Each Runnable.stream() called; tokens flow as yielded
batch(["q1",…]) ──► Each Runnable.batch() called; inputs run in parallel
astream("q")    ──► async generator — each Runnable.astream() called
```

### RAG Chain Data Flow

```
User Query (str)
      │
      ▼
┌─────────────────────┐
│   DatabricksVector  │  .as_retriever(search_kwargs={"k": 4})
│   Search Retriever  │  returns List[Document]
└────────┬────────────┘
         │ List[Document]
         ▼
┌─────────────────────┐
│  RunnableLambda /   │  format_docs(docs) → str
│  format_docs        │  joins page_content with newlines
└────────┬────────────┘
         │ str (formatted context)
         ▼
┌─────────────────────┐
│  ChatPromptTemplate │  from_messages([("system","…"), ("human","{question}\nContext:{context}")])
│  + Passthrough      │  RunnablePassthrough merges {question} + {context}
└────────┬────────────┘
         │ ChatPromptValue (list of messages)
         ▼
┌─────────────────────┐
│   ChatDatabricks    │  endpoint="databricks-dbrx-instruct", max_tokens=512
└────────┬────────────┘
         │ AIMessage
         ▼
┌─────────────────────┐
│  StrOutputParser    │  .content → str
└────────┬────────────┘
         │
         ▼
    Final Answer (str)
```

### Component Responsibility Map

```
Component                   Responsibility                  Where in Databricks
─────────────────────────────────────────────────────────────────────────────
ChatDatabricks              LLM call via Foundation         databricks_langchain
                            Model API endpoint              ChatDatabricks(endpoint=…)

DatabricksVectorSearch      Vector similarity search;       databricks_langchain
.as_retriever()             returns List[Document]          DatabricksVectorSearch(…)
                                                            .as_retriever()

ChatPromptTemplate          Fills named variables into      langchain_core.prompts
                            a list of chat messages         ChatPromptTemplate.from_messages()

MessagesPlaceholder         Injects message history         langchain_core.prompts
                            into prompt at invocation time  MessagesPlaceholder("history")

RunnablePassthrough         Passes input unchanged;         langchain_core.runnables
                            used to fork input to          RunnablePassthrough()
                            multiple branches (RunnableMap)

RunnableLambda              Wraps a Python function         langchain_core.runnables
                            as a Runnable                   RunnableLambda(fn)

StrOutputParser             Extracts .content string        langchain_core.output_parsers
                            from AIMessage                  StrOutputParser()

PydanticOutputParser        Validates LLM output against    langchain_core.output_parsers
                            a Pydantic schema               PydanticOutputParser(pydantic_object=…)
```

---

## Key Concepts

### LCEL — LangChain Expression Language

LCEL is a declarative composition syntax for building chains from Runnable components using Python's `|` (bitwise OR) operator, overloaded by the `Runnable` base class. Under the hood, `A | B` calls `A.__or__(B)`, which returns a `RunnableSequence` containing both components. When you call `chain.invoke(input)`, the sequence calls `A.invoke(input)` and passes the result as input to `B.invoke(result)` — synchronously. When you call `chain.stream(input)`, the sequence calls `A.stream(input)` and pipes the resulting generator through `B.transform(generator)`, allowing tokens to flow as they arrive without buffering the full output. When you call `chain.batch([i1, i2, …])`, the sequence uses `ThreadPoolExecutor` to invoke each input through the full chain in parallel. All of these execution modes are inherited automatically by any chain built with `|` — you do not implement them per chain. In Databricks agent code, LCEL chains appear in LangGraph node functions and in MLflow-logged `ResponsesAgent` implementations; the `ChatDatabricks` model is a drop-in Runnable that participates in any LCEL chain.

### ChatPromptTemplate and MessagesPlaceholder

`ChatPromptTemplate` is a Runnable that takes a dictionary of variable values and returns a `ChatPromptValue` (a list of formatted chat messages). It is constructed with `ChatPromptTemplate.from_messages(messages)`, where `messages` is a list of `(role, template_string)` tuples — `role` is `"system"`, `"human"`, or `"ai"`, and `template_string` uses `{variable_name}` placeholders. At invocation time, the template calls Python's `str.format_map()` on each template string, substituting the values from the input dictionary. `MessagesPlaceholder` is a special placeholder that inserts a pre-existing list of `BaseMessage` objects (e.g., conversation history) into the prompt at a named position. Under the hood, when the template encounters a `MessagesPlaceholder("history")`, it reads `input["history"]` (which must be `List[BaseMessage]`) and splices those messages into the output list at that position. In Databricks RAG chains, `ChatPromptTemplate.from_messages([("system", "Answer using context: {context}"), ("human", "{question}")])` is the standard pattern, and it appears in every chain logged to MLflow as the prompt component.

### Retriever Interface

A retriever is any object that implements the Runnable interface and whose `.invoke(query: str) -> List[Document]` method returns a list of `Document` objects (each with `page_content: str` and `metadata: dict`) given a string query. The interface is abstract: the underlying implementation may be dense vector search, sparse BM25, or a hybrid — the chain does not care. Under the hood, `DatabricksVectorSearch.as_retriever(search_kwargs={"k": 4})` returns a `VectorStoreRetriever` wrapper that calls the Databricks Vector Search REST API with the given query and top-k parameter, deserializes the response into `Document` objects, and returns them. Because it is a Runnable, it participates in LCEL chains directly: `retriever | format_docs | prompt | llm | parser`. In Databricks, `DatabricksVectorSearch` is initialized with the Unity Catalog index name, the workspace token, and an embedding function; `as_retriever()` accepts `search_kwargs` such as `{"k": 4, "query_type": "ANN"}` to control retrieval behaviour.

### RunnablePassthrough and RunnableLambda

`RunnablePassthrough` is a no-op Runnable that returns its input unchanged; its primary use is inside `RunnableParallel` (also written as `{"key": runnable}`) to forward the original input to a downstream component that needs it alongside a transformed branch. For example, in a RAG chain, `{"context": retriever | format_docs, "question": RunnablePassthrough()}` creates a parallel branch: one branch retrieves and formats documents, the other passes the original question string through — both results are merged into a single dict fed to the prompt. Under the hood, `RunnableParallel` uses `ThreadPoolExecutor` to run both branches concurrently, then merges results. `RunnableLambda` is a Runnable wrapper for any Python callable: `RunnableLambda(lambda docs: "\n\n".join(d.page_content for d in docs))` wraps a formatting function so it can participate in an LCEL chain with a defined input type (`List[Document]`) and output type (`str`). In Databricks agent code, `RunnableLambda` is the standard way to inject custom preprocessing or post-processing logic into a chain without breaking the streaming contract.

### Output Parsers

Output parsers are Runnables that sit between the LLM and the downstream consumer, transforming an `AIMessage` (the LLM's raw output) into a typed Python object. `StrOutputParser` extracts the `.content` string attribute from an `AIMessage` and returns it as a plain Python `str`; it is the correct choice when downstream consumers expect a string. `JsonOutputParser` calls `json.loads()` on the message content string, returning a `dict`; it optionally accepts a Pydantic schema to guide LLM output format via `get_format_instructions()`. `PydanticOutputParser` validates the parsed JSON against a Pydantic model, raising a `OutputParserException` if the structure does not match; this is the correct choice when downstream code accesses structured fields by name. Under the hood, all three implement `parse(text: str)` and are exposed as Runnables via `Runnable.invoke(AIMessage) -> T`. In Databricks pipelines, `StrOutputParser` is used for streaming chat responses to users, while `PydanticOutputParser` is used for tool argument extraction and structured data output that flows into downstream tools.

### ChatDatabricks

`ChatDatabricks` is the `langchain_community` (or `databricks_langchain`) LangChain wrapper for the Databricks Foundation Model API serving endpoints. It is a `BaseChatModel` subclass and therefore a full Runnable: it accepts `List[BaseMessage]` as input and returns `AIMessage` as output, and it supports `stream()` and `astream()` for token-level streaming. Under the hood, it calls the Databricks Model Serving REST endpoint (`/serving-endpoints/{endpoint}/invocations`) using the workspace URL and a personal access token (or OAuth token); it handles the OpenAI-compatible chat completion schema automatically. Initialization: `ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens=512, temperature=0.0)` — `endpoint` is the serving endpoint name in the workspace (including external model endpoints like `databricks-claude-sonnet-4-5`). In Databricks agent code, `ChatDatabricks` appears as the LLM node in LangGraph graphs, in LCEL RAG chains logged to MLflow, and in `create_react_agent(ChatDatabricks(...), tools=[...])` patterns for tool-calling agents.


---

## Key Parameters / Configuration Knobs

| Parameter | What it controls | Decision rule |
|---|---|---|
| `ChatDatabricks(endpoint=…)` | Which Foundation Model API endpoint receives requests | Use `"databricks-dbrx-instruct"` for open-weight general tasks; use `"databricks-claude-sonnet-4-5"` when tool-calling accuracy or complex reasoning is required; use a custom endpoint name for fine-tuned or provisioned models. |
| `ChatDatabricks(max_tokens=…)` | Maximum tokens in the LLM response | Set to 512 for standard RAG answers; set to 2048+ for summarization or multi-step reasoning tasks. Unset (None) means the endpoint default applies — dangerous in production as it may exceed quota. |
| `ChatDatabricks(temperature=…)` | Sampling randomness | Set `0.0` for deterministic RAG and tool-calling chains; set `0.3–0.7` for creative generation. Higher values increase hallucination risk in factual pipelines. |
| `DatabricksVectorSearch.as_retriever(search_kwargs={"k": N})` | Number of documents retrieved per query | Set `k=4` as a baseline; increase to 8–10 if the context window allows and the answer requires synthesis across many documents. Setting `k` too high degrades answer quality by flooding the context with noise. |
| `ChatPromptTemplate.from_messages(messages)` | The list of chat messages and their variable slots | Always include a system message that constrains the LLM to use only the provided context; use `MessagesPlaceholder("history")` for multi-turn chains. Order matters: system first, then history, then human. |
| `RunnableLambda` / `as_tool=True` | Whether to expose a Runnable as an agent tool | Use `as_tool=True` only when the Runnable's output is consumed by an agent tool-calling loop; use plain `RunnableLambda` when it is an internal transformation step. |


---

## Worked Example: Requirement → Decision

**Given:** A support team at a software company wants to deploy a RAG chatbot that answers questions about internal documentation stored in a Databricks Vector Search index. The chatbot must stream responses to the UI, must cite which documents it used, and must fail gracefully if no relevant documents are found. The team is working in a Databricks workspace with a provisioned Vector Search endpoint and access to `databricks-dbrx-instruct`.

**Step 1 — Identify the goal:**
Build a streaming LCEL RAG chain that retrieves relevant documents from Databricks Vector Search, formats them into a prompt, calls `ChatDatabricks`, streams the answer, and returns both the answer text and the source document metadata.

**Step 2 — Define inputs:**
- Input to chain: `{"question": str}` — the user's question string.
- Retriever input: the question string (passed via `RunnablePassthrough`).
- Prompt inputs: `{"context": str, "question": str}`.

**Step 3 — Define outputs:**
- Final answer: `str` (via `StrOutputParser`) streamed token by token.
- Source documents: `List[Document]` returned in a parallel branch for the UI to render citations.

**Step 4 — Apply constraints:**
- Streaming is required: sequential `.invoke()` calls would break streaming.
- The prompt must instruct the LLM to answer only from context, to avoid hallucinations.
- A `RunnableParallel` / dict branch is needed to expose both the answer and the sources from a single `chain.stream()` call.
- `max_tokens` must be set explicitly to avoid quota overrun on the shared endpoint.

**Step 5 — Select the approach:**
Use LCEL with `RunnableParallel({"context": retriever | format_docs, "question": RunnablePassthrough()}) | prompt | llm | StrOutputParser()`. *Rationale:* LCEL is the only approach that preserves token-level streaming through the full chain; calling `retriever.invoke()` then `llm.invoke()` sequentially breaks streaming because the entire retrieval result must be available before the LLM can start. *Alternative 1:* LangGraph with a retrieval node and generation node — valid for more complex routing (conditional retrieval, re-ranking), but overkill for a simple single-turn RAG pipeline. *Alternative 2:* Manually calling each component — loses streaming, loses automatic async, and requires re-implementing error handling.

---

## Implementation

```python
# Scenario: Production streaming RAG chain with Databricks Vector Search and ChatDatabricks
# Constraint: Must stream tokens to UI in real time and return source document citations

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from databricks_langchain import ChatDatabricks, DatabricksVectorSearch

# --- Step 1: Initialise the retriever from a Unity Catalog Vector Search index ---
vector_store = DatabricksVectorSearch(
    index_name="main.docs.product_docs_index",   # Unity Catalog 3-level namespace
    embedding=None,                               # index already has embeddings stored
    columns=["content", "title", "source_url"],  # metadata columns to return
)
retriever = vector_store.as_retriever(
    search_kwargs={"k": 4}                        # retrieve top-4 most relevant chunks
)

# --- Step 2: Define a context formatter as a RunnableLambda ---
def format_docs(docs):
    """Join retrieved Document objects into a single context string."""
    if not docs:
        return "No relevant documentation found."
    return "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source_url', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )

format_context = RunnableLambda(format_docs)

# --- Step 3: Define the prompt ---
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful support assistant. Answer the question using ONLY the "
     "provided documentation context. If the context does not contain the answer, "
     "say 'I don't have documentation on that.' Do not invent information.\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])

# --- Step 4: Initialise the LLM ---
llm = ChatDatabricks(
    endpoint="databricks-dbrx-instruct",
    max_tokens=512,
    temperature=0.0,    # deterministic for factual RAG
)

# --- Step 5: Compose the full LCEL chain ---
# RunnableParallel runs retrieval and passes question through in parallel
rag_chain = (
    {"context": retriever | format_context, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- Step 6: Invoke modes (all work automatically because of LCEL) ---

# Synchronous single-call
answer = rag_chain.invoke("How do I reset my password?")
print(answer)

# Streaming (tokens appear as generated — UI can render progressively)
for chunk in rag_chain.stream("How do I reset my password?"):
    print(chunk, end="", flush=True)

# Async streaming (for FastAPI / async web frameworks)
import asyncio
async def stream_async():
    async for chunk in rag_chain.astream("How do I reset my password?"):
        print(chunk, end="", flush=True)
asyncio.run(stream_async())
```

```python
# Anti-pattern: Sequential .invoke() calls that look like a chain but break streaming and async
# This pattern is extremely common in tutorials and demo notebooks but is wrong for production

from databricks_langchain import ChatDatabricks, DatabricksVectorSearch
from langchain_core.prompts import ChatPromptTemplate

retriever = DatabricksVectorSearch(
    index_name="main.docs.product_docs_index"
).as_retriever(search_kwargs={"k": 4})

llm = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens=512)
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using this context:\n{context}"),
    ("human", "{question}"),
])

# WRONG: Sequential .invoke() calls
def answer_question_wrong(question: str) -> str:
    docs = retriever.invoke(question)                         # call 1 — blocks
    context = "\n\n".join(d.page_content for d in docs)
    formatted_prompt = prompt.invoke({"context": context, "question": question})  # call 2
    response = llm.invoke(formatted_prompt)                   # call 3 — blocks, NO streaming
    return response.content

# What breaks:
# 1. Streaming: llm.invoke() waits for the full response before returning.
#    You cannot yield tokens to the UI incrementally.
# 2. Async: you cannot use `await llm.ainvoke()` inside a sync def without extra plumbing.
# 3. Batch parallelism: calling answer_question_wrong() in a loop processes one question at a time.
# 4. Observability: LangSmith tracing treats this as three disconnected calls, not one chain trace.

# CORRECT: LCEL composition — all four issues disappear automatically
rag_chain_correct = (
    {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
     "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
# Now rag_chain_correct.stream(), .astream(), .batch() all work without extra code.
```

```python
# Scenario: Custom RunnableLambda for context formatting with token budget enforcement
# Constraint: Context window is limited; must truncate retrieved docs to stay under 2000 tokens

from langchain_core.runnables import RunnableLambda
from typing import List
from langchain_core.documents import Document

MAX_CONTEXT_CHARS = 8000   # rough proxy for ~2000 tokens at 4 chars/token

def format_docs_with_budget(docs: List[Document]) -> str:
    """Format retrieved docs into a context string, truncating to respect the token budget.
    Preserves source citations and prioritises the most relevant document (index 0).
    """
    if not docs:
        return "No relevant context available."

    parts = []
    total_chars = 0
    for i, doc in enumerate(docs):
        snippet = f"[Doc {i+1}: {doc.metadata.get('title', 'unknown')}]\n{doc.page_content}"
        if total_chars + len(snippet) > MAX_CONTEXT_CHARS:
            # Truncate the current document to fit the remaining budget
            remaining = MAX_CONTEXT_CHARS - total_chars
            if remaining > 200:   # only add if there is meaningful content left
                parts.append(snippet[:remaining] + "… [truncated]")
            break
        parts.append(snippet)
        total_chars += len(snippet)

    return "\n\n---\n\n".join(parts)

# Wrap as a Runnable — participates in LCEL chains with full streaming support
context_formatter = RunnableLambda(format_docs_with_budget)

# Usage in a chain
from databricks_langchain import ChatDatabricks, DatabricksVectorSearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

retriever = DatabricksVectorSearch(
    index_name="main.docs.product_docs_index"
).as_retriever(search_kwargs={"k": 6})   # retrieve more, then budget-trim

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer strictly from this context:\n{context}"),
    ("human", "{question}"),
])
llm = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens=512, temperature=0.0)

chain = (
    {"context": retriever | context_formatter, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

---

## Common Pitfalls & Misconceptions

- **"LCEL `|` is just syntactic sugar for calling components in sequence."** — Beginners see the pipe and assume it is a cosmetic shorthand for a `for` loop. The `|` operator does not just execute components in sequence: it creates a `RunnableSequence` whose `.stream()` method pipes generators through `.transform()` at each step, allowing token-level streaming without buffering. Calling components manually in a loop produces the wrong behaviour (blocking, no streaming) even if the final string output looks identical.

- **"I can add streaming later by swapping `.invoke()` for `.stream()`."** — Engineers often prototype with `.invoke()` and assume they can add streaming as a drop-in change. Streaming only works end-to-end when every component in the chain supports it via the Runnable interface. Non-LCEL chains (sequential `.invoke()` calls) cannot be made streaming by changing just the final call; the entire chain must be rebuilt as an LCEL expression.

- **"RunnablePassthrough copies the input, which is expensive for large payloads."** — Beginners avoid `RunnablePassthrough` thinking it serialises a large document or message list. `RunnablePassthrough` is a reference pass — it returns the same Python object without copying it. The cost is a single dictionary key assignment, not a deep copy.

- **"ChatDatabricks needs a Databricks cluster to run."** — Engineers new to Databricks assume model calls require active Spark compute. `ChatDatabricks` calls the Model Serving REST endpoint, which runs on serverless GPU infrastructure. No cluster or Spark session is required; all that is needed is a workspace URL and a token.

- **"I should use `JsonOutputParser` for all structured outputs."** — `JsonOutputParser` parses the LLM's string output as JSON but does not validate it against a schema. When downstream code accesses fields by name (e.g., `result.category`), a `KeyError` from malformed LLM output will surface far from the LLM call. Use `PydanticOutputParser` when structure must be guaranteed; save `JsonOutputParser` for exploratory or loosely-typed pipelines.

- **"MessagesPlaceholder is only for chatbots; RAG chains don't need it."** — Engineers building single-turn RAG chains omit `MessagesPlaceholder`, then must rewrite their prompt template when requirements add conversation history. Using `MessagesPlaceholder("history")` with a default of `[]` costs nothing for single-turn chains and makes the upgrade to multi-turn trivial.

---

## Key Definitions

| Term | Definition |
|---|---|
| **LCEL** | LangChain Expression Language; the declarative composition syntax that overloads `|` on the `Runnable` base class to create `RunnableSequence` objects with automatic streaming, batching, and async support. |
| **Runnable** | The abstract base class in `langchain_core.runnables` that all LangChain components implement; defines `invoke`, `batch`, `stream`, `ainvoke`, `abatch`, and `astream` as the contract. |
| **RunnableSequence** | A `Runnable` created by the `|` operator; executes its constituent runnables in order, passing each output as the next input; inherits all execution modes from the Runnable contract. |
| **RunnableParallel** | A `Runnable` created by passing a `dict` of `{key: Runnable}` in an LCEL chain; executes all branches concurrently and merges results into a dict. Also called `RunnableMap`. |
| **RunnablePassthrough** | A no-op `Runnable` that returns its input unchanged; used to forward the original input to a downstream branch that needs it alongside transformed branches. |
| **RunnableLambda** | A `Runnable` wrapper for any Python callable; allows arbitrary functions to participate in LCEL chains while preserving the Runnable contract. |
| **ChatDatabricks** | The LangChain chat model wrapper for Databricks Foundation Model API endpoints; a `BaseChatModel` subclass from `databricks_langchain` (or `langchain_community`). |
| **DatabricksVectorSearch** | The LangChain `VectorStore` wrapper for Databricks Vector Search; exposes `.as_retriever()` to return a `VectorStoreRetriever` Runnable. |
| **Retriever** | Any `Runnable` that implements `invoke(query: str) -> List[Document]`; LangChain's standard abstraction for document lookup, regardless of the underlying storage technology. |
| **StrOutputParser** | A `Runnable` output parser that extracts `.content` from an `AIMessage` and returns a plain Python `str`. |
| **PydanticOutputParser** | A `Runnable` output parser that parses LLM output as JSON and validates it against a Pydantic model, raising `OutputParserException` on failure. |
| **MessagesPlaceholder** | A prompt component that inserts a pre-existing `List[BaseMessage]` into a `ChatPromptTemplate` at a named position, used for conversation history injection. |

---

## Summary / Quick Recall

- LCEL `|` creates a `RunnableSequence` — streaming, batch, and async work automatically on any LCEL chain.
- Sequential `.invoke()` calls break streaming and async; they are the most common anti-pattern in LangChain code.
- `ChatDatabricks(endpoint=…, max_tokens=…, temperature=…)` is the LLM component; always set `max_tokens` explicitly in production.
- `DatabricksVectorSearch.as_retriever(search_kwargs={"k": N})` is the retriever; it implements the Runnable interface and participates in LCEL chains directly.
- `RunnableParallel` (`{"key": runnable}` dict syntax) runs branches concurrently and merges results; required to pass both the query and the retrieved context to the prompt.
- `RunnableLambda` wraps any Python function as a Runnable; use it for custom preprocessing or formatting without breaking the streaming contract.
- `StrOutputParser` → plain string; `JsonOutputParser` → dict; `PydanticOutputParser` → validated Pydantic model. Choose based on what downstream code expects.

---

## Self-Check Questions

1. What does the LCEL `|` operator actually create when it composes two Runnables?

   <details><summary>Answer</summary>

   The `|` operator calls `A.__or__(B)`, which returns a `RunnableSequence` object — a first-class Runnable that holds references to `A` and `B` and implements the full Runnable contract (`invoke`, `batch`, `stream`, `ainvoke`, `abatch`, `astream`). It does not immediately execute the components; it creates a declarative description of the pipeline. This is why calling `.stream()` on the resulting chain works automatically: the `RunnableSequence.stream()` method pipes the streaming generator from `A.stream()` through `B.transform()`, enabling token-level streaming through the entire chain. The most tempting wrong answer is "it creates a function that calls A then B" — this is true for `.invoke()` but misses the mechanism that makes `.stream()` and `.batch()` work without any additional code.

   </details>

2. You are building a RAG chain. The prompt template expects `{"context": str, "question": str}`. The retriever returns `List[Document]`. How do you structure the LCEL chain so that the original user question reaches the prompt alongside the formatted context?

   <details><summary>Answer</summary>

   Use `RunnableParallel` (dict syntax) to run two branches in parallel: one that retrieves and formats documents, and one that passes the question through unchanged:

   ```python
   chain = (
       {"context": retriever | format_docs, "question": RunnablePassthrough()}
       | prompt | llm | StrOutputParser()
   )
   ```

   `{"context": ..., "question": RunnablePassthrough()}` creates a `RunnableParallel`. When the chain receives `"How do I reset my password?"`, the parallel step runs `retriever.invoke(question)` (producing docs, then formatted to str) and `RunnablePassthrough().invoke(question)` (returning the question string) concurrently, then merges the results into `{"context": "...", "question": "How do I reset my password?"}`, which exactly matches the prompt's expected input schema. Without `RunnablePassthrough()`, the question would be lost after the retriever step.

   </details>

3. **Which TWO** of the following statements correctly describe a difference between using LCEL composition (`retriever | prompt | llm | parser`) versus sequential `.invoke()` calls for each component?
   - A. LCEL composition prevents the chain from being called with `.invoke()` — it only supports `.stream()`.
   - B. LCEL composition enables token-level streaming through `.stream()` and async execution through `.astream()` automatically; sequential `.invoke()` calls block and cannot stream.
   - C. Sequential `.invoke()` calls process multiple queries in parallel when called in a loop; LCEL `.batch()` is single-threaded.
   - D. LCEL composition allows LangSmith to trace the entire chain as a single logical unit with parent-child spans; sequential `.invoke()` calls produce three disconnected traces.
   - E. Sequential `.invoke()` calls allow you to inspect intermediate results more easily, making them preferable for debugging in production.

   <details><summary>Answer</summary>

   **Correct answers: B and D.**

   B is correct because LCEL's `RunnableSequence.stream()` pipes generators through each component's `.transform()` method, enabling tokens to flow to the caller as they are generated rather than waiting for the full LLM response. Sequential `.invoke()` calls always block at the LLM step.

   D is correct because LangSmith's tracing integrates with the Runnable interface at the `RunnableSequence` level; when you call `chain.invoke()`, LangSmith creates a single root span with child spans for each component. Sequential `.invoke()` calls create three root spans with no parent-child relationship, making it impossible to correlate retrieval latency with generation latency for a single request.

   A is incorrect: LCEL chains support all execution modes — `.invoke()`, `.batch()`, `.stream()`, `.astream()`, `.ainvoke()`.

   C is the opposite of truth: sequential `.invoke()` in a loop is single-threaded; LCEL `.batch()` uses `ThreadPoolExecutor` to process multiple inputs in parallel.

   E is partially true (you can inspect intermediate values) but "preferable for production debugging" is wrong — sequential calls lose tracing, streaming, and batch parallelism, making them worse for production in every dimension.

   </details>

4. A team has deployed a RAG chain using `ChatDatabricks(endpoint="databricks-dbrx-instruct")` without setting `max_tokens`. In production, some queries return very long responses that exceed the endpoint's per-request token quota, causing `HTTP 429` errors. What is the correct fix, and why is it better than catching the exception and retrying?

   <details><summary>Answer</summary>

   Set `max_tokens` explicitly at chain construction time: `ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens=512)`. This is better than catching the exception and retrying for two reasons: (1) **Prevention over recovery** — the token limit is enforced on the request payload before it reaches the endpoint, so the 429 never occurs; (2) **Cost and latency** — a retry doubles the number of LLM calls and adds hundreds of milliseconds of latency per affected request, whereas a hard token cap adds zero overhead. Catching exceptions and retrying also masks a misconfiguration as a transient error, making the root cause invisible in monitoring. The alternative of raising `max_tokens` to a very high value risks hitting endpoint quota for other concurrent requests sharing the serving endpoint.

   </details>

5. You need to choose between `StrOutputParser`, `JsonOutputParser`, and `PydanticOutputParser` for a chain that extracts a structured ticket classification (`category`, `priority`, `requires_escalation`) from LLM output, which is then consumed by a downstream routing function that accesses `result.category`. Which parser should you use and why are the other two wrong for this use case?

   <details><summary>Answer</summary>

   Use `PydanticOutputParser` with a Pydantic model defining `category`, `priority`, and `requires_escalation` with strict types. The downstream function accesses `result.category` (attribute access on a typed object), which means the output must be a validated Python object, not a raw string or dict.

   `StrOutputParser` is wrong because it returns a plain string; `result.category` would raise `AttributeError`.

   `JsonOutputParser` is wrong because it returns a `dict` (not a Pydantic object); `result.category` would raise `AttributeError` (dicts use `result["category"]`), and more importantly, `JsonOutputParser` does not validate the structure — the LLM could omit `requires_escalation` entirely, and the error would surface only when the routing function tries to access it, far from the LLM call. `PydanticOutputParser` catches the missing field immediately at parse time and raises `OutputParserException`, keeping errors close to their source.

   </details>

---

## Further Reading

- [LangChain Runnable Interface — docs.langchain.com](https://docs.langchain.com/oss/python/langchain/overview) — *verified 2026-07-11* — Comprehensive guide to the Runnable abstraction and composition with `|`.
- [Databricks Agent Framework: Author an AI Agent](https://docs.databricks.com/aws/en/agents/agent-framework/author-agent) — *verified 2026-07-11* — How to author, deploy, and evaluate agents using Python on Databricks, including LangChain/LangGraph examples with `ChatDatabricks`.
- [Databricks: Use Agents on Databricks](https://docs.databricks.com/aws/en/agents/agent-framework/build-agents) — *verified 2026-07-11* — Overview of agent tooling: Foundation Models, Vector Search, MLflow Tracing, and evaluation in the Databricks Agent Framework.
