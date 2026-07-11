# LAB-07: Building an End-to-End LCEL RAG Chain with LangChain Components

**Lab:** LAB-07 | **Section:** 04 — Application Development | **Module:** 01 — Frameworks and Prompting | **Est. time:** 90 min

---

## Objective

Build a complete LCEL RAG chain using `ChatDatabricks` (or a local mock for offline testing), `ChatPromptTemplate` with `MessagesPlaceholder`, `RunnablePassthrough`, `RunnableLambda`, and `StrOutputParser`, demonstrating token-level streaming and diagnosing the classic sequential-`.invoke()` anti-pattern.

---

## Prerequisites

- Completed reading of `notes.md` for this chapter (LangChain Components)
- Completed LAB-06 (LangGraph Fundamentals) — familiarity with the Runnable interface
- Python 3.11+ installed locally
- One of the following:
  - A Databricks workspace with a personal access token and a Foundation Model API endpoint (`databricks-dbrx-instruct` or equivalent)
  - **OR** an Anthropic or OpenAI API key (the lab provides a local mock path for offline use)

---

## Setup

Install dependencies and configure your environment. The lab supports two paths: **Databricks** (uses `ChatDatabricks`) and **Local** (uses `ChatAnthropic` / `ChatOpenAI` as a drop-in replacement). Both paths produce identical chain behaviour because every component implements the Runnable interface.

```bash
# Setup: install LangChain core components and the Databricks integration
pip install langchain langchain-core langchain-community
pip install databricks-langchain          # provides ChatDatabricks + DatabricksVectorSearch

# Local fallback (if you do not have a Databricks workspace)
pip install langchain-anthropic           # OR: pip install langchain-openai

# Set credentials — choose the path matching your environment
# --- Databricks path ---
export DATABRICKS_HOST="https://<your-workspace>.azuredatabricks.net"
export DATABRICKS_TOKEN="<your-personal-access-token>"

# --- Local path (Anthropic) ---
# export ANTHROPIC_API_KEY="your-key-here"

# --- Local path (OpenAI) ---
# export OPENAI_API_KEY="your-key-here"
```

Verify the installation and choose your LLM:

```python
# Verify: confirm LangChain core and your chosen LLM integration are importable.
# Both paths are valid — uncomment the block that matches your environment.

import langchain_core
print(f"langchain-core version: {langchain_core.__version__}")

# ── Path A: Databricks workspace ──────────────────────────────────────────
try:
    from databricks_langchain import ChatDatabricks
    llm = ChatDatabricks(
        endpoint="databricks-dbrx-instruct",
        max_tokens=512,
        temperature=0.0,
    )
    LLM_PATH = "databricks"
    print(f"LLM: ChatDatabricks (endpoint=databricks-dbrx-instruct)")
except Exception as e:
    LLM_PATH = None
    print(f"ChatDatabricks unavailable: {e}")

# ── Path B: Local Anthropic (offline / CI) ────────────────────────────────
if LLM_PATH is None:
    try:
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0, max_tokens=512)
        LLM_PATH = "anthropic"
        print(f"LLM: ChatAnthropic (claude-haiku-4-5-20251001)")
    except Exception as e:
        print(f"ChatAnthropic unavailable: {e}")

# ── Path C: Local OpenAI (offline / CI) ───────────────────────────────────
if LLM_PATH is None:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=512)
    LLM_PATH = "openai"
    print(f"LLM: ChatOpenAI (gpt-4o-mini)")

assert llm is not None, "No LLM could be initialised — check your credentials."
print(f"\nUsing LLM path: {LLM_PATH}")
# Expected: no ImportError, one "LLM: ..." line, and "Using LLM path: <path>"
```

---

## Steps

### Step 1 — Build a Mock Retriever for Local Testing

Before wiring the full chain, implement a mock retriever that respects the LangChain Retriever interface (`invoke(query: str) -> List[Document]`). This makes the lab runnable without a live Databricks Vector Search index. When running on Databricks, swap this with `DatabricksVectorSearch.as_retriever()`.

```python
# Scenario: Local RAG development requires a retriever that matches the
# LangChain Retriever contract so the LCEL chain can be tested offline.
# Constraint: must return List[Document] and be composable with | in an LCEL chain.

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

# ── Mock knowledge base (replace with DatabricksVectorSearch in production) ──
KNOWLEDGE_BASE = {
    "password": [
        Document(
            page_content=(
                "To reset your password, navigate to Account Settings → Security. "
                "Click 'Reset Password' and enter your registered email address. "
                "A reset link will be sent within 5 minutes. Links expire after 30 minutes."
            ),
            metadata={"source_url": "docs/account-security.md", "title": "Password Reset"},
        ),
        Document(
            page_content=(
                "If you have not received the password reset email after 10 minutes, "
                "check your spam folder or contact support@example.com."
            ),
            metadata={"source_url": "docs/account-security.md", "title": "Reset Troubleshooting"},
        ),
    ],
    "billing": [
        Document(
            page_content=(
                "Billing cycles run on the 1st of each month. Invoices are emailed to the "
                "primary account holder. To update payment methods, go to Billing → Payment Methods."
            ),
            metadata={"source_url": "docs/billing.md", "title": "Billing Overview"},
        ),
    ],
    "api": [
        Document(
            page_content=(
                "API keys are created in Developer Settings → API Keys. Each key has a "
                "configurable rate limit (default: 1000 req/min). Keys can be revoked at any time."
            ),
            metadata={"source_url": "docs/api-reference.md", "title": "API Key Management"},
        ),
    ],
}

def _mock_retrieve(query: str, k: int = 3) -> list[Document]:
    """
    Keyword-based mock retriever. In production, replace with:
        DatabricksVectorSearch(index_name="…").as_retriever(search_kwargs={"k": k})
    """
    query_lower = query.lower()
    for keyword, docs in KNOWLEDGE_BASE.items():
        if keyword in query_lower:
            return docs[:k]
    return [
        Document(
            page_content="No relevant documentation found for this query.",
            metadata={"source_url": "unknown", "title": "No Results"},
        )
    ]

# Wrap the function as a Runnable so it participates in LCEL chains with |
mock_retriever = RunnableLambda(_mock_retrieve)

# ── Production swap (Databricks Vector Search) ────────────────────────────
# Uncomment when running in a Databricks workspace with a Vector Search index:
#
# from databricks_langchain import DatabricksVectorSearch
# vector_store = DatabricksVectorSearch(
#     index_name="main.support_docs.product_docs_index",
#     columns=["content", "title", "source_url"],
# )
# mock_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Quick sanity check
test_docs = mock_retriever.invoke("How do I reset my password?")
print(f"Mock retriever returned {len(test_docs)} document(s)")
print(f"First doc title: {test_docs[0].metadata['title']}")
# Expected: 2 document(s); First doc title: Password Reset
```

---

### Step 2 — Build the Context Formatter and ChatPromptTemplate

Define the `format_docs` formatter as a `RunnableLambda` and the system prompt using `ChatPromptTemplate` with `MessagesPlaceholder` for conversation history.

```python
# Scenario: A support chatbot needs to format retrieved documents into a
# clean context string, enforce a token budget, and inject optional
# conversation history — all without breaking LCEL streaming.

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from typing import List

MAX_CONTEXT_CHARS = 6000  # rough proxy for ~1500 tokens at 4 chars/token

def format_docs(docs: List[Document]) -> str:
    """
    Join retrieved Document objects into a single context string.
    Enforces a character budget so the final prompt stays within context window limits.
    Preserves source URLs for citation rendering.
    """
    if not docs:
        return "No relevant documentation found."

    parts = []
    total_chars = 0
    for i, doc in enumerate(docs):
        citation = doc.metadata.get("source_url", "unknown")
        snippet = f"[Source {i + 1}: {citation}]\n{doc.page_content}"
        if total_chars + len(snippet) > MAX_CONTEXT_CHARS:
            remaining = MAX_CONTEXT_CHARS - total_chars
            if remaining > 200:
                parts.append(snippet[:remaining] + "… [truncated]")
            break
        parts.append(snippet)
        total_chars += len(snippet)

    return "\n\n---\n\n".join(parts)

# Wrap as a Runnable — participates in LCEL chains with full streaming support
format_context = RunnableLambda(format_docs)

# Define the prompt with MessagesPlaceholder for optional conversation history.
# MessagesPlaceholder("history") splices a List[BaseMessage] into the prompt at
# invocation time. Passing history=[] (the default) costs nothing for single-turn use
# but makes the upgrade to multi-turn trivial.
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful support assistant. Answer the user's question using ONLY "
        "the provided documentation context below. If the context does not contain "
        "the answer, respond with: 'I don't have documentation on that topic.'\n\n"
        "Do not invent information that is not in the context.\n\n"
        "Documentation context:\n{context}",
    ),
    MessagesPlaceholder("history", optional=True),   # inject prior turns if provided
    ("human", "{question}"),
])

# Test the prompt renders correctly before wiring the chain
test_prompt_value = prompt.invoke({
    "context": "Test context.",
    "question": "Test question?",
    "history": [],
})
print(f"Prompt message count: {len(test_prompt_value.messages)}")
print(f"System message preview: {test_prompt_value.messages[0].content[:80]}…")
print(f"Human message: {test_prompt_value.messages[-1].content}")
# Expected: 2 messages (system + human); no MessagesPlaceholder in output
```

---

### Step 3 — Compose the LCEL RAG Chain with `|`

Wire retriever → formatter → prompt → LLM → parser into a single LCEL chain using `RunnableParallel` (dict syntax) and `RunnablePassthrough`.

```python
# Scenario: A single-turn RAG chain must retrieve context, inject the original
# question into the prompt, call the LLM, and return a plain string answer —
# all while preserving token-level streaming for the UI.
# Constraint: the prompt expects {"context": str, "question": str, "history": list},
# but the chain receives only the question string as input.

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# RunnableParallel (dict syntax) runs two branches concurrently:
#   "context"  branch: retriever → format_context  (str)
#   "question" branch: RunnablePassthrough          (str, unchanged)
# The merged dict {"context": "…", "question": "…"} is then fed to the prompt.
rag_chain = (
    {
        "context": mock_retriever | format_context,  # retrieve → format
        "question": RunnablePassthrough(),           # pass question through unchanged
        "history": RunnablePassthrough()             # will be overridden per-call if needed
                   | RunnableLambda(lambda _: []),   # default to empty list
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Quick test with .invoke() — confirms the chain is wired correctly
test_answer = rag_chain.invoke("How do I reset my password?")
print(f"Chain test answer (first 120 chars):")
print(test_answer[:120])
print()
# Expected: a coherent answer about resetting a password, sourced from the mock KB
```

---

### Step 4 — Demonstrate Token-Level Streaming

Use `.stream()` to yield tokens as they arrive from the LLM, then show `.astream()` for async frameworks.

```python
# Scenario: A web UI needs to render the LLM response progressively as tokens
# arrive rather than waiting for the full response — requires true token-level
# streaming through every component in the chain.

import sys
import asyncio

QUESTION = "How do I manage API keys?"

print(f"=== Synchronous streaming (.stream()) ===")
print(f"Question: {QUESTION}\n")
print("Answer: ", end="", flush=True)

full_response_chunks = []
for chunk in rag_chain.stream(QUESTION):
    # chunk is a string token fragment (StrOutputParser output)
    print(chunk, end="", flush=True)
    full_response_chunks.append(chunk)

full_response = "".join(full_response_chunks)
print(f"\n\nTotal characters streamed: {len(full_response)}")
print(f"Total chunks yielded: {len(full_response_chunks)}")
# Expected: tokens printed progressively; total chunks > 1 (not one big string)

print(f"\n=== Async streaming (.astream()) ===")
print("(simulates async web framework usage, e.g. FastAPI + Server-Sent Events)\n")

async def stream_answer_async(question: str) -> str:
    """Async generator pattern — yields tokens to a web client in real time."""
    collected = []
    print("Answer: ", end="", flush=True)
    async for chunk in rag_chain.astream(question):
        print(chunk, end="", flush=True)
        collected.append(chunk)
    print()  # newline after streaming
    return "".join(collected)

async_answer = asyncio.run(stream_answer_async("How does billing work?"))
print(f"\nAsync response length: {len(async_answer)} characters")
# Expected: same progressive token output via async generator
```

---

### Step 5 — Demonstrate the Anti-Pattern: Sequential `.invoke()` Calls

Show why manually calling each component breaks streaming, async, batching, and tracing — then demonstrate the correct LCEL fix.

```python
# Anti-pattern: Building a "chain" with sequential .invoke() calls.
# This pattern is common in tutorials and appears to work for single-turn,
# single-request use cases — but it silently breaks streaming, async,
# batch parallelism, and LangSmith distributed tracing.

from langchain_core.prompts import ChatPromptTemplate as _CPT

_anti_prompt = _CPT.from_messages([
    ("system", "Answer using this context:\n{context}"),
    ("human", "{question}"),
])

def answer_question_wrong(question: str) -> str:
    """
    Sequential .invoke() pattern — DO NOT use in production.
    Each call is a separate, disconnected operation.
    """
    # Call 1: retrieve — blocks until all docs are returned
    docs = mock_retriever.invoke(question)

    # Call 2: format — pure Python, no Runnable contract
    context = "\n\n".join(d.page_content for d in docs)

    # Call 3: prompt — blocks until the ChatPromptValue is returned
    formatted_prompt = _anti_prompt.invoke({"context": context, "question": question})

    # Call 4: LLM — blocks until the FULL response is generated
    # .invoke() on an LLM always waits for the complete response before returning.
    # There is NO way to stream tokens from this call.
    response = llm.invoke(formatted_prompt)

    return response.content

# What breaks with this pattern:
# 1. STREAMING: llm.invoke() buffers the entire response. You cannot yield tokens.
#    Replacing this with llm.stream() only streams the last step — the whole function
#    still blocks on retrieval, prompt formatting, etc.
# 2. ASYNC:     You cannot use `await llm.ainvoke()` inside a sync def without
#               extra plumbing (asyncio.run() nesting restrictions in notebooks).
# 3. BATCH:     Wrapping this function in a list comprehension processes one query
#               at a time. LCEL .batch() uses ThreadPoolExecutor automatically.
# 4. TRACING:   LangSmith sees four disconnected root spans. Latency attribution
#               (which step was slow?) is impossible without manual instrumentation.

# Verify it produces an answer (so we can compare to LCEL output)
wrong_answer = answer_question_wrong("How do I reset my password?")
print("Anti-pattern answer (first 120 chars):")
print(wrong_answer[:120])
print()
print("NOTE: This answer was produced by blocking each step in sequence.")
print("      It cannot stream tokens to a UI or run queries in parallel.")

# ── Correct version: LCEL composition ────────────────────────────────────
# All four problems disappear because RunnableSequence implements
# .stream(), .astream(), and .batch() using the streaming contract
# built into every component.

print("\n--- Correct LCEL version (same answer, streaming-capable) ---")
print("Answer: ", end="", flush=True)
for chunk in rag_chain.stream("How do I reset my password?"):
    print(chunk, end="", flush=True)
print()
```

---

### Step 6 — Multi-Turn Conversation with MessagesPlaceholder

Feed conversation history into the chain via the `history` slot in the prompt, demonstrating how `MessagesPlaceholder` upgrades a single-turn chain to multi-turn without rewriting the chain.

```python
# Scenario: A support chatbot must maintain conversation context across
# multiple turns so follow-up questions ("What if that doesn't work?")
# resolve correctly using prior context.
# Constraint: the LCEL chain is already built — no rewiring, just pass history.

from langchain_core.messages import HumanMessage, AIMessage

# ── Multi-turn chain variant ───────────────────────────────────────────────
# Replace the history branch with a passthrough that accepts real history.
multi_turn_chain = (
    {
        "context": mock_retriever | format_context,
        "question": RunnablePassthrough(),
        "history": RunnableLambda(lambda _: []),  # replaced at call time below
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Turn 1 — initial question
print("=== Multi-turn conversation demo ===\n")
print("Turn 1 — User: How do I reset my password?\n")
turn1_answer = multi_turn_chain.invoke("How do I reset my password?")
print(f"Assistant: {turn1_answer[:200]}\n")

# Turn 2 — follow-up; inject Turn 1 history into the prompt
history = [
    HumanMessage(content="How do I reset my password?"),
    AIMessage(content=turn1_answer),
]

# Build a chain variant that injects the history list at call time
# using a partial invocation pattern with a wrapper lambda.
def invoke_with_history(question: str, history_msgs: list) -> str:
    """Invoke the RAG chain with injected conversation history."""
    docs = mock_retriever.invoke(question)
    context = format_docs(docs)
    # Construct the prompt input manually to inject history
    prompt_input = {
        "context": context,
        "question": question,
        "history": history_msgs,
    }
    formatted = prompt.invoke(prompt_input)
    response = llm.invoke(formatted)
    return response.content

print("Turn 2 — User: What if I don't receive the reset email?\n")
turn2_answer = invoke_with_history(
    "What if I don't receive the reset email?",
    history_msgs=history,
)
print(f"Assistant: {turn2_answer[:300]}\n")
print(f"Total messages in Turn 2 prompt: {len(history) + 2} (system + history + human)")
# Expected: the Turn 2 answer references the reset email flow,
#           and the prompt contains 4 messages (system, human1, ai1, human2)
```

---

### Step 7 — Batch Execution and Output Parser Comparison

Use `.batch()` to process multiple questions in parallel, then compare `StrOutputParser`, `JsonOutputParser`, and `PydanticOutputParser`.

```python
# Scenario: A nightly QA pipeline must evaluate the chain against a batch of
# test questions. Processing them sequentially would be 3× slower than parallel.
# Constraint: all questions are independent — no shared state between calls.

from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# ── Batch with StrOutputParser ────────────────────────────────────────────
batch_questions = [
    "How do I reset my password?",
    "How does billing work?",
    "How do I manage API keys?",
]

print("=== Batch execution (.batch()) ===")
print(f"Processing {len(batch_questions)} questions in parallel…\n")

batch_answers = rag_chain.batch(batch_questions)
for q, a in zip(batch_questions, batch_answers):
    print(f"Q: {q}")
    print(f"A: {a[:100]}…\n")
# Expected: all 3 answers returned; LCEL uses ThreadPoolExecutor under the hood

# ── JsonOutputParser ──────────────────────────────────────────────────────
# Use when the downstream consumer expects a dict and structure is loosely typed.
json_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a support classifier. Given a user question and context, "
        "return a JSON object with exactly these fields:\n"
        '  {{"category": "<billing|security|api|other>", "confidence": <0.0-1.0>}}\n'
        "Return ONLY valid JSON, no explanation.\n\nContext:\n{context}",
    ),
    ("human", "{question}"),
])

json_chain = (
    {
        "context": mock_retriever | format_context,
        "question": RunnablePassthrough(),
    }
    | json_prompt
    | llm
    | JsonOutputParser()   # parses LLM string output as JSON → dict
)

classification = json_chain.invoke("How do I reset my password?")
print(f"=== JsonOutputParser output ===")
print(f"Type: {type(classification).__name__}")   # dict
print(f"Value: {classification}")
# Expected: {'category': 'security', 'confidence': 0.9} (or similar)

# ── PydanticOutputParser ──────────────────────────────────────────────────
# Use when downstream code accesses fields by name (e.g., result.category)
# and schema compliance must be enforced at parse time.
class TicketClassification(BaseModel):
    category: str = Field(description="One of: billing, security, api, other")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    requires_escalation: bool = Field(description="True if the issue needs a human agent")

pydantic_parser = PydanticOutputParser(pydantic_object=TicketClassification)

pydantic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a support ticket classifier. Given the context and question, "
        "classify the ticket.\n\n{format_instructions}\n\nContext:\n{context}",
    ),
    ("human", "{question}"),
]).partial(format_instructions=pydantic_parser.get_format_instructions())

pydantic_chain = (
    {
        "context": mock_retriever | format_context,
        "question": RunnablePassthrough(),
    }
    | pydantic_prompt
    | llm
    | pydantic_parser   # parses + validates against TicketClassification schema
)

ticket = pydantic_chain.invoke("I can't log in and it's urgent")
print(f"\n=== PydanticOutputParser output ===")
print(f"Type: {type(ticket).__name__}")                       # TicketClassification
print(f"category: {ticket.category}")                         # attribute access
print(f"confidence: {ticket.confidence}")
print(f"requires_escalation: {ticket.requires_escalation}")
# Expected: TicketClassification object with all three fields populated and validated
```

---

## Validation

Run all assertions after completing every step. All checks must pass.

```python
# Validation: confirm the LCEL RAG chain is correctly built and all execution
# modes work. All assertions must pass for the lab to be considered complete.

print("\n=== Validation Checks ===\n")

from langchain_core.runnables import RunnableSequence

# 1. Chain is a RunnableSequence (LCEL composition confirmed)
assert isinstance(rag_chain, RunnableSequence), \
    "FAIL: rag_chain is not a RunnableSequence — LCEL composition was not used"
print("✓ rag_chain is a RunnableSequence (LCEL composition confirmed)")

# 2. .invoke() returns a non-empty string
invoke_result = rag_chain.invoke("How do I reset my password?")
assert isinstance(invoke_result, str) and len(invoke_result) > 20, \
    "FAIL: .invoke() did not return a meaningful string answer"
print(f"✓ .invoke() returned string answer ({len(invoke_result)} chars)")

# 3. .stream() yields multiple chunks (token-level streaming confirmed)
stream_chunks = list(rag_chain.stream("How does billing work?"))
assert len(stream_chunks) > 1, \
    f"FAIL: .stream() returned only {len(stream_chunks)} chunk(s) — streaming may not be working"
assert all(isinstance(c, str) for c in stream_chunks), \
    "FAIL: .stream() chunks are not strings (StrOutputParser not at end of chain?)"
print(f"✓ .stream() yielded {len(stream_chunks)} string chunks (token-level streaming confirmed)")

# 4. .batch() returns the correct number of results
batch_results = rag_chain.batch(["question 1 about password", "question 2 about billing"])
assert len(batch_results) == 2, \
    f"FAIL: .batch() returned {len(batch_results)} results, expected 2"
assert all(isinstance(r, str) for r in batch_results), \
    "FAIL: .batch() results are not all strings"
print(f"✓ .batch() returned {len(batch_results)} results in parallel")

# 5. Mock retriever returns List[Document]
from langchain_core.documents import Document as _Doc
retrieved = mock_retriever.invoke("password")
assert isinstance(retrieved, list) and all(isinstance(d, _Doc) for d in retrieved), \
    "FAIL: mock_retriever did not return List[Document]"
print(f"✓ mock_retriever returns List[Document] ({len(retrieved)} docs for 'password')")

# 6. format_context is a RunnableLambda (participates in LCEL with |)
from langchain_core.runnables import RunnableLambda as _RL
assert isinstance(format_context, _RL), \
    "FAIL: format_context is not a RunnableLambda"
print("✓ format_context is a RunnableLambda (Runnable interface confirmed)")

# 7. Prompt contains MessagesPlaceholder for history
from langchain_core.prompts import MessagesPlaceholder as _MP
has_placeholder = any(
    isinstance(msg_template, _MP)
    for msg_template in prompt.messages
)
assert has_placeholder, \
    "FAIL: prompt does not contain a MessagesPlaceholder for conversation history"
print("✓ ChatPromptTemplate contains MessagesPlaceholder('history')")

# 8. Anti-pattern function produces an answer (confirming it is functional but wrong pattern)
anti_result = answer_question_wrong("How do I reset my password?")
assert isinstance(anti_result, str) and len(anti_result) > 10, \
    "FAIL: anti-pattern function did not return an answer"
print("✓ anti-pattern (sequential .invoke()) produces an answer — but cannot stream")

# 9. Async streaming works end-to-end
async def _async_check() -> str:
    chunks = []
    async for chunk in rag_chain.astream("How do I manage API keys?"):
        chunks.append(chunk)
    return "".join(chunks)

async_result = asyncio.run(_async_check())
assert isinstance(async_result, str) and len(async_result) > 10, \
    "FAIL: .astream() did not produce a valid result"
print(f"✓ .astream() completed successfully ({len(async_result)} chars)")

# 10. PydanticOutputParser returns a validated Pydantic object
assert hasattr(ticket, "category") and hasattr(ticket, "requires_escalation"), \
    "FAIL: PydanticOutputParser did not return a TicketClassification object"
print(f"✓ PydanticOutputParser returned TicketClassification(category='{ticket.category}')")

print("\n=== All validation checks passed ===")
```

---

## Teardown

No persistent resources are created during this lab. All state is in RAM.

```python
# Teardown: this lab creates no persistent resources (no Databricks Vector Search
# index, no MLflow experiment, no database files). All objects are in-process RAM.
# Deleting the references is sufficient.

del rag_chain
del mock_retriever
del format_context
del prompt
del llm

print("Lab teardown complete. All in-memory resources released.")
print()
print("If you uncommented the DatabricksVectorSearch block in Step 1:")
print("  - The Vector Search index is NOT deleted by this lab (it was pre-existing).")
print("  - The Databricks personal access token is stored only as an env variable.")
print("  - No cleanup is required on the Databricks side.")
```

---

## Reflection Questions

1. In Step 3, the `history` branch uses `RunnableLambda(lambda _: [])` to default to an empty list. If you removed this branch entirely and called `rag_chain.invoke("some question")`, what error would you see and at which component in the chain? What does this tell you about the relationship between a `ChatPromptTemplate`'s declared variables and the dict it receives at invocation time?

2. In Step 5, the anti-pattern answer and the LCEL answer are textually identical for a single synchronous call. If the answers are the same, what is the concrete, measurable cost of using the anti-pattern in a production system serving 100 concurrent users? Name at least two specific failure modes that would appear under load that are invisible in single-user testing.

3. Step 7 shows three output parsers. You are building a pipeline where the LLM extracts `{"action": "refund", "amount": 49.99, "reason": "duplicate charge"}` from a customer message, and the extracted dict is passed directly to a payment API that requires all three fields to be present. Which parser is the correct choice, and what specific error would `JsonOutputParser` allow to reach the payment API that `PydanticOutputParser` would catch before the API call?