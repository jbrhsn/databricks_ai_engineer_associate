# LangChain Components for RAG and Agent Pipelines — Interview Prep

**Section:** Application Development | **Role target:** Senior AI/ML Engineer, Databricks Solutions Architect, GenAI Platform Engineer

---

## Core Conceptual Questions

These test whether you understand the fundamentals.

| Question | Key points to cover | Common weak-answer trap |
|---|---|---|
| What is LCEL and why does it exist? | (1) Declarative composition with `|` that creates `RunnableSequence`; (2) the real value is streaming, async, batch — not syntax; (3) all components implement Runnable so `|` is polymorphic | Saying "it makes chaining simpler" without explaining *why* streaming/async require it |
| What is the Runnable interface and what methods does it define? | `invoke`, `batch`, `stream`, `ainvoke`, `abatch`, `astream`; all LangChain components implement it; `|` composes Runnables into `RunnableSequence` which inherits all methods | Only mentioning `invoke`; not knowing that `.stream()` requires Runnable composition to actually stream |
| How does `RunnablePassthrough` work in a RAG chain? | Passes input unchanged; used inside `RunnableParallel` (dict syntax) to forward the original query to the prompt alongside the retrieved context | Confusing with `RunnableLambda`; thinking it "copies" data rather than forwarding a reference |
| What is the difference between `StrOutputParser`, `JsonOutputParser`, and `PydanticOutputParser`? | `StrOutputParser` → plain string from `.content`; `JsonOutputParser` → dict from `json.loads()` (no schema validation); `PydanticOutputParser` → validated typed object; choose based on downstream consumer's type expectations | Saying "use PydanticOutputParser for everything" without acknowledging the validation overhead; or picking `JsonOutputParser` when downstream code does attribute access |
| How does `ChatDatabricks` authenticate to the Databricks Model Serving endpoint? | Uses workspace URL + PAT (or OAuth token); calls `/serving-endpoints/{endpoint}/invocations` with OpenAI-compatible schema; no cluster required — serverless GPU | Assuming you need a running Spark cluster or Databricks Runtime environment |

---

## Applied / Scenario Questions

**Q:** A colleague has built a RAG chain that works in a Jupyter notebook but, when deployed as a FastAPI endpoint, responds with the full answer 5 seconds after the request rather than streaming tokens. The chain is written as three sequential `.invoke()` calls: `retriever.invoke()`, `prompt.invoke()`, `llm.invoke()`. What is wrong, and how would you fix it?

**Strong answer framework:**
- **Root cause:** Sequential `.invoke()` calls block at each step; `llm.invoke()` waits for the full response before returning. LCEL's `chain.stream()` / `chain.astream()` requires Runnable composition to pipe streaming generators between components.
- **Fix:** Rewrite as an LCEL chain with `|`: `{"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()`. Use `chain.astream(input)` in the FastAPI endpoint with `StreamingResponse`.
- **Show tradeoff awareness:** Acknowledge that the sequential pattern has one minor benefit — you can inspect intermediate values more easily in a notebook. But in production, that advantage is outweighed by the loss of streaming, async, and unified LangSmith tracing.

---

**Q:** You are building a multi-turn RAG chatbot. The user's conversation history needs to be included in each prompt, and the context should come from a Databricks Vector Search index. How would you structure the prompt template and the chain?

**Strong answer framework:**
- Use `ChatPromptTemplate.from_messages([("system", "…{context}…"), MessagesPlaceholder("history"), ("human", "{question}")])`.
- The chain input is `{"question": str, "history": List[BaseMessage]}`.
- `RunnableParallel`: one branch retrieves and formats context from the user's current question; the other passes `history` and `question` through via `RunnablePassthrough()`.
- Highlight that `MessagesPlaceholder` with `default=[]` makes the template backward-compatible with single-turn calls.
- Show tradeoff awareness: as conversation history grows, the context window shrinks. Mention that a production system would trim history to the last N turns or summarise old turns.

---

## System Design / Architecture Question

**Q:** Design a production RAG pipeline on Databricks that serves a customer-facing support chatbot. The pipeline must: (1) stream responses, (2) return source document citations, (3) handle 50 concurrent users, and (4) log all interactions for evaluation. Describe the key components, how they connect, and what could go wrong.

**Approach:**
1. **Clarify requirements:** Token streaming to a web UI, citations as a separate field in the response, evaluation via MLflow, deployed on Databricks Apps or Model Serving.
2. **Propose structure:**
   - `DatabricksVectorSearch.as_retriever(search_kwargs={"k": 4})` for retrieval.
   - `RunnableParallel({"context": retriever | format_docs, "question": RunnablePassthrough(), "source_docs": retriever})` to capture both the formatted context for the prompt and the raw `List[Document]` for citation rendering.
   - `ChatPromptTemplate` with a grounding system message.
   - `ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens=512, temperature=0.0)`.
   - `StrOutputParser()` for the streaming answer.
   - Wrap the chain in MLflow `ResponsesAgent` or log with `mlflow.langchain.autolog()` for tracing.
3. **Justify choices and name tradeoffs explicitly:**
   - LCEL `|` enables `.astream()` for concurrent requests via `asyncio.gather()`.
   - `max_tokens=512` prevents individual requests from monopolising the serving endpoint's quota.
   - The double-retriever in `RunnableParallel` adds one extra retrieval call per request; a cleaner design stores the retrieved docs in a LangGraph state node and reuses them.
   - MLflow autologging adds ~10ms of overhead per request for trace serialisation; acceptable for observability, but worth measuring under load.

---

## Vocabulary That Signals Expertise

Use these terms naturally — don't force them:
- **`RunnableSequence`** — the actual Python object created by `|`; shows you know what the operator returns
- **`RunnableParallel`** / **dict syntax** — the correct name for `{"key": runnable}` composition; shows you know it runs branches concurrently
- **`.transform()` method** — how `RunnableSequence.stream()` pipes generators through each component; shows you understand the streaming mechanism
- **`ThreadPoolExecutor` / `asyncio.gather()`** — the concurrency primitives LCEL uses under `.batch()` and `.abatch()`; shows you can reason about performance
- **`BaseChatModel` subclass** — the correct description of `ChatDatabricks`; shows you understand the type hierarchy
- **`OutputParserException`** — the exception `PydanticOutputParser` raises on validation failure; shows you know error handling

---

## Vocabulary That Signals Weakness

Avoid these — they signal outdated or shallow understanding:
- **"LangChain chains"** (without specifying LCEL or `RunnableSequence`) — vague; sounds like you mean the deprecated `LLMChain` or `SequentialChain` classes
- **"I just call `.invoke()` on each component"** — flags you as unfamiliar with streaming requirements or the Runnable interface
- **"I'll add streaming later"** — signals you don't know that streaming requires LCEL composition from the start
- **"Databricks model needs a cluster"** — reveals a misconception about how Model Serving endpoints work (serverless, no cluster required)
- **"I use `JsonOutputParser` for structured output"** — acceptable only if followed by awareness that it doesn't validate; otherwise signals you don't know `PydanticOutputParser`

---

## STAR Answer Frame

**Situation:** I was reviewing a team's RAG prototype that was being prepared for a customer demo. The chain was written as sequential `.invoke()` calls in a Flask endpoint. During a pre-demo load test with 10 concurrent users, P99 response latency was 12 seconds and the UI showed a blank screen until the full answer arrived.

**Task:** I was responsible for identifying the root cause, proposing a fix, and implementing it within a one-day window before the demo.

**Action:** I identified that the sequential `.invoke()` pattern was blocking at the LLM step and that the Flask endpoint was serving requests synchronously. I refactored the chain to LCEL: `{"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()`. I replaced the Flask endpoint with a FastAPI `StreamingResponse` using `chain.astream()`. I also set `max_tokens=512` on `ChatDatabricks`, which had been unset and causing some requests to generate 2000-token responses that saturated the serving endpoint.

**Result:** P99 latency dropped from 12 seconds to 2.8 seconds. The demo showed token-by-token streaming in the UI, which the customer's team cited as a key differentiator. The `max_tokens` fix eliminated two categories of 429 errors from the logs. Total code change: 18 lines.

---

## Red Flags Interviewers Watch For

- Describing LCEL as "just a shorthand" without knowing it enables streaming — signals you have used LangChain only in notebooks.
- Not knowing what `RunnablePassthrough` is for — signals you have only built linear chains, not parallel branches (RAG requires a parallel branch).
- Confusing `ChatDatabricks` with a Spark API — signals unfamiliarity with Databricks Model Serving; the model endpoint is a REST API independent of Spark.
- Saying "I'd add a retry loop around the parser" for `PydanticOutputParser` failures without mentioning `OutputFixingParser` — shows you haven't used LangChain's built-in error recovery.
- Designing a RAG chain without mentioning a system prompt that grounds the LLM to the context — signals you haven't deployed a RAG system that actually needed to avoid hallucinations.
