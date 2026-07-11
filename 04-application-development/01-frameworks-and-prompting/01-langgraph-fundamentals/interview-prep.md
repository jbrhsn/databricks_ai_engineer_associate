# LangGraph Fundamentals — Interview Prep

**Section:** 04 — Application Development | **Role target:** Senior AI/ML Engineer, Solutions Architect, Senior Data Engineer

---

## Core Conceptual Questions

These test whether you understand the fundamentals.

| Question | Key points to cover | Common weak-answer trap |
|---|---|---|
| "What is LangGraph, and how does it differ from a LangChain LCEL chain?" | (1) Graph-based vs sequential pipeline; (2) explicit shared TypedDict state vs implicit context flow; (3) conditional edges enable loops; (4) checkpointing is first-class in LangGraph but bolt-on in LCEL; (5) nodes are isolated functions communicating only through state. | Saying "LangGraph is just a newer version of LangChain" — they are complementary, not hierarchical. LCEL is appropriate for linear pipelines; LangGraph is the right choice when you need looping, conditional routing, or persistent memory. |
| "What is a checkpointer, and why would you use `SqliteSaver` instead of `InMemorySaver`?" | (1) Checkpointer saves full state snapshot after every super-step; (2) `InMemorySaver` is RAM-only, lost on restart; (3) `SqliteSaver` writes to disk, survives restarts; (4) both require `thread_id` in config to function; (5) `PostgresSaver` for production concurrency. | Saying "checkpointers are just for conversation history" — they also enable fault-tolerant resumption, time-travel debugging, and human-in-the-loop workflows. |
| "What is the purpose of `START` and `END` in a LangGraph graph?" | (1) Virtual sentinel constants, not real node functions; (2) `START` identifies which node receives the initial input — `add_edge(START, "first_node")`; (3) `END` is the terminal destination — routing to `END` halts that execution path; (4) graph finishes when all active paths reach `END`. | Saying "`START` is the constructor" or "`END` is a special return value" — both are graph topology constants that must be imported and used as edge endpoints. |
| "How does a conditional edge work, and when would you use one?" | (1) Registered via `add_conditional_edges(source_node, router_fn, mapping)`; (2) after `source_node` runs, `router_fn(state)` is called; (3) return value is looked up in the optional mapping dict to resolve to a node name; (4) use case: decide whether to call tools again or stop after an LLM response. | Giving an example that only handles two outcomes. A strong answer notes that `router_fn` can return a list of node names to fan out to multiple nodes in parallel, and that `END` is always one valid return value. |
| "Explain the difference between `stream_mode='updates'` and `stream_mode='values'`." | (1) `updates`: yields only the dict keys changed by each node — bandwidth-efficient, one dict per node per step; (2) `values`: yields the full state snapshot after each step — easier for debugging; (3) `messages`: yields `(token, metadata)` tuples for LLM token streaming; (4) use `updates` in production, `values` in development. | Only describing one mode. A strong answer explains the trade-off: `values` includes all unchanged keys on every step (redundant data), while `updates` only shows what changed (easier to process programmatically). |

---

## Applied / Scenario Questions

**Q: You are asked to build a Databricks support agent that takes a user's question, looks up relevant documentation in a vector store, generates an answer, validates that the answer cites a real document, and loops back if validation fails. Design the LangGraph graph.**

**Strong answer framework:**
- **State schema:** `TypedDict` with `messages: Annotated[list, operator.add]`, `retrieved_docs: list[str]`, `validation_passed: bool`, `retry_count: int`
- **Nodes:** (1) `retrieve` — calls Unity Catalog Vector Search with the user's question; (2) `generate` — LLM generates an answer citing retrieved docs; (3) `validate` — checks that the answer contains at least one doc ID from `retrieved_docs`; (4) `fallback` — generates a safe "I could not verify this" response if `retry_count >= 2`
- **Edges:** `START` → `retrieve` → `generate` → `validate`; conditional edge after `validate`: if `validation_passed` → `END`, if not and `retry_count < 2` → `retrieve` (retry loop), if `retry_count >= 2` → `fallback` → `END`
- **How to show trade-off awareness:** Note that looping back to `retrieve` is expensive (vector search + LLM call per retry). A production version would use `stream_mode="updates"` to surface the retry to the user in real time and consider capping retries at 2–3 to control cost.

---

**Q: A team reports that their LangGraph agent is producing different outputs for the same input on repeated invocations despite using a deterministic model (temperature=0). What would you investigate?**

**Strong answer framework:**
- **First hypothesis: `thread_id` is stable across runs.** If the same `thread_id` is passed to multiple test runs, later runs accumulate prior messages in state, producing different context for the LLM — even at temperature=0. Fix: use a fresh `thread_id` per test run, or use `InMemorySaver` with a new saver instance per test.
- **Second hypothesis: reducer accumulating unexpected values.** A field with `operator.add` reducer keeps growing with each invocation on the same thread. If `retrieved_docs` accumulates across turns, the LLM sees a growing (and eventually noisy) document list.
- **Third hypothesis: tool non-determinism.** The tool itself (e.g., a vector search returning different top-k results due to index updates) is non-deterministic, not the LLM. Check tool outputs first before blaming LLM temperature.
- **Mention streaming as a diagnostic tool:** `stream_mode="updates"` will show which node produced which state update; this usually isolates the non-determinism to a specific node in under five minutes.

---

## System Design / Architecture Questions

**Q: Design a stateful multi-turn customer support agent for Databricks that must handle ~500 concurrent users, persist conversation history across browser sessions, support human escalation with an approval step before sending a refund, and be observable via MLflow Tracing.**

**Approach:**
1. **Clarify requirements:** Is the LLM call latency budget under 3s per turn? Are escalations synchronous (block until human approves) or asynchronous (notify human, continue later)? Do we need cross-session memory within a 24h window, or indefinitely?
2. **Propose structure:**
   - **State schema:** `MessagesState` subclass with additional fields: `escalation_needed: bool`, `escalation_approved: bool`, `ticket_id: str`, `refund_amount: float`
   - **Nodes:** `agent` (LLM with tools), `tools` (execute tool calls), `escalation_gate` (creates ticket, sets `escalation_needed=True`, calls `interrupt()` to pause for human), `refund_executor` (runs after approval), `fallback` (graceful degradation at recursion limit)
   - **Checkpointer:** `PostgresSaver` — handles 500 concurrent users writing checkpoints without SQLite's write-lock bottleneck
   - **Human-in-the-loop:** Use LangGraph's `interrupt()` primitive inside `escalation_gate`; the graph pauses until resumed with `Command(resume=approved_value)` from the human review UI
   - **Observability:** Wrap compiled graph in MLflow `ResponsesAgent` for automatic MLflow Tracing; set `LANGSMITH_TRACING=true` for LangSmith traces during development
   - **Deployment:** Databricks Apps with `ResponsesAgent` and `databricks bundle deploy`; Unity AI Gateway for LLM call governance and cost attribution
3. **Justify choices and name trade-offs explicitly:**
   - `PostgresSaver` over `SqliteSaver`: concurrent write safety at 500 users; trade-off is infrastructure overhead (requires a Postgres instance)
   - `interrupt()` for human escalation: synchronous by default (blocks the thread); for high-volume escalations, prefer async (write ticket to queue, have a separate worker resume the thread) to avoid tying up threads
   - `MessagesState` subclass over raw `TypedDict`: saves writing the `add_messages` reducer manually; trade-off is slightly less flexibility if your message type diverges from LangChain's `BaseMessage`

---

## Vocabulary That Signals Expertise

Use these terms naturally — don't force them:
- **Super-step** — when / why to use it: describing when the checkpointer saves a snapshot ("a checkpoint is written after each super-step, not after each node")
- **Reducer** — when discussing state update semantics ("the `messages` field uses `add_messages` as its reducer, which deduplicates by message ID rather than blindly appending")
- **`interrupt()`** — when describing human-in-the-loop patterns ("the graph pauses at the `interrupt()` call and resumes when `Command(resume=...)` is passed")
- **Thread** — when discussing persistent memory ("each user session maps to a LangGraph thread identified by a stable UUID")
- **Conditional edge + routing function** — when describing agent decision logic ("the `should_continue` routing function checks `state['messages'][-1].tool_calls` — if non-empty, it routes to `tools`; if empty, it returns `END`")
- **`StateSnapshot`** — when describing observability or debugging ("calling `graph.get_state(config)` returns a `StateSnapshot` with the full current state and the list of next nodes to execute")

---

## Vocabulary That Signals Weakness

Avoid these — they signal outdated or shallow understanding:
- **"LangGraph replaces LangChain"** — LangGraph is the orchestration runtime; LangChain provides components (models, tools, retrievers) used inside LangGraph nodes. They are complementary, not competing.
- **"The state is just a list of messages"** — This conflates `MessagesState` (one common state schema) with the general pattern. The state is any `TypedDict`; messages are just one common field.
- **"I use a global variable to track conversation history"** — This immediately signals ignorance of the thread model and signals production reliability concerns.
- **"LangGraph is only for multi-agent systems"** — LangGraph is equally valuable for single-agent stateful workflows; the multi-agent use case is just one application.
- **"Checkpointing is optional if you don't need memory"** — Checkpointing also enables fault tolerance and human-in-the-loop workflows; framing it as only about memory undersells the feature.

---

## STAR Answer Frame

**Situation:** Our production support chatbot was built as a LangChain LCEL chain. After adding tool-calling capabilities (order lookup, refund processing), we began seeing silent failures — users reported receiving garbled or incomplete answers, but our logs showed no exceptions. The agent appeared to be silently entering tool-call loops.

**Task:** I was responsible for diagnosing the failure mode and proposing a robust architecture that would prevent recurrence and make future failures immediately visible.

**Action:** I added detailed logging to the chain and discovered the agent was entering tool-call loops because the tool results were not being appended to the conversation history — the chain was treating each tool call as a fresh context rather than accumulating results in state. I proposed migrating to LangGraph: defined a `TypedDict` state with `messages: Annotated[list[BaseMessage], operator.add]`, implemented a `should_continue` conditional edge that checked `state["messages"][-1].tool_calls`, and attached `SqliteSaver` for persistence. The migration took two engineer-days. I added `stream_mode="updates"` logging in staging so every node transition was visible in the application logs.

**Result:** Post-migration, the silent tool-call loops disappeared immediately — the conditional edge routing function caught the loop condition on the second iteration and routed to `END`. When a new edge case appeared three weeks later (a tool returning an empty string that confused the LLM), we diagnosed it in under five minutes using the `stream_mode="updates"` log showing exactly which node produced the bad state. Our incident resolution time for agent failures dropped from 4 hours (average) to 15 minutes.

---

## Red Flags Interviewers Watch For

- **Cannot explain why a `thread_id` is required for multi-turn memory** — this is fundamental to how LangGraph's checkpointer retrieves and stores state; not knowing it suggests the candidate has only run demos, not built production agents.
- **Claims `recursion_limit` is the primary way to stop an agent** — a strong candidate designs an explicit `END` condition via conditional edges; relying on `GraphRecursionError` is an anti-pattern for production.
- **Does not know the difference between `InMemorySaver` and `SqliteSaver`** — this is a basic production readiness question; inability to distinguish them suggests limited real-world deployment experience.
- **Cannot name the three parts of `add_conditional_edges` (source node, router function, optional mapping dict)** — this is the core looping/routing primitive in LangGraph; not knowing it suggests surface-level framework familiarity.
- **Describes LangGraph agents as "just another chatbot framework"** — strong candidates articulate the explicit state machine model and the debuggability/testability advantage over implicit chain-based approaches.
