# Multi-Agent Systems and Genie Agents — Interview Prep

**Section:** 04 — Application Development | **Role target:** Senior AI/ML Engineer, ML Platform Engineer, Solutions Architect

---

## Core Conceptual Questions

These test whether you understand the fundamentals.

| Question | Key points to cover | Common weak-answer trap |
|---|---|---|
| What is the difference between the supervisor pattern and the swarm/agent-as-tool pattern in multi-agent systems? | Supervisor: central router node reads shared state and dispatches work via `Command`/conditional edges; all agents share one `StateGraph`; observability is end-to-end. Agent-as-tool: sub-agent is a `@tool` function, no shared state graph, parent sees only the final return string; simpler but opaque. | Saying "the supervisor just calls agents one by one" — misses that routing is dynamic and state-driven, not hardcoded sequential. |
| How does LangGraph prevent a supervisor from routing to the same agent indefinitely? | Explicit `recursion_limit` at compile time; structured output forces a bounded `Literal` enum including `"FINISH"`; supervisor prompt includes full `messages` history so it can detect completed work; `interrupt_before` allows human override if the loop escalates. | Saying "the LLM just knows when to stop" — the model has no inherent termination logic; the graph enforces it. |
| What is the `add_messages` reducer and why does it matter for multi-agent systems? | Annotates the `messages` state key so that node return values are appended (not replaced). Without it, every node overwrites the history, making the supervisor blind to prior work. It is essential for a shared, ordered conversation history across all sub-agents. | Saying "messages is just a list" without explaining what happens when two nodes write to it concurrently or sequentially. |
| How does the Genie Conversation API integrate into a multi-agent system? | Wrap the API as a `@tool`: `POST .../start-conversation` returns `conversation_id` and `message_id`; poll `GET .../messages/{msg_id}` until `status=COMPLETED`; extract answer from `attachments`. Unity Catalog security is enforced per requesting user, not per service principal — a key governance advantage. | Treating Genie as a synchronous API ("just POST and read the response") — misses the polling requirement and the security implications. |
| How do you choose between LangGraph and CrewAI for a multi-agent system? | LangGraph: production-grade, precise state control, `interrupt_before` for HITL, native MLflow tracing via `ResponsesAgent`, explicit routing graph. CrewAI: faster prototyping, declarative role/task model, less boilerplate for simple workflows. Govern by: need for per-user data access control → LangGraph; need for human approval gates → LangGraph; proof-of-concept with role-based agents → CrewAI. | "CrewAI is simpler so I'd use it for production" — ignores the lack of first-class HITL support and the difficulty of enforcing Unity Catalog per-user security through CrewAI's implicit context propagation. |

---

## Applied / Scenario Questions

**Q:** You are asked to debug a LangGraph multi-agent system where the graph runs for a while and then raises `GraphRecursionError`. The graph has a supervisor, a retrieval agent, and a synthesis agent. What do you investigate first?

**Strong answer framework:**
- First check whether the supervisor is receiving the full `messages` history — if it is only seeing the latest message, it cannot detect that retrieval has already run and will keep routing to the retrieval agent.
- Second, verify that the retrieval agent's output is actually being written to the `messages` channel (not to a custom state key that the supervisor does not read).
- Third, confirm the supervisor's structured output includes `"FINISH"` as a valid option and that the routing function maps `"FINISH"` to `END`.
- Fourth, check `recursion_limit` — was it set explicitly? If not, default 25 may be too low for the intended number of routing cycles.
- Show tradeoff awareness: artificially raising `recursion_limit` without fixing the root cause is a band-aid; the correct fix is ensuring the supervisor has the information to terminate.

---

**Q:** A security reviewer flags your multi-agent system because it has a `run_sql` tool that uses a service principal with broad warehouse permissions. How do you remediate this?

**Strong answer framework:**
- Replace the `run_sql` tool with a Genie Agent tool wrapper that calls `POST /api/2.0/genie/spaces/{space_id}/start-conversation` using the requesting user's OAuth token (not the service principal token).
- Explain that Genie Agents enforce Unity Catalog row-level security per the identity in the request — each user only sees data they are authorised to access, enforced at query execution time.
- Note that Genie Agents generate read-only SQL — they cannot issue `INSERT`, `UPDATE`, or `DELETE` statements.
- Show tradeoff awareness: Genie Agents require a well-configured space (datasets, example SQL, instructions) — the remediation is not zero-cost, and you will need to set up and test the Genie Agent before replacing the tool.

---

## System Design / Architecture Questions

**Q:** Design a multi-agent system on Databricks that answers complex supply chain questions. The system must query structured inventory data, retrieve from unstructured supplier contract PDFs, synthesise an answer, and require manager approval before returning the answer to the user.

**Approach:**

1. **Clarify requirements:** What latency is acceptable? How many users concurrently? What is the sensitivity of inventory data?

2. **Propose structure:**
   - `StateGraph` with four nodes: `supervisor`, `inventory_agent`, `document_agent`, `synthesis_agent`.
   - `supervisor` uses structured output (`RouteDecision` Pydantic model) to dispatch to `inventory_agent` or `document_agent` based on current message state.
   - `inventory_agent` wraps a Genie Agent tool call — enforces Unity Catalog row security for inventory tables.
   - `document_agent` wraps a vector search tool for PDF retrieval.
   - `synthesis_agent` combines retrieved facts into a final answer.
   - Compile with `interrupt_before=["synthesis_agent"]` — execution pauses, manager reviews the intermediate state in the MLflow trace, resumes with `app.invoke(None, config={"thread_id": ...})`.
   - Set `recursion_limit=20` explicitly.
   - Wrap in `ResponsesAgent` for deployment on Databricks Apps.

3. **Justify choices and name tradeoffs explicitly:**
   - Genie for inventory data: governance out-of-the-box; limitation is setup time for the knowledge store.
   - `interrupt_before`: human-in-the-loop with no code changes to the agents; limitation is that the graph must use a checkpointed state (e.g. `SqliteSaver` or Databricks-managed state) to persist between the pause and resume.
   - `ResponsesAgent`: compatible with AI Playground, Agent Evaluation, and MLflow tracing.

---

## Vocabulary That Signals Expertise

Use these terms naturally — don't force them:
- **`add_messages` reducer** — when explaining how shared state accumulates conversation history
- **`Command(goto=...)` / conditional edges** — when distinguishing dynamic routing from static pipelines
- **`interrupt_before`** — when describing human-in-the-loop without restructuring the graph
- **`recursion_limit`** — when explaining how to prevent infinite supervisor loops
- **`space_id` / Genie Conversation API** — when describing governed SQL analytics as a sub-agent
- **`ResponsesAgent` (MLflow)** — when describing how multi-agent graphs are deployed and traced on Databricks
- **compound AI system** — when describing Genie's internal architecture (compound system, not single LLM)

---

## Vocabulary That Signals Weakness

Avoid these — they signal outdated or shallow understanding:
- **"Just increase the context window"** — treating context limits as the core problem; the core problem is isolation and observability
- **"The agent decides when to stop"** — agents have no inherent termination logic; the graph's `recursion_limit` and `FINISH` routing enforce termination
- **"Genie Spaces"** without noting the rename — the product is now called **Genie Agents** as of July 2026; using the old name signals you haven't read the current docs
- **"I would use a for-loop to call each agent"** — sequential `.invoke()` chaining destroys shared state and makes tracing impossible
- **"CrewAI is more production-ready than LangGraph"** — the opposite is true for Databricks deployments requiring per-user governance and HITL

---

## STAR Answer Frame

**Situation:** I was working on a multi-agent analytics assistant for a finance team. The system included a supervisor, a document retrieval agent, and an analytics agent. Three days before release, the team reported that the graph occasionally ran indefinitely without returning a result, eventually hitting a timeout.

**Task:** I was responsible for diagnosing and fixing the infinite-loop behaviour without breaking the routing logic or rewriting the supervisor prompt.

**Action:** I added MLflow tracing to every node and replayed the failing queries. The traces revealed the supervisor was not seeing the retrieval agent's output in its messages — it was being written to a custom `retrieval_results` key, not to the `messages` channel. The supervisor's structured output included `"FINISH"` only if it saw a message from the retrieval agent in the history; without that message, it kept routing back to retrieval. I added the `add_messages` reducer to the `retrieval_results` key as well, and also updated the supervisor prompt to check for a retrieval-tagged message. I also set `recursion_limit=12` explicitly and added a `GraphRecursionError` catch in the caller to surface clean error messages instead of silent timeouts.

**Result:** The fix eliminated the infinite loops entirely. The explicit `recursion_limit` also caught two previously hidden edge cases in staging that would have caused customer-facing timeouts. Post-deployment MLflow traces showed clean termination on every query in the first two weeks of production traffic.

---

## Red Flags Interviewers Watch For

- **Cannot explain the difference between `add_messages` and `operator.add`** — suggests the candidate has copied code without understanding state management
- **Claims Genie Agent mode is available via the API** — as of July 2026, Agent mode is UI-only; this mistake signals they haven't verified against current docs
- **Proposes sequential agent chaining as a "simpler" multi-agent approach** — signals no understanding of state sharing or observability tradeoffs
- **Cannot name a concrete Genie API endpoint or polling pattern** — suggests hand-wavy knowledge of the Genie integration without actual implementation experience
- **Describes `recursion_limit` as "a LangGraph performance setting"** — it is a correctness guard against infinite loops, not a performance parameter
