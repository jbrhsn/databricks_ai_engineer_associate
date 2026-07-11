# MCP Servers and Persistent Memory — Interview Prep

**Section:** 05 — Assembling and Deploying Applications | **Role target:** Senior ML Engineer, AI Platform Engineer, Solutions Architect

> ⚠️ **Fast-evolving:** MCP and Databricks managed MCP servers are in Public Preview as of July 2026. Verify current API details before interviews at companies actively using Databricks AI.

---

## Core Conceptual Questions

These test whether you understand the fundamentals.

| Question | Key points to cover | Common weak-answer trap |
|---|---|---|
| What is MCP and what problem does it solve? | Open JSON-RPC 2.0 standard; standardises how AI agents discover and call external tools/resources/prompts; eliminates bespoke per-integration auth and error handling; the USB-C analogy for AI connectivity | Describing it only as "a way to give agents tools" — confuses MCP with LangChain tool wrappers; misses the governance, discoverability, and standardisation dimensions |
| What are the three MCP primitive types and when do you use each? | Tool = executable action (`tools/call`); Resource = read-only context data (`resources/read`); Prompt = reusable template (`prompts/get`); only Tool invocation feeds back into the LLM as a tool message | Saying all three trigger an LLM call — Resources and Prompts are pure data fetches; also avoid treating Prompt as a system prompt replacement (it is a fetched template, not an injected instruction) |
| What is the difference between stdio and Streamable HTTP MCP transports? | stdio: client spawns server as subprocess, stdin/stdout exchange, zero network overhead, local only, single client; Streamable HTTP: server is an independent process, HTTP POST + optional SSE, multi-client, supports auth headers, used by all Databricks managed servers | Saying HTTP is always better — stdio is fully production-ready for local single-user tools and has lower latency; the choice is about deployment topology |
| What is a LangGraph checkpointer and why does it matter for agents? | Snapshots `TypedDict` state at every super-step; keyed by `thread_id`; enables multi-turn memory, fault-tolerant resumption, human-in-the-loop; `MemorySaver` = in-process (dev only); durable options = `SqliteSaver`, `AsyncPostgresSaver`, custom Delta-backed | Saying "checkpointer is how LangGraph does logging" — it is a state persistence layer, not a logging mechanism; also avoid saying MemorySaver is fine for production |
| What is the difference between short-term and long-term agent memory? | Short-term: conversation history in LangGraph state, injected into context window every turn, lost on process restart without external checkpointer, grows with turns; Long-term: external store (Delta/Vector Search/Redis), retrieved selectively, persists across sessions, does not consume context window proportionally | Saying "just use a bigger context window" — misses the cost/latency implication and the selectivity requirement; also conflating checkpointer with long-term memory |

---

## Applied / Scenario Questions

**Q:** You are building a LangGraph agent that needs to query a Genie Agent for billing data and an AI Search index for support documentation. How do you wire up these two data sources using Databricks MCP, and what authentication approach do you use when deploying on Databricks Apps?

**Strong answer framework:**
- Instantiate `DatabricksMultiServerMCPClient` with two `DatabricksMCPServer` entries: one pointing at the Genie Agent managed MCP server URL (`/api/2.0/mcp/genie/{space_id}`) and one at the AI Search server URL (`/api/2.0/mcp/ai-search/{catalog}/{schema}/{index}`)
- Use `async with mcp_client:` context manager, call `await mcp_client.get_tools()` to discover tools from both servers dynamically
- Pass `workspace_client = WorkspaceClient()` (no explicit profile in production; Databricks Apps injects credentials) — for on-behalf-of-user, use `ModelServingUserCredentials`
- Wire tools into `create_react_agent(llm, tools, checkpointer=...)`
- In `databricks.yml`, declare the Genie space (`CAN_RUN`) and AI Search index (`SELECT`) as resources under the app so the app's service principal is granted access at deploy time
- Show trade-off awareness: `DatabricksMultiServerMCPClient` handles the auth and tool-list aggregation; the alternative of hard-coded LangChain tool wrappers bypasses Unity AI Gateway governance

**Q:** A deployed agent intermittently "forgets" earlier turns for some users but works perfectly for others. How do you diagnose and fix this?

**Strong answer framework:**
- First hypothesis: `MemorySaver` singleton shared across process replicas — each replica has its own in-process heap; round-robin routing sends user's follow-up to a different replica with empty state
- Verification: check if the app has more than one replica; reproduce by invoking the agent twice with the same `thread_id` from two different processes
- Fix: replace `MemorySaver` with `AsyncPostgresSaver` (or a custom Delta-backed checkpointer) so all replicas read/write the same store
- Secondary hypothesis: `thread_id` not set consistently — some code paths use a hardcoded test ID or omit the config dict; verify with tracing
- Show trade-off: external checkpointer adds network latency on every state read/write (~5–20 ms for Postgres); acceptable for conversational agents, potentially significant for high-frequency agentic loops

---

## System Design / Architecture Questions

**Q:** Design a production customer-support agent for 200 concurrent users that needs: (1) real-time access to billing data in Databricks, (2) semantic search over a support KB, (3) per-session conversational memory, and (4) Unity Catalog governance over all data access.

**Approach:**
1. **Clarify requirements:** What is the expected session duration? Is cross-session memory needed? What is the SLA on response latency?
2. **Propose structure:**
   - Deployment: Databricks Apps (process lifecycle management for checkpointer)
   - LLM endpoint: Databricks Model Serving (`databricks-claude-sonnet-4-5` or similar)
   - Tool connectivity: `DatabricksMultiServerMCPClient` with three managed servers: Genie Agent (billing), AI Search (KB), UC functions (custom business logic)
   - Memory: `AsyncPostgresSaver` or Redis-backed checkpointer keyed by `session_id` for short-term; Delta table with merge upsert in a `memory_write` node for long-term user preferences if cross-session recall is needed
   - Auth: OAuth on-behalf-of-user (`ModelServingUserCredentials`) so each user's Unity Catalog permissions are enforced on their MCP calls
   - Observability: MLflow tracing on the agent; Unity AI Gateway logs for MCP calls; inference tables for response quality monitoring
3. **Justify choices and name trade-offs explicitly:**
   - Managed MCP servers over custom wrappers: governance without custom auth code, automatic tool discovery
   - Databricks Apps over Model Serving: needed for external checkpointer + app-level resource declarations; trade-off is higher operational complexity vs fully managed endpoint
   - AsyncPostgresSaver over Delta-backed checkpointer: lower write latency (Postgres vs Delta MERGE); trade-off is an additional Postgres dependency to manage
   - On-behalf-of-user auth over service principal: security best practice (principle of least privilege per user); trade-off is that each user must have grants on the underlying data assets

---

## Vocabulary That Signals Expertise

Use these terms naturally — don't force them:
- **JSON-RPC 2.0** — when explaining the MCP wire protocol (not just "REST" or "API calls")
- **Capability negotiation** — the `initialize` handshake phase where client and server declare what they support
- **Streamable HTTP transport** — the current name (not "HTTP+SSE", which is the deprecated 2024-11-05 transport)
- **`tools/list` / `tools/call`** — the specific MCP discovery and invocation methods, shows protocol-level familiarity
- **`thread_id`** — the LangGraph config key that scopes checkpointer snapshots; signals you know how multi-turn memory actually works
- **`AsyncPostgresSaver`** — signals you know the production checkpointer options beyond `MemorySaver`
- **`DatabricksMultiServerMCPClient`** — signals familiarity with the Databricks-specific MCP SDK
- **Unity AI Gateway** — the correct product name for Databricks' MCP governance control plane
- **On-behalf-of-user authentication** / **`ModelServingUserCredentials`** — signals awareness of the auth model for user-scoped permissions in deployed agents
- **Super-step** — LangGraph terminology for a single execution unit that the checkpointer snapshots

---

## Vocabulary That Signals Weakness

Avoid these — they signal outdated or shallow understanding:
- **"HTTP+SSE transport"** — this is the deprecated protocol version from 2024-11-05; the current name is Streamable HTTP transport
- **"MemorySaver is fine for production"** — immediate red flag; shows the candidate has not thought about multi-replica deployment
- **"MCP is just a fancy tool wrapper"** — misses the protocol standardisation, governance, and dynamic discoverability dimensions
- **"The agent calls the MCP server directly"** — the LLM does not call MCP; the client intercepts the LLM's tool-call output and routes it
- **"Long-term memory means injecting all conversation history"** — signals no understanding of context window constraints or selective retrieval
- **"You have to restart the agent to pick up new tools"** — MCP supports `tools/list_changed` notifications; the client can refresh dynamically
- **"Delta tables are too slow for agent memory"** — depends entirely on the access pattern; a simple SELECT by primary key on a small preference table is milliseconds; only bulk scans are slow

---

## STAR Answer Frame

**Situation:** I was building a customer-analytics agent for an enterprise client on Databricks. The agent needed to query three different data sources: a Genie space for NL-to-SQL queries, a Vector Search index for product documentation, and a set of UC functions for custom pricing logic. Each data source had its own team maintaining it, and the security team required unified audit logging for all AI data access.

**Task:** Design and implement the agent's tool connectivity layer and memory architecture so the agent could access all three data sources without bespoke auth code, with audit trails, and with per-session conversational memory for a concurrent multi-user deployment.

**Action:** I replaced the three hand-rolled LangChain tool wrappers with a `DatabricksMultiServerMCPClient` pointing at three managed MCP server URLs (Genie, AI Search, and UC functions). This moved auth and audit to Unity AI Gateway, eliminating ~300 lines of custom wrapper code. For memory, I identified that `MemorySaver` was in use and the app was running two replicas — I replaced it with `AsyncPostgresSaver` pointing at a Databricks-provisioned Postgres instance, keyed by user session ID. I added a `memory_read` node that queried a Delta table for the user's saved preferences at session start, and a `memory_write` node that upserted extracted entities at session end.

**Result:** Zero authentication incidents in the first three months post-launch. The security team approved the deployment in one review cycle (previously taking three cycles) because they could inspect all agent data access from the Unity AI Gateway MCP tab. Session memory failures dropped from ~30% of sessions to under 1% after the checkpointer migration. The UC functions team published two new tools during the period — the agent picked them up automatically via `tools/list` without any code change on the agent side.

---

## Red Flags Interviewers Watch For

- **Cannot name a MCP primitive other than Tool** — suggests surface-level familiarity from a single tutorial rather than production experience or thorough study
- **Does not mention `thread_id` when asked about multi-turn memory** — the most common gap; candidates often describe the checkpointer concept correctly but do not know the config key that activates it
- **Says "MCP replaces LangGraph tool nodes"** — MCP is a transport protocol; LangGraph nodes are still the orchestration layer that calls MCP tools; they coexist
- **Cannot distinguish Databricks Apps from Model Serving for stateful agents** — Model Serving is stateless by default; any candidate proposing a fully stateful agent on Model Serving without an external checkpointer has not deployed one in production
- **Uses "stateful" and "persistent" interchangeably** — stateful (process holds in-memory state) and persistent (state survives process restart) are different properties; a `MemorySaver`-backed agent is stateful but not persistent
- **Cannot explain what happens when the same `thread_id` is used across multiple replicas** — this is the fundamental multi-replica memory correctness question; inability to answer it suggests the candidate has only run agents in single-process development environments
