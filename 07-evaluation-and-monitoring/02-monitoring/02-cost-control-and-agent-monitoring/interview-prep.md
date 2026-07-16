# Cost Control and Agent Monitoring — Interview Prep

**Section:** 07 Evaluation and Monitoring | **Role target:** Senior AI/ML Engineer, ML Platform Engineer, Solutions Architect (GenAI)

---

## Core Conceptual Questions

| Question | Key points to cover | Common weak-answer trap |
|---|---|---|
| What is the difference between a rate limit and a spend budget in Unity AI Gateway? | Rate limits are per-minute consumption ceilings (QPM/TPM) that protect capacity and fairness; spend budgets are monthly dollar thresholds that protect financial exposure. Rate limits reject requests with 429; budgets send alerts or block (Genie only). They solve different problems and both are needed. | Saying "rate limits control cost" — they control *rate*, not cumulative spend. A user within a per-minute limit can still accumulate large monthly spend. |
| What does `mlflow.langchain.autolog()` actually do under the hood? | Monkey-patches LangChain/LangGraph execution to emit OpenTelemetry-compatible spans automatically. Captures inputs, outputs, token counts, latency, and status for each LLM call, tool invocation, and graph node. Spans are assembled into a hierarchical trace and flushed asynchronously to the configured storage backend. | Saying "it logs metrics to MLflow" — autologging specifically produces *traces* (span hierarchies), not just scalar metrics. The distinction matters because traces carry structured call graphs. |
| Why should production MLflow traces be stored in Unity Catalog rather than the experiment artifact store? | No 100,000 trace cap; traces are stored as OTel Delta tables (SQL-queryable with Spark, AI/BI, Genie); Unity Catalog governs access with the same privileges as data tables; OpenTelemetry compatible for cross-tool observability; production monitoring requires this backend for long-term retention. | Saying "experiment storage is fine for small teams" — the 100K cap and lack of SQL access are architectural constraints, not just scale concerns. |
| What is `recursion_limit` in LangGraph and why is it a production cost control? | `recursion_limit` is a compile-time configuration on a `StateGraph` that raises `GraphRecursionError` after N node visits. It is a cost control because a looping agent (e.g. one where the LLM repeatedly calls a tool that returns empty results) would otherwise make unbounded LLM calls, each consuming tokens. Setting it to 20–25 bounds worst-case LLM call count per invocation. | Saying "LangGraph doesn't loop by default" — it does loop whenever a conditional edge returns a node name rather than `END`. Without a recursion limit, a misconfigured condition creates an infinite loop. |
| How do you attribute per-user token spend on Databricks? | Query `system.billing.usage` filtering `product = 'UNITY_AI_GATEWAY'` and group by `identity_metadata.run_as` (the principal that made the request). For request-level detail, join with inference table records (logged per request to a Unity Catalog Delta table). Cross-reference with MLflow traces by `request_id` to get per-trace token breakdown. | Querying only MLflow traces for spend attribution — traces show token counts per invocation but are not the billing source of truth. `system.billing.usage` is the authoritative spend record. |

---

## Applied / Scenario Questions

**Q:** A deployed LangGraph customer-support agent has been running for two weeks. The monthly spend is 3× the estimate, but no errors are reported and user feedback is positive. How do you diagnose and fix this?

**Strong answer framework:**
- **Step 1 — Get attribution before acting:** Query `system.billing.usage` grouping by `identity_metadata.run_as` to identify whether the overrun is concentrated in a few users or distributed. If concentrated, investigate those users' trace patterns specifically.
- **Step 2 — Find expensive invocations:** Use `mlflow.search_traces()` with `order_by=["attributes.total_tokens DESC"]` to surface the highest-token invocations. Inspect the span trees of the top 10.
- **Step 3 — Diagnose the pattern:** Count LLM spans per trace. If most traces have 8–12 LLM spans when 3–4 is expected, the agent is looping. If the token count is high but LLM call count is normal, the prompts or completions are unexpectedly large — check if accumulated conversation history is being appended to every prompt.
- **Step 4 — Apply targeted controls:** For loop issues: reduce `recursion_limit` and add a token budget guard in agent state. For large prompts: implement prompt trimming or history summarization. For specific users: set per-user TPM limits in Unity AI Gateway.
- **Show tradeoff awareness:** The team should not tighten limits blindly — a user with legitimately complex queries needs enough tokens to get a useful response. Trace data informs *where* to constrain without degrading quality for normal users.

---

**Q:** Your team is setting up production monitoring for a LangGraph agent for the first time. Walk me through the full setup from "just deployed" to "traces visible in Unity Catalog with sampling-based quality scorers running."

**Strong answer framework:**
1. Call `mlflow.set_experiment("/Shared/agents/my-agent")` and `mlflow.langchain.autolog()` in the model serving container's model loading code (not just in the notebook)
2. In the MLflow UI, bind the experiment to a Unity Catalog trace location (Experiments > Edit > Trace storage) so traces land in OTel Delta tables
3. Verify traces are appearing: invoke the agent a few times, then check the **Traces** tab in the MLflow UI and run `mlflow.search_traces(experiment_names=[...])` to confirm
4. Register a production monitoring scorer: define a `@scorer` function in a Databricks notebook, call `.register(name="my_scorer")` and `.start(sampling_config=ScorerSamplingConfig(sample_rate=0.2))`
5. For critical scorers (safety), use `sample_rate=1.0`; for expensive LLM judges, use `0.05–0.2`
6. Configure Unity AI Gateway rate limits and a budget threshold to protect against cost spikes while the agent is being monitored

---

## System Design / Architecture Questions

**Q:** Design a cost observability and enforcement system for a team of 50 data scientists sharing a pool of Databricks model serving endpoints for LLM-based agents. The team needs per-user spend visibility, automatic alerting, and protection against runaway agents, without blocking legitimate high-intensity workloads.

**Approach:**
1. **Clarify requirements:** Is there a dollar budget per user per month? What defines "runaway" (e.g. single invocation > 10,000 tokens, or > 50 LLM calls)? What response time is acceptable for alerting?
2. **Propose structure:**
   - Unity AI Gateway model services with per-user default TPM limits (e.g. 50,000 TPM) and higher per-service-principal limits for automated pipelines
   - Unity AI Gateway budget with per-user threshold at 80% of per-user allocation → alert email; team lead notified, not blocked
   - MLflow tracing with Unity Catalog backend on every agent endpoint; `mlflow.langchain.autolog()` in all serving containers
   - Custom `@scorer` for loop detection (`llm_call_count > 15` → flag trace) registered at `sample_rate=1.0` on the production monitoring job
   - Weekly attribution dashboard from `system.billing.usage` joined with MLflow trace data, surfaced via AI/BI or a Databricks SQL dashboard
3. **Justify choices and name tradeoffs explicitly:**
   - Per-user TPM rather than QPM because token spend is the cost driver, not request count
   - Alert without block because blocking production pipelines mid-run creates data consistency issues; alerting with manual review is preferred for an internal team
   - 100% sampling for loop detection (cheap to evaluate, high severity) vs 10–20% for quality scorers (expensive LLM judges)
   - Unity Catalog trace storage over experiment artifact store because 50 data scientists × multiple agents × production traffic would exceed 100K traces quickly

---

## Vocabulary That Signals Expertise

Use these terms naturally — don't force them:
- **"span hierarchy"** — when discussing how to read a trace to find a failure; signals you understand traces as trees, not flat logs
- **"TPM vs QPM"** — distinguishing token-based vs request-based rate limiting; signals you understand cost-per-token economics
- **"near-real-time enforcement decoupled from `system.billing.usage` refresh"** — when explaining budget alerts; signals you understand the eventual consistency in the billing pipeline
- **"Unity Catalog OTel Delta tables"** — when describing trace storage; signals you know the production-recommended backend and its SQL-queryable nature
- **"recursion_limit on the compiled graph"** — not "add a loop counter to my code"; signals you use the framework's built-in mechanism
- **"`ScorerSamplingConfig(sample_rate=...)`"** — when discussing production monitoring; signals you understand the coverage vs cost tradeoff in continuous evaluation
- **"inference table"** — the AI Gateway feature that logs request/response payloads; distinct from MLflow traces

## Vocabulary That Signals Weakness

Avoid these — they signal outdated or shallow understanding:
- **"just set max_tokens"** — signals unfamiliarity with multi-step agent cost dynamics; max_tokens is one of many controls
- **"rate limits prevent cost overruns"** — conflates rate (per-minute) with cumulative spend; signals surface-level familiarity
- **"we log to CloudWatch / Datadog for agent monitoring"** — not wrong for infrastructure, but misses the agent-specific trace hierarchy that makes span-level debugging possible; signals infrastructure monitoring mental model applied to a different problem
- **"MLflow tracks metrics"** — accurate but incomplete; the production-relevant feature is *traces* (hierarchical span trees), not scalar metrics
- **"we watch the billing dashboard"** — reactive and non-actionable; signals no proactive attribution or alerting infrastructure

---

## STAR Answer Frame

**Situation:** Our team deployed a LangGraph research assistant to production. Within the first month, LLM spend exceeded the quarterly budget estimate, but the agent appeared to be working correctly from the user's perspective.

**Task:** I was responsible for identifying the cause of the cost overrun and implementing controls that would prevent recurrence without degrading the user experience.

**Action:** I enabled `mlflow.langchain.autolog()` in the model serving container (it had only been in development notebooks previously), bound the MLflow experiment to Unity Catalog trace storage, then queried `system.billing.usage` grouped by `identity_metadata.run_as` to find the heaviest users. I used `mlflow.search_traces()` to retrieve the 50 highest-token traces for those users and inspected the span trees. The culprit was a tool call node that returned an empty list — the agent interpreted this as "try again" and re-invoked the tool 8–12 times per session before giving up. I added a `recursion_limit = 20` to the compiled graph, implemented a token budget guard in agent state that short-circuited after 4,000 accumulated tokens, and set per-user TPM limits of 60,000 in Unity AI Gateway with a budget alert at 80% of the team's monthly allocation.

**Result:** Average tokens per invocation dropped from ~6,200 to ~1,400 (77% reduction). Monthly spend returned to within 10% of the forecast. Zero degradation in user-reported quality scores because the loop fix did not affect successful invocations — it only eliminated unnecessary retries on the empty-tool-response path.

---

## Red Flags Interviewers Watch For

- Recommending `max_tokens` as the primary cost control without mentioning loop detection, per-user rate limits, or trace-based attribution — suggests the candidate has only deployed single-call RAG pipelines, not multi-step agents
- Describing Unity Catalog and MLflow as separate, independent systems rather than an integrated observability stack — suggests unfamiliarity with the Databricks platform architecture
- Inability to distinguish between what a *rate limit* and a *budget* control — two different knobs for two different failure modes; confusing them in an interview is a clear gap signal
- Saying "we would add more logging" as the answer to agent observability — logs are unstructured and do not capture the span hierarchy needed for root-cause analysis; the correct answer is structured tracing
- Not mentioning that production monitoring scorers must be registered from a Databricks notebook (serialization requirement) — signals the candidate has read the docs but not the operational constraints
