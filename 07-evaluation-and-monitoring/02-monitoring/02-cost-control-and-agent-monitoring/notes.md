# Cost Control and Agent Monitoring

**Section:** 07 Evaluation and Monitoring | **Module:** 02 Monitoring | **Est. time:** 2.5 hrs | **Exam mapping:** Domain 6 — Evaluation and Monitoring (12%)

---

## TL;DR

LLM agents in production burn tokens through every tool call, retry, and loop — and without explicit controls, costs compound silently across users and use-cases. Databricks provides two complementary layers: Unity AI Gateway enforces rate limits and spend budgets at the infrastructure level, while MLflow Tracing makes every agent invocation — every LLM call, every tool execution, every span of latency — observable and queryable. Together they let you detect a runaway agent in minutes rather than at the end-of-month bill. **The one thing to remember: a token budget without a trace is just a prayer — you need both enforcement (Gateway) and visibility (MLflow Tracing) to actually control costs.**

---

## ELI5 — Explain It Like I'm 5

Imagine you hire a bunch of kids to run errands. Each kid has a walkie-talkie (the LLM), a list of places they can visit (tools), and a daily spending allowance (rate/budget limit). The allowance stops them from blowing all the money in one trip. But to know *why* the money ran out — which errand took 20 stops instead of 3, which kid got lost in a loop at the hardware store — you need a trip log: every stop, every question asked, every minute spent. That trip log is exactly what MLflow Tracing gives you. The common misconception is that setting a token limit is enough to control costs; it is not — limits cap the blast radius but they cannot tell you which agent node is the problem, which is the only information that lets you fix it.

---

## Learning Objectives

By the end of this chapter you will be able to:
- [ ] Configure Unity AI Gateway rate limits (QPM and TPM) at service, user, and group levels to enforce consumption ceilings
- [ ] Set up spend budgets in Unity AI Gateway with shared and per-user thresholds and alert notifications
- [ ] Enable MLflow autologging for LangGraph agents and interpret the resulting trace structure (spans, tool calls, LLM calls)
- [ ] Extract per-request token counts, latency breakdowns, and tool call success rates from MLflow traces
- [ ] Diagnose agent failures — runaway loops, malformed tool outputs, node errors — by reading a trace hierarchy

---

## Visual Overview

### Agent Monitoring Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PRODUCTION REQUEST                           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │    Unity AI Gateway            │
              │  ┌────────────────────────┐    │
              │  │  Rate Limits (QPM/TPM) │    │
              │  │  Spend Budgets         │    │
              │  │  Traffic Routing       │    │
              │  └────────────────────────┘    │
              └──────────────┬─────────────────┘
                             │ allowed through
                             ▼
              ┌────────────────────────────────┐
              │    LangGraph Agent             │
              │  ┌────────────────────────┐    │
              │  │  Node A ──► Tool Call  │    │
              │  │  Node B ──► LLM Call   │    │
              │  │  Node C ──► LLM Call   │    │
              │  └────────────────────────┘    │
              └──────────────┬─────────────────┘
                             │ trace data
                             ▼
              ┌────────────────────────────────┐
              │  MLflow Tracing                │
              │  ┌────────────────────────┐    │
              │  │  Spans (hierarchy)     │    │
              │  │  Token counts          │    │
              │  │  Latency per span      │    │
              │  │  Tool call status      │    │
              │  └────────────────────────┘    │
              └──────────────┬─────────────────┘
                             │ stored in
                             ▼
              ┌────────────────────────────────┐
              │  Unity Catalog (Delta tables)  │
              │  MLflow Experiment UI          │
              │  system.billing.usage          │
              └────────────────────────────────┘
```

### Rate Limit Priority Hierarchy

```
Incoming request
│
├── Service-level limit (global ceiling — QPM or TPM for all users)
│   └── If exceeded ──► HTTP 429 for ALL users
│
└── Per-user limits (evaluated after service limit passes)
    ├── Individual user / service principal limit (highest priority)
    ├── User group limit (shared across group members)
    └── Default user limit (fallback if no custom rule matches)
```

### Trace Span Hierarchy for a LangGraph Agent

```
Trace: agent_invocation  [total latency, total tokens]
│
├── Span: graph_run
│   ├── Span: node__retrieve_context   [latency, tool call status]
│   │   └── Span: tool_call__search    [input, output, status=OK/ERROR]
│   │
│   ├── Span: node__generate_response  [latency]
│   │   └── Span: llm_call             [prompt_tokens, completion_tokens, model]
│   │
│   └── Span: node__validate_output    [latency]
│       └── Span: llm_call             [prompt_tokens, completion_tokens]
│
└── Attributes: status, total_tokens, request_id, user_id
```

### Cost Attribution Query Flow

```
system.billing.usage
        │
        │ filter: product = 'UNITY_AI_GATEWAY'
        ▼
Unity AI Gateway usage records
        │
        │ join on: service_name, principal_id
        ▼
Spend by team / use-case / user
        │
        │ join on: request_id (inference table)
        ▼
Request-level cost attribution
        │
        │ cross-reference
        ▼
MLflow trace spans ──► identify expensive agent patterns
```

---

## Key Concepts

### Token Budgeting

**What is it?** Token budgeting is the practice of setting explicit upper bounds on the number of tokens — prompt tokens + completion tokens — consumed per LLM call, per agent invocation, or per user session. It prevents unbounded spend from runaway loops, over-long prompts, or verbose completions.

**How does it work mechanistically?** At the prompt level, `max_tokens` (or `max_completion_tokens` depending on the API) tells the model server to stop generating after N tokens, returning whatever the model has produced so far. This is enforced server-side before the response leaves the inference endpoint. For cumulative spend across a session or user, you must instrument tracking yourself: either read the `usage` field from each API response (which contains `prompt_tokens` and `completion_tokens`) and accumulate them in agent state, or query the inference table after the fact. When a cumulative threshold is crossed, the agent logic should short-circuit rather than continue calling the LLM.

**Where does it appear in Databricks?** In the LangGraph agent state, you maintain a `token_count` accumulator and check it before each node that calls an LLM. The `max_tokens` parameter is passed directly to the model serving endpoint in the request body. Inference tables (enabled via AI Gateway) record `request_body` and `response_body` including token counts, queryable as Delta tables in Unity Catalog.

### Unity AI Gateway Rate Limiting

> ⚠️ **Fast-evolving:** Unity AI Gateway is in Beta as of July 2026. Rate limit configuration UI and API may change. Verify against [Databricks release notes](https://docs.databricks.com/aws/en/release-notes/) before production use.

**What is it?** Rate limiting in Unity AI Gateway enforces consumption ceilings at the service level (all users combined) and at the user/group level, measured in Queries Per Minute (QPM) or Tokens Per Minute (TPM). When a limit is exceeded, the service returns HTTP 429 Too Many Requests.

**How does it work mechanistically?** The rate limiter operates at low latency by recording usage *after* a response is sent rather than checking ahead of time. This means short bursts can slightly exceed the configured limit immediately after a service starts or restarts; over a longer window, the average converges. If both QPM and TPM limits are specified on the same service, the more restrictive one is enforced. Service-level limits are enforced globally — if the service ceiling is hit, all users are blocked regardless of their individual limits. Custom per-user or per-group limits override the default user limit; if a user is covered by both a user-specific and a group-specific limit, the user-specific one wins.

**Where does it appear in Databricks?** Rate limits are configured in the Unity AI Gateway UI (under Data and AI Governance > AI Governance) or via the Unity Catalog REST API on model service objects. A maximum of 20 rate limits and 5 group-level limits are supported per service.

### Unity AI Gateway Spend Budgets

> ⚠️ **Fast-evolving:** Budget management in Unity AI Gateway is in Beta as of July 2026. Per-user overrides and usage blocking are currently only available for Genie budgets, not general AI Gateway budgets.

**What is it?** Spend budgets are monthly dollar-denominated limits attached to Unity AI Gateway services. They provide alerting and, for Genie, hard blocking when spend thresholds are crossed. They scope to the `UNITY_AI_GATEWAY` product in `system.billing.usage`.

**How does it work mechanistically?** A budget object consists of one or more thresholds, each with a type (shared or per-user), a dollar amount, and an action (send alert email or block usage). The budget enforcement engine tracks near-real-time spend; when a threshold is reached, it triggers the configured action. Spend data flows through `system.billing.usage` which updates every few hours — the near-real-time enforcement is decoupled from the table refresh. One limitation: provisioned throughput and external-model inference are not currently tracked in Unity AI Gateway budgets.

**Where does it appear in Databricks?** Budgets are created in the account console under Budgets. Up to 1,000 budgets per account are supported. `system.billing.usage` is the underlying table; filtering on `product = 'UNITY_AI_GATEWAY'` scopes to AI Gateway spend.

### MLflow Trace Monitoring for LangGraph Agents

> ⚠️ **Fast-evolving:** MLflow 3 tracing API and the Unity Catalog trace storage backend are actively evolving (last doc update July 15, 2026). The `mlflow.langchain.autolog()` call and `mlflow.set_experiment()` remain the primary entry points but behavior may differ between MLflow 2 and MLflow 3.

**What is it?** MLflow Tracing is an end-to-end observability system that captures inputs, outputs, intermediate steps, token usage, and latency for every invocation of a GenAI application. For LangGraph agents, it records the entire graph execution as a hierarchical tree of spans.

**How does it work mechanistically?** When `mlflow.langchain.autolog()` is called, MLflow monkey-patches the LangChain/LangGraph execution internals to emit spans automatically. Each node execution, each LLM call, and each tool invocation becomes a child span of the parent agent invocation span. Spans carry attributes including `llm.token_count.prompt`, `llm.token_count.completion`, latency in milliseconds, and status (`OK`, `ERROR`). The spans are assembled into a trace and flushed to the configured storage backend — either the MLflow experiment artifact store (capped at 100,000 traces) or Unity Catalog Delta tables (recommended for production, no cap, SQL-queryable).

**Where does it appear in Databricks?** Call `mlflow.set_experiment("/path/to/experiment")` and `mlflow.langchain.autolog()` before graph invocation. Traces appear in the MLflow UI under the experiment's **Traces** tab. When stored in Unity Catalog, they are queryable via `mlflow.search_traces()` in Python or directly via SQL on the OTel Delta tables.

### Agent-Specific KPIs

**What is it?** Agent-specific KPIs are metrics derived from trace data that characterize the health and efficiency of a running agent. Unlike simple HTTP-level metrics, they reflect the multi-step, tool-calling nature of agents.

**How does it work mechanistically?** KPIs are computed from spans in the trace: total request latency is the root span duration; per-node latency is each child span duration; LLM calls per invocation is the count of LLM-type spans; tool call success rate is the fraction of tool spans with status `OK`; loop detection fires when the count of a particular node span exceeds a configured `max_iterations` threshold (LangGraph raises `GraphRecursionError` at compile-time configurable recursion limits). These metrics can be extracted post-hoc by querying MLflow traces or computed inline using custom scorers registered for production monitoring.

**Where does it appear in Databricks?** `mlflow.search_traces()` returns trace objects with span attributes. The production monitoring service (MLflow 3, Beta) supports custom `@scorer` functions that compute numeric or categorical KPIs over trace data and attach them as feedback. The MLflow experiment UI displays trace duration and error status for every invocation.

### Debugging Agent Failures from Traces

**What is it?** Trace-based debugging is the practice of reading the span hierarchy of a failed agent invocation to identify which node, LLM call, or tool was the proximate cause of failure — rather than guessing from logs or output text.

**How does it work mechanistically?** A failed invocation has its root span status set to `ERROR`. Drilling into child spans reveals the first `ERROR` span, which indicates the failure origin. The span's `events` attribute contains the exception type and message. If a tool call returned malformed output, the tool span will be `OK` but the subsequent LLM call span may carry a parsing error or the next node span may show unexpected output attributes. Loop failures surface as a large number of identical node spans before the `GraphRecursionError` event.

**Where does it appear in Databricks?** In the MLflow UI Traces tab, each trace renders as an interactive span tree. Click on a span to see its input, output, attributes, and events. For automated failure alerting, run production monitoring scorers on traces filtered to `attributes.status = 'ERROR'` or register a custom code scorer that checks for specific failure signatures.

---

## Key Parameters / Configuration Knobs

| Parameter | What it controls | Decision rule |
|---|---|---|
| `max_tokens` (inference endpoint request) | Maximum completion tokens returned per LLM call | Set to the smallest value consistent with valid responses for that node; do not leave as `None` in production — unset `max_tokens` allows the model to fill the full context window |
| `QPM` (Unity AI Gateway, service level) | Maximum queries per minute across all users of a model service | Set to the throughput your serving endpoint can handle without latency degradation; start at your p95 load and enforce with a 20% headroom buffer |
| `QPM` / `TPM` (Unity AI Gateway, per-user) | Maximum queries or tokens per minute for an individual user or group | Set per-user limits at 10–20% of the service limit for exploratory users; grant higher limits to production workloads via service-principal-specific rules |
| `sample_rate` (`ScorerSamplingConfig`) | Fraction of production traces evaluated by a monitoring scorer | Use `1.0` for safety/security scorers; use `0.05–0.2` for expensive LLM judges; balance coverage vs compute cost |
| `recursion_limit` (LangGraph `CompiledGraph`) | Maximum number of node visits before `GraphRecursionError` | Set to `2 × expected_steps + buffer`; a limit of 25 is a safe default for most ReAct-style agents; lower it during debugging to surface loops faster |
| `filter_string` (`ScorerSamplingConfig`) | Which traces a production scorer processes | Always add `attributes.status = 'OK'` for quality scorers (exclude errored traces from quality measurement) and remove it only for error-analysis scorers |
| Shared budget threshold (Unity AI Gateway budget) | Monthly dollar ceiling across all users | Set a `send alert` threshold at 80% of budget and a hard cap only where blocking is acceptable; for Genie workloads enable `block_usage` at 100% |

---

## Worked Example: Requirement → Decision

**Given:** A team deploys a LangGraph customer-support agent that makes 2–4 LLM calls per invocation plus one tool call to a retrieval system. After two weeks in production, the team notices the monthly LLM spend is 3× the forecast. They have no visibility into which users or which agent invocations are responsible.

**Step 1 — Identify the goal:** Attribute the excess token spend to specific users or agent paths, then enforce limits to prevent recurrence.

**Step 2 — Define inputs:**
- LangGraph agent code (deployed to Databricks Model Serving)
- MLflow experiment where the agent logs runs
- `system.billing.usage` showing `UNITY_AI_GATEWAY` product records
- Unity AI Gateway model service configuration

**Step 3 — Define outputs:**
- A cost attribution query showing token spend broken down by `principal_id` (user) and `service_name`
- A set of MLflow traces showing which invocations consumed the most tokens, and which nodes within those invocations
- Rate limit and budget configuration that caps per-user spend going forward

**Step 4 — Apply constraints:**
- The agent must remain available to all users — hard blocking is not acceptable
- Autologging was not enabled at deploy time, so traces are sparse; historical data is limited
- Unity AI Gateway rate limits are in Beta and the team is on AWS

**Step 5 — Select the approach:**

1. Enable `mlflow.langchain.autolog()` and bind the MLflow experiment to a Unity Catalog trace location — this gives forward-looking trace coverage and makes traces SQL-queryable without trace count limits
2. Query `system.billing.usage` filtering on `product = 'UNITY_AI_GATEWAY'` and group by `identity_metadata.run_as` to find top-spending principals
3. Use `mlflow.search_traces()` with `filter_string="attributes.total_tokens > 5000"` to surface the most expensive agent invocations; drill into span trees to identify the runaway node
4. Set a service-level QPM limit (protecting the endpoint) and a per-user TPM default limit (protecting per-user spend); set a spend budget with an 80% alert threshold and a 100% alert (not block) at the service level

*Rationale vs alternatives:* Simply lowering `max_tokens` globally would degrade response quality for all users and would not identify the root cause. Trace analysis first, then targeted enforcement, is the correct order — limits without observability are guesses.

---

## Implementation

```python
# Scenario: Enable MLflow autologging for a LangGraph agent so every invocation
# produces a queryable trace in Unity Catalog — required before any
# token-spend or latency analysis is possible.

import mlflow
import mlflow.langchain
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

# Point the experiment at a workspace experiment (Unity Catalog trace location
# is configured separately in the MLflow experiment settings).
mlflow.set_experiment("/Shared/agents/customer-support-prod")

# Enable automatic tracing for all LangChain/LangGraph calls in this process.
# This monkey-patches graph execution to emit spans automatically.
mlflow.langchain.autolog()


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    total_tokens: int        # accumulate tokens to enforce a per-invocation budget
    tool_calls_made: int     # track tool call count for KPI reporting


def call_llm(state: AgentState) -> AgentState:
    """LLM node — enforces a per-invocation token budget before calling."""
    # Guard: refuse to call LLM if cumulative tokens already exceed budget
    TOKEN_BUDGET = 4000
    if state["total_tokens"] >= TOKEN_BUDGET:
        # Surface as a structured message rather than silently truncating
        return {
            "messages": [{"role": "assistant", "content": "[BUDGET_EXCEEDED]"}],
            "total_tokens": state["total_tokens"],
            "tool_calls_made": state["tool_calls_made"],
        }

    from langchain_community.chat_models import ChatDatabricks
    llm = ChatDatabricks(
        endpoint="databricks-meta-llama-3-3-70b-instruct",
        max_tokens=512,   # always set — never leave as None in production
        temperature=0.0,
    )
    response = llm.invoke(state["messages"])
    # Read token usage from the response metadata (populated by Databricks serving)
    tokens_used = response.response_metadata.get("token_usage", {})
    total = tokens_used.get("total_tokens", 0)

    return {
        "messages": [response],
        "total_tokens": state["total_tokens"] + total,
        "tool_calls_made": state["tool_calls_made"],
    }


def call_tool(state: AgentState) -> AgentState:
    """Tool node — executes retrieval and tracks call count for KPI."""
    # ... tool execution logic ...
    return {
        "messages": state["messages"],
        "total_tokens": state["total_tokens"],
        "tool_calls_made": state["tool_calls_made"] + 1,
    }


def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if "[BUDGET_EXCEEDED]" in str(last_message):
        return END
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool"
    return END


graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("tool", call_tool)
graph.set_entry_point("llm")
graph.add_conditional_edges("llm", should_continue, {"tool": "tool", END: END})
graph.add_edge("tool", "llm")

# Set recursion_limit to cap loop depth — prevents GraphRecursionError from
# appearing only after hundreds of silent LLM calls.
app = graph.compile()
app.config["recursion_limit"] = 20

# Each invocation produces a trace visible in the MLflow UI and queryable via
# mlflow.search_traces() or SQL on the Unity Catalog OTel Delta tables.
result = app.invoke({
    "messages": [{"role": "user", "content": "How do I reset my password?"}],
    "total_tokens": 0,
    "tool_calls_made": 0,
})
```

```python
# Scenario: Query MLflow traces to build a cost attribution report showing
# which agent invocations consumed the most tokens — the first step in
# understanding a cost overrun before configuring rate limits.

import mlflow
from mlflow.entities import SpanType
import pandas as pd

mlflow.set_experiment("/Shared/agents/customer-support-prod")

# Retrieve the 1000 most recent traces that completed successfully
traces = mlflow.search_traces(
    experiment_names=["/Shared/agents/customer-support-prod"],
    filter_string="attributes.status = 'OK'",
    max_results=1000,
    order_by=["attributes.timestamp_ms DESC"],
)

# Build a cost summary from trace-level attributes
rows = []
for trace in traces:
    root = trace.info
    # Sum prompt + completion tokens across all LLM spans in this trace
    llm_spans = [
        s for s in trace.data.spans
        if s.span_type == SpanType.LLM
    ]
    prompt_tokens = sum(
        s.attributes.get("llm.token_count.prompt", 0) for s in llm_spans
    )
    completion_tokens = sum(
        s.attributes.get("llm.token_count.completion", 0) for s in llm_spans
    )
    tool_spans = [
        s for s in trace.data.spans
        if s.span_type == SpanType.TOOL
    ]
    tool_success_rate = (
        sum(1 for s in tool_spans if s.status.status_code == "OK") / len(tool_spans)
        if tool_spans else None
    )
    rows.append({
        "trace_id": root.request_id,
        "timestamp": root.timestamp_ms,
        "latency_ms": root.execution_time_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "llm_call_count": len(llm_spans),
        "tool_call_count": len(tool_spans),
        "tool_success_rate": tool_success_rate,
    })

df = pd.DataFrame(rows)
print(df.sort_values("total_tokens", ascending=False).head(20))
# Top spenders by token count surface which invocations to inspect for
# runaway loops or unexpectedly large prompts.
```

```python
# Anti-pattern: storing traces only in the MLflow experiment artifact store
# with no Unity Catalog binding, then trying to query cost data at scale.

# BAD — this caps you at 100,000 traces, offers no SQL access, and means
# your cost data disappears when the experiment is deleted.
mlflow.set_experiment("/Users/me/my_agent_experiment")
mlflow.langchain.autolog()
# traces go to experiment artifact store only — no SQL, no governance, no Delta

# Correct approach: bind the experiment to a Unity Catalog trace location
# so traces land in OTel Delta tables with no storage cap and full SQL access.
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
experiment = client.get_experiment_by_name("/Shared/agents/customer-support-prod")

# In Databricks, set the Unity Catalog trace location from the experiment UI
# (Experiments > Edit > Trace storage > Unity Catalog) or via the API.
# Once set, mlflow.search_traces() and direct SQL queries both work:
#
#   SELECT trace_id, duration_ms, SUM(token_count) as total_tokens
#   FROM main.agents.traces_otel  -- the UC Delta table bound to this experiment
#   GROUP BY trace_id, duration_ms
#   ORDER BY total_tokens DESC
#
# What breaks with the anti-pattern: at 100,001 traces the oldest are silently
# evicted; you cannot JOIN trace data with system.billing.usage; Unity Catalog
# access controls do not apply so any workspace user can see all traces.
mlflow.set_experiment("/Shared/agents/customer-support-prod")
mlflow.langchain.autolog()
```

---

## Common Pitfalls & Misconceptions

- **Setting `max_tokens` and calling it "cost control"** — Beginners conflate the per-call token cap with a comprehensive cost strategy because `max_tokens` is the most visible knob in the API docs. The correct mental model: `max_tokens` limits one LLM call's output; it does nothing to prevent the agent from making 50 LLM calls in a loop, each under the cap, resulting in total spend 50× the per-call budget.

- **Assuming rate limits prevent all cost overruns** — New engineers often treat QPM/TPM limits as budget controls because they are configured in the same UI. Rate limits protect capacity and enforce fairness; they do not cap monthly dollar spend. A single user can stay within their per-minute limit while issuing thousands of requests per day. Use budget thresholds (Unity AI Gateway budgets) for dollar-denominated controls.

- **Not enabling autologging before deployment** — Teams frequently add `mlflow.langchain.autolog()` only during development and forget to include it in the production serving container. Without it, production traces are absent and cost forensics are impossible. The correct mental model: autologging must be called in the same Python process that executes the agent — including the model serving endpoint's model loading code.

- **Storing traces in the experiment artifact store at scale** — Beginners reach for the simplest `mlflow.set_experiment()` call without reading the storage implications. The 100,000-trace cap means a moderately busy production agent (e.g. 10 req/min) will fill the store in under a week, silently evicting the oldest traces — the ones most needed for forensics. Use Unity Catalog trace storage for any production workload.

- **Treating a 429 response as a permanent failure** — When an agent receives HTTP 429 from the model serving endpoint, developers new to distributed systems sometimes abort the invocation and report an error to the user. Rate limit errors are transient; the correct mental model is to implement exponential backoff with jitter and retry — the rate limiter windows reset on the per-minute boundary.

- **Reading total latency instead of per-span latency** — Beginners optimize the wrong thing: total agent latency is dominated by LLM call time, but the fixable cost driver is often a slow or repeated tool call. The correct model: always decompose latency at the span level before drawing optimization conclusions.

---

## Key Definitions

| Term | Definition |
|---|---|
| **Span** | A single timed unit of work within a trace — e.g. one LLM call, one tool invocation, or one graph node execution. Spans are nested into a hierarchy within a trace. |
| **Trace** | The complete record of a single agent invocation: the root span plus all child spans, with their inputs, outputs, attributes, and timing. |
| **QPM (Queries Per Minute)** | A rate limit dimension measuring the number of requests per minute, irrespective of token size. |
| **TPM (Tokens Per Minute)** | A rate limit dimension measuring cumulative tokens (prompt + completion) consumed per minute. |
| **Inference table** | A Unity Catalog Delta table, enabled via AI Gateway, that logs every request body and response body for a model serving endpoint. Used for payload auditing and cost attribution. |
| **Spend budget** | A Unity AI Gateway object that sets monthly dollar-denominated thresholds on AI Gateway usage, triggering alert emails or (for Genie) usage blocking when crossed. |
| **Production monitoring** | An MLflow 3 Beta feature that continuously evaluates a sample of production traces against registered scorers, attaching quality feedback to each evaluated trace. |
| **`GraphRecursionError`** | A LangGraph exception raised when a compiled graph's `recursion_limit` is exceeded — used as the primary loop-detection mechanism. |
| **`ScorerSamplingConfig`** | MLflow dataclass specifying `sample_rate` (fraction of traces to evaluate) and optional `filter_string` for a production monitoring scorer. |
| **Token budget** | An explicit ceiling on cumulative token spend, enforced either server-side (`max_tokens`) or application-side (accumulated token count check in agent state). |

---

## Summary / Quick Recall

- Unity AI Gateway enforces QPM/TPM rate limits at service, user, and group levels — when the service limit is hit, all users get 429; per-user limits only protect fairness within that ceiling.
- Spend budgets in Unity AI Gateway are monthly dollar limits with alert and (for Genie) blocking actions; they track `UNITY_AI_GATEWAY` product rows in `system.billing.usage`.
- `mlflow.langchain.autolog()` must be called in the production serving process — not just in notebooks — to capture production traces.
- Unity Catalog trace storage is always preferred over experiment artifact storage for production: no 100K cap, SQL-queryable, governed by Unity Catalog permissions.
- Agent KPIs derived from traces: total latency (root span), per-node latency (child spans), LLM call count (LLM-type span count), tool success rate (fraction of TOOL spans with status OK).
- Loop detection in LangGraph uses `recursion_limit` on the compiled graph; a `GraphRecursionError` in the trace signals runaway iteration.
- Token budget + rate limit + trace observability = the three-legged stool; removing any one leg makes production cost control fragile.

---

## Self-Check Questions

1. What happens when a Unity AI Gateway service-level QPM limit is exceeded, compared to when a per-user QPM limit is exceeded?

   <details><summary>Answer</summary>

   **Correct answer:** When the service-level QPM limit is exceeded, *all users* of the service receive an HTTP 429 response — the entire service is throttled. When a per-user QPM limit is exceeded, only that specific user receives 429 responses; other users continue to be served normally.

   **Why the distractor is wrong:** A common wrong answer is that both limits simply queue requests. Rate limits in Unity AI Gateway do not queue — they reject with 429 and expect clients to implement retry with backoff. Another common error is conflating QPM (request count) with TPM (token count): exceeding a QPM limit stops requests; exceeding a TPM limit stops requests that would consume more tokens than remaining capacity.

   </details>

2. A LangGraph agent in production is taking 45 seconds per invocation but individual LLM API calls return in under 3 seconds each. Using MLflow trace data, how would you identify the root cause?

   <details><summary>Answer</summary>

   **Correct approach:** Retrieve the trace for a slow invocation and examine the span hierarchy. If total latency is 45 seconds but individual LLM spans are each under 3 seconds, the slow path is either (a) a tool call span — inspect TOOL spans for high duration, which would indicate the retrieval system is the bottleneck, or (b) a large number of LLM spans indicating a loop — count LLM-type spans; if the agent is calling the LLM 15+ times when 3–4 is expected, `recursion_limit` was not preventing runaway iteration.

   **Why simply lowering `max_tokens` is wrong:** That would reduce per-call token spend and might shorten completions, but it would not reduce *latency* caused by a slow tool call or eliminate extra LLM calls from a loop. The trace span hierarchy is the correct diagnostic tool.

   </details>

3. **Which TWO** of the following storage configurations for MLflow traces are appropriate for a production LangGraph agent handling 500+ requests per day? Select all that apply.

   - A. Store traces in the default MLflow experiment artifact store with no additional configuration
   - B. Bind the MLflow experiment to a Unity Catalog trace location so traces land in OTel Delta tables
   - C. Enable inference tables on the model serving endpoint via AI Gateway for payload logging
   - D. Write trace data manually to a CSV file after each invocation
   - E. Use `mlflow.log_artifact()` to save trace JSON files to DBFS

   <details><summary>Answer</summary>

   **Correct answers: B and C.**

   **B** is correct because Unity Catalog trace storage has no trace count cap, is SQL-queryable, and is governed by Unity Catalog access controls — all required properties for a production workload at 500+ requests/day. The default experiment artifact store (A) caps at 100,000 traces and would be filled in about 200 days at this rate, with no way to extend it — worse, older traces are evicted silently.

   **C** is correct because inference tables log request/response payloads including token counts to a Unity Catalog Delta table, enabling cost attribution queries and payload auditing independently of MLflow tracing.

   **D** is wrong because CSV files do not scale, are not queryable with SQL, provide no governance, and are not integrated with MLflow's trace search APIs.

   **E** is wrong because `mlflow.log_artifact()` to DBFS is a file-level operation with no structured query support, no integration with `mlflow.search_traces()`, and no Unity Catalog governance.

   </details>

4. A team wants to enforce that no individual user spends more than $50/month on LLM inference through Unity AI Gateway, but they do not want to block any user's access. How should they configure cost controls, and what is the limitation of this approach?

   <details><summary>Answer</summary>

   **Correct configuration:** Create a Unity AI Gateway budget scoped to the `UNITY_AI_GATEWAY` product, set a per-user threshold of $50 with the action `send alert`. This triggers an email notification when any individual user approaches the limit.

   **Limitation:** Per-user `block_usage` enforcement is currently available only for Genie budgets, not general Unity AI Gateway budgets. Therefore, the $50 threshold can only *alert* — it cannot automatically stop a user from exceeding it. To enforce a hard per-user dollar cap for general AI Gateway services, the team must combine the alert with a separate per-user TPM rate limit in the rate limit configuration to slow token consumption rate, and monitor the alert emails for manual intervention.

   **Why "just set a per-user TPM rate limit" alone is wrong:** A TPM rate limit caps the rate of token consumption per minute, not total monthly spend. A user within their per-minute limit can still accumulate large monthly spend over thousands of requests.

   </details>

5. Your LangGraph agent occasionally produces an error trace where the root span shows `status = ERROR` but the LLM spans all show `status = OK`. Which agent component is the most likely failure point, and how does the trace hierarchy confirm this?

   <details><summary>Answer</summary>

   **Correct answer:** The most likely failure point is a tool call node. If all LLM spans are `OK` but the root trace is `ERROR`, a non-LLM span is raising an exception. Inspect the TOOL-type spans — a TOOL span with `status = ERROR` and an exception message in its `events` attribute pinpoints the exact tool invocation that failed. The span's `output` attribute may show the raw exception or a malformed tool response that the subsequent LLM span could not parse.

   **Why "the LLM returned a malformed response" is wrong:** If the LLM returned malformed output, the LLM span would show `status = OK` (the API call itself succeeded), but the *subsequent node* that tried to parse the output would show `ERROR`. This is still not an LLM span failure — it is a parsing/node failure. Reading the error span's position in the hierarchy (which child of which parent) precisely identifies the failure location.

   **Why loop detection (GraphRecursionError) is wrong here:** A `GraphRecursionError` would surface as a large number of repeated spans before the error, not as `OK` LLM spans with a single isolated root `ERROR`.

   </details>

---

## Further Reading

- [MLflow Tracing — GenAI observability](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/) — *verified 2026-07-16* — Overview of MLflow tracing storage options, Unity Catalog vs experiment backends
- [Add traces to applications: automatic and manual tracing](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/app-instrumentation/) — *verified 2026-07-16* — When to use `autolog()` vs manual decorator tracing
- [Trace agents deployed on Databricks](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/prod-tracing) — *verified 2026-07-16* — Production deployment pattern for MLflow tracing with Custom Agents and CPU serving
- [AI governance with Unity AI Gateway](https://docs.databricks.com/aws/en/ai-gateway/) — *verified 2026-07-16* — Rate limits, budgets, traffic routing, and inference tables in Unity AI Gateway
- [Configure rate limits for AI services using Unity AI Gateway](https://docs.databricks.com/aws/en/ai-gateway/rate-limits) — *verified 2026-07-16* — QPM/TPM limit configuration, priority hierarchy, and 429 behavior
- [Manage budgets for Unity AI Gateway](https://docs.databricks.com/aws/en/ai-gateway/budgets) — *verified 2026-07-16* — Shared and per-user budget thresholds, limitations on current Beta
- [Monitor GenAI apps in production](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/production-monitoring) — *verified 2026-07-16* — Production monitoring scorers, `ScorerSamplingConfig`, and sampling strategy
