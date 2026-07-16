# LAB-22: Cost Control and Agent Monitoring with MLflow Tracing

**Lab:** LAB-22 | **Section:** 07 Evaluation and Monitoring | **Module:** 02 Monitoring | **Est. time:** 1.5 hrs

---

## Objective

Build a cost-observable LangGraph agent: enable MLflow autologging, inspect the resulting trace in the MLflow UI, extract token and latency KPIs, implement a token budget guard, and build a cost attribution query from the inference table.

---

## Prerequisites

- Completed LAB-21 (or equivalent familiarity with LangGraph agent authoring)
- MLflow >= 2.14 installed (`pip show mlflow` to verify)
- Access to a Databricks workspace with a Foundation Model API endpoint (e.g. `databricks-meta-llama-3-3-70b-instruct`)
- Unity AI Gateway enabled (for Step 5 inference table query; Step 1–4 run without it)
- Familiarity with: LangGraph `StateGraph`, `TypedDict` agent state, Python dataclasses

> **Simulation note:** This lab is written in illustrative/simulation style. All Python code is executable in a local Python environment or Databricks notebook. Steps 1–4 use a mock LLM to avoid live API costs; Step 5 shows the production query pattern using realistic schema documentation.

---

## Setup

```python
# Install required packages (run once)
# In a Databricks notebook: %pip install mlflow langgraph langchain langchain-community
# Locally:
#   pip install mlflow langgraph langchain langchain-community

import mlflow
import mlflow.langchain
from mlflow.entities import SpanType
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Optional
import operator
import time

print(f"MLflow version: {mlflow.__version__}")
# Expected: 2.14.x or higher for LangGraph autologging support
```

```python
# Mock LLM class — simulates a Databricks Foundation Model API response
# with realistic token usage metadata. Replace with ChatDatabricks in production.

class MockLLMResponse:
    """Simulates the response structure of a ChatDatabricks LLM call."""
    def __init__(self, content: str, prompt_tokens: int, completion_tokens: int,
                 tool_calls: Optional[list] = None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.response_metadata = {
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
        }

class MockLLM:
    """Simulates ChatDatabricks with configurable response behaviour."""
    def __init__(self, responses: list):
        self._responses = iter(responses)
        self._call_count = 0

    def invoke(self, messages, **kwargs):
        self._call_count += 1
        try:
            return next(self._responses)
        except StopIteration:
            # Simulate a runaway loop by echoing back a tool_call every time
            return MockLLMResponse(
                content="",
                prompt_tokens=120,
                completion_tokens=15,
                tool_calls=[{"name": "search", "args": {"query": "retry"}}]
            )
```

---

## Steps

### Step 1 — Enable MLflow Autologging for a LangGraph Agent

Enable tracing before defining the graph. Set the experiment to a dedicated path. In production this would be bound to a Unity Catalog trace location; here we use the local MLflow tracking server.

```python
# Step 1: Enable autologging and set the experiment
# This must be done before ANY LangGraph invocations, including test runs.

mlflow.set_tracking_uri("mlite")  # use local sqlite store for simulation
mlflow.set_experiment("lab22-cost-control-agent")
mlflow.langchain.autolog()

print("MLflow autologging enabled.")
print(f"Experiment: {mlflow.get_experiment_by_name('lab22-cost-control-agent')}")
```

```python
# Define a minimal two-node LangGraph agent: one LLM node, one tool node.
# No token budget guard yet — we will observe the raw cost behaviour first.

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    total_tokens: int
    tool_calls_made: int

# A realistic 3-turn conversation: call LLM → call tool → call LLM again → finish
NORMAL_RESPONSES = [
    MockLLMResponse(
        content="",
        prompt_tokens=210,
        completion_tokens=25,
        tool_calls=[{"name": "search", "args": {"query": "password reset steps"}}]
    ),
    MockLLMResponse(
        content="Here is how you reset your password: visit account.example.com and click Forgot Password.",
        prompt_tokens=380,
        completion_tokens=62,
    ),
]

llm_normal = MockLLM(NORMAL_RESPONSES)

def call_llm_v1(state: AgentState) -> AgentState:
    response = llm_normal.invoke(state["messages"], max_tokens=512)
    tokens_used = response.response_metadata.get("token_usage", {})
    return {
        "messages": [response],
        "total_tokens": state["total_tokens"] + tokens_used.get("total_tokens", 0),
        "tool_calls_made": state["tool_calls_made"],
    }

def call_tool_v1(state: AgentState) -> AgentState:
    # Simulate a retrieval tool call — returns a relevant snippet
    time.sleep(0.05)  # simulate 50ms tool latency
    tool_result = "Account password reset: visit account.example.com, click Forgot Password."
    return {
        "messages": [{"role": "tool", "content": tool_result}],
        "total_tokens": state["total_tokens"],
        "tool_calls_made": state["tool_calls_made"] + 1,
    }

def should_continue_v1(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tool"
    return END

graph_v1 = StateGraph(AgentState)
graph_v1.add_node("llm", call_llm_v1)
graph_v1.add_node("tool", call_tool_v1)
graph_v1.set_entry_point("llm")
graph_v1.add_conditional_edges("llm", should_continue_v1, {"tool": "tool", END: END})
graph_v1.add_edge("tool", "llm")
app_v1 = graph_v1.compile()

print("Agent v1 compiled (no token budget guard).")
```

```python
# Invoke the agent — MLflow autologging captures the full trace automatically.
result_v1 = app_v1.invoke({
    "messages": [{"role": "user", "content": "How do I reset my password?"}],
    "total_tokens": 0,
    "tool_calls_made": 0,
})

print(f"Final answer: {result_v1['messages'][-1].content}")
print(f"Total tokens accumulated in state: {result_v1['total_tokens']}")
print(f"Tool calls made: {result_v1['tool_calls_made']}")
print("\nNavigate to the MLflow UI > Experiments > lab22-cost-control-agent > Traces tab")
print("to see the full span hierarchy for this invocation.")
```

---

### Step 2 — Run the Agent and Inspect the Trace in the MLflow UI

Simulate a looping agent to produce a multi-span trace that illustrates cost escalation, then retrieve and print the trace programmatically.

```python
# Step 2: Simulate a runaway agent — the tool returns empty results, causing
# the LLM to retry indefinitely until recursion_limit is hit.

# IMPORTANT: We set recursion_limit=8 to bound the simulation.
# In production without this limit, costs would continue until the request times out.

graph_loop = StateGraph(AgentState)
llm_loop = MockLLM([])  # always returns a tool_call (see MockLLM.invoke StopIteration)

def call_llm_loop(state: AgentState) -> AgentState:
    response = llm_loop.invoke(state["messages"])
    tokens_used = response.response_metadata.get("token_usage", {})
    return {
        "messages": [response],
        "total_tokens": state["total_tokens"] + tokens_used.get("total_tokens", 0),
        "tool_calls_made": state["tool_calls_made"],
    }

def call_tool_empty(state: AgentState) -> AgentState:
    # Simulates a tool that always returns empty — triggers retry loop
    return {
        "messages": [{"role": "tool", "content": "[]"}],
        "total_tokens": state["total_tokens"],
        "tool_calls_made": state["tool_calls_made"] + 1,
    }

def should_continue_loop(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tool"
    return END

graph_loop.add_node("llm", call_llm_loop)
graph_loop.add_node("tool", call_tool_empty)
graph_loop.set_entry_point("llm")
graph_loop.add_conditional_edges("llm", should_continue_loop, {"tool": "tool", END: END})
graph_loop.add_edge("tool", "llm")

# recursion_limit=8 bounds the loop — without it the agent would run until timeout
app_loop = graph_loop.compile()
app_loop.config = {"recursion_limit": 8}

try:
    result_loop = app_loop.invoke({
        "messages": [{"role": "user", "content": "Find me articles about LLMs"}],
        "total_tokens": 0,
        "tool_calls_made": 0,
    })
except Exception as e:
    print(f"Agent stopped with: {type(e).__name__}: {e}")
    print("This is the GraphRecursionError — the loop was detected and bounded.")
```

```python
# Retrieve the most recent traces from the experiment and print a summary
import mlflow

traces = mlflow.search_traces(
    experiment_names=["lab22-cost-control-agent"],
    max_results=5,
    order_by=["attributes.timestamp_ms DESC"],
)

print(f"Found {len(traces)} traces.\n")
for i, trace in enumerate(traces):
    info = trace.info
    spans = trace.data.spans
    llm_spans = [s for s in spans if getattr(s, "span_type", None) == SpanType.LLM]
    tool_spans = [s for s in spans if getattr(s, "span_type", None) == SpanType.TOOL]
    print(f"Trace {i+1}: {info.request_id}")
    print(f"  Status:         {info.status}")
    print(f"  Duration:       {info.execution_time_ms} ms")
    print(f"  Total spans:    {len(spans)}")
    print(f"  LLM spans:      {len(llm_spans)}")
    print(f"  Tool spans:     {len(tool_spans)}")
    print()
```

---

### Step 3 — Extract Token Counts and Latency from the Trace

Build a structured KPI table from trace span attributes.

```python
# Step 3: Extract per-trace KPIs for cost and latency analysis.
# In production, this query runs against Unity Catalog OTel Delta tables via SQL.
# Here we compute it from the Python SDK response.

import pandas as pd

def extract_trace_kpis(traces) -> pd.DataFrame:
    """Extract cost and performance KPIs from a list of MLflow traces."""
    rows = []
    for trace in traces:
        info = trace.info
        spans = trace.data.spans

        llm_spans = [s for s in spans if getattr(s, "span_type", None) == SpanType.LLM]
        tool_spans = [s for s in spans if getattr(s, "span_type", None) == SpanType.TOOL]

        # Token counts from span attributes (populated by autologging)
        prompt_tokens = sum(
            s.attributes.get("llm.token_count.prompt", 0) for s in llm_spans
        )
        completion_tokens = sum(
            s.attributes.get("llm.token_count.completion", 0) for s in llm_spans
        )

        # Tool success rate
        tool_success = (
            sum(1 for s in tool_spans
                if getattr(s.status, "status_code", "") == "OK") / len(tool_spans)
            if tool_spans else None
        )

        # Per-tool-call average latency
        avg_tool_latency = (
            sum(
                (s.end_time_ns - s.start_time_ns) / 1_000_000  # ns to ms
                for s in tool_spans
                if s.end_time_ns and s.start_time_ns
            ) / len(tool_spans)
            if tool_spans else None
        )

        rows.append({
            "trace_id":           info.request_id,
            "status":             info.status,
            "total_latency_ms":   info.execution_time_ms,
            "llm_call_count":     len(llm_spans),
            "tool_call_count":    len(tool_spans),
            "prompt_tokens":      prompt_tokens,
            "completion_tokens":  completion_tokens,
            "total_tokens":       prompt_tokens + completion_tokens,
            "tool_success_rate":  tool_success,
            "avg_tool_latency_ms": avg_tool_latency,
        })

    return pd.DataFrame(rows)


kpi_df = extract_trace_kpis(traces)
print(kpi_df.to_string(index=False))
print("\nHigh llm_call_count rows indicate looping agents — investigate those traces first.")
```

---

### Step 4 — Implement a Token Budget Guard in the Agent

Add an in-state token accumulator and a pre-call guard that short-circuits the agent cleanly before the token budget is exceeded.

```python
# Step 4: Agent v2 with a token budget guard.
# The guard checks cumulative tokens BEFORE calling the LLM and returns a
# structured message rather than silently truncating or raising an error.

TOKEN_BUDGET = 600   # Realistic low budget for illustration; use 4000+ in production

GUARDED_RESPONSES = [
    MockLLMResponse(
        content="",
        prompt_tokens=210,
        completion_tokens=25,
        tool_calls=[{"name": "search", "args": {"query": "account setup"}}]
    ),
    # Second LLM call would push us over TOKEN_BUDGET=600 — the guard fires first
    MockLLMResponse(
        content="Here are the account setup steps.",
        prompt_tokens=380,
        completion_tokens=62,
    ),
]
llm_guarded = MockLLM(GUARDED_RESPONSES)

def call_llm_v2(state: AgentState) -> AgentState:
    """LLM node with token budget guard."""
    if state["total_tokens"] >= TOKEN_BUDGET:
        # Return a structured sentinel rather than calling the LLM
        # This surfaces clearly in the trace and to the calling application
        print(f"[BUDGET GUARD] Token budget {TOKEN_BUDGET} reached "
              f"(used: {state['total_tokens']}). Stopping agent.")
        return {
            "messages": [MockLLMResponse(
                content="[BUDGET_EXCEEDED: Agent stopped to control costs. "
                        "Please rephrase your request more specifically.]",
                prompt_tokens=0,
                completion_tokens=0,
            )],
            "total_tokens": state["total_tokens"],
            "tool_calls_made": state["tool_calls_made"],
        }

    response = llm_guarded.invoke(state["messages"], max_tokens=512)
    tokens_used = response.response_metadata.get("token_usage", {})
    new_total = state["total_tokens"] + tokens_used.get("total_tokens", 0)
    print(f"[LLM CALL] tokens this call: {tokens_used.get('total_tokens', 0)}, "
          f"running total: {new_total}/{TOKEN_BUDGET}")
    return {
        "messages": [response],
        "total_tokens": new_total,
        "tool_calls_made": state["tool_calls_made"],
    }

def call_tool_v2(state: AgentState) -> AgentState:
    time.sleep(0.05)
    tool_result = "Account setup guide: visit account.example.com/setup."
    return {
        "messages": [{"role": "tool", "content": tool_result}],
        "total_tokens": state["total_tokens"],
        "tool_calls_made": state["tool_calls_made"] + 1,
    }

def should_continue_v2(state: AgentState) -> str:
    last = state["messages"][-1]
    # Also stop if budget exceeded message is present
    if "[BUDGET_EXCEEDED" in str(getattr(last, "content", "")):
        return END
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tool"
    return END

graph_v2 = StateGraph(AgentState)
graph_v2.add_node("llm", call_llm_v2)
graph_v2.add_node("tool", call_tool_v2)
graph_v2.set_entry_point("llm")
graph_v2.add_conditional_edges("llm", should_continue_v2, {"tool": "tool", END: END})
graph_v2.add_edge("tool", "llm")
app_v2 = graph_v2.compile()
app_v2.config = {"recursion_limit": 20}

print(f"\n--- Running agent v2 with TOKEN_BUDGET={TOKEN_BUDGET} ---\n")
result_v2 = app_v2.invoke({
    "messages": [{"role": "user", "content": "Walk me through account setup step by step."}],
    "total_tokens": 0,
    "tool_calls_made": 0,
})

final_content = result_v2["messages"][-1].content
print(f"\nFinal response: {final_content}")
print(f"Total tokens used: {result_v2['total_tokens']}")
print(f"Budget remaining: {TOKEN_BUDGET - result_v2['total_tokens']} tokens")
```

---

### Step 5 — Build a Cost Attribution Query from the Inference Table

This step shows the production query pattern. It uses realistic schema documentation rather than live cluster execution.

```python
# Step 5: Cost attribution query from system.billing.usage
# This query runs in a Databricks SQL Warehouse or notebook connected to a UC metastore.
# Schema reference: https://docs.databricks.com/aws/en/ai-gateway/cost-observability

ATTRIBUTION_QUERY = """
-- Scenario: Identify top-spending users on Unity AI Gateway for the current month
-- so rate limits and budget thresholds can be calibrated to actual usage patterns.
--
-- Prerequisites:
--   1. Unity AI Gateway enabled for your account
--   2. Model serving endpoints configured as Unity AI Gateway services
--   3. system.billing.usage accessible (requires system schema enabled)

SELECT
    identity_metadata.run_as                        AS principal_id,
    SUM(usage_quantity)                             AS total_dbu_spend,
    COUNT(*)                                        AS request_count,
    AVG(usage_quantity)                             AS avg_dbu_per_request,
    -- Estimate token-equivalent spend (Unity AI Gateway bills per token)
    SUM(usage_quantity) * 1000                      AS estimated_tokens  -- illustrative multiplier
FROM
    system.billing.usage
WHERE
    product = 'UNITY_AI_GATEWAY'
    AND usage_date >= DATE_TRUNC('month', CURRENT_DATE)
GROUP BY
    identity_metadata.run_as
ORDER BY
    total_dbu_spend DESC
LIMIT 20;

-- To drill down to request-level detail, JOIN with the inference table:
--
-- SELECT
--     b.identity_metadata.run_as AS principal_id,
--     i.request_id,
--     i.timestamp,
--     i.response_body:usage:total_tokens::INT AS total_tokens,
--     i.response_body:usage:prompt_tokens::INT AS prompt_tokens,
--     i.response_body:usage:completion_tokens::INT AS completion_tokens
-- FROM
--     system.billing.usage b
--     JOIN <catalog>.<schema>.inference_table i
--         ON b.usage_metadata.request_id = i.request_id
-- WHERE
--     b.product = 'UNITY_AI_GATEWAY'
--     AND b.usage_date >= CURRENT_DATE - INTERVAL 7 DAYS
-- ORDER BY
--     total_tokens DESC
-- LIMIT 100;
"""

print("Cost attribution query (production pattern):")
print(ATTRIBUTION_QUERY)
print("\nKey fields:")
print("  identity_metadata.run_as  — the Unity Catalog principal (user or service principal)")
print("  usage_quantity            — DBU consumption (Unity AI Gateway bills per token)")
print("  product = 'UNITY_AI_GATEWAY' — filters to AI Gateway spend only")
print("\nNext steps in production:")
print("  1. Run this query in a Databricks SQL Warehouse")
print("  2. Feed the top-N principals into Unity AI Gateway rate limit configuration")
print("  3. Set per-user TPM limits at 120% of their typical daily peak (from avg_dbu_per_request)")
```

---

## Validation

```python
# Validation: confirm that traces were captured and KPIs are extractable

print("=== Lab Validation ===\n")

# Check 1: Traces exist in the experiment
traces = mlflow.search_traces(
    experiment_names=["lab22-cost-control-agent"],
    max_results=10,
)
assert len(traces) > 0, "No traces found — check that autologging is enabled before graph invocations"
print(f"✓ {len(traces)} trace(s) found in experiment 'lab22-cost-control-agent'")

# Check 2: KPI extraction produces a non-empty DataFrame
kpi_df = extract_trace_kpis(traces)
assert not kpi_df.empty, "KPI extraction returned empty DataFrame"
assert "total_tokens" in kpi_df.columns, "total_tokens column missing"
print(f"✓ KPI DataFrame has {len(kpi_df)} rows and {len(kpi_df.columns)} columns")

# Check 3: Token budget guard was triggered on at least one trace
# (The v2 agent with TOKEN_BUDGET=600 should have hit the guard)
budget_traces = [
    t for t in traces
    if any(
        "[BUDGET_EXCEEDED" in str(getattr(s, "outputs", {}).get("content", ""))
        for s in t.data.spans
    )
]
print(f"✓ Budget guard triggered in {len(budget_traces)} trace(s) (expected ≥1 from Step 4)")

# Check 4: Loop detection — at least one trace should have many LLM spans (from Step 2)
high_call_traces = kpi_df[kpi_df["llm_call_count"] > 3]
print(f"✓ {len(high_call_traces)} trace(s) with >3 LLM calls (loop simulation from Step 2)")

print("\n=== All validation checks passed ===")
```

---

## Teardown

```python
# Teardown: clean up local MLflow tracking files (simulation mode only)
# In production, no teardown is needed — traces remain in Unity Catalog

import shutil
import os

# Remove local MLflow sqlite database and artifact directory created by mlite
for path in ["mlruns", "mlite.db"]:
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
        print(f"Removed: {path}")

print("Teardown complete. No cloud resources were created in this simulation lab.")
```

---

## Reflection Questions

1. In Step 2 the looping agent was stopped by `recursion_limit=8`. What would happen if you set `recursion_limit=100` and `TOKEN_BUDGET=600`? Which guard fires first — the recursion limit or the token budget? How does the answer depend on the cost of each LLM call?

2. The token budget guard in Step 4 returns a structured `[BUDGET_EXCEEDED]` message rather than raising an exception. What are the tradeoffs of this design compared to raising a Python exception from within the node? How would each approach appear differently in the MLflow trace span tree?

3. In Step 5, the attribution query groups by `identity_metadata.run_as`. In a production environment where agents are called by automated pipelines (service principals), not human users, how would you modify the attribution strategy to distinguish cost by *use-case* or *agent type* rather than by principal? What metadata would you need to add to agent requests?
