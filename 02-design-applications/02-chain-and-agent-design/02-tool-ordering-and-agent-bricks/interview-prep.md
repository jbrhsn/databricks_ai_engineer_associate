# Interview Prep: Tool Ordering and Agent Bricks

## Core Question: Explain Tool Ordering and Why It Matters in Production Agents

### What They're Asking

The interviewer wants to understand whether you grasp that agent performance is not just about the model or the tools—it's also about how information is presented to the model. They're testing:
1. Do you understand LLM cognition (attention, positional bias)?
2. Can you diagnose and fix a real performance problem without retraining?
3. Do you think systemically about agent observability and iteration?

### Your Answer (Structured)

**Start with a concrete example:**
"I worked on an agent that queries a data warehouse, searches vectors, and sends recommendations. The LLM was calling the wrong tool ~25% of the time. I discovered the tools were alphabetized: `describe_table`, `get_recommendation`, `list_tables`, … `query_warehouse`. The most-used tool was buried 4th.

I reordered by frequency: `query_warehouse` (60% of calls) first, `search_vectors` second (20%), etc. No code changes. Just reordering. Accuracy jumped to 95%, and token costs dropped 18%."

**Explain the mechanism:**
"This works because of positional bias in LLMs. Transformer attention allocates more compute to earlier tokens. When tools appear earlier in the list, the model reasons about them more deeply. Later tools get skimmed or ignored, especially in long lists. So if the right tool is buried, the LLM commits to a wrong tool early and doesn't reconsider."

**Connect to production:**
"In production, tool ordering is a prompt-engineering lever, not just a UI concern. Reordering is zero-cost and high-impact: 15–40% accuracy gains, 10–25% token savings, minimal deployment risk. It should be part of your agent tuning checklist, right after system prompts and model selection."

**Mention scaling:**
"For agents with 50+ tools, reordering alone isn't enough. You also need hierarchical tool selection: classify the user's intent, filter to 5–10 relevant tools, then order those by frequency. Bricks make this pattern easy to implement and test."

---

## Scenario Question: Design an Agent for a Multi-Domain Use Case

### The Scenario

*Your team is building an agent for Databricks customers. It needs to support:*
- *SQL queries (most common, ~60% of calls)*
- *Vector search (frequent, ~20% of calls)*
- *AI recommendations (15% of calls)*
- *Admin tasks (5%): schedule jobs, export data, manage permissions*
- *Discovery tasks (<5%): list tables, describe schema, audit logs*

*The team is considering two approaches:*
1. *Pass all 20+ tools to the LLM and let it decide.*
2. *Use a router brick to filter tools by intent, then order the filtered set.*

*Which approach would you recommend, and why?*

### Your Answer

**Recommend Approach 2 (router + filtered ordering).**

**Rationale:**

"Approach 1 (all tools to LLM) has three problems:

1. **Attention overload**: The LLM receives 20 tool descriptions. Positional bias means only the first 5–8 get serious reasoning. The rest are noise. You lose the value of having 20 tools; the model can't effectively consider them all.

2. **Token waste**: Each tool description takes ~50–100 tokens. 20 tools = 1000+ tokens just on tool definitions. That's 12–20% of a typical context window wasted on tools the user probably doesn't need.

3. **Safety blind spots**: Admin tools like `schedule_job` or `export_data` should not be available unless the user explicitly enables them. If they're in the global tool list, the LLM might call them unexpectedly, especially if they're early in the list and the user's intent is ambiguous.

**Approach 2 is better:**

1. **Intent routing**: The router brick classifies the user's query (query, search, recommend, admin, discovery). This is lightweight: a few-shot classification, ~100ms. It filters the tool list from 20 to 5–8 relevant tools.

2. **Safe tool availability**: Admin tools are only available if the router (and upstream guardrails) explicitly enable them. This is safer and more predictable.

3. **Optimized ordering**: The tool brick orders the 5–8 filtered tools by frequency. Now the LLM's positional bias works in your favor: the most-used tools are first, and there's less noise.

4. **Faster and cheaper**: Fewer tools per LLM call = fewer tokens = faster response time.

**Example workflow:**
- User: "What are the top products by revenue?"
- Router: Intent = "query". Enable tools: [query_warehouse, list_tables, describe_table].
- Tool Brick: Order by frequency: [query_warehouse (60%), list_tables (5%), describe_table (<1%)].
- LLM: Sees 3 focused tools, queries the warehouse immediately.
- Result: Fast, accurate, cheap.

**Trade-off:** The router adds ~100ms latency and requires tuning (few-shot examples). But the savings in token cost and accuracy improvement (15–25% typically) justify the cost."

---

## System Design: Build a Production-Ready Agent with Agent Bricks

### The Question

*Design a production-ready agent for Databricks customers that:*
- *Supports querying warehouses, searching vectors, and generating recommendations.*
- *Can be deployed to both LangGraph (for stateful, long-running workflows) and LangChain (for lightweight use cases).*
- *Is versioned, logged, and observable (governance requirement).*
- *Should be iterable without rewriting the core agent logic when requirements change.*

*What architecture would you use? How would you decompose it into bricks?*

### Your Answer (Architecture Sketch)

**High-level design:**

```
Orchestration Framework (LangGraph or LangChain adapter)
    │
    ├─► Router Brick
    │   ├─ Input: user query + context
    │   ├─ Logic: classify intent → select relevant tools
    │   └─ Output: selected tool set + routing signal
    │
    ├─► Memory Brick
    │   ├─ Input: agent state + new user message
    │   ├─ Logic: load prior context (if any) + store new interaction
    │   └─ Output: updated state with context
    │
    ├─► Tool Brick
    │   ├─ Input: tool set + routing signal
    │   ├─ Logic: order tools by frequency → present to LLM
    │   └─ Output: ordered tool list + execution handler
    │
    └─► Execution Brick
        ├─ Input: LLM's tool call
        ├─ Logic: invoke tool, catch errors, retry if needed
        └─ Output: tool result → fed back to LLM
```

**Key design decisions:**

1. **Brick independence**: Each brick has a clear input/output contract. Router doesn't care about memory; Memory doesn't care about Tool ordering. This enables isolated testing and safe changes.

2. **Framework agnosticism**: Bricks use a framework-neutral state schema (dict or dataclass). The adapter layer (LangGraph or LangChain) translates between brick state and framework state at deploy time. Same bricks, different adapters.

3. **Observability**: Each brick emits structured logs (input state, output state, execution time, errors). Databricks workspace dashboards consume these logs and display real-time agent behavior.

4. **Versioning**: Bricks are versioned in MLflow Model Registry. A "composition" is a set of brick versions + configuration. You can pin a composition to a specific git commit and reproduce it exactly 6 months later.

**Deployment flow:**

```
Brick Composition (defined once, framework-neutral)
    │
    ├─ LangGraph Adapter ──► Deploy on LangGraph (long-running, stateful)
    ├─ LangChain Adapter ──► Deploy on LangChain (lightweight, request/response)
    └─ MLflow Registry ───► Version and track all deployments
```

**Iteration loop:**

1. Metrics show Tool Brick's ordering strategy (currently frequency-based) has edge cases. New strategy: semantic relevance per query.
2. Update Tool Brick's code (one place). Bricks are recompiled via both adapters.
3. New LangGraph and LangChain agents inherit the fix automatically.
4. Run A/B test: old vs. new ordering strategy. Monitor for 1 week.
5. If new strategy wins, promote it. If not, roll back (one-line config change).

**Advantages over monolithic agent:**

- **Reuse**: The Router Brick is used by 5+ customer agents. Bug fix in Router → all 5 agents improve immediately.
- **Safety**: Each brick is tested independently. Integration tests verify brick composition. Reduced regression risk.
- **Maintainability**: A new engineer can understand a single brick (Router, Memory, Tool) without needing to understand the entire agent.
- **Governance**: Every agent invocation is logged with its brick composition version. Compliance team can audit: "Which agent version was running on date X?"

---

## Vocabulary: Know These Terms

- **Positional Bias**: LLMs prefer tools/options that appear early in a list, independent of relevance.
- **Tool Ordering Strategy**: Rule for arranging tools (frequency, priority, semantic relevance).
- **Agent Brick**: Self-contained component implementing one agent concern (routing, memory, tools).
- **Brick Composition**: DAG of chained bricks; state flows from one brick to the next.
- **Framework Agnosticism**: Bricks work with any framework (LangGraph, LangChain, LlamaIndex) without modification.
- **Router Brick**: Classifies user intent and pre-filters available tools.
- **Tool Brick**: Presents tools to LLM in optimized order; handles tool invocation.
- **Memory Brick**: Loads and stores context (conversation history, long-term memory) across turns.
- **Context Window Saturation**: Phenomenon where longer lists cause LLM to skim later items due to finite attention.
- **Brick Adapter**: Layer translating between brick's framework-neutral state and framework-specific state (e.g., LangGraph StatefulGraph).

---

## STAR: Situation, Task, Action, Result

### Example: "Tell Me About a Time You Optimized an Agent"

**Situation:**
"I was on a team deploying a customer-facing agent on Databricks. It had 15 tools for data exploration: queries, searches, recommendations, admin tasks. After 2 weeks in production, we noticed accuracy was only ~75%, below our 90% SLA target."

**Task:**
"I was asked to debug the issue. We had the right model, the right tools, the right data. But something was wrong with how the agent was choosing tools."

**Action:**
"I pulled execution traces from LangSmith and observed that the LLM was calling low-frequency tools (admin tasks) instead of high-frequency tools (queries). I hypothesized positional bias: the tools were alphabetized, so `admin_task` appeared before `query_warehouse` in the tool list.

I reordered the tools by historical call frequency: `query_warehouse` (60%) first, then `search_vectors` (20%), then others. I didn't change the model, the tools, or the architecture—just reordering the list.

I deployed the change to a test customer for 3 days, monitoring accuracy and token cost. Accuracy jumped to 94%, and token cost dropped 15%."

**Result:**
"We deployed the fix to all 50 production agents. Accuracy improved from 75% to 93% on average. We reduced monthly API costs by $12K due to fewer wasted tool calls. The lesson: prompt engineering (including tool order) is as important as model selection. I also built a continuous monitoring job to detect if tool frequencies shift over time, triggering automatic reordering."

---

## Red Flags: Avoid These Statements in an Interview

- **"Tool order shouldn't matter; the LLM should figure it out."** This shows you don't understand positional bias or LLM cognition. Red flag.

- **"We always alphabetize tools for consistency."** This shows you prioritize appearance over performance. Red flag.

- **"Agent Bricks lock us into Databricks."** Misunderstands the entire point (framework agnosticism). Red flag.

- **"I pass all 50 tools to the LLM; it's more flexible."** This is token waste and noise. Shows poor system thinking. Red flag.

- **"Reordering tools after deployment is risky."** It's not risky if you test it first. This shows overly conservative thinking without data. Red flag.

- **"I don't need observability; agents either work or they don't."** Agents are probabilistic systems. You need telemetry to iterate. Red flag.

---

## What They Want to Hear

1. **Concrete thinking**: You describe a real scenario and a specific fix, not theoretical generalities.

2. **Systems thinking**: You understand that agent performance is a composition of model, tools, ordering, routing, memory, and observability. No single lever wins.

3. **Production awareness**: You think about versioning, governance, monitoring, and iteration cycles, not just one-shot accuracy.

4. **Modularity**: You see Agent Bricks as a practical solution to the problem of reuse and framework evolution, not just an abstraction exercise.

5. **Data-driven**: You mention metrics, A/B tests, and observability. You don't claim improvements without evidence.

---

