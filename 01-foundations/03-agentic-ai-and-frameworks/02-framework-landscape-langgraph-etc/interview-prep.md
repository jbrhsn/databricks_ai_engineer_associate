# Framework Landscape: LangGraph, LangChain, CrewAI — Interview Prep

**Section:** Foundations | **Role target:** GenAI Engineer / ML Engineer

## Core Conceptual Questions

These test whether you understand the fundamentals.

| Question | Key points to cover | Common weak-answer trap |
|---|---|---|
| How do LangGraph, LangChain, and CrewAI relate? | They are layers, not competitors: LangChain = component library + light harness; LangGraph = low-level orchestration (state/nodes/edges/cycles); CrewAI = higher-abstraction role-based multi-agent. LangChain's own agents are built on LangGraph. | Framing them as three rival products you "pick one of," or claiming LangGraph replaced LangChain. |
| What does LangGraph give you that a plain chain doesn't? | Explicit `State` with reducers, conditional edges and cycles, a checkpointer for persistence, `interrupt()` for human-in-the-loop, and `recursion_limit` as a safety net. | Saying "it's just a nicer way to write chains" — missing state, cycles, and HITL. |
| When would you choose CrewAI? | When the work genuinely decomposes into collaborating roles; choose `sequential` for a fixed pipeline or `hierarchical` (with a manager agent) for dynamic delegation. Trade control for speed-to-build. | Defaulting to CrewAI for everything because "multi-agent is more powerful." |
| What makes the Databricks Agent Framework "framework-agnostic"? | It supports LangGraph, LangChain, LlamaIndex, and OpenAI; the unifying contract is an MLflow `ResponsesAgent`/`ChatAgent` wrapper, not any single framework. | Believing Databricks requires a special/proprietary agent format. |
| How do you deploy an agent to Databricks Model Serving? | Wrap in MLflow `ResponsesAgent`, log with models-from-code (`log_model(python_model="agent.py")`), register to Unity Catalog, create a serving endpoint. Signature is auto-inferred. | Saying "just pickle it" or assuming you rewrite the agent per framework. |
| What is a conditional edge and why does it matter? | An edge whose routing function inspects state at runtime to pick the next node; routing back forms the agent's reasoning cycle. | Confusing it with a static edge, or not connecting it to how loops are formed. |

## Applied / Scenario Questions

**Q:** You're asked to build an agent that retrieves policy docs, drafts a reply, and pauses for human approval on refunds over $500, deployed on Databricks with monitoring. Walk me through your framework choice.

**Strong answer framework:**
- Identify the control requirements first: a conditional branch (refund threshold) and a durable human-in-the-loop pause — both point to LangGraph.
- Use a `StateGraph` with a conditional edge for the threshold, `interrupt()` + a `checkpointer` and stable `thread_id` for the approval gate and resume-after-hours.
- Put LangChain components (retriever, `ChatDatabricks`) *inside* the nodes — layers, not either/or.
- Wrap the compiled graph in an MLflow `ResponsesAgent`, log via models-from-code, register to Unity Catalog, deploy to Model Serving; monitoring/tracing comes from the MLflow integration.
- Show tradeoff awareness: a plain chain can't pause/resume or loop; CrewAI hides the exact conditional/HITL control this needs.

## System Design / Architecture Questions (if applicable)

**Q:** Design an agent platform where product teams can build in whichever framework they prefer but ops needs one consistent deploy/monitor path.

**Approach:**
1. **Clarify requirements:** multiple frameworks (LangGraph, LangChain, LlamaIndex, OpenAI), one deployment contract, tracing/eval/monitoring, governance.
2. **Propose structure:** Standardize on the MLflow `ResponsesAgent`/`ChatAgent` interface as the boundary; teams implement `predict`/`predict_stream` however they like. Log every agent via models-from-code, register to Unity Catalog, serve on Model Serving. Route LLM traffic through the AI Gateway for centralized governance.
3. **Justify choices and name tradeoffs:** The wrapper makes the framework a swappable internal detail (low lock-in) and gives uniform tracing/eval; the tradeoff is that teams must conform to the Responses schema and wrap their agents, which is a small, one-time cost versus per-framework deploy pipelines.

## Vocabulary That Signals Expertise

Use these terms naturally — don't force them:
- **State / reducer** — signals you understand how LangGraph accumulates vs. overwrites data across nodes.
- **Conditional edge / cycle / super-step** — signals real knowledge of the graph execution model.
- **Checkpointer + thread_id** — signals you know how persistence, multi-turn memory, and resume work.
- **`interrupt()` / human-in-the-loop** — signals you can build controllable, gated agents.
- **Models-from-code / `ResponsesAgent` / `ChatAgent`** — signals you know the framework-agnostic Databricks deployment contract.
- **Sequential vs. hierarchical process** — signals precise CrewAI understanding.
- **Framework-agnostic** — signals you see the frameworks as layers behind a stable interface.

## Vocabulary That Signals Weakness

Avoid these — they signal outdated or shallow understanding:
- "LangGraph replaced LangChain" — they are layers; LangChain agents are built on LangGraph.
- "Multi-agent is always better" — signals a bias toward over-engineering.
- "Just pickle the agent and serve it" — ignores the MLflow signature/interface contract.
- "You have to rewrite it for Databricks" — the Agent Framework is framework-agnostic.
- "It's basically a `while` loop with an LLM" — ignores state, persistence, HITL, and step limits.

## STAR Answer Frame

**Situation:** A team had shipped a hierarchical CrewAI crew to handle routine support Q&A, and it was slow and hard to debug.
**Task:** I was asked to cut latency and cost without losing answer quality, and make the flow observable.
**Action:** I mapped the requirement and found no real role collaboration — it was a single retrieve-and-answer pass. I replaced the crew with a plain LangChain LCEL chain, and where a later feature needed a conditional retry loop I moved that path (only that path) to a LangGraph `StateGraph` with a conditional edge and checkpointer. I wrapped the result in an MLflow `ResponsesAgent` and deployed via models-from-code to Model Serving for tracing.
**Result:** Token cost and p95 latency dropped sharply (the manager-agent overhead was gone), traces became readable end-to-end, and the team adopted a "least orchestration first" rule for new agents.

## Red Flags Interviewers Watch For

- Cannot articulate *why* LangGraph over a chain — no mention of state, cycles, or HITL.
- Reaches for multi-agent/CrewAI by default without asking whether roles genuinely collaborate.
- Thinks the frameworks are mutually exclusive and can't explain the layering.
- Doesn't know the Databricks deployment contract is an MLflow interface (`ResponsesAgent`/`ChatAgent`) and assumes per-framework rewrites.
- Ignores observability, evaluation, and resumability when justifying a framework choice.
- Proposes a hand-rolled agent loop without accounting for persistence, retries, or step limits.
