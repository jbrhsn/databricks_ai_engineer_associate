# Agentic AI Concepts — Interview Prep

**Section:** Foundations | **Role target:** GenAI Engineer / ML Engineer

## Core Conceptual Questions

These test whether you understand the fundamentals.

| Question | Key points to cover | Common weak-answer trap |
|---|---|---|
| What is an AI agent, and how does it differ from an LLM call and a deterministic chain? | Agent = LLM + reason→act→observe loop + tools + goal, choosing control flow at runtime; LLM call = one shot, no tools; chain = developer-fixed steps. | Saying an agent is "just a better prompt" or "an LLM with memory" — missing the loop-and-tools structure. |
| Walk me through the agentic loop. | Reason (plan) → act (tool/LLM call) → observe (result) → repeat until final answer or step cap; ReAct interleaves reasoning traces with actions. | Describing it as a single forward pass; forgetting the termination condition entirely. |
| How does function calling actually work? | Model receives typed tool schemas, emits a structured JSON call, *your code* executes it and appends the result, model re-invoked to summarize. | Believing the model runs the function itself — a governance/security red flag. |
| Short-term vs long-term memory in agents? | Short-term = conversation/thread state (LangGraph `messages` + checkpointer/`thread_id`); long-term = external store (vector index/table) retrieved as a tool. | Treating "memory" as one undifferentiated blob or as something the context window holds forever. |
| Single-agent vs multi-agent — when each? | Single-agent = one LLM, cohesive domain, the enterprise default; multi-agent = distinct domains or overloaded tool sets, supervisor routes/hands off. | Reaching for multi-agent as the "advanced/better" option without a scaling justification. |
| When should you NOT use an agent? | Fixed/deterministic workflow, latency-sensitive, cost-sensitive, reliability-critical — use a chain; agents add loops, tokens, and unpredictability. | Assuming more autonomy is always better; unable to name a case for a chain. |

## Applied / Scenario Questions

**Q:** Your team's agent occasionally runs up huge token bills and once "hung" on a bad tool response. Users also report it sometimes took an irreversible action they didn't approve. How do you fix this?

**Strong answer framework:**
- Add an iteration/`recursion_limit` cap so a runaway loop fails gracefully instead of burning tokens — Databricks warns loops can occur in *any* tool-calling scenario.
- Insert a human-in-the-loop approval gate before irreversible/risky actions (LangGraph `interrupt()` + `Command(resume=...)`); sandbox record-writing tools (UC functions / Lakeguard).
- Instrument with MLflow Tracing so you can *see* where the loop spins and which tool call misfired.
- Tradeoff awareness: for the fixed sub-tasks, question whether an agent is even warranted — a deterministic chain removes the loop-risk class entirely.

**Q:** An agent keeps ignoring a tool or calling it with wrong arguments. Diagnose it.

**Strong answer framework:**
- First suspect the **tool schema/description**, not the model — routing is driven by the docstring / SQL `COMMENT` and typed parameters.
- Lower `temperature` (0.0–0.1) to stabilize argument generation; raising it makes this worse.
- Check tool-set size and JSON-schema limits (max 32 tools; 16-key schema cap; flatten deeply nested schemas).

## System Design / Architecture Questions (if applicable)

**Q:** Design a customer-support GenAI system on Databricks that answers policy questions, looks up order status, and can issue refunds.

**Approach:**
1. **Clarify requirements:** volume/latency budget, which actions are irreversible (refunds), governance/PII constraints, one domain or many.
2. **Propose structure:** single-agent (cohesive support domain) with tools as Unity Catalog functions — `lookup_order`, `get_policy`, `issue_refund`; author with LangGraph via the framework-agnostic Agent Framework; models via Foundation Model API function calling.
3. **Justify choices and name tradeoffs:** single-agent over multi-agent because it's one domain and the tool set is small (well under 32); deterministic-chain the pure policy-lookup path to save latency/cost; HITL gate on `issue_refund` (irreversible); step cap + low temperature for stability; MLflow Tracing + Agent Evaluation for observability and quality; escalate to multi-agent only if the domain later splits (billing vs technical vs shopping).

## Vocabulary That Signals Expertise

Use these terms naturally — don't force them:
- **Autonomy spectrum / continuum** — signals you see agency as a dial, not a binary.
- **Reason → act → observe loop / ReAct** — the precise mechanism, not hand-waving.
- **Tool schema / grounding** — shows you know routing quality comes from descriptions.
- **Recursion limit / iteration cap / graceful degradation** — production loop-safety literacy.
- **Human-in-the-loop (HITL)** — risk control for irreversible actions.
- **Framework-agnostic Agent Framework; Unity Catalog function tools; MLflow Tracing** — Databricks-specific fluency.
- **Compound AI system** — the "combine patterns" mental model.

## Vocabulary That Signals Weakness

Avoid these — they signal outdated or shallow understanding:
- **"AGI" / "the agent thinks/knows"** — anthropomorphizing; the model emits tokens and structured calls, it doesn't "think."
- **"Just make it fully autonomous"** — ignores cost, latency, and reliability trade-offs.
- **"The agent runs the tool"** — factually wrong; your code executes the tool.
- **"Multi-agent is more advanced so we should use it"** — pattern-chasing without justification.
- **"Prompt engineering will fix the looping"** — conflates prompt quality with control-flow safety.

## STAR Answer Frame

**Situation:** Our support bot's costs spiked and it occasionally hung and once issued a duplicate action, all under a fixed lookup-then-answer workflow.  
**Task:** I owned reliability and cost for the agent without regressing answer quality.  
**Action:** I re-implemented the fixed path as a deterministic chain, kept the agent only for the branch that chooses among refund/escalate/chase tools, set a `recursion_limit` with `RemainingSteps` graceful degradation, added a HITL approval gate on the irreversible refund tool, lowered `temperature` to 0.0, tightened each Unity Catalog function's `COMMENT`, and wired MLflow Tracing.  
**Result:** Per-request token cost dropped sharply, the hang class disappeared (bounded loop), unapproved irreversible actions went to zero, and tool mis-calls fell after the description fixes — with no measurable quality loss on answers.

## Red Flags Interviewers Watch For

- Defaults to "build an agent" for every task, unable to argue for a deterministic chain.
- Can't state a loop termination condition or why a step cap matters.
- Thinks the model executes tools (security/governance blind spot).
- Treats tool descriptions as cosmetic rather than as routing/correctness infrastructure.
- Picks multi-agent for prestige, not for distinct domains or tool-set limits.
- No observability plan — can't say how they'd debug a looping or misfiring agent (no MLflow Tracing).
- Ignores irreversible-action risk — no HITL, no sandboxing of write tools.
