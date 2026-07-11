# Agentic AI Concepts — Thought Leadership

**Section:** Foundations | **Target audience:** senior engineers, tech leads, and platform architects evaluating GenAI | **Target publication:** LinkedIn / personal blog

## Hook / Opening Thesis

The hardest agent-engineering skill isn't building an agent — it's knowing when *not* to. I've watched more GenAI budgets die to enthusiastic autonomy than to any model limitation. An agent is not a maturity milestone you graduate to; it's a cost you take on deliberately, and most of the time a boring deterministic chain wins.

## Key Claims (3–5)

1. Agency is a spending decision, not a capability upgrade — every loop iteration is real latency, tokens, and unpredictability you're buying.
2. The single-agent loop is the right default for *most* enterprise use cases; multi-agent is a scaling answer to distinct domains and overloaded tool sets, not a sophistication badge.
3. Tool descriptions are correctness infrastructure, not documentation polish — the model routes on the description, so a vague docstring is a latent bug.
4. "Framework-agnostic" is the strategically correct posture for a platform, and Databricks betting the Agent Framework on it (LangGraph, LangChain, LlamaIndex, OpenAI) is a signal that the orchestration layer is commoditizing.
5. The unglamorous controls — a step cap and a human-in-the-loop gate — separate demoware from production far more reliably than a cleverer prompt.

## Supporting Evidence & Examples

Databricks' own *Agent system design patterns* guidance leads with "start simple" and frames LLM → deterministic chain → single-agent → multi-agent as a continuum of *complexity and cost*, not a ladder of quality. It explicitly names single-agent as the "sweet spot" for enterprise and warns that infinite loops "can occur in any tool-calling scenario" — which is why LangGraph ships a `recursion_limit` (default 1000, which is a *safety net*, not a setting you should leave alone) and a `RemainingSteps` value for graceful degradation.

I've seen the tool-description failure first-hand: an agent that "randomly" ignored a lookup function turned out to have a one-line docstring. Rewriting the Unity Catalog `COMMENT` to state exactly when and how to use it fixed the routing with zero code change. Databricks devotes a whole section of its tool docs to "effective vs ineffective tool documentation" for precisely this reason.

On the cost point: Databricks' function-calling docs note that more tools mean more input tokens, and the design-patterns page is blunt that "each additional LLM or tool call increases token usage and response time." That's the whole argument against autonomy-by-default in one sentence.

## The Original Angle

Most agent content sells you *up* the complexity curve — here's how to build a multi-agent swarm. I want to sell you *down* it. The genuinely senior move is to treat the autonomy spectrum as a dial you turn as far *left* (toward deterministic) as the requirements allow, and justify every click rightward. I can say this credibly because I've paid the bills for the alternative: the outages and cost spikes almost never came from the model being dumb; they came from us granting freedom the task never needed.

## Counterarguments to Address

*"But agents are the future — you're just describing legacy pipelines."* No. I'm describing where autonomy earns its keep versus where it's cargo-culted. The future is compound AI systems that *combine* patterns — a mostly deterministic chain with one dynamic tool-calling step is a real Databricks-recommended design, and it's more agentic than a chain without being a runaway.

*"A good enough model won't loop or misfire, so caps are training wheels."* Model quality reduces failure probability; it doesn't make it zero, and production is where the tail lives. A step cap costs nothing when the agent behaves and saves you when it doesn't — that asymmetry alone justifies it.

## Practical Takeaways for the Reader

- Before building an agent, write the deterministic version first; adopt the agent only for the steps that genuinely vary at runtime.
- Set a `recursion_limit` / iteration cap on day one, and use `RemainingSteps`-style graceful degradation over hard crashes.
- Treat tool `COMMENT`s and docstrings as part of your correctness test surface, not afterthoughts.
- Keep tool sets minimal per agent; when you approach the ceiling or cross domains, split into a multi-agent system rather than overloading one.
- Wire MLflow Tracing before you need it — you cannot debug a loop you can't see.

## Call to Action

If you're about to build your first agent this quarter, do one experiment first: implement the deterministic-chain version of the same task, measure its latency and cost, and only then justify — in writing — which specific step needs runtime autonomy. What's the last "agent" you built that a chain would have served better? I'd genuinely like to hear where the line fell for you.

## Further Reading / References

- [Agent system design patterns — Databricks](https://docs.databricks.com/aws/en/agents/agent-system-design-patterns) — anchors the "start simple, add autonomy deliberately" argument and the loop-guard warning.
- [Function calling on Databricks](https://docs.databricks.com/aws/en/machine-learning/model-serving/function-calling) — supports the token-cost and tool-count claims.
- [Create AI agent tools using Unity Catalog functions](https://docs.databricks.com/aws/en/agents/agent-framework/create-custom-tool) — the effective-vs-ineffective tool documentation evidence.
- [Graph API overview — LangGraph](https://docs.langchain.com/oss/python/langgraph/graph-api) — the `recursion_limit` and `RemainingSteps` mechanics behind the "always cap it" takeaway.
