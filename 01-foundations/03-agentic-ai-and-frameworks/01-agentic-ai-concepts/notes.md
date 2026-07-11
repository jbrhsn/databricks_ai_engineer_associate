# Agentic AI Concepts

**Section:** Foundations | **Module:** Agentic AI & Frameworks | **Est. time:** 2.5 hrs | **Exam mapping:** Supports Application Development (30%) & Assembling & Deploying (22%)

> ⚠️ Fast-evolving: verify against current official docs before relying on this. Agent Framework, Databricks function calling (Public Preview), and MCP change frequently.

---

## TL;DR

An AI agent is an LLM given a loop, tools, and a goal: it reasons about what to do, calls a tool, observes the result, and repeats until the task is done or a limit is hit — unlike a single LLM call (one shot, no actions) or a deterministic chain (fixed steps the developer wired). Agency is a *continuum*, not a switch: deterministic chains are predictable and cheap, autonomous agents are flexible but costly, less predictable, and prone to loops. On Databricks the Mosaic AI Agent Framework is framework-agnostic — you author with LangGraph, LangChain, LlamaIndex, or OpenAI, expose tools as Unity Catalog functions or via MCP, call models with Foundation Model API function calling, and trace every step with MLflow.

**The one thing to remember: an agent is an LLM in a reason → act → observe loop with tools; add autonomy only when a deterministic chain genuinely cannot do the job, because every extra loop costs latency, money, and predictability.**

---

## ELI5 — Explain It Like I'm 5

A fixed recipe card tells you exactly what to do, step by step, and never changes — that is a deterministic chain. A single LLM call is like asking a friend one question and getting one answer: no cooking, just talking. An agent is a chef who tastes the soup, decides it needs salt, walks to the pantry to fetch some (that walk is a *tool call*), tastes again, and keeps adjusting until the soup is right. The chef doesn't follow a fixed card — they *loop*: taste, act, taste again. The common mistake is thinking an agent is just a smarter prompt or that "more freedom" always makes a better chef; in reality a chef with unlimited freedom and no timer might keep fussing forever, waste ingredients, and still serve dinner late — sometimes the fixed recipe is exactly what you want.

---

## Learning Objectives

By the end of this chapter you will be able to:
- [ ] Distinguish an AI agent from a plain LLM call and a deterministic chain by their control flow.
- [ ] Explain the reason → act → observe agentic loop and the ReAct pattern under the hood.
- [ ] Compare single-agent vs multi-agent designs and place a workload on the autonomy spectrum.
- [ ] Design a tool (function schema + Unity Catalog function) that an agent can call reliably.
- [ ] Diagnose when NOT to use an agent, citing determinism, cost, latency, and reliability trade-offs.

---

## Visual Overview

### The Agentic Loop (reason → act → observe)

```
             ┌──────────────────────────────────────────┐
             │                                          ▼
User goal ──► REASON ──► decide: need a tool? ──► ACT (tool call) ──► OBSERVE (tool result)
             ▲   │                                                         │
             │   └── no tool needed ──► FINAL ANSWER ──► User              │
             └─────────────────────── feed result back ───────────────────┘
                        (repeat until answer or step limit hit)
```

### LLM Call vs Deterministic Chain vs Agent

```
LLM + prompt          Prompt ──► LLM ──► Answer
                      (one shot, no tools, no loop)

Deterministic chain   Input ──► Retrieve ──► Augment ──► LLM ──► Answer
                      (developer fixes the steps; LLM never chooses the path)

Single agent          Goal ──► LLM decides ──► tool? ──► loop ──► Answer
                      (LLM chooses which tools, in what order, when to stop)
```

### Decision Tree — "Should this be an agent?"

```
Is the task a fixed, well-defined workflow?
├── Yes ──► Are the steps always the same regardless of input?
│           ├── Yes ──► Deterministic chain (predictable, cheap, auditable)
│           └── No  ──► Chain with one dynamic tool-calling step
└── No  ──► Does it need on-the-fly decisions / variable tool use?
            ├── Yes ──► One cohesive domain? 
            │           ├── Yes ──► Single-agent system (the usual sweet spot)
            │           └── No, many distinct domains ──► Multi-agent system
            └── No  ──► Single LLM call + prompt
```

---

## Key Concepts

### Agent vs LLM Call vs Deterministic Chain

**What it is.** A spectrum of GenAI system designs that trade predictability for flexibility: a bare LLM+prompt (one shot), a deterministic chain (fixed developer-defined steps such as basic RAG), a single-agent system (LLM orchestrates its own tool use), and a multi-agent system (specialized agents coordinated by a supervisor).

**How it works under the hood.** In a deterministic chain the *developer* decides which tools are called, in what order, and with which parameters — the LLM never chooses the path, so behavior is identical for every request. In an agent the *LLM* decides at runtime whether to call a tool, which one, and when to stop; this decision is re-made on every loop iteration using the accumulated conversation and tool outputs. That runtime decision-making is exactly what makes an agent flexible and also what makes it less predictable.

**Where in Databricks.** Databricks documents these as four "levels of complexity" on the *Agent system design patterns* page. Custom Agents built with the Mosaic AI Agent Framework are agnostic to the pattern — you start with a chain and evolve toward single- then multi-agent as requirements grow.

### The Agentic Loop and the ReAct Pattern

**What it is.** The core execution cycle of an agent: **reason** (plan the next step), **act** (call a tool or the LLM), **observe** (read the tool result), then repeat. ReAct ("Reasoning + Acting") is the specific pattern of interleaving a reasoning trace with tool actions.

**How it works under the hood.** The LLM is prompted with the goal, the available tool schemas, and the running history. It emits either a final answer or a structured tool call. The framework executes the tool, appends the result as a new message, and calls the LLM again — so the loop is literally "append tool output, re-invoke model." The loop terminates when the model returns a final answer with no tool call, or when a step/recursion limit fires.

**Where in Databricks.** LangGraph (primary framework) models this loop explicitly as a graph: nodes do work, edges route, and a conditional edge sends control back to the model node after each tool call. `create_react_agent` from `langgraph.prebuilt` builds this loop for you; MLflow Tracing records each reason/act/observe step as a span.

### Tools, Function Calling, and Tool Schemas

**What it is.** A tool is a callable capability (query data, run code, hit an external API) exposed to the LLM through a schema — a name, description, and typed parameters. Function calling is the mechanism by which the model emits a structured request to invoke one.

**How it works under the hood.** You pass a JSON-schema `tools` list to the model. The model does **not** run the function — it returns a JSON object of arguments that adhere to the schema; your code parses it, executes the function, appends the result, and re-invokes the model to summarize. The quality of the model's tool selection depends heavily on the *description*: a vague description leads to wrong or missing calls. This is why grounding (feeding retrieved facts and clear tool docs) matters — the agent can only act well on what it can see and understand.

**Where in Databricks.** Foundation Model APIs expose OpenAI-compatible function calling via the `tools` and `tool_choice` parameters (max 32 tools, JSON schema limited to 16 keys, Public Preview optimized for single-turn). Tools are commonly authored as Unity Catalog functions (`create_python_function`, or SQL `CREATE FUNCTION` with a `COMMENT` clause that becomes the tool description) and surfaced through `UCFunctionToolkit` or a managed MCP server.

### Memory: Short-Term vs Long-Term

**What it is.** Short-term (conversation/working) memory is the message history and tool outputs within a single task or thread; long-term memory is knowledge persisted across sessions (facts, past decisions, user preferences).

**How it works under the hood.** Short-term memory lives in the running state passed between loop iterations — in LangGraph, a `messages` channel with the `add_messages` reducer accumulates the conversation, and a checkpointer persists that state per `thread_id` so a conversation can resume. Long-term memory is typically an external store (a vector index or table) the agent retrieves from as a tool, not something held in the context window.

**Where in Databricks.** LangGraph's `MessagesState` + checkpointer handles short-term/thread memory; long-term memory is usually a Mosaic AI Vector Search index or a Unity Catalog table queried through a retrieval tool.

### Planning and Task Decomposition

**What it is.** The agent's ability to break a goal into sub-steps and sequence them, rather than answering in one shot.

**How it works under the hood.** In the reasoning step the model produces a plan ("look up the order, then check the return policy, then decide"), then executes sub-steps as tool calls, revising the plan as observations arrive. Planning is emergent from prompting in a single-agent loop, or explicit as separate nodes/agents in a graph.

**Where in Databricks.** The *Agent system design patterns* call-center example shows plan → find information → reason → act → reason. In LangGraph this maps to distinct nodes and conditional edges; a multi-agent system can dedicate a supervisor to planning and route sub-tasks to specialist agents.

### Single-Agent vs Multi-Agent & the Autonomy Spectrum

**What it is.** A single-agent system is one LLM orchestrating one coordinated flow; a multi-agent system is two or more specialized agents coordinated by a supervisor (LLM or rule-based router). Autonomy is the continuum from fully deterministic (developer controls flow) to fully autonomous (model controls flow).

**How it works under the hood.** In multi-agent designs the supervisor decides which specialist handles a request or when to hand off; each agent can own a distinct prompt, context, and tool subset. This is useful when one agent's tool schema would be overloaded (recall the 32-tool ceiling) or when domains differ radically. The cost is orchestration complexity and harder tracing/debugging across endpoints.

**Where in Databricks.** Databricks names single-agent the "good default" and sweet spot for enterprise use; multi-agent is for large cross-functional domains. LangGraph implements handoffs via `Command(goto=..., graph=Command.PARENT)` between subgraphs.

### Human-in-the-Loop & When NOT to Use an Agent

**What it is.** Human-in-the-loop (HITL) pauses the agent for approval before a risky or irreversible action. "When not to use an agent" is the discipline of choosing a simpler design when autonomy buys nothing.

**How it works under the hood.** HITL is implemented by interrupting execution, surfacing the proposed action to a human, and resuming with their decision. In LangGraph this is `interrupt()` inside a node plus `Command(resume=...)` to continue, backed by a checkpointer that holds state while paused. You skip agents entirely when the task is deterministic (a fixed pipeline), latency-sensitive (each extra LLM call adds delay), cost-sensitive (each loop burns tokens), or reliability-critical (unpredictable tool loops are unacceptable).

**Where in Databricks.** The design-patterns page explicitly recommends requiring human approval for risky actions and sandboxing record-updating or code-running tools; Unity Catalog functions provide sandboxed (Lakeguard) execution for production.

### Key Parameters / Configuration Knobs

| Parameter | What it controls | Decision rule |
|---|---|---|
| `recursion_limit` (LangGraph) | Max super-steps in one graph execution before `GraphRecursionError` (default 1000) | Set to a small number (e.g. 10–25) for bounded agents; use `RemainingSteps` to degrade gracefully rather than crash on complex-but-legitimate tasks. |
| `tool_choice` (Foundation Model API) | Whether/which tool the model may call (`auto`, `required`, `none`, or a named function) | Use `auto` for general agents; `required` when a tool call is mandatory; `none` to force a plain text turn (e.g. final summary). |
| Number of tools in `tools` | Size of the tool set the model chooses from (max 32) | Give the agent only the tools it needs; if you approach the ceiling or span domains, split into a multi-agent system. |
| `temperature` | Randomness of generation, including tool-argument generation | Set low (e.g. 0.0–0.1) for tool-calling agents so argument generation and routing are stable and repeatable. |
| Iteration / step cap (single-agent loop) | Max tool-call cycles before forced stop | Always set one — Databricks warns infinite loops can occur in any tool-calling scenario; a cap converts a runaway cost into a graceful failure. |
| `thread_id` / checkpointer | Persistence key for short-term (conversation) memory | Set a stable `thread_id` per conversation to enable resume and HITL; omit persistence for stateless one-shot calls. |

### Worked Example: Requirement → Decision

**Given:** A support team wants a bot that answers "What's the status of invoice INV-1042?" by looking the invoice up in a governed table and replying in plain English. The lookup is always the same operation; the only variable is the invoice ID.

- **Step 1 — Identify the goal:** Return an invoice's status from a known table, phrased for a human.
- **Step 2 — Define inputs:** A user question containing (or implying) an invoice ID; a Unity Catalog table `finance.ar.invoices`.
- **Step 3 — Define outputs:** A short natural-language status answer, plus a traced record of the lookup.
- **Step 4 — Apply constraints:** Latency-sensitive (support chat), cost-sensitive (high volume), reliability-critical (finance data must not be guessed), and the workflow is fixed (always: extract ID → look up → phrase).
- **Step 5 — Select the approach:** Use a **deterministic chain** (or a single dynamic function-calling step), *not* an autonomous multi-step agent. Rationale: the steps never change, so an agent's runtime path-choosing adds latency, token cost, and loop risk with zero benefit — reserve the agentic loop for the *variant* of this task where the bot must also decide between many tools (issue a refund, escalate, chase payment) based on the answer.

---

## Implementation

```python
# Scenario: A support agent must answer varied questions in ONE domain and decide
# at runtime whether to look a customer up — a single-agent loop over a UC-function tool.
# Constraint: bounded autonomy — cap steps, low temperature, trace every step.
import mlflow
from databricks_langchain import ChatDatabricks, UCFunctionToolkit
from langgraph.prebuilt import create_react_agent

mlflow.langchain.autolog()  # record each reason/act/observe step as a trace span

# UC function `main.default.lookup_customer_info` was registered with a clear COMMENT,
# which becomes the tool description the model reasons over.
tools = UCFunctionToolkit(
    function_names=["main.default.lookup_customer_info"]
).tools

llm = ChatDatabricks(endpoint="databricks-claude-sonnet-4-5", temperature=0.0)
agent = create_react_agent(llm, tools)  # builds the reason → act → observe loop

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the status of Acme Corp's account?"}]},
    config={"recursion_limit": 12},  # bound the loop; fail gracefully instead of forever
)
print(result["messages"][-1].content)
```

```python
# Anti-pattern: give an autonomous agent a fixed, well-defined task and no step cap.
# The task (always look up one invoice) never varies, yet the agent is free to loop —
# a malformed tool result can send it retrying indefinitely, burning tokens and latency
# with no correctness benefit, and there is no recursion_limit to stop it.
agent = create_react_agent(llm, tools)
agent.invoke({"messages": [{"role": "user", "content": "Status of INV-1042?"}]})
# No recursion_limit -> defaults to 1000 super-steps; a stuck loop can run up huge cost.

# Correct approach: for a fixed workflow, use a deterministic chain (no runtime path choice),
# and if you must keep an agent, always cap the loop.
def invoice_status_chain(invoice_id: str) -> str:  # Retrieve -> Augment -> Generate
    row = client.execute_function(
        "finance.ar.get_invoice_status", parameters={"invoice_id": invoice_id}
    ).value
    prompt = f"State the invoice status in one sentence for a customer:\n{row}"
    return llm.invoke(prompt).content
# Predictable, cheap, auditable, and impossible to loop. Reserve the agent (with a step cap)
# for the variant where the bot must CHOOSE among refund / escalate / chase-payment tools.
```

---

## Common Pitfalls & Misconceptions

- **"An agent is just a smarter prompt."** — Beginners equate autonomy with prompt quality because both improve answers. The correct model is structural: an agent adds a *loop* and *tools* around the LLM so it can take actions and react to results, which a prompt alone can never do.
- **"More autonomy is always better."** — It feels like giving the model freedom makes it smarter, so people reach for agents by default. In reality most production systems deliberately constrain autonomy; a deterministic chain is often more predictable, cheaper, faster, and easier to audit than an agent.
- **"The LLM runs the tool."** — Because function calling looks like the model "using" a tool, beginners assume it executes code. The model only emits a structured JSON request; *your code* runs the function and feeds the result back, so you own execution, permissions, and sandboxing.
- **"No step cap needed — it'll stop when it's done."** — New builders trust the model to terminate, but a malformed tool result or ambiguous goal can loop forever. Always set an iteration/`recursion_limit`; Databricks warns infinite loops can occur in any tool-calling scenario.
- **"Multi-agent is the advanced, better choice."** — Multi-agent sounds more capable, so it's over-adopted. It only pays off across radically distinct domains or when one agent's tool schema is overloaded; otherwise it just adds orchestration and tracing complexity over a single-agent "sweet spot."
- **"Tool descriptions don't matter much."** — Beginners write terse tool docs because the code works in a unit test. But the model chooses tools *from the description*; vague or missing docstrings/`COMMENT`s cause wrong or skipped calls, so tool documentation is part of correctness, not polish.

---

## Key Definitions

| Term | Definition |
|---|---|
| AI agent | An LLM placed in a reason → act → observe loop with tools and a goal, choosing at runtime which tools to call and when to stop. |
| Deterministic chain | A GenAI pipeline whose steps, order, and parameters are fixed by the developer; the LLM never chooses the control flow. |
| Agentic loop | The cycle of reasoning about the next step, acting (tool/LLM call), observing the result, and repeating until done or capped. |
| ReAct | A pattern that interleaves an explicit reasoning trace with tool actions in the same loop ("Reasoning + Acting"). |
| Tool / function calling | A callable capability exposed to the model via a typed schema; the model emits a structured call, and application code executes it. |
| Tool schema | The name, description, and typed parameters that tell the model when and how to use a tool. |
| Grounding | Supplying the model with retrieved facts and clear tool descriptions so its actions and answers rest on real data. |
| Short-term memory | Conversation/working state within one task or thread (message history, tool outputs). |
| Long-term memory | Knowledge persisted across sessions, typically in an external store the agent retrieves from as a tool. |
| Single-agent system | One LLM orchestrating one coordinated flow of tool calls and decisions. |
| Multi-agent system | Two or more specialized agents coordinated by a supervisor that routes or hands off tasks. |
| Autonomy spectrum | The continuum from fully deterministic (developer-controlled flow) to fully autonomous (model-controlled flow). |
| Human-in-the-loop (HITL) | Pausing the agent for human approval before a risky or irreversible action, then resuming. |
| Mosaic AI Agent Framework | Databricks' framework-agnostic toolkit to author, deploy, and evaluate agents (LangGraph, LangChain, LlamaIndex, OpenAI). |

---

## Summary / Quick Recall

- Agent = LLM + loop + tools + goal; LLM call = one shot; chain = fixed developer-defined steps.
- The loop is reason → act → observe, repeated until a final answer or a step cap fires.
- Function calling means the model emits a structured call; *your code* runs the tool and returns the result.
- Tool selection quality depends on the tool description — write clear docstrings/`COMMENT`s.
- Single-agent is the enterprise sweet spot; multi-agent only for distinct domains or overloaded tool sets.
- Always cap iterations (`recursion_limit`); autonomy costs latency, tokens, and predictability.
- Databricks Agent Framework is framework-agnostic; tools = UC functions/MCP, models = Foundation Model API function calling, observability = MLflow Tracing.

---

## Self-Check Questions

1. What distinguishes an AI agent from a deterministic chain?

   <details><summary>Answer</summary>

   In an agent, the **LLM decides at runtime** which tools to call, in what order, and when to stop, looping over reason → act → observe. In a deterministic chain the **developer fixes** the steps, order, and parameters, so the LLM never chooses the control flow and behavior is identical for every request. The tempting wrong answer — "an agent just has a better prompt" — is wrong because the difference is structural (a loop with tools and runtime decisions), not prompt quality.

   </details>

2. A team needs a bot that always performs the same three-step lookup-and-summarize workflow for every request, with tight latency and cost limits. Which design should they choose and why?

   <details><summary>Answer</summary>

   A **deterministic chain** (Retrieve → Augment → Generate). Because the steps never vary, an agent's runtime path-choosing adds LLM calls, latency, token cost, and loop risk for zero benefit. Choosing a single-agent system here is the tempting mistake — it "feels" more capable, but it degrades exactly the latency/cost/predictability the team requires.

   </details>

3. You wire a function-calling tool but the agent keeps ignoring it or calling it with wrong arguments. Applying the concepts, what is the most likely fix?

   <details><summary>Answer</summary>

   Improve the **tool schema / description** (the docstring or SQL `COMMENT` and typed parameters). The model selects and fills tools *from their descriptions*; vague docs cause skipped or malformed calls. Raising `temperature` is the tempting but wrong move — higher randomness makes argument generation *less* stable, the opposite of what you want for tool calling (keep it low, e.g. 0.0–0.1).

   </details>

4. **Which TWO** of the following are valid reasons to choose a multi-agent system over a single-agent system?
   - A. The application spans radically distinct domains (e.g. finance, devops, marketing).
   - B. You want lower latency and fewer LLM calls.
   - C. One agent's tool set would exceed a practical/schema tool limit.
   - D. You need the simplest design that is easiest to trace and debug.
   - E. The workflow is fixed and fully deterministic.

   <details><summary>Answer</summary>

   **A and C.** Multi-agent shines when domains differ radically (each agent owns a prompt/context/tool subset) and when a single agent's tool schema would be overloaded (recall the 32-tool ceiling). B and D are wrong because multi-agent *adds* orchestration overhead, more LLM calls, and harder cross-endpoint tracing — the opposite of lower latency and simpler debugging. E is the most tempting distractor but points to a deterministic chain, not any agent at all.

   </details>

5. An agent that can update customer records occasionally takes an irreversible action the user didn't intend, and once looped indefinitely on a malformed tool result. Which TWO controls best address these two failures, and why?

   <details><summary>Answer</summary>

   A **human-in-the-loop approval gate** for the risky write action, and an **iteration/`recursion_limit` cap** for the loop. HITL (LangGraph `interrupt()` + `Command(resume=...)`) forces human sign-off before irreversible actions — Databricks explicitly recommends requiring approval for risky actions and sandboxing record-updating tools. A step cap converts a runaway loop into a graceful failure instead of unbounded token spend. Simply lowering `temperature` is the tempting non-fix: it slightly stabilizes routing but neither prevents an unwanted irreversible write nor guarantees loop termination.

   </details>

---

## Further Reading

- [Agent system design patterns — Databricks](https://docs.databricks.com/aws/en/agents/agent-system-design-patterns) — *verified 2026-07-11* — the LLM → chain → single-agent → multi-agent autonomy continuum, reason/act loop, HITL, and loop-guard advice.
- [Build AI agents on Databricks — Databricks](https://docs.databricks.com/aws/en/agents/) — *verified 2026-07-11* — overview of the framework-agnostic Mosaic AI Agent Framework, AI Playground, MCP, and MLflow Tracing.
- [Create AI agent tools using Unity Catalog functions — Databricks](https://docs.databricks.com/aws/en/agents/agent-framework/create-custom-tool) — *verified 2026-07-11* — authoring tools as UC functions, tool documentation, `UCFunctionToolkit`, serverless vs local execution.
- [Function calling on Databricks — Databricks](https://docs.databricks.com/aws/en/machine-learning/model-serving/function-calling) — *verified 2026-07-11* — OpenAI-compatible `tools`/`tool_choice`, JSON schema limits, and the function-calling loop.
- [Graph API overview — LangGraph](https://docs.langchain.com/oss/python/langgraph/graph-api) — *verified 2026-07-11* — StateGraph nodes/edges, `MessagesState`, `recursion_limit`, `RemainingSteps`, and `Command` routing.
- [MLflow Tracing for LLM and Agent Observability — MLflow](https://mlflow.org/docs/latest/genai/tracing/) — *verified 2026-07-11* — framework-agnostic per-step tracing with one-line autolog for LangChain/LangGraph and others.

