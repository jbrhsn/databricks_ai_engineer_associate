# Framework Landscape: LangGraph, LangChain, CrewAI

**Section:** Foundations | **Module:** Agentic AI & Frameworks | **Est. time:** 2.5 hrs | **Exam mapping:** Supports Application Development (30%) & Assembling & Deploying (22%)

> ⚠️ Fast-evolving: verify against current official docs before relying on this. The Databricks Agent Framework, MLflow `ResponsesAgent`/`ChatAgent`, and the LangGraph/LangChain APIs change frequently — the mechanisms below are current as of 2026-07-11.

---

## TL;DR

The orchestration "framework landscape" is not a menu where you pick one winner — it is a stack of layers you compose. LangChain is the **component library** (models, prompts, tools, retrievers, LCEL) and a lightweight agent harness; LangGraph is the **low-level orchestration layer** that gives you an explicit `State`, nodes, edges, cycles, checkpointing, and human-in-the-loop control; CrewAI is a **higher-abstraction, role-based multi-agent** framework you trade control for speed with. On Databricks, none of these lock you in: the Agent Framework is framework-agnostic, and you deploy whatever you build by wrapping it in an MLflow `ResponsesAgent` (or `ChatAgent`) and logging it with the models-from-code pattern to Model Serving. **The one thing to remember: choose the framework by how much explicit control over state and flow you need — LangGraph for stateful/controllable agents, a plain LangChain chain when a single deterministic pass is enough, CrewAI when you want fast role-based multi-agent at the cost of fine control.**

---

## ELI5 — Explain It Like I'm 5

Imagine you are building with Lego. LangChain is the **box of bricks** — the wheels, the little people, the windows — plus a small pre-glued car you can grab if you just need a car. LangGraph is the **instruction booklet and the baseplate wiring**: it decides which brick connects to which, when to loop back and add another piece, and where to stop and ask a grown-up "does this look right?" before continuing. CrewAI is a **pre-built Lego set in a themed box** — a pirate ship with a captain and a crew already assigned roles — you snap it together fast, but you cannot easily rewire how the captain talks to the crew. The common mistake is thinking you must pick only one box off the shelf; in reality you usually pour the LangChain bricks *into* the LangGraph booklet, and you only reach for the pre-built CrewAI set when the themed shape it gives you is exactly what you want.

---

## Learning Objectives

By the end of this chapter you will be able to:
- [ ] Explain how LangGraph, LangChain, and CrewAI relate as layers rather than competitors, and where each sits in the stack.
- [ ] Compare the three frameworks on control, speed-to-build, and multi-agent fit, and select one for a given requirement.
- [ ] Implement a minimal LangGraph `StateGraph` with a conditional edge and a compiled graph, and a plain LangChain agent/chain for the simple case.
- [ ] Explain why the Databricks Agent Framework is framework-agnostic and how you deploy any of these via MLflow `ResponsesAgent` and models-from-code to Model Serving.
- [ ] Diagnose when reaching for a multi-agent crew or a hand-rolled loop is the wrong call versus a controllable graph.

---

## Visual Overview

### LangGraph State-Graph (nodes, edges, cycle)

```
                   ┌──────────────────────────────┐
     user input    │           State              │  (shared dict/TypedDict,
        │          │  messages, retrieved, done   │   updated via reducers)
        ▼          └──────────────────────────────┘
     START ──► retrieve ──► generate ──► ┌ should_retry? ┐
                  ▲                      │               │
                  │                      ├─ yes ──────────┘  (conditional edge = cycle)
                  └──────────────────────┘
                                         └─ no ──► END
```

### Layering: LangChain components inside LangGraph nodes

```
┌───────────────────────────── LangGraph (orchestration) ─────────────────────────────┐
│   State  │  node_retrieve        │  node_generate         │  conditional edge / END   │
│          │  ┌──────────────────┐ │  ┌───────────────────┐ │                           │
│          │  │ LangChain:       │ │  │ LangChain:        │ │   routing_function(state) │
│          │  │  retriever +     │ │  │  ChatModel + LCEL │ │   ──► "retrieve" | END     │
│          │  │  output parser   │ │  │  prompt | model   │ │                           │
│          │  └──────────────────┘ │  └───────────────────┘ │                           │
└──────────────────────────────────────────────────────────────────────────────────────┘
        deploy the compiled graph ──► MLflow ResponsesAgent ──► Databricks Model Serving
```

### Framework decision tree (control vs speed vs multi-agent)

```
Do you need explicit control over state, loops, or human-in-the-loop?
├── Yes ──► LangGraph (StateGraph: nodes, edges, checkpointer, interrupt)
└── No  ──► Is one deterministic pass (retrieve → prompt → model → parse) enough?
            ├── Yes ──► Plain LangChain chain / create_agent
            └── No  ──► Do you want fast role-based multi-agent, low control?
                        ├── Yes ──► CrewAI (crew of role agents, sequential/hierarchical)
                        └── No  ──► LangGraph multi-agent (explicit handoffs)
```

---

## Key Concepts

### LangGraph — the low-level orchestration layer (PRIMARY)

**What it is.** LangGraph models an agent workflow as a *graph*: a shared `State`, `Nodes` (functions that do work), and `Edges` (functions that decide what runs next). It is the primary framework for stateful, controllable agents because it exposes cycles, conditional branching, persistence, and human-in-the-loop as first-class primitives.

**How it works under the hood.** You define a `State` schema (a `TypedDict`, `dataclass`, or Pydantic model) whose keys each have a *reducer* — the default reducer overwrites, while `add_messages`/`operator.add` accumulate. You add nodes with `builder.add_node(...)` and wire them with `add_edge` (static) or `add_conditional_edges(node, routing_function)` (dynamic). Execution runs as message-passing "super-steps" (Pregel-style): a node becomes active when it receives state on an incoming edge, runs, emits an update applied by the reducer, and passes messages onward; the graph halts when no messages are in transit. Conditional edges that route back to an earlier node create **cycles** (the reasoning loop of an agent). You must call `.compile()` before running, which is also where you attach a `checkpointer` for persistence.

**Where it appears.** `from langgraph.graph import StateGraph, START, END`; `builder.compile(checkpointer=InMemorySaver())`; `graph.invoke(inputs, config={"recursion_limit": 25})`; `interrupt("Do you approve?")` + `Command(resume=...)` for HITL. On Databricks it appears wrapped inside an MLflow `ResponsesAgent` (`mlflow.langchain.autolog()`), and `databricks_langchain.ChatDatabricks` supplies the model.

### LangChain — the component library and lightweight harness (SECONDARY)

**What it is.** LangChain is the standard-interface component library — chat models, prompts, tools, retrievers, output parsers — plus LCEL (the pipe-composable expression language) and `create_agent`, a minimal, configurable agent harness. It is secondary: you use its components *inside* LangGraph nodes, or use a plain chain when a single deterministic pass is all you need.

**How it works under the hood.** LCEL lets you compose `Runnable` objects with the `|` operator (`prompt | model | parser`), giving you batching, async, and streaming for free because each function becomes a `Runnable`. `create_agent(model, tools, system_prompt)` builds a tool-calling agent loop; critically, **LangChain's agents are themselves built on top of LangGraph**, so they inherit durable execution, persistence, and HITL. This is the concrete proof that these are layers, not rivals.

**Where it appears.** `from langchain.agents import create_agent`; `from langchain_core.tools import tool`; `from databricks_langchain import ChatDatabricks`; the `|` LCEL pipe. In a LangGraph app, LangChain shows up as the model call and retriever *inside* `node_generate` and `node_retrieve`.

### CrewAI — role-based multi-agent (BREADTH / COMPARISON)

**What it is.** CrewAI is a higher-abstraction framework where a *Crew* is a group of role-defined `Agent`s (each with a `role`, `goal`, `backstory`) executing a list of `Task`s under a `process`. You trade fine-grained control for speed-to-build and an intuitive "team of specialists" mental model.

**How it works under the hood.** You declare agents and tasks (in code or `crew.jsonc`), then `crew.kickoff(inputs=...)` runs them. The `process` attribute controls flow: `sequential` runs tasks in listed order passing context forward; `hierarchical` inserts a manager agent (requiring `manager_llm` or `manager_agent`) that delegates and validates. CrewAI also supports `checkpoint=True` / `CheckpointConfig` to resume long runs and `crewai replay -t <task_id>` to replay from a task. The abstraction hides the graph, so you cannot easily inject arbitrary conditional routing the way LangGraph allows.

**Where it appears.** `from crewai import Agent, Crew, Task, Process`; `Crew(agents=[...], tasks=[...], process=Process.sequential)`. On Databricks, a CrewAI crew is wrapped the same way as anything else — inside an MLflow `ResponsesAgent`/`ChatAgent` for deployment. *(LlamaIndex fills the same "breadth" slot for data/RAG-centric pipelines and is likewise supported by the Agent Framework.)*

### Databricks Agent Framework — framework-agnostic authoring + MLflow deployment

**What it is.** The Databricks Agent Framework is a set of tools for authoring, deploying, and monitoring agents that is **framework-agnostic**: it supports LangGraph, LangChain, LlamaIndex, and OpenAI (Agents SDK) equally. The unifying contract is not a framework — it is an MLflow interface.

**How it works under the hood.** You wrap whatever agent you built in MLflow's `ResponsesAgent` (recommended) or `ChatAgent` interface by subclassing it and implementing `predict`/`predict_stream`. You log it with the **models-from-code** pattern (`mlflow.pyfunc.log_model(python_model="agent.py", name="agent")`), which auto-infers a signature, appends `{"task": "agent/v1/responses"}` metadata, and produces an artifact compatible with AI Playground, Agent Evaluation, and Agent Monitoring. Registering to Unity Catalog and creating a serving endpoint deploys it to **Model Serving**. Because the wrapper is the contract, swapping LangGraph for CrewAI changes only the code *inside* `predict`.

**Where it appears.** `from mlflow.pyfunc import ResponsesAgent`; `mlflow.pyfunc.log_model(python_model="agent.py", ...)`; `mlflow.register_model(...)`; `get_deploy_client("databricks").create_endpoint(...)`. The template `agent-openai-agents-sdk` and the LangGraph `create_react_agent` examples in the docs show the same wrapper over different frameworks.

### Key Parameters / Configuration Knobs

| Parameter | What it controls | Decision rule |
|---|---|---|
| `recursion_limit` (LangGraph, on `config`) | Max super-steps before `GraphRecursionError` | Raise above the default when a legitimate agent loop needs many turns; keep it low as a safety net for cyclic graphs that could run away. Pass it as a top-level `config` key, not inside `configurable`. |
| `checkpointer` + `thread_id` (LangGraph) | Persistence of state across turns/resumes | Set a checkpointer (e.g. `InMemorySaver`, or a durable store in prod) and a stable `thread_id` when you need multi-turn memory, HITL pauses, or crash recovery; omit for stateless single-shot graphs. |
| `interrupt()` / `Command(resume=...)` (LangGraph) | Human-in-the-loop pause/resume | Use when a human must approve or edit before the graph continues; requires a checkpointer. Keep side effects before `interrupt` idempotent because the node re-runs on resume. |
| `process` = `sequential` \| `hierarchical` (CrewAI) | How crew tasks are coordinated | Use `sequential` for a fixed pipeline of tasks; use `hierarchical` (with `manager_llm`/`manager_agent`) when a manager should dynamically delegate and validate. |
| `python_model` (MLflow `log_model`) | Models-from-code deployment artifact | Point it at your `agent.py` that calls `set_model(agent)`; this is the framework-agnostic path that makes the deployment identical regardless of LangGraph/LangChain/CrewAI/OpenAI underneath. |

### Worked Example: Requirement → Decision

**Given:** You must build a customer-support agent that retrieves policy docs, drafts an answer, and — for refund requests over $500 — pauses for a human agent to approve before replying. It will be deployed on Databricks Model Serving and monitored.

- **Step 1 — Identify the goal:** A stateful agent with a conditional branch (refund threshold) and a human approval gate, deployable and observable.
- **Step 2 — Define inputs:** Conversation `messages`, a retriever over the policy knowledge base, a refund-amount field parsed from the request.
- **Step 3 — Define outputs:** A `ResponsesAgentResponse` (assistant message), with intermediate tool/approval steps traced.
- **Step 4 — Apply constraints:** Must support human-in-the-loop pause/resume, must be resumable if the human takes hours, must deploy through MLflow to Model Serving with monitoring.
- **Step 5 — Select the approach:** Build with **LangGraph** — its conditional edges express the refund branch, `interrupt()` + a `checkpointer`/`thread_id` express the approval gate and durable pause, and LangChain's `ChatDatabricks` + retriever live inside the nodes. Wrap the compiled graph in an MLflow `ResponsesAgent` and log via models-from-code. *Rationale vs alternatives:* a plain LangChain chain cannot pause-and-resume or loop conditionally; CrewAI's role abstraction hides exactly the conditional/HITL control this requirement demands. LangGraph is the only option giving explicit control over state and flow.

---

## Implementation

```python
# Scenario: A support agent must loop — retrieve, generate, and retry retrieval
# if the answer is ungrounded — which requires a cycle and a conditional edge.
# A plain chain can't loop, so we use a LangGraph StateGraph.
from typing import Annotated, TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from databricks_langchain import ChatDatabricks

class State(TypedDict):
    messages: Annotated[list, add_messages]  # reducer accumulates turns
    grounded: bool

llm = ChatDatabricks(endpoint="databricks-claude-sonnet-4-5")

def retrieve(state: State) -> dict:
    # (LangChain retriever would run here) — component inside a node
    return {"messages": [{"role": "system", "content": "context..."}]}

def generate(state: State) -> dict:
    resp = llm.invoke(state["messages"])
    return {"messages": [resp], "grounded": "context" in resp.content}

def should_retry(state: State) -> Literal["retrieve", "__end__"]:
    return "retrieve" if not state["grounded"] else END

builder = StateGraph(State)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_conditional_edges("generate", should_retry)  # cycle back or stop
graph = builder.compile(checkpointer=InMemorySaver())     # MUST compile
graph.invoke({"messages": [{"role": "user", "content": "refund policy?"}]},
             config={"recursion_limit": 10, "configurable": {"thread_id": "t1"}})
```

```python
# Scenario: Deploy ANY of the above frameworks to Databricks Model Serving.
# The framework-agnostic contract is an MLflow ResponsesAgent + models-from-code.
# agent.py
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (ResponsesAgentRequest, ResponsesAgentResponse,
                                     output_to_responses_items_stream, to_chat_completions_input)
from mlflow.models import set_model
import mlflow

class LangGraphResponsesAgent(ResponsesAgent):
    def __init__(self, agent): self.agent = agent
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        outputs = [e.item for e in self.predict_stream(request)
                   if e.type == "response.output_item.done"]
        return ResponsesAgentResponse(output=outputs)
    def predict_stream(self, request):
        cc = to_chat_completions_input([i.model_dump() for i in request.input])
        for _, events in self.agent.stream({"messages": cc}, stream_mode=["updates"]):
            for node in events.values():
                yield from output_to_responses_items_stream(node["messages"])

mlflow.langchain.autolog()
set_model(LangGraphResponsesAgent(graph))   # `graph` = the compiled LangGraph above
# then: mlflow.pyfunc.log_model(python_model="agent.py", name="agent")
```

```python
# Anti-pattern: reaching for a multi-agent CrewAI crew when the task is one
# deterministic pass — it adds a manager/role LLM overhead, extra latency, and
# hides the flow you didn't need to hide, all to answer a single question.
from crewai import Agent, Crew, Task, Process
crew = Crew(agents=[Agent(role="Answerer", goal="answer", backstory="...")],
            tasks=[Task(description="Answer {q}", expected_output="answer", agent=...)],
            process=Process.hierarchical, manager_llm="openai/gpt-4o")  # overkill

# Correct approach: one deterministic LCEL chain — cheaper, faster, fully traceable.
from databricks_langchain import ChatDatabricks
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
chain = (ChatPromptTemplate.from_template("Answer concisely: {q}")
         | ChatDatabricks(endpoint="databricks-claude-sonnet-4-5")
         | StrOutputParser())
chain.invoke({"q": "What is the refund window?"})
# What breaks in the anti-pattern: a hierarchical crew spins up a manager loop and
# role agents for a task with no collaboration, multiplying token cost and adding
# non-deterministic delegation where a single pass was sufficient.
```

---

## Common Pitfalls & Misconceptions

- **"I have to pick one framework."** — Beginners treat LangGraph, LangChain, and CrewAI as competing products to choose between. They are layers: LangChain components run *inside* LangGraph nodes, and LangChain's own agents are built *on* LangGraph — compose them rather than choosing blindly.
- **"CrewAI/multi-agent is more powerful, so use it by default."** — More agents feels more capable, so people over-reach for crews. Multi-agent adds coordination cost and reduces control; use it only when the work genuinely decomposes into collaborating roles, otherwise a single chain or graph is cheaper and more controllable.
- **"I'll just write my own `while` loop for the agent cycle."** — Hand-rolling a loop seems simpler than learning a graph. But you then re-implement state persistence, retry, HITL pauses, and step limits by hand; LangGraph's `StateGraph` + `checkpointer` + `recursion_limit` gives you a controllable, resumable cycle for free.
- **"Deploying means rewriting for Databricks."** — People assume Databricks needs a special agent format. The Agent Framework is framework-agnostic — you wrap your existing agent in an MLflow `ResponsesAgent`/`ChatAgent` and log via models-from-code; the framework underneath is irrelevant to the deployment contract.
- **"Forgetting to `.compile()` the graph."** — New LangGraph users try to `invoke` a builder. A `StateGraph` must be compiled first (`builder.compile(...)`), which validates structure and attaches the checkpointer — the builder itself is not runnable.

---

## Key Definitions

| Term | Definition |
|---|---|
| LangGraph | A low-level orchestration framework that models agent workflows as a graph of `State`, nodes, and edges, supporting cycles, persistence, and human-in-the-loop. |
| LangChain | A component library (models, prompts, tools, retrievers, output parsers, LCEL) and a lightweight `create_agent` harness; its agents are built on LangGraph. |
| CrewAI | A higher-abstraction, role-based multi-agent framework where a Crew of role-defined agents executes tasks under a sequential or hierarchical process. |
| State (LangGraph) | The shared, schema-typed data structure (with per-key reducers) that every node reads and updates as the graph runs. |
| Conditional edge | An edge whose `routing_function` inspects state at runtime to decide the next node(s); routing back to an earlier node forms a cycle. |
| Checkpointer | The LangGraph component that persists state at super-step boundaries, enabling multi-turn memory, HITL pauses, and resume. |
| Databricks Agent Framework | A framework-agnostic toolset (supports LangGraph, LangChain, LlamaIndex, OpenAI) for authoring, deploying, and monitoring agents on Databricks. |
| ResponsesAgent | The recommended MLflow interface that wraps any agent for logging, tracing, evaluation, and serving on Databricks; compatible with the OpenAI Responses schema. |
| Models-from-code | The MLflow logging pattern (`log_model(python_model="agent.py", ...)`) that packages agent code as a deployable artifact independent of framework. |

---

## Summary / Quick Recall

- LangGraph = control (state, cycles, conditional edges, checkpointer, HITL) — the primary choice for stateful agents.
- LangChain = components + LCEL + a light harness; its agents are built on LangGraph, proving these are layers not rivals.
- CrewAI = fast role-based multi-agent; trade control for speed, choose `sequential` vs `hierarchical` process.
- LlamaIndex fills the same breadth slot for data/RAG-centric pipelines; also Agent-Framework-supported.
- Databricks Agent Framework is framework-agnostic — the deployment contract is MLflow `ResponsesAgent`/`ChatAgent`, not any one framework.
- Deploy via models-from-code (`log_model(python_model="agent.py")`) → register to Unity Catalog → Model Serving.
- Decide by control vs speed-to-build vs multi-agent: LangGraph → plain LangChain chain → CrewAI.

---

## Self-Check Questions

1. Which statement best describes how LangGraph and LangChain relate?

   <details><summary>Answer</summary>

   **Correct:** LangChain is a component library (and light harness) whose components run inside LangGraph nodes, and LangChain's own agents are built on top of LangGraph — they are layers, not competitors. The most tempting wrong answer is "they are rival frameworks you choose between," which is false: LangChain explicitly builds its `create_agent` agents on LangGraph's durable execution and persistence. "LangGraph replaced LangChain" is also wrong — both are maintained and used together.

   </details>

2. You must build an agent that pauses for human approval before sending a high-value refund, then resumes hours later. Which framework and mechanism fit best?

   <details><summary>Answer</summary>

   **Correct:** LangGraph with `interrupt()` plus a `checkpointer` and stable `thread_id` — this pauses the graph durably and resumes with `Command(resume=...)`. A plain LangChain chain is wrong because it runs to completion in one pass and cannot pause/resume with persisted state. CrewAI's role abstraction does not expose a clean per-step human-approval gate the way LangGraph's `interrupt` does.

   </details>

3. You need to answer a single question by retrieving one document and formatting the result — no loops, no branching. What is the most appropriate build?

   <details><summary>Answer</summary>

   **Correct:** A plain LangChain LCEL chain (`prompt | model | parser`) — it is one deterministic pass, cheap and fully traceable. Using a LangGraph `StateGraph` is unnecessary overhead for no cycle or conditional routing, and a CrewAI hierarchical crew is the worst fit because it spins up a manager and role agents (extra latency and token cost) for work that needs no collaboration.

   </details>

4. **Which TWO** statements about deploying agents on Databricks are correct?
   - A. The Agent Framework only supports agents written in LangGraph.
   - B. You wrap the agent in an MLflow `ResponsesAgent` (or `ChatAgent`) interface for logging, tracing, and serving.
   - C. You log the agent with the models-from-code pattern (`log_model(python_model="agent.py", ...)`).
   - D. Switching from LangGraph to CrewAI requires a different Databricks deployment path.
   - E. Personal access tokens are the required way to authenticate queries to a deployed agent.

   <details><summary>Answer</summary>

   **Correct: B and C.** The Agent Framework is framework-agnostic, so the deployment contract is the MLflow `ResponsesAgent`/`ChatAgent` wrapper (B) logged via models-from-code (C) — this is identical regardless of the underlying framework. A is wrong because LangGraph, LangChain, LlamaIndex, and OpenAI are all supported. D is the most tempting distractor but wrong: because the wrapper is the contract, only the code *inside* `predict` changes when you swap frameworks. E is wrong — for agents on Databricks Apps, OAuth tokens are used, not PATs.

   </details>

5. A team defaults to a CrewAI hierarchical crew for every agent "because multi-agent is more powerful." When is this the wrong trade-off, and what is the correct mental model?

   <details><summary>Answer</summary>

   **Correct:** It is wrong whenever the task does not genuinely decompose into collaborating roles — e.g. a single deterministic retrieve-and-answer, or a task needing precise conditional/HITL control. A hierarchical crew adds a manager LLM loop, delegation non-determinism, and token/latency cost. The correct mental model: pick the *least* orchestration that meets the requirement — a plain LangChain chain for one pass, LangGraph when you need explicit state/cycles/HITL control, and CrewAI only when role-based collaboration is the actual shape of the work. The tempting error is equating "more agents" with "more capable"; more agents means more coordination cost and less control.

   </details>

---

## Further Reading

- [LangGraph Graph API overview](https://docs.langchain.com/oss/python/langgraph/graph-api) — *verified 2026-07-11* — Official reference for `State`, nodes, edges, conditional edges, `Command`, `recursion_limit`, and compilation.
- [LangChain overview (create_agent, built on LangGraph)](https://docs.langchain.com/oss/python/langchain/overview) — *verified 2026-07-11* — Official explanation of the component library, `create_agent` harness, and how LangChain agents sit on LangGraph.
- [CrewAI — Crews](https://docs.crewai.com/concepts/crews) — *verified 2026-07-11* — Official reference for crews, agents, tasks, sequential vs hierarchical process, and checkpointing.
- [Databricks — Author an AI agent and deploy it](https://docs.databricks.com/aws/en/agents/agent-framework/author-agent) — *verified 2026-07-11* — Official Agent Framework guide showing framework-agnostic authoring (OpenAI SDK, LangGraph) and deployment.
- [MLflow — ResponsesAgent for Model Serving](https://mlflow.org/docs/latest/genai/serving/responses-agent) — *verified 2026-07-11* — Official MLflow docs for the `ResponsesAgent` interface, wrapping a LangGraph agent, and models-from-code logging.
