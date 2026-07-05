# Agent Bricks and Databricks-Native Agents

**Section:** Application Development | **Module:** Chains, Agents, and Frameworks | **Est. time:** 2 hrs
**Exam mapping:** Databricks GenAI Engineer Associate — Application Development domain (~30%); Databricks-native agent deployment patterns, Knowledge Assistant, and `ResponsesAgent` are exam-relevant

## Learning Objectives

By the end of this chapter, you will be able to:

- Create and configure a Knowledge Assistant agent (no-code) via the Databricks workspace UI
- Explain when to use Knowledge Assistant vs. a custom LangGraph agent
- Wrap a LangGraph agent with `mlflow.pyfunc.ResponsesAgent` for Databricks deployment compatibility
- Describe what `ResponsesAgent` provides: automatic tracing, Playground compatibility, streaming support
- Query a deployed agent endpoint via Python and the REST API
- Explain the Supervisor Agent pattern in Databricks (multi-agent via the UI)
- Describe the role of Databricks Apps as the hosting layer for custom agent chat UIs

## Core Concepts

### The Databricks Agent Deployment Stack

Databricks (2026-07-01) provides three levels of agent building:

| Level | Product | When to use |
|-------|---------|-------------|
| **No-code** | Knowledge Assistant | Domain Q&A over documents; no custom logic needed |
| **Low-code** | Supervisor Agent (UI) | Multi-agent orchestration over Databricks tools (Genie Spaces, UC functions, AI Search) |
| **Full-code** | Custom agent + `ResponsesAgent` + Databricks Apps | Any complex agent requiring custom LangGraph logic, custom tools, or custom UI |

**The deployment target for full-code agents is Databricks Apps** — a hosting environment that runs your agent code alongside a built-in chat UI. MLflow `ResponsesAgent` is the interface that makes any agent compatible with Databricks features.

---

### Knowledge Assistant — No-Code RAG Chatbot

**Knowledge Assistant** is a managed Databricks product that creates a production-quality
RAG chatbot from your documents with no code. Based on the Databricks docs (updated 2026-06-30):

**What it does:**
- Takes files (PDF, Markdown, DOCX, TXT, PPTX) or an AI Search index as input
- Automatically parses, embeds, and indexes the documents
- Creates a chatbot that answers questions with citations
- Supports iterative quality improvement via natural language guidelines

**How to create one (workspace UI):**
1. **Agents** (left sidebar) → **Create Agent** → **Knowledge Assistant**
2. Configure name and description
3. Add knowledge source(s):
   - Type: Files in a Volume | Files in a UC Table | AI Search Index
   - For volumes: path to a UC Volume directory; supports up to 10 sources
4. Add optional instructions (natural language rules for how the agent responds)
5. Click **Create Agent** — ingestion runs (takes up to a few hours for large corpora)

**Key limitations (from Databricks docs 2026-06-30):**
- Files > 50 MB are skipped during ingestion
- PDF/DOCX/PPT files > 500 pages are skipped
- Files with names starting with `_` or `.` are skipped
- AI Search indexes must use one of: `databricks-gte-large-en`, `databricks-bge-large-en`, or `databricks-qwen3-embedding-0-6b`

**Quality improvement workflow:**
1. Test the agent in the Build tab
2. Add example questions in the Examples tab
3. Add guidelines (natural language instructions for how to answer specific question types)
4. Guidelines take effect immediately — no re-training required

**Querying the Knowledge Assistant endpoint via Python:**
```python
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# Query the endpoint created by Knowledge Assistant
response = w.serving_endpoints.query(
    name="<knowledge-assistant-endpoint-name>",
    inputs={
        "messages": [
            {"role": "user", "content": "What is the token limit for BGE embeddings?"}
        ]
    }
)
print(response.predictions[0]["candidates"][0]["message"]["content"])
```

---

### `mlflow.pyfunc.ResponsesAgent` — The Deployment Wrapper

When building custom agents (LangGraph, OpenAI Agents SDK, or any Python framework),
Databricks recommends wrapping the agent in the **`ResponsesAgent`** interface from MLflow.

From Databricks docs (2026-07-01):
> "Databricks recommends MLflow `ResponsesAgent` to build agents. `ResponsesAgent` lets you
> build agents with any third-party framework, then integrate it with Databricks AI features
> for robust logging, tracing, evaluation, deployment, and monitoring capabilities."

**What `ResponsesAgent` provides:**
- **Automatic MLflow tracing** — every invocation is traced and logged automatically
- **AI Playground compatibility** — the agent can be tested in the Playground UI
- **Agent Evaluation integration** — works with MLflow's LLM judge evaluation
- **Streaming support** — emit delta events for real-time token streaming
- **Multi-agent support** — works as a sub-agent in Supervisor Agent patterns
- **OpenAI Responses schema compatibility** — standard interface across all agent types

**Implementing `ResponsesAgent` around a LangGraph agent:**

```python
import mlflow
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.agent import (
    ChatAgentRequest,
    ChatAgentResponse,
    ChatAgentMessage,
)
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from databricks_langchain import ChatDatabricks
from langchain_core.tools import tool
from typing import Generator

# ── Build the underlying LangGraph agent ─────────────────────────────────────
@tool
def search_docs(query: str) -> str:
    """Search the documentation knowledge base for relevant information."""
    # In production: call AI Search index
    return f"Found documentation relevant to: {query}"

llm = ChatDatabricks(endpoint="databricks-meta-llama-3-1-70b-instruct")
_langgraph_agent = create_react_agent(model=llm, tools=[search_docs])

# ── Wrap with ResponsesAgent ──────────────────────────────────────────────────
class MyRAGAgent(ResponsesAgent):
    """
    A RAG agent wrapped with ResponsesAgent for Databricks deployment.
    Inheriting from ResponsesAgent enables automatic MLflow tracing,
    Playground compatibility, and streaming.
    """

    def predict(
        self,
        context,
        request: ChatAgentRequest,
        params=None,
    ) -> ChatAgentResponse:
        """Handle a non-streaming request."""
        # Convert request messages to LangChain format
        messages = [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages
        ]
        result = _langgraph_agent.invoke({"messages": messages})
        last_message = result["messages"][-1]
        return ChatAgentResponse(
            messages=[
                ChatAgentMessage(
                    role="assistant",
                    content=last_message.content
                )
            ]
        )

    def predict_stream(
        self,
        context,
        request: ChatAgentRequest,
        params=None,
    ) -> Generator[ChatAgentResponse, None, None]:
        """Handle a streaming request — yield delta events."""
        messages = [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages
        ]
        # Stream token-by-token using LangGraph's stream_events
        for chunk in _langgraph_agent.stream({"messages": messages}):
            # Extract content chunks from agent node output
            if "agent" in chunk:
                agent_msgs = chunk["agent"].get("messages", [])
                for msg in agent_msgs:
                    if hasattr(msg, "content") and msg.content:
                        yield ChatAgentResponse(
                            messages=[ChatAgentMessage(
                                role="assistant",
                                content=msg.content
                            )]
                        )

# ── Log to MLflow (for registration and deployment) ───────────────────────────
agent = MyRAGAgent()

with mlflow.start_run():
    model_info = mlflow.pyfunc.log_model(
        artifact_path="agent",
        python_model=agent,
        # Signature is automatically inferred for ResponsesAgent
    )
    print(f"Logged model: {model_info.model_uri}")
```

**Registering in Unity Catalog and deploying:**
```python
# Register the logged model to Unity Catalog
import mlflow

mlflow.set_registry_uri("databricks-uc")
model_version = mlflow.register_model(
    model_uri=model_info.model_uri,
    name="main.rag_agents.my_rag_agent"
)

# Deploy to Model Serving (covered in Section 04)
# For now, test locally:
loaded_agent = mlflow.pyfunc.load_model(model_info.model_uri)
result = loaded_agent.predict({
    "messages": [{"role": "user", "content": "What is AI Search?"}]
})
print(result)
```

---

### Databricks Apps — Hosting Agents with Built-In Chat UI

**Databricks Apps** is the hosting environment that runs your custom agent code and serves
a built-in chat UI. From Databricks docs (2026-07-01):

> "Every conversational agent template includes a built-in chat UI with no additional setup
> required. The chat UI supports streaming responses, markdown rendering, Databricks
> authentication, and optional persistent chat history."

**Architecture:**
```
Databricks Apps runtime
  ├── Chat UI (React frontend, auto-provided)
  ├── MLflow AgentServer (async FastAPI, handles /responses endpoint)
  └── Your agent code (ResponsesAgent wrapped LangGraph agent)
```

**Using the OpenAI Agents SDK vs. LangGraph in Databricks Apps:**

Databricks shows two framework options in their 2026 docs:

**LangGraph (this course's focus):**
```python
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from databricks_langchain import ChatDatabricks

@tool
def get_current_time() -> str:
    """Get the current date and time."""
    from datetime import datetime
    return datetime.now().isoformat()

agent = create_react_agent(
    ChatDatabricks(endpoint="databricks-claude-sonnet-4-5"),
    tools=[get_current_time],
)
```

**OpenAI Agents SDK (alternative):**
```python
from agents import Agent, function_tool

@function_tool
def get_current_time() -> str:
    """Get the current date and time."""
    from datetime import datetime
    return datetime.now().isoformat()

agent = Agent(
    name="My agent",
    instructions="You are a helpful assistant.",
    model="databricks-claude-sonnet-4-5",
    tools=[get_current_time],
)
```

Both frameworks are supported. The exam tests the LangGraph path.

---

### Supervisor Agent — Low-Code Multi-Agent UI

The **Supervisor Agent** is a Databricks-managed multi-agent pattern built from the UI:
> "Build a supervisor agent that orchestrates Genie Spaces, agent endpoints, Unity Catalog
> functions, MCP servers, and custom agents."

For the exam, know that:
- It is the low-code alternative to building a custom multi-agent `StateGraph`
- It can orchestrate any combination of: AI Search endpoints, Genie Spaces (natural language SQL), custom agent endpoints, UC functions, and MCP servers
- Custom agents must be deployed as agent endpoints (via `ResponsesAgent` + Model Serving) before they can be used in a Supervisor Agent

---

### Unity AI Gateway for Agents (Beta, 2026)

From Databricks docs (2026-07-01), **Unity AI Gateway** is a governance layer for LLM traffic:

```
Agent code
  → Unity AI Gateway endpoint  ← service policies (guardrails, PII, spend caps)
  → Foundation Model API / external LLM
```

**What it provides (Beta):**
- Route agent LLM calls through a governed endpoint (swap models without code changes)
- Per-user spend caps (hard limits stop requests when budget exceeded)
- Built-in service policies: PII detection, prompt injection defense, unsafe content filtering
- Unified audit log of all LLM interactions

**To use in a LangGraph agent:**
```python
from databricks_langchain import ChatDatabricks

# Point to a Unity AI Gateway endpoint instead of a Foundation Model API endpoint directly
llm = ChatDatabricks(
    model="<ai-gateway-endpoint-name>",
    use_ai_gateway=True
)
```

For the exam: know what Unity AI Gateway is (governance proxy for LLM traffic) and how to
enable it (`use_ai_gateway=True` on `ChatDatabricks`). Deep policy configuration is Section 05.

---

## Deep Dive / Advanced Topics

### Choosing Between Knowledge Assistant and Custom Agent

| Dimension | Knowledge Assistant | Custom LangGraph Agent |
|-----------|--------------------|-----------------------|
| Development speed | Minutes | Hours–days |
| Document Q&A quality | Excellent (Instructed Retriever approach) | Depends on implementation |
| Custom business logic | Not supported | Full control |
| Multi-source reasoning | Up to 10 knowledge sources | Unlimited (any tools) |
| Multi-step task execution | Not supported (Q&A only) | Full agent loop |
| Integration with other systems | Via MCP (limited) | Any tool, any API |
| Governance/customization | Via natural language guidelines | Code-level |

**Decision rule:** Start with Knowledge Assistant for domain Q&A use cases. If you need custom routing logic, multi-step tool use, custom UI behavior, or integration with non-document data sources, build a custom LangGraph agent.

### `ResponsesAgent` vs. `ChatModel` MLflow Interface

MLflow offers multiple pyfunc interfaces for serving LLM applications:

| Interface | Use case |
|-----------|---------|
| `ChatModel` | Simple stateless LLM endpoints; single-turn Q&A |
| `ResponsesAgent` | Full agents with tool calls, streaming, multi-turn, HITL |
| `PythonModel` | General-purpose; no agent-specific features |

**Use `ResponsesAgent`** for any agent that:
- Calls tools
- Streams responses
- Needs MLflow tracing of intermediate steps
- Will be tested in AI Playground
- Will be evaluated with Agent Evaluation

## Worked Examples & Practice

### Testing a Knowledge Assistant Endpoint

After creating a Knowledge Assistant via the UI, query it programmatically:

```python
from databricks.sdk import WorkspaceClient
import json

w = WorkspaceClient()

def query_knowledge_assistant(endpoint_name: str, question: str) -> str:
    """Query a Knowledge Assistant endpoint and return the answer with citations."""
    response = w.serving_endpoints.query(
        name=endpoint_name,
        inputs={
            "messages": [
                {"role": "user", "content": question}
            ]
        }
    )
    result = response.predictions[0]
    answer = result["candidates"][0]["message"]["content"]
    return answer

# Test
endpoint_name = "knowledge-assistant-databricks-docs"
questions = [
    "What are the requirements for Knowledge Assistant?",
    "How does hybrid search work in AI Search?",
    "What is the maximum file size for Knowledge Assistant ingestion?"
]

for q in questions:
    answer = query_knowledge_assistant(endpoint_name, q)
    print(f"Q: {q}")
    print(f"A: {answer[:300]}...")
    print()
```

**Failure mode to observe:** Query a Knowledge Assistant with a question completely outside its knowledge source scope (e.g., ask about a competitor product when the sources only contain Databricks docs). The agent should respond with "I don't know" or a polite scope limitation — because Knowledge Assistant uses the Instructed Retriever approach which retrieves only from the configured sources. Compare this to a base LLM which would hallucinate an answer. This is the value of grounded RAG.

## Common Pitfalls & Misconceptions

- **Pitfall:** Using Knowledge Assistant for use cases that require multi-step tool use or custom logic → **Why it happens:** Knowledge Assistant is easy to set up → **Fix:** Knowledge Assistant is a Q&A tool, not a general-purpose agent. If the user's task requires calling an API, running a SQL query, or executing a sequence of steps, build a custom LangGraph agent.

- **Pitfall:** Deploying a LangGraph agent without wrapping it in `ResponsesAgent` and being unable to use it in AI Playground → **Why it happens:** The agent "works" when called directly → **Fix:** AI Playground, Agent Evaluation, and Supervisor Agent all require the `ResponsesAgent` interface. Wrapping is a one-time step that unlocks the full Databricks agent ecosystem.

- **Pitfall:** Trying to use an AI Search index with a non-supported embedding model in Knowledge Assistant → **Why it happens:** The index was created with a custom embedding model → **Fix:** Knowledge Assistant only supports three embedding models: `databricks-gte-large-en`, `databricks-bge-large-en`, or `databricks-qwen3-embedding-0-6b`. If your index uses a different model, either re-index with a supported model or build a custom LangGraph RAG agent instead.

- **Pitfall:** Adding knowledge sources to Knowledge Assistant and expecting immediate availability → **Why it happens:** Modern systems are often near-real-time → **Fix:** Knowledge Assistant ingestion can take **up to a few hours** for large corpora. After adding or updating files, you must also click **Sync** — incremental sync only processes newly added files, it does not automatically detect changes.

## Key Definitions

| Term | Definition |
|---|---|
| Knowledge Assistant | A Databricks no-code managed RAG chatbot that answers questions about documents with citations; configurable via the workspace UI |
| Agent Bricks | The Databricks product family for pre-built agent building blocks; includes Knowledge Assistant and Supervisor Agent |
| `ResponsesAgent` | An MLflow pyfunc interface (`mlflow.pyfunc.ResponsesAgent`) that wraps any agent framework to integrate with Databricks features: tracing, Playground, streaming, evaluation |
| Databricks Apps | The Databricks hosting environment for custom agent applications; provides a built-in chat UI, MLflow AgentServer, and deployment tooling |
| Supervisor Agent | A Databricks low-code UI tool for building multi-agent systems that orchestrate Genie Spaces, UC functions, agent endpoints, and MCP servers |
| Unity AI Gateway | A Databricks governance proxy for LLM traffic (Beta, 2026); provides guardrails, spend caps, rate limits, and audit logging for all LLM calls |
| `MLflow AgentServer` | An async FastAPI server embedded in Databricks Apps that handles the `/responses` endpoint for querying agents |
| Instructed Retriever | The Databricks Knowledge Assistant's retrieval approach; uses system-level reasoning to improve retrieval quality beyond traditional RAG |

## Summary / Quick Recall

- **Knowledge Assistant:** no-code RAG chatbot from docs; Q&A only; up to 10 sources; 50 MB file limit; 3 supported embedding models
- **ResponsesAgent:** MLflow wrapper for any agent framework; required for Playground, evaluation, and streaming; implement `predict()` and `predict_stream()`
- **Databricks Apps:** hosting environment for custom agents; built-in chat UI included; uses MLflow AgentServer
- **Supervisor Agent:** low-code multi-agent UI; orchestrates UC functions, Genie, AI Search, custom agent endpoints
- **Unity AI Gateway:** LLM governance proxy (Beta); enable with `use_ai_gateway=True` on `ChatDatabricks`
- Sync is required after adding/updating Knowledge Assistant files — not automatic
- Custom agents need `ResponsesAgent` wrapping + MLflow logging before they can be deployed

## Self-Check Questions

1. You need to build a Q&A chatbot over 500 internal support documents in DOCX format. The documents are already in a Unity Catalog Volume. What is the fastest path to a working agent?

<details>
<summary>Answer</summary>
Use Knowledge Assistant. In the Databricks workspace, go to Agents → Create Agent → Knowledge Assistant. Point it at the UC Volume containing the DOCX files. Knowledge Assistant automatically handles parsing, chunking, embedding, and indexing. It supports DOCX files (up to 500 pages per file) and creates a citation-aware chatbot endpoint within a few hours. No code required. This is faster than building a custom LangGraph RAG agent for a straightforward document Q&A use case.
</details>

2. You built a LangGraph agent and it works perfectly when called from Python. But when you try to test it in AI Playground, nothing appears. What step did you miss?

<details>
<summary>Answer</summary>
The agent was not wrapped with `mlflow.pyfunc.ResponsesAgent`. AI Playground expects the agent to implement the ResponsesAgent interface (specifically, a `predict()` or `predict_stream()` method returning `ChatAgentResponse`). Wrap your LangGraph agent in a class that inherits from `ResponsesAgent`, log it with `mlflow.pyfunc.log_model()`, register it in Unity Catalog, and deploy it to Model Serving. Once deployed as a `ResponsesAgent`, it is compatible with AI Playground.
</details>

3. What are the three embedding models supported by Databricks Knowledge Assistant for AI Search indexes? Why is this constraint important?

<details>
<summary>Answer</summary>
The three supported models are: `databricks-gte-large-en`, `databricks-bge-large-en`, and `databricks-qwen3-embedding-0-6b`. This constraint matters because Knowledge Assistant generates embeddings for user queries using the same model that was used to build the index. If it cannot use the same model (e.g., because a custom model was used), the query and document embeddings would be from different vector spaces and similarity search would produce nonsense results. Constraining to known supported models ensures embedding space compatibility.
</details>

4. Explain the relationship between `ResponsesAgent`, `MLflow AgentServer`, and Databricks Apps. How do they fit together?

<details>
<summary>Answer</summary>
They form a deployment stack: (1) `ResponsesAgent` is the Python class interface you implement — it wraps your agent logic and exposes `predict()` and `predict_stream()` methods. (2) `MLflow AgentServer` is an async FastAPI server that is embedded in Databricks Apps; it receives HTTP requests at `/responses` and calls your `ResponsesAgent.predict()` or `predict_stream()`. (3) Databricks Apps is the hosting environment that runs the combined server + your agent code + a built-in chat UI frontend, all under Databricks authentication and infrastructure. The flow: user sends message via chat UI → Apps routes to AgentServer → AgentServer calls ResponsesAgent.predict() → result streams back to UI.
</details>

5. A team wants to use Unity AI Gateway to enforce that agents cannot exceed $500/month in LLM spend. Is this possible? How?

<details>
<summary>Answer</summary>
Yes. Unity AI Gateway (Beta, 2026) supports hard spend caps — when the budget is reached, requests are stopped rather than just alerting after the fact. Configure the cap in the Unity AI Gateway budget settings for the gateway endpoint. In the agent code, point `ChatDatabricks` at the gateway endpoint (`model="<gateway-endpoint>"`, `use_ai_gateway=True`). All LLM calls route through the gateway, and the gateway enforces the cap. Note: Unity AI Gateway is in Beta as of 2026-07; verify the feature is enabled in your workspace admin settings before relying on it.
</details>

## Further Reading

- [Databricks — Knowledge Assistant docs](https://docs.databricks.com/en/agents/agent-bricks/knowledge-assistant.html) — full Knowledge Assistant reference including limitations, quality improvement, and SDK management (updated 2026-06-30)
- [Databricks — Author an AI agent and deploy it on Databricks Apps](https://docs.databricks.com/en/agents/agent-framework/author-agent.html) — step-by-step custom agent deployment guide with LangGraph and OpenAI SDK examples (updated 2026-07-01)
- [MLflow — ResponsesAgent reference](https://mlflow.org/docs/latest/genai/serving/responses-agent) — official interface specification and implementation patterns
- [Databricks — Unity AI Gateway](https://docs.databricks.com/en/ai-gateway/index.html) — governance, guardrails, spend management for LLM traffic (updated 2026-07-01)
- [Databricks — Build agents overview](https://docs.databricks.com/en/agents/agent-framework/build-agents.html) — comparison of all agent-building options on Databricks
