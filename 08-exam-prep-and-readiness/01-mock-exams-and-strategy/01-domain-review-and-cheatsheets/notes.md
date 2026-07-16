# Domain Review and Cheatsheets

**Section:** 08 Exam Prep and Readiness | **Module:** 01 Mock Exams and Strategy | **Est. time:** 3 hrs | **Exam mapping:** Supporting content — all 6 domains

---

## TL;DR

The Databricks Certified Generative AI Engineer Associate exam covers six domains weighted from 8% to 30%, with Application Development alone accounting for nearly one-third of the exam. Candidates who pass consistently know which APIs, configuration fields, and decision rules are tested in each domain — not just the concepts in isolation. This chapter consolidates every domain into a single rapid-recall reference: per-domain cheatsheets, a master API parameter table, and a prioritised study-plan framework. **The one thing to remember: Domain 3 (Application Development, 30%) is where exams are won or lost — allocate study time proportional to domain weight, not to your comfort level.**

---

## ELI5 — Explain It Like I'm 5

Imagine you are preparing for a really big test that covers six different subjects at school, and each subject is worth a different number of marks. One subject — let's call it Maths — is worth 30 marks all by itself, while another subject like PE is only worth 8 marks. A smart student does not spend the same amount of time on every subject; they spend the most time on Maths because getting that right moves the needle most. This chapter is your cheat-sheet folder: one page per subject, with only the things most likely to show up on the test. The biggest mistake students make is spending all their revision time on PE because it feels easier and more concrete — but that only earns 8 marks. Aim your study hours at the 30-mark subject first, then work down by marks.

---

## Learning Objectives

By the end of this chapter you will be able to:
- [ ] Identify the top 3–4 highest-yield topics and specific Databricks APIs for each of the six exam domains
- [ ] Apply a time-boxed, weight-proportional study plan given a practice-exam score breakdown and a fixed number of days remaining
- [ ] Recall the exam-critical parameter values for `mlflow.evaluate`, `databricks.agents.deploy`, Vector Search index creation, AI Gateway, and inference tables
- [ ] Distinguish cross-domain confusables (e.g., `chunk_relevance` vs `groundedness`; Delta Sync vs Direct Access index) under timed exam conditions
- [ ] Use spaced-repetition and active-recall workflows instead of passive re-reading to consolidate knowledge

---

## Visual Overview

### Domain Weight Bar Chart

```
Domain                          Weight   Bar
────────────────────────────────────────────────────────────
D3: Application Development      30%   ██████████████████████████████
D4: Assembling & Deploying        22%   ██████████████████████
D1: Design Applications           14%   ██████████████
D2: Data Preparation              14%   ██████████████
D6: Evaluation & Monitoring       12%   ████████████
D5: Governance                     8%   ████████
────────────────────────────────────────────────────────────
Total: 100%    45 questions / 90 min / $200
```

### Topic-to-API Mapping Tree

```
Exam Topics
├── D1 Design Applications (14%)
│   ├── RAG vs fine-tuning decision ──► compound AI system patterns
│   └── Chunking + prompt template design
├── D2 Data Preparation (14%)
│   ├── Chunking strategies (fixed/semantic/recursive)
│   ├── Embedding model selection
│   └── Vector Search index types ──► Delta Sync | Direct Access
├── D3 Application Development (30%)  ◄── highest weight
│   ├── Prompt engineering (few-shot, CoT, structured output)
│   ├── LangGraph StateGraph ──► nodes, edges, conditional_edges
│   ├── LangChain LCEL chains
│   ├── RAG chain assembly
│   ├── Tool/function calling & ReAct agents
│   └── mlflow.pyfunc.PythonModel + PyFunc wrapping
├── D4 Assembling & Deploying (22%)
│   ├── databricks.agents.deploy() ──► UC model registration
│   ├── Model Serving endpoint config
│   ├── AI Gateway setup + routing
│   └── CI/CD for prompt lifecycle
├── D5 Governance (8%)
│   ├── PII detection (Presidio) + AI Gateway guardrails
│   └── Unity Catalog column masking + licensing
└── D6 Evaluation & Monitoring (12%)
    ├── mlflow.evaluate() + model_type="databricks-agent"
    ├── 5 built-in judges + make_genai_metric()
    └── auto_capture_config + MLflow tracing
```

### Exam-Question Decision Flow

```
Exam question arrives
        │
        ▼
Which domain does this belong to?
        │
        ├──► D3 (30%)? ──► Think: LangGraph / LCEL / PyFunc / prompt engineering
        │
        ├──► D4 (22%)? ──► Think: agents.deploy() / Model Serving config / AI Gateway
        │
        ├──► D1/D2 (14% each)? ──► Think: RAG vs fine-tune / chunking / Vector Search
        │
        ├──► D6 (12%)? ──► Think: mlflow.evaluate / judges / inference tables
        │
        └──► D5 (8%)? ──► Think: PII / guardrails / UC column masking
```

---

## Key Concepts

### Domain 1 — Design Applications (14%)

**What does this domain test?** The ability to choose the right architecture and tooling for a GenAI application before writing code: when to use RAG vs fine-tuning, how to design compound AI system patterns, which tool to select, how to structure prompt templates, and how to pick a chunking strategy for different document types.

**Highest-yield topics:**

1. **RAG vs fine-tuning decision** — RAG is preferred when facts change frequently or must be cited; fine-tuning is preferred when a new skill (style, format, domain-specific reasoning) must be baked in. The exam tests this as a scenario: "customer wants real-time pricing answers" → RAG; "customer wants a model that always responds in legal language" → fine-tuning.
2. **Compound AI system patterns** — Chains, agents, and multi-agent pipelines. The Databricks Agent Framework is framework-agnostic (LangGraph, LangChain, LlamaIndex, OpenAI). Know when each pattern is appropriate.
3. **Chunking strategy for document types** — Fixed-size for uniform text; recursive/semantic for structured documents (PDFs, HTML) with meaningful headings; overlap prevents context loss at boundaries.
4. **Prompt template design** — System message establishes role and constraints; few-shot examples bootstrap behaviour; output format instructions (JSON schema) reduce post-processing failures.

**Where in Databricks:** AI Playground for prototype iteration; `PromptTemplate` and `ChatPromptTemplate` in LangChain; `mlflow.models.set_model()` for logging; Databricks Vector Search for the retrieval step.

---

### Domain 2 — Data Preparation (14%)

**What does this domain test?** How raw documents become retrieval-ready embeddings in a Vector Search index: chunking, embedding model selection, index type choice, metadata filtering, and document pre-processing.

**Highest-yield topics:**

1. **Chunking strategies** — Fixed-size splits at a character count (fast, predictable, may break sentences); recursive splits on `\n\n`, `\n`, ` ` in priority order (LangChain `RecursiveCharacterTextSplitter`); semantic/sentence-boundary splits for accuracy-sensitive applications. `chunk_overlap` prevents context loss at boundaries.
2. **Embedding model selection** — Open models (e.g., `gte-large`, `e5-large-v2`) for cost and data-sovereignty; proprietary models (OpenAI `text-embedding-ada-002`) for baseline quality. Match embedding model between index creation and query time — mismatches silently degrade recall.
3. **Delta Sync vs Direct Access index** — Delta Sync index automatically syncs when the source Delta table updates (managed pipeline); Direct Access index requires manual embedding upserts via API (used when embeddings are pre-computed externally). The exam tests which is appropriate for a given scenario.
4. **Metadata filtering** — At query time, pass `filters` to narrow the vector search to a subset of documents (e.g., `{"region": "EMEA"}`). Reduces latency and prevents leakage across tenant boundaries.

**Where in Databricks:** `VectorSearchClient` from `databricks.vector_search.client`; index created with `create_delta_sync_index()` or `create_direct_access_index()`; queried with `similarity_search(query_text=..., filters=...)`.

---

### Domain 3 — Application Development (30%)

**What does this domain test?** The hands-on ability to build working GenAI applications: prompt engineering, constructing LangGraph graphs, composing LCEL chains, assembling RAG chains, implementing tool calling and ReAct agents, wrapping models as PyFunc, and managing memory.

**Highest-yield topics:**

1. **Prompt engineering** — Few-shot examples in the prompt improve performance on structured tasks; Chain-of-Thought (CoT) via "think step by step" improves reasoning accuracy; structured output instructions (JSON schema or function signature) make response parsing deterministic.
2. **LangGraph StateGraph** — Define a `TypedDict` state schema; add nodes with `graph.add_node(name, fn)`; connect with `graph.add_edge(src, dst)`; branch with `graph.add_conditional_edges(src, routing_fn, {result: dst_node})`; compile with `graph.compile()`. The compiled graph is a `Runnable`.
3. **LangChain LCEL chains** — Compose runnables with `|` operator: `prompt | llm | output_parser`. Each link is a `Runnable`. LCEL enables streaming, batching, and async out of the box. Use `RunnablePassthrough` to pass inputs unchanged alongside transformations.
4. **RAG chain assembly** — Retriever → prompt with context → LLM → output parser. In LCEL: `({"context": retriever, "question": RunnablePassthrough()}) | prompt | llm | StrOutputParser()`.
5. **Tool/function calling + ReAct** — Define tools as Python functions decorated with `@tool`; bind to an LLM with `llm.bind_tools(tools)`; ReAct loop: reason → act (tool call) → observe (tool result) → repeat until final answer.
6. **PyFunc wrapping** — Subclass `mlflow.pyfunc.PythonModel`, implement `predict(self, context, model_input)`. Log with `mlflow.pyfunc.log_model(python_model=..., artifacts=..., pip_requirements=...)`. Required for custom pre/post-processing logic.
7. **Memory management** — `ConversationBufferMemory` for full history; `ConversationSummaryMemory` for long sessions; pass history as `messages` in LangGraph state. Exam tests when to truncate vs summarise.

**Where in Databricks:** LangGraph and LangChain used inside Databricks notebooks; MLflow autologging captures LangChain traces; `mlflow.langchain.log_model()` logs LCEL chains; `mlflow.pyfunc.log_model()` for custom Python models.

---

### Domain 4 — Assembling & Deploying Applications (22%)

**What does this domain test?** Taking a trained/logged model and getting it running in production: registering to Unity Catalog, deploying via `agents.deploy()` or Model Serving, configuring AI Gateway, wiring up MCP servers, and implementing CI/CD for prompt lifecycle management.

**Highest-yield topics:**

1. **`databricks.agents.deploy()`** — Registers the model in Unity Catalog, creates a Model Serving endpoint, provisions authentication, enables the Review App, enables real-time MLflow tracing, and enables inference tables — all in one call. Use `mlflow>=3.1.3` and `databricks-agents>=1.1.0`. Pass `scale_to_zero_enabled=True` to reduce costs for idle endpoints.
2. **MLflow model registration to Unity Catalog** — `mlflow.register_model(model_uri, "catalog.schema.model_name")`. Unity Catalog model registry enforces lineage, access control, and versioning. Required before `agents.deploy()`.
3. **Model Serving endpoint config** — `served_models` list with `model_name`, `model_version`, `workload_size` (`Small`/`Medium`/`Large`), and `scale_to_zero_enabled`. `auto_capture_config` enables inference tables.
4. **AI Gateway setup** — Routes traffic to multiple LLM providers; enforces rate limits, guardrails, and PII filtering at the gateway layer. Configures `route_type` (`llm/v1/completions`, `llm/v1/chat`, `llm/v1/embeddings`).
5. **CI/CD for prompt lifecycle** — Treat prompt templates as versioned code; use MLflow experiment tracking to compare prompt variants; gate promotion from staging to production on evaluation metric thresholds.

**Where in Databricks:** `from databricks import agents; agents.deploy(uc_model_name, model_version)`; MLflow UI → Models tab; Serving UI → Create endpoint; AI Gateway UI → Routes tab.

---

### Domain 5 — Governance (8%)

**What does this domain test?** How to prevent sensitive data from leaking through an AI system and how to enforce compliance controls: PII detection, AI Gateway guardrails, prompt injection mitigations, Unity Catalog policies, licensing, and regulatory mapping.

**Highest-yield topics:**

1. **PII detection and masking (Presidio)** — Use `AnalyzerEngine` to detect entity types (PERSON, EMAIL, PHONE_NUMBER, etc.); `AnonymizerEngine` to mask/replace. Apply as a pre-processing step before sending user input to the LLM, and as a post-processing step on LLM output.
2. **AI Gateway guardrails** — Configure `input_filter` and `output_filter` in the AI Gateway route config. Supports PII filtering, toxicity detection, and keyword blocklists. Applied at the gateway layer so guardrails are model-agnostic.
3. **Prompt injection mitigations** — Separate system instructions from user content using distinct message roles; validate/sanitise user input; use a fixed system prompt that explicitly instructs the model to ignore override attempts.
4. **Unity Catalog column masking** — `ALTER TABLE … ALTER COLUMN … SET MASK <masking_function>`. Returns masked values to users without the `UNMASK` privilege. Used to protect PII in source Delta tables used as RAG corpora.
5. **Licensing** — Open-weights models (Meta Llama, Mistral) have custom licenses that may restrict commercial use; proprietary APIs (OpenAI, Anthropic) have data-processing agreements. Know which tier is appropriate for regulated industries.

**Where in Databricks:** AI Gateway route config YAML; `presidio-analyzer` and `presidio-anonymizer` Python packages; Unity Catalog → Data Explorer → Column Masking; MLflow AI Gateway `databricks-uc` route type.

---

### Domain 6 — Evaluation & Monitoring (12%)

**What does this domain test?** How to measure whether a GenAI application is working correctly in both offline evaluation and online production monitoring: running `mlflow.evaluate()`, interpreting the five built-in LLM judges, creating custom scorers, capturing inference data, and using MLflow tracing.

**Highest-yield topics:**

1. **`mlflow.evaluate()` with `model_type="databricks-agent"`** — Runs built-in LLM judges over an evaluation dataset. Returns a DataFrame of per-row scores and a summary MLflow run. Requires at minimum `request` and `response` columns; `expected_response` and `retrieved_context` unlock additional judges.
2. **Five built-in judges** — `correctness` (requires `expected_response`): is the answer factually right? `groundedness`: is the answer supported by `retrieved_context`? `chunk_relevance` (maps to `retrieval/llm_judged/chunk_relevance/precision/average`): are retrieved chunks relevant to the query? `relevance_to_query` (formerly `answer_relevance`): does the response address the question? `safety`: is the output free of harmful content?
3. **Custom scorers with `make_genai_metric()`** — Define a custom LLM-judged metric by supplying a `definition`, `grading_prompt`, `examples`, and `model`. Returns a scorer callable passable to `mlflow.evaluate(extra_metrics=[...])`.
4. **Inference tables (`auto_capture_config`)** — Enable on a Model Serving endpoint to log every request/response to a Unity Catalog Delta table. Schema includes `databricks_request_id`, `request` (JSON), `response` (JSON), `status_code`, `execution_time_ms`. Databricks now recommends AI Gateway inference tables over legacy inference tables.
5. **MLflow tracing** — `@mlflow.trace` decorator or `mlflow.start_span()` context manager instruments each step of the agent pipeline. Traces visible in the MLflow Experiments UI and stored in the inference table for production monitoring.

**Where in Databricks:** `mlflow.evaluate(data=eval_df, model_type="databricks-agent")`; `databricks.agents.evals.judges.groundedness(...)` for direct judge invocation; AI Gateway inference table at `<catalog>.<schema>.<endpoint_name>_payload`.

---

## Key Parameters / Configuration Knobs

| API / Config | Key Parameter | Exam-relevant value or rule |
|---|---|---|
| `mlflow.evaluate()` | `model_type` | Set `"databricks-agent"` to activate all built-in LLM judges; omit and judges do not run |
| `mlflow.evaluate()` | `data` | DataFrame with `request`, `response`; add `expected_response` to unlock `correctness`; add `retrieved_context` to unlock `groundedness` and `chunk_relevance` |
| `mlflow.evaluate()` | `extra_metrics` | Pass list of `make_genai_metric()` outputs to add custom judges alongside built-ins |
| `databricks.agents.deploy()` | `uc_model_name` | Must be fully qualified Unity Catalog name: `catalog.schema.model_name` |
| `databricks.agents.deploy()` | `scale_to_zero_enabled` | `True` reduces cost for idle endpoints; increases cold-start latency — use `False` for latency-sensitive production |
| Model Serving `served_models` | `workload_size` | `Small` (4 CPUs), `Medium` (8 CPUs), `Large` (16 CPUs) — choose based on model memory footprint and concurrency target |
| Model Serving `auto_capture_config` | `catalog_name`, `schema_name`, `table_name_prefix` | Defines Unity Catalog location of inference table; default table name is `<endpoint_name>_payload` |
| Vector Search `create_delta_sync_index()` | `pipeline_type` | `TRIGGERED` (manual refresh) vs `CONTINUOUS` (auto-sync on Delta commit) — use `CONTINUOUS` when near-real-time retrieval freshness is required |
| Vector Search query | `filters` | Pass `{"column": "value"}` dict to restrict retrieval to a metadata subset; critical for multi-tenant or permission-scoped RAG |
| LangGraph `StateGraph` | `state_schema` | `TypedDict` class defining state keys; all nodes must read from and write to this schema — mismatched keys cause `KeyError` at runtime |
| LangGraph `graph.add_conditional_edges()` | routing function return value | Must return a key that exists in the `{return_value: destination_node}` mapping dict passed as the third argument |
| AI Gateway route config | `route_type` | `llm/v1/chat`, `llm/v1/completions`, `llm/v1/embeddings` — must match the endpoint's expected request format |
| AI Gateway route config | `rate_limits` | `calls` per `renewal_period` (`minute` or `day`) — set at the route level to enforce per-user or per-team quotas |
| `mlflow.pyfunc.log_model()` | `pip_requirements` | List of exact package versions; mismatch between training and serving environment is the most common deployment failure |

---

## Worked Example: Requirement → Decision

**Given:** A candidate has 14 days until the exam. They took a practice test and scored 60% overall. Their section-by-section breakdown shows: D1 65%, D2 70%, D3 45%, D4 55%, D5 80%, D6 60%.

**Step 1 — Identify the goal:** Raise the overall score to at least 70% (passing threshold) within 14 days, targeting the domains with the highest combination of weight and current weakness.

**Step 2 — Define inputs:**
- Score breakdown (D1: 65%, D2: 70%, D3: 45%, D4: 55%, D5: 80%, D6: 60%)
- Domain weights (D3: 30%, D4: 22%, D1: 14%, D2: 14%, D6: 12%, D5: 8%)
- Time available: 14 days, approximately 1–2 focused hours per day
- Budget constraint: $200 exam fee already paid — failing means repaying

**Step 3 — Define outputs:**
- A prioritised daily study schedule allocating time proportional to (weight × weakness score)
- A list of the 3–5 most impactful API calls and decision rules to drill

**Step 4 — Apply constraints:**
- Time-boxed: no more than 2 hours per day (retention degrades with longer cramming sessions)
- D5 at 80% is already near-passing — spending time on the 8% domain is low ROI
- Must use active recall (practice questions) not passive re-reading for efficient retention
- Focus on Databricks-specific API details, not general ML theory

**Step 5 — Select the remediation approach:**
Allocate study days in proportion to `weight × (100 − current_score)`: D3 gets the most days (30% × 55 gap = 16.5 points of potential gain), D4 second (22% × 45 gap = 9.9), D6 third (12% × 40 gap = 4.8). Spend 6 days on D3 (LangGraph, PyFunc, LCEL, ReAct), 4 days on D4 (agents.deploy, Model Serving config, AI Gateway), 2 days on D6 (mlflow.evaluate, judges, inference tables), 1 day on D1/D2 together, and 1 day for a full timed practice exam.

Rationale vs alternatives: Spending equal time on all domains wastes time on D5 (already 80%, 8% weight) and under-invests in D3 (45%, 30% weight). Spending all time on D3 ignores the 22% D4 gap. The weighted-gap formula ensures the greatest point gain per hour invested.

---

## Implementation

```python
# Scenario: Running a full agent evaluation with built-in judges and one custom scorer
# Goal: measure groundedness, correctness, and a custom "citation_quality" metric
# Constraint: evaluation dataset has both expected_response and retrieved_context columns

import mlflow
from mlflow.metrics.genai import make_genai_metric, EvaluationExample

# Define a custom scorer for citation quality
citation_quality = make_genai_metric(
    name="citation_quality",
    definition="Does the response cite specific facts from the retrieved context?",
    grading_prompt=(
        "Score 1 (yes) if the response explicitly references facts from the context. "
        "Score 0 (no) if the response is generic or ignores the context."
    ),
    examples=[
        EvaluationExample(
            input="What is the MLflow tracking URI?",
            output="The tracking URI is set via mlflow.set_tracking_uri(), as stated in the docs.",
            score=1,
            justification="Response cites a specific API from context.",
        ),
    ],
    model="endpoints:/databricks-claude-3-7-sonnet",
    greater_is_better=True,
)

# Evaluation dataset — must include request, response, expected_response, retrieved_context
eval_df = spark.table("prep.eval.rag_evaluation_set").toPandas()

with mlflow.start_run(run_name="domain_review_eval"):
    results = mlflow.evaluate(
        data=eval_df,
        model_type="databricks-agent",    # activates all built-in LLM judges
        extra_metrics=[citation_quality],  # adds custom scorer
    )

print(results.metrics)
# Key metrics to check:
# response/llm_judged/groundedness/rating/percentage  -> should be > 0.8
# response/llm_judged/correctness/rating/percentage   -> should be > 0.75
# retrieval/llm_judged/chunk_relevance/precision/average -> should be > 0.7
```

```python
# Scenario: Deploying a registered agent to a Model Serving endpoint
# Goal: zero-downtime deploy with inference table logging and scale-to-zero disabled
# Constraint: endpoint must be immediately available (no cold-start delay)

import mlflow
from databricks import agents

# Step 1: Log and register the agent to Unity Catalog
with mlflow.start_run():
    model_info = mlflow.pyfunc.log_model(
        artifact_path="agent",
        python_model=MyRAGAgent(),           # subclass of mlflow.pyfunc.PythonModel
        pip_requirements=["langchain==0.3.0", "databricks-vectorsearch==0.40"],
    )

registered = mlflow.register_model(
    model_uri=model_info.model_uri,
    name="prod_catalog.agents.rag_support_bot",  # Unity Catalog fully qualified name
)

# Step 2: Deploy — one call creates endpoint, auth, Review App, tracing, inference table
deployment = agents.deploy(
    model_name="prod_catalog.agents.rag_support_bot",
    model_version=registered.version,
    scale_to_zero_enabled=False,    # keep warm for latency-sensitive SLAs
)
print(deployment.query_endpoint)   # REST API URL for production traffic
```

```python
# Anti-pattern: enabling inference tables by hardcoding auto_capture_config
# in a low-level REST payload while bypassing agents.deploy()
# BREAKS: loses Review App, MLflow tracing integration, automatic auth provisioning,
#         and production monitoring — all of which agents.deploy() sets up automatically.

import requests

# WRONG: manual endpoint creation bypasses all agent framework integrations
payload = {
    "name": "my-agent-endpoint",
    "config": {
        "served_models": [{"model_name": "...", "model_version": "1"}],
        "auto_capture_config": {"catalog_name": "...", "schema_name": "..."},
    }
}
requests.post("https://<workspace>/api/2.0/serving-endpoints", json=payload)

# Correct approach: always use agents.deploy() for agent workloads
# It wraps the REST API AND wires up all Databricks GenAI integrations:
from databricks import agents
deployment = agents.deploy("catalog.schema.model", model_version=1)
# agents.deploy() internally calls the same REST endpoint BUT also:
#   - enables MLflow real-time tracing
#   - provisions the Review App
#   - provisions short-lived authentication credentials
#   - enables AI Gateway inference tables (not legacy inference tables)
```

---

## Common Pitfalls & Misconceptions

- **Spending study time by comfort, not by domain weight** — Candidates naturally gravitate toward topics they partially understand (D1/D2 chunking, D5 governance) because those feel more concrete. The correct mental model is to allocate study hours proportional to `weight × weakness`: D3 at 30% must dominate your plan even though LCEL and LangGraph feel abstract.

- **Confusing `groundedness` with `correctness`** — Candidates mix these two judges because both assess answer quality. `groundedness` asks "is every claim in the response supported by the retrieved context?" — it does not require an expected answer. `correctness` asks "does the response match the expected answer?" — it requires `expected_response` in the eval dataset. A response can be grounded (faithful to context) but still incorrect (context was wrong), and vice versa.

- **Confusing `chunk_relevance` with `relevance_to_query`** — Both use the word "relevance". `chunk_relevance` (a retrieval metric) evaluates whether the chunks returned by the retriever were relevant to the question — it is scored at the retrieval step. `relevance_to_query` (a response metric) evaluates whether the final LLM response addressed the user's question — it is scored at the generation step. Getting these backwards on a multi-select question is the primary failure mode.

- **Treating Delta Sync and Direct Access as interchangeable** — Candidates assume both index types behave identically and differ only in cost. The critical difference is ownership of the embedding pipeline: Delta Sync manages embeddings automatically using a Databricks-hosted embedding model; Direct Access requires you to compute and upsert embeddings via API. Choosing the wrong type for a scenario that requires external embeddings means the index will have no data.

- **Forgetting to match embedding models at index creation and query time** — The exam frequently tests what happens when you use `text-embedding-ada-002` to build the index but `gte-large` to embed the query. The cosine similarity scores are meaningless because the vectors live in different semantic spaces. The symptom is near-zero retrieval relevance with no error thrown.

- **Assuming `agents.deploy()` is optional for production** — Candidates believe they can deploy a registered MLflow model via any Model Serving endpoint and achieve the same result. `agents.deploy()` is not just a convenience wrapper: it provisions authentication credentials, wires up the Review App, enables AI Gateway inference tables, and configures MLflow real-time tracing — none of which happen with a plain `POST /api/2.0/serving-endpoints` call.

- **Using `ConversationBufferMemory` for long sessions without considering cost** — The buffer grows unboundedly and eventually exceeds the context window, causing silent truncation or errors. The correct mental model is: use `ConversationBufferMemory` only for sessions where message count is bounded and known; use `ConversationSummaryMemory` or explicit state truncation in LangGraph for open-ended conversations.

---

## Key Definitions

| Term | Definition |
|---|---|
| RAG (Retrieval-Augmented Generation) | Architecture where relevant documents are retrieved from an external store at inference time and injected into the prompt, allowing the LLM to answer with up-to-date or proprietary knowledge without fine-tuning |
| Fine-tuning | Supervised training of a pre-trained model on a domain-specific dataset to update its weights and bake in new skills or behaviour patterns |
| Delta Sync Index | A Vector Search index type that automatically keeps embeddings synchronised with the source Delta table using a Databricks-managed pipeline |
| Direct Access Index | A Vector Search index type where the caller is responsible for computing embeddings and upserting them via the Vector Search API |
| chunk_overlap | The number of characters or tokens shared between adjacent chunks to prevent context loss at chunk boundaries |
| LangGraph StateGraph | A directed graph where each node is a Python function that reads from and writes to a shared typed state dict; used to build stateful, multi-step agent loops |
| conditional_edges | A LangGraph construct that routes the graph to different next nodes based on the return value of a routing function applied to the current state |
| LCEL (LangChain Expression Language) | A composable interface where Runnables (prompts, LLMs, parsers) are chained with the `\|` operator to form pipelines with built-in streaming and batching |
| mlflow.pyfunc.PythonModel | The MLflow base class for custom Python models; subclassed to implement `predict()` for custom inference logic with arbitrary dependencies |
| PyFunc | Shorthand for MLflow's Python function model flavour; enables any Python code to be packaged as a standard MLflow model and served on Model Serving |
| agents.deploy() | A high-level Databricks SDK function that registers a Unity Catalog model as a production agent endpoint, wiring up auth, Review App, tracing, and inference tables in one call |
| auto_capture_config | The Model Serving endpoint configuration block that enables inference table logging; specifies the Unity Catalog catalog, schema, and table name prefix |
| Inference table | A Unity Catalog Delta table automatically populated with every request and response to a Model Serving endpoint; used for monitoring, debugging, and training data collection |
| groundedness | An MLflow LLM judge that evaluates whether every factual claim in the LLM response is supported by the retrieved context; does not require an expected answer |
| correctness | An MLflow LLM judge that compares the LLM response to a provided expected answer; requires `expected_response` in the evaluation dataset |
| chunk_relevance | An MLflow LLM judge (retrieval metric) that evaluates whether each retrieved chunk was relevant to the user query; scored as `precision/average` across all chunks |
| relevance_to_query | An MLflow LLM judge (response metric) that evaluates whether the final LLM response addressed the user's question; no expected answer required |
| make_genai_metric() | MLflow function for creating a custom LLM-judged evaluation metric by supplying a definition, grading prompt, and examples |
| AI Gateway | A Databricks-managed proxy layer that routes requests to multiple LLM providers, enforces rate limits, applies guardrails, and logs traffic to inference tables |
| PII (Personally Identifiable Information) | Data that can identify a specific individual; must be detected and masked before sending to external LLMs under GDPR/HIPAA and similar regulations |
| Presidio | An open-source Microsoft library used for PII detection (`AnalyzerEngine`) and anonymisation (`AnonymizerEngine`) in pre/post-processing pipelines |
| ReAct | A prompting and agent architecture pattern combining Reasoning (CoT-style thought) and Acting (tool calls) in alternating steps until the agent reaches a final answer |

---

## Summary / Quick Recall

- **D3 (Application Development, 30%) is the make-or-break domain** — LangGraph StateGraph, LCEL chains, PyFunc, and ReAct agents are the highest-density exam topics
- **`mlflow.evaluate()` requires `model_type="databricks-agent"`** to activate built-in LLM judges; without it, no judges run
- **`groundedness` ≠ `correctness`**: groundedness checks faithfulness to retrieved context (no expected answer needed); correctness checks agreement with a reference answer (requires `expected_response`)
- **`chunk_relevance` (retrieval metric) ≠ `relevance_to_query` (response metric)** — both use "relevance" but measure different pipeline stages
- **Delta Sync index = Databricks manages embeddings automatically; Direct Access index = you compute and upsert embeddings** — wrong choice for a scenario causes an empty index
- **`agents.deploy()` does more than deploy** — it also provisions auth, Review App, MLflow tracing, and AI Gateway inference tables; plain Model Serving API calls miss all of these
- **Allocate study time by `weight × weakness`**, not by comfort level — D5 at 8% should never dominate your revision schedule

---

## Self-Check Questions

1. An LLM judge returned a failing score for `groundedness` on a RAG application response. Which statement most accurately describes what this means?

   A. The response did not match the expected answer in the evaluation dataset
   B. The retrieved chunks were not relevant to the user's query
   C. At least one factual claim in the response is not supported by the retrieved context
   D. The response contained harmful or unsafe content

   <details><summary>Answer</summary>

   **Correct: C.**

   `groundedness` specifically asks: "Is every claim in the LLM response supported by the retrieved context?" It checks faithfulness of the response to the context — not whether the response is correct in an absolute sense, and not whether the retrieval step worked well.

   **Why A is wrong:** A describes `correctness`, which requires an `expected_response` reference answer and is a distinct judge.

   **Why B is wrong:** B describes `chunk_relevance`, a retrieval-stage metric that evaluates the chunks themselves, not the final response.

   **Why D is wrong:** D describes the `safety` judge, which is independent of groundedness.

   </details>

2. A team is building a RAG system over a Delta table of product documentation that is updated daily. They want the vector index to automatically stay in sync with each Delta table commit. Which index type and `pipeline_type` should they configure?

   A. Direct Access index, any pipeline type
   B. Delta Sync index, `pipeline_type="TRIGGERED"`
   C. Delta Sync index, `pipeline_type="CONTINUOUS"`
   D. Direct Access index, `pipeline_type="CONTINUOUS"`

   <details><summary>Answer</summary>

   **Correct: C.**

   A Delta Sync index lets Databricks manage the embedding pipeline automatically. Setting `pipeline_type="CONTINUOUS"` causes the index to sync on every Delta table commit, which matches the "automatically stay in sync" requirement.

   **Why B is wrong:** `TRIGGERED` requires a manual refresh call — it does not auto-sync on commit. If the team needs automatic synchronisation, `TRIGGERED` means they would need to schedule a job to call the refresh API.

   **Why A and D are wrong:** Direct Access requires the team to compute embeddings and upsert them via API — there is no built-in auto-sync. The team would have to build and maintain the sync pipeline themselves.

   </details>

3. **Which TWO** of the following are required columns in the evaluation DataFrame passed to `mlflow.evaluate(data=eval_df, model_type="databricks-agent")` to enable both the `groundedness` judge and the `correctness` judge to run?

   A. `request`
   B. `response`
   C. `expected_response`
   D. `retrieved_context`
   E. `trace`

   <details><summary>Answer</summary>

   **Correct: C (`expected_response`) and D (`retrieved_context`).**

   Both `request` and `response` are baseline required columns for any evaluation run. The question asks specifically which two *additional* columns enable `groundedness` AND `correctness` together.

   - `retrieved_context` unlocks `groundedness` (checks response against context) and `chunk_relevance` (checks whether chunks matched the query).
   - `expected_response` unlocks `correctness` (checks response against ground truth).

   **Why E (`trace`) is wrong:** A `trace` column is optional and used for cost and latency metrics, not for LLM judge activation. `trace` is auto-generated when a `model` argument is passed, not a required input column.

   **Why A and B alone are wrong:** `request` and `response` are always required but are already given in the question's DataFrame — they enable only `relevance_to_query` and `safety` by default, not `groundedness` or `correctness`.

   </details>

4. A candidate scores 60% on a practice exam with this breakdown: D1 65%, D2 72%, D3 44%, D4 53%, D5 81%, D6 61%. They have 10 days remaining. Ranked by expected point gain per hour, which domain should receive the most study time?

   A. D1 (Design Applications, 14%) — because it feels most concrete
   B. D2 (Data Preparation, 14%) — because chunking strategy questions are frequent
   C. D3 (Application Development, 30%) — because it has both the highest weight and the largest gap
   D. D5 (Governance, 8%) — because it is already near-passing and easy to solidify

   <details><summary>Answer</summary>

   **Correct: C.**

   The expected point gain from improving a domain is `weight × gap`. D3's gap is 56 percentage points (100% − 44%) at 30% weight = 16.8 potential points — the highest of any domain. By contrast, D5's 19-point gap at 8% weight yields only 1.5 potential points.

   **Why A is wrong:** D1 at 14% weight with a 35-point gap yields 4.9 potential points — meaningful but far below D3.

   **Why D is wrong:** Investing time in a domain that is already near-passing and carries the lightest weight is the classic low-ROI trap. Governance at 8% has limited upside even if you achieve 100%.

   </details>

5. A team deploys a RAG agent using `agents.deploy()`. Six months later, a compliance audit requires all production LLM inference data to be retained in a queryable Delta table. The team checks the endpoint and finds inference table logging was not enabled at deploy time. What is the correct remediation action?

   A. Re-run `agents.deploy()` with the same model version — it will update the existing endpoint and add inference table logging
   B. Edit the endpoint configuration via the Serving UI to enable the AI Gateway inference table, or use the API to update `auto_capture_config`
   C. Create a brand new endpoint; inference tables cannot be added to an existing endpoint
   D. Inference tables are enabled automatically by `agents.deploy()` and cannot be disabled

   <details><summary>Answer</summary>

   **Correct: B.**

   AI Gateway inference tables can be enabled (or migrated from legacy inference tables) on an existing endpoint by editing its configuration through the Databricks Serving UI ("Edit AI Gateway" → "Enable inference tables") or via the REST API to update `auto_capture_config`. There is no need to recreate the endpoint.

   **Why A is partially wrong:** Re-running `agents.deploy()` with the same model name will update the endpoint with the new version's configuration — but the question specifies the same model version, and the intent is to add inference logging without a model change. The cleaner, non-disruptive path is to edit the existing endpoint config directly.

   **Why C is wrong:** Inference tables can be added to existing endpoints; recreating the endpoint would cause unnecessary downtime.

   **Why D is wrong:** While `agents.deploy()` does enable inference tables by default in current versions, it is possible to deploy without them (or to have deployed with an older SDK version that did not enable them automatically). The statement "cannot be disabled" is also factually incorrect per the Databricks documentation.

   </details>

---

## Further Reading

- [Databricks Certified Generative AI Engineer Associate Exam Guide (March 2026)](https://www.databricks.com/sites/default/files/2026-03/Databricks-Certified-Generative-AI-Engineer-Associate-Exam-Guide-Mar26.pdf) — *verified 2026-07-16* — Authoritative exam blueprint with domain weights and objective wording
- [How quality, cost, and latency are assessed by Agent Evaluation (MLflow 2)](https://docs.databricks.com/aws/en/generative-ai/agent-evaluation/llm-judge-metrics.html) — *verified 2026-07-16* — Definitive reference for built-in LLM judges, metric names, and aggregation formulas
- [Deploy an agent for generative AI applications (Model Serving)](https://docs.databricks.com/aws/en/generative-ai/agent-framework/deploy-agent.html) — *verified 2026-07-16* — `agents.deploy()` API reference, deployment actions table, and authentication configuration
- [Use agents on Databricks](https://docs.databricks.com/aws/en/generative-ai/agent-framework/build-genai-apps.html) — *verified 2026-07-16* — Overview of the full agent build-deploy-evaluate lifecycle on Databricks
- [Inference tables for monitoring and debugging models](https://docs.databricks.com/aws/en/machine-learning/model-serving/inference-tables.html) — *verified 2026-07-16* — Inference table schema, `auto_capture_config` fields, and migration path to AI Gateway inference tables
- [Databricks AI Search (Vector Search)](https://docs.databricks.com/aws/en/generative-ai/vector-search/index.html) — *verified 2026-07-16* — Index types (Delta Sync vs Direct Access), similarity algorithms, and hybrid search configuration
