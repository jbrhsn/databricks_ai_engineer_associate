# MLflow Agent Framework — Interview Prep

**Section:** 04 Application Development — Agents, MLflow & Genie | **Role target:** Senior ML Engineer, MLOps Engineer, AI Platform Engineer, Solutions Architect

---

## Core Conceptual Questions

These test whether you understand the fundamentals.

| Question | Key points to cover | Common weak-answer trap |
|---|---|---|
| What does `mlflow.langchain.log_model()` actually store, and how is it different from pickling a chain? | (1) Stores the Python file (models-from-code) or cloudpickle of the chain object; (2) also records `requirements.txt`, `conda.yaml`, `MLmodel` manifest with input/output signature; (3) links to MLflow run for lineage; (4) models-from-code stores the definition, not the live state, so it survives environment changes | "It just saves the model weights" — LangChain/LangGraph agents have no weights to save; the artifact is code + dependencies |
| What is the Unity Catalog Model Registry and how does it differ from the legacy workspace Model Registry? | (1) UC Registry uses three-part namespace (catalog.schema.model); (2) uses aliases (`@champion`, `@challenger`) instead of lifecycle stages (`Staging`, `Production`); (3) inherits UC GRANT/REVOKE access control; (4) `transition_model_version_stage()` is deprecated — use `set_registered_model_alias()` instead | Saying "you promote models to Staging/Production" — this is the legacy API that doesn't work on UC-backed registries |
| How does `mlflow.langchain.autolog()` work, and what does it miss? | (1) Registers callbacks into LangChain/LangGraph's callback system; (2) automatically emits spans for every Runnable node, LLM call, retriever call; (3) misses custom Python functions outside the Runnable abstraction; (4) fix: combine with `@mlflow.trace` for custom logic | "It instruments everything" — it instruments only LangChain/LangGraph framework primitives |
| What LLM-judge metrics does `mlflow.evaluate(model_type="databricks-agent")` compute, and what columns trigger each? | (1) Groundedness: requires `retrieved_context` + `response`; (2) Answer correctness: requires `expected_response` + `response`; (3) Chunk relevance: requires `retrieved_context` + `request`; (4) Results in `result.tables["eval_results"]` and `result.metrics` | "It automatically calls the model and scores the answers" — you must supply responses (or pass a model URI); the judge scores existing responses, not new ones unless you pass `model=` |
| What is the `ResponsesAgent` interface and why does Databricks recommend it? | (1) MLflow abstract class with `predict()` and `predict_stream()` methods; (2) produces OpenAI Responses-schema-compatible model signature automatically; (3) enables AI Playground, Review App, and production monitoring without code changes; (4) framework-agnostic — wraps LangGraph, LangChain, OpenAI SDK equally | "It's a LangChain-specific wrapper" — `ResponsesAgent` is framework-agnostic and lives in `mlflow.pyfunc` |

---

## Applied / Scenario Questions

**Q:** You've logged a LangGraph RAG agent to MLflow. When you load it with `mlflow.pyfunc.load_model()` on a different Databricks cluster, it raises `ModuleNotFoundError: No module named 'databricks_langchain'`. What went wrong and how do you fix it?

**Strong answer framework:**
- The agent was logged without explicit `pip_requirements` (or with an incomplete list), so the `requirements.txt` in the artifact does not include `databricks-langchain`.
- When `mlflow.pyfunc.load_model()` is called, MLflow installs the recorded requirements into a virtual environment. If `databricks-langchain` is missing from the requirements, it is not installed.
- Fix: Re-log with `mlflow.langchain.log_model(..., pip_requirements=["langchain>=0.3.0", "langgraph>=0.2.0", "databricks-langchain>=0.3.0"])`.
- Show tradeoff awareness: `pip_requirements` replaces the entire requirements list; `extra_pip_requirements` appends to the inferred list. For complex dependency graphs, `extra_pip_requirements` is safer because it doesn't accidentally drop MLflow's own runtime requirements.

---

**Q:** A business stakeholder reports that the deployed RAG agent sometimes answers questions confidently with information that isn't in the HR policy documents. How would you diagnose whether this is a retrieval failure or a hallucination, and what tooling would you use?

**Strong answer framework:**
- Enable/confirm MLflow Tracing is active (via `mlflow.langchain.autolog()` + `@mlflow.trace` on retrieval functions).
- Open the Traces tab in the MLflow Experiment and filter for the affected queries.
- Inspect the RETRIEVER span: check what chunks were actually retrieved, their similarity scores, and their `doc_uri`. If the right document wasn't retrieved → retrieval failure (fix: improve query rewriting, chunk overlap, or embedding model).
- Inspect the LLM span: if correct chunks were retrieved but the answer doesn't match → LLM hallucination on the provided context (fix: stronger system prompt, reduce temperature, add explicit "only use provided context" instruction).
- Run `mlflow.evaluate(model_type="databricks-agent")` with `retrieved_context` from the traces. Low `percent_grounded` with correct retrieval = hallucination; low `percent_grounded` with wrong retrieval = retrieval failure.
- Show tradeoff awareness: distinguishing these two failure modes requires tracing; without it, you can only see the wrong output, not the cause.

---

## System Design / Architecture Questions

**Q:** Design an end-to-end MLOps pipeline for a LangGraph RAG agent from development notebook to production Model Serving, including quality gates and rollback capability.

**Approach:**
1. **Clarify requirements:** What is the SLA? What quality threshold gates promotion? What is the rollback time budget? Is A/B testing required or is it hard cutover?
2. **Propose structure:**
   - Dev: Notebook logs agent with `mlflow.langchain.log_model(lc_model="./agent.py", registered_model_name="prod_catalog.agents.hr_bot")`, runs `mlflow.evaluate()` with an eval set, promotes to `@candidate` alias.
   - CI/CD: Automated pipeline runs eval on the `@candidate` version; if `percent_grounded >= 0.80` AND `percent_correct >= 0.75`, calls `set_registered_model_alias("hr_bot", "champion", version=N)`.
   - Model Serving: Endpoint references `models:/prod_catalog.agents.hr_bot/@champion` — alias swap is zero-downtime.
   - Rollback: `set_registered_model_alias("hr_bot", "champion", version=N-1)` restores previous version in seconds; no redeployment needed.
   - Production monitoring: MLflow 3 production monitoring runs the same judges on live traces async; alerts when `percent_grounded` drops below 0.75 on rolling 24h window.
3. **Justify choices:** Alias-based promotion is zero-downtime and atomic. Models-from-code logging eliminates environment drift between dev and prod. Reusing the same judge configuration for offline eval and online monitoring ensures a consistent quality definition.

---

## Vocabulary That Signals Expertise

Use these terms naturally — don't force them:

- **Models-from-code** — when explaining why LangGraph agents should be logged as file paths rather than live objects
- **Unity Catalog alias** vs. **lifecycle stage** — signals you know the UC migration from MLflow workspace Registry
- **Champion/challenger** — the alias-based canary pattern for safe production promotion
- **Groundedness** — the specific metric name for retrieval-supported-answer scoring; don't say "accuracy" for this
- **Span hierarchy** / **span tree** — the correct term for the nested structure of traces; "logs" is imprecise
- **`registered_model_name` as a UC three-part name** — signals you know the UC naming convention (`catalog.schema.model`)
- **`ResponsesAgent`** — the standard interface; calling it "the MLflow agent wrapper" is vague

---

## Vocabulary That Signals Weakness

Avoid these — they signal outdated or shallow understanding:

- **"Promote to Staging/Production"** — the UC Model Registry replaced lifecycle stages with aliases; using stage terminology signals you haven't worked with UC-backed registries
- **"Save the model as a pickle"** — signals unfamiliarity with MLflow's models-from-code approach and the serialization issues with complex agents
- **"MLflow just tracks metrics"** — signals you only know the traditional MLflow ML tracking use case, not the GenAI-specific tracing and evaluation features
- **"Run mlflow.evaluate() to generate responses"** — conflates model invocation with evaluation; `mlflow.evaluate()` scores existing responses (or invokes a logged model), it doesn't "generate" on its own
- **"Autolog handles everything"** — fails to acknowledge that custom Python functions outside LangChain/LangGraph Runnables are invisible to autolog

---

## STAR Answer Frame

**Situation:** My team had deployed a LangGraph RAG agent for internal HR policy questions. After two weeks in production, the support team started receiving complaints that the agent was giving inconsistent answers about the expense reimbursement policy — sometimes citing the correct 30-day submission window, sometimes citing a 60-day window that appeared in an older archived document.

**Task:** I was responsible for (1) diagnosing whether the failure was in retrieval (wrong document retrieved) or generation (hallucination against correct context), (2) implementing a fix, and (3) ensuring we could detect this class of failure automatically going forward.

**Action:** I opened the MLflow Experiment Traces tab and filtered for queries containing "expense reimbursement." In the RETRIEVER spans, I immediately saw that the vector search index had not been refreshed after the policy document was updated; similarity scores for the old archived document were marginally higher than for the new policy (0.83 vs. 0.79). This was a retrieval failure, not a hallucination. I rebuilt the vector index with the updated documents and re-ran `mlflow.evaluate()` with an updated eval set that included the expense reimbursement question. I then added a CI check that runs this eval set on every vector index rebuild, with groundedness ≥ 80% as the gate before the new index is promoted. I also set the `@mlflow.trace(span_type=SpanType.RETRIEVER)` decorator to log `similarity_score` as a custom span attribute, making future retrieval quality regressions immediately visible in the Traces tab without needing to dig into raw index logs.

**Result:** The groundedness score for expense-related queries went from 61% (old index) to 94% (rebuilt index). The CI eval gate caught a subsequent minor index drift two weeks later, blocking a bad deployment automatically before it reached users. The support team reported zero complaints in the following month.

---

## Red Flags Interviewers Watch For

- **Can't distinguish between MLflow tracking (metrics/params) and MLflow Tracing (spans/traces)** — these are fundamentally different subsystems with different data models; conflating them suggests surface-level knowledge.
- **Doesn't know what `retrieved_context` is needed for** — if you can't explain which eval metrics require it and why, you likely haven't run a real agent evaluation.
- **Says "just call mlflow.evaluate() on the model and it will score everything"** — suggests you haven't read the actual input schema; eval needs structured input columns, not just a model URI.
- **Can't name what `@champion` alias replaces** — not knowing that it replaces deprecated `Staging`/`Production` lifecycle stages signals you haven't worked with UC-backed registries.
- **Thinks `mlflow.langchain.autolog()` and `@mlflow.trace` are redundant / do the same thing** — they are complementary: autolog covers the framework layer, `@mlflow.trace` covers custom code outside the framework.
