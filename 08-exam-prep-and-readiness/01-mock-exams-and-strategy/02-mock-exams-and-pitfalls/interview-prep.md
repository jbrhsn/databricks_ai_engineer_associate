# Mock Exams and Exam Pitfalls — Interview Prep

**Section:** 08 Exam Prep | **Role target:** Senior ML Engineer, GenAI Platform Engineer, AI/ML Solutions Architect

---

## Core Conceptual Questions

These questions test whether you understand the underlying mechanisms — not just that you know the names.

| Question | Key points to cover | Common weak-answer trap |
|---|---|---|
| "What is the difference between `mlflow.register_model()` and `databricks.agents.deploy()`?" | `register_model` creates a versioned, auditable entry in Unity Catalog; `agents.deploy()` creates a serving endpoint with review app and inference tables. They are sequential steps, not alternatives. | Saying "they both deploy the model" — conflating registration (governance step) with serving (runtime step). |
| "When would you choose Direct Access Vector Search index over Delta Sync?" | Direct Access: when the ingestion pipeline can push updates via SDK immediately after each event; required for sub-minute freshness. Delta Sync: managed, scheduled, eventual consistency — correct for batch-updated document stores where latency tolerance exists. | Defaulting to "Delta Sync is always easier" without addressing the freshness constraint. |
| "What does `model_type='databricks-agent'` do in `mlflow.evaluate()`?" | It activates the Databricks AI judge suite (groundedness, relevance, safety, correctness) and enables Agent Evaluation metrics. Without it, `mlflow.evaluate()` runs standard numeric metrics only. | Saying "it sets the model type for serving" — confusing the evaluation parameter with the serving endpoint flavor. |
| "How does LangGraph state differ from a standard Python dict passed between functions?" | LangGraph state is a typed schema (`TypedDict`) with per-field reducers. The reducer determines how new values are merged (e.g., `add_messages` appends; default assignment overwrites). Reducers enable concurrent node execution with deterministic state merges. | "It's basically just a dict that gets passed around" — missing the reducer/schema concept entirely. |
| "What is the role of AI Gateway in a Databricks GenAI architecture?" | AI Gateway is a proxy layer that unifies access to foundation models (Databricks-hosted and external), applies guardrails (input/output filtering, PII detection), enforces rate limits, and logs usage. It is separate from Unity Catalog, which handles structured data governance. | "It's where you store your models" — confusing AI Gateway with Model Registry or Model Serving. |

---

## Applied / Scenario Questions

**Q:** "We're building a customer support agent using LangGraph and LangChain. It needs to maintain conversation history, call two external tools, and be deployed to production with monitoring. Walk me through your architecture and the specific Databricks components you'd use at each step."

**Strong answer framework:**
- Start with the LangGraph graph structure: `StateGraph` with `TypedDict` state schema, `add_messages` reducer on the messages field, nodes for the LLM call and each tool, conditional edges that route based on whether the last message contains tool calls
- LangChain for components inside nodes: prompt templates, the LLM client (pointing at a Databricks Foundation Model endpoint or external model via AI Gateway), tool definitions as JSON schema with Python execution functions
- Deployment: log with `mlflow.langchain.log_model()` or `mlflow.pyfunc.log_model()` (pyfunc if the inference logic is non-standard), register to Unity Catalog with `mlflow.register_model()`, deploy with `databricks.agents.deploy()` which provisions the serving endpoint, review app for human feedback, and inference tables for offline analysis
- Monitoring: `mlflow.evaluate()` with `model_type="databricks-agent"` on sampled production traces; groundedness and relevance judges run against the inference table data
- How to show trade-off awareness: discuss when to use Foundation Model API (pay-per-token, no infra) vs. provisioned throughput (predictable cost at high QPS), and when Direct Access vs. Delta Sync Vector Search is appropriate

**Q:** "A colleague says your Vector Search retrieval is returning stale documents — articles updated 10 minutes ago aren't showing up. How do you diagnose and fix this?"

**Strong answer framework:**
- First, determine index type: Delta Sync Index has a sync lag; Direct Access Index should be immediate if SDK calls are executing correctly
- For Delta Sync: check the sync status in the Vector Search UI or via the API; verify the Delta table has new commits; check whether an explicit sync was triggered or if it's on a schedule
- For Direct Access: check that the ingestion pipeline is calling the upsert SDK method after every write; verify the index is not in an error state
- Fix decision: if near-real-time freshness is a hard requirement and the current index is Delta Sync, switch to Direct Access and modify the ingestion pipeline to push updates via SDK
- Show depth: mention that Delta table OPTIMIZE operations can affect sync timing, and that Vector Search indexes have eventual consistency guarantees even after a successful sync call

---

## System Design / Architecture Questions

**Q:** "Design a RAG evaluation pipeline for a production agent that processes 10,000 queries per day. The team needs to track quality over time and get alerts when groundedness drops below a threshold."

**Approach:**
1. **Clarify requirements:** What judge coverage is needed (groundedness only, or full suite)? Is evaluation online (real-time) or offline (batch on sampled traces)? What is the acceptable evaluation latency?
2. **Propose structure:** Enable inference tables on the serving endpoint to capture request/response pairs to a Delta table. Sample 5–10% of daily traffic (500–1,000 traces) for cost management. Run `mlflow.evaluate(data=sampled_df, model_type="databricks-agent")` in a scheduled Databricks job. Write aggregated metrics to a Delta table using the dashboard pattern from the docs. Create a Databricks SQL dashboard or Lakeview dashboard on that metrics table.
3. **Justify choices and name trade-offs explicitly:** Sampling rate is a trade-off between cost (AI judges bill per evaluation) and statistical reliability (at 5%, a 1,000-sample daily run gives sufficient power to detect a 5-point groundedness drop). Online evaluation on every query would be prohibitively expensive. The scheduled job approach introduces a lag (metrics are 24 hours old) — if the team needs faster alerting, increase sampling rate and run hourly. The alternative of streaming evaluation using MLflow 3 / production monitoring is worth mentioning as the forward-looking approach.

---

## Vocabulary That Signals Expertise

Use these terms naturally — don't force them:

- **"Reducer"** — when explaining LangGraph state merge semantics; signals you understand the concurrent execution model, not just the API
- **"Inference table"** — when discussing production monitoring; signals you know the Agents deployment stack beyond just the serving endpoint
- **"Distractor pattern"** — in the context of exam prep; signals you think about wrong answers structurally, not just by elimination of what you don't recognise
- **"Confusable pair"** — when discussing API design; signals you think about the cognitive model of the user, not just the correctness of the implementation
- **"AI judge"** — the Databricks term for an LLM-based evaluator in Agent Evaluation; using this instead of generic "LLM evaluation" signals familiarity with the platform

---

## Vocabulary That Signals Weakness

Avoid these — they signal outdated or shallow understanding:

- **"I'd just use the Databricks UI to deploy it"** — signals no understanding of the programmatic deployment pipeline (register → deploy → monitor)
- **"MLflow tracks the model"** — too vague; the question is always "tracks it where, with what metadata, and accessible via which API"
- **"Vector Search is like Pinecone but in Databricks"** — signals a vendor-comparison mental model rather than architectural understanding of Delta Sync vs. Direct Access index types and their trade-offs
- **"You can use `model_type` to configure the LLM"** — conflates evaluation configuration with serving configuration; a flag for shallow understanding of the MLflow evaluation API

---

## STAR Answer Frame

**Situation:** I was preparing to take the Databricks Certified Generative AI Engineer Associate exam while simultaneously building a RAG-based internal knowledge agent for my team. We had a production deployment issue where Vector Search was returning stale documents after article updates.

**Task:** I needed to diagnose the freshness issue, fix it, and ensure I understood the architectural distinction well enough to answer exam questions about it correctly under time pressure.

**Action:** I identified that we were using a Delta Sync Index with a scheduled refresh interval of 30 minutes. The business requirement was sub-5-minute freshness after article ingestion. I switched to a Direct Access Index and modified the ingestion pipeline to call the Vector Search SDK's `upsert` method immediately after each Delta write. For exam prep, I built a confusable-pairs table entry for this exact distinction — Delta Sync (eventual, managed) vs. Direct Access (real-time, SDK-driven) — and tied the distinguishing rule to the stem constraint "requires real-time freshness."

**Result:** The production retrieval latency dropped from 30 minutes to under 30 seconds for new articles. On the exam, I encountered a Vector Search freshness question and applied the distinguishing rule in under 15 seconds.

---

## How to Handle a Question You Don't Know the Exact Answer To

This is the interview equivalent of a flagged exam question. The worst response is silence or a guess delivered as certainty — both undermine trust. The best response has three parts:

1. **Anchor to what you do know.** "I know that Vector Search has two index types with different consistency models. I'm not certain of the exact SDK method name for Direct Access upserts, but the pattern would be..."

2. **Reason out loud to the boundary of your knowledge.** "Given that the requirement is real-time freshness, I'd rule out Delta Sync because it's schedule-based. I'd look for an SDK method that takes a list of document-embedding pairs and writes directly to the index — something like an upsert call."

3. **Close the loop.** "I'd want to verify the exact method signature in the Databricks Vector Search SDK docs before implementing. Is this the kind of detail you'd like me to look up, or are you more interested in the architectural decision?"

This approach demonstrates reasoning under uncertainty — which is more valuable than perfect recall of API names. It also shows intellectual honesty, which is what interviewers are actually probing for when they ask a question the candidate is likely to be uncertain about.

---

## Red Flags Interviewers Watch For

- **Stating the correct answer without being able to explain the wrong one.** If you say "use Direct Access Index," the follow-up is always "why not Delta Sync?" Inability to articulate the distinction signals memorisation without understanding.
- **Using "it depends" without immediately naming the specific variables it depends on.** "It depends on your use case" tells the interviewer nothing. "It depends on whether your ingestion pipeline can be modified to call the SDK — if yes, Direct Access; if no, and your latency tolerance is minutes, Delta Sync" demonstrates structured thinking.
- **Confusing the governance layer with the serving layer.** If an interviewer describes a PII filtering requirement for an LLM chatbot and you reach for Unity Catalog column masking instead of AI Gateway guardrails (or vice versa), it signals you haven't internalised the two-layer architecture.
- **Fluency on happy-path workflows but silence on failure modes.** Strong candidates can describe what breaks when the wrong `model_type` is passed to `mlflow.evaluate()`, what happens if `register_model()` is called without Unity Catalog set as the registry URI, or why a LangGraph agent loses conversation history when the state schema uses an overwrite reducer instead of `add_messages`. Interviewers use failure-mode questions to separate practitioners from readers.
