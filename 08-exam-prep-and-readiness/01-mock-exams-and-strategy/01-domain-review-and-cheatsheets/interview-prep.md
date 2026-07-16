# Domain Review and Cheatsheets — Interview Prep

**Section:** 08 Exam Prep and Readiness | **Role target:** Senior ML/AI Engineer, MLOps Engineer, Solutions Architect (Generative AI)

---

## Core Conceptual Questions

These test whether you understand the full GenAI stack — not just individual components in isolation.

| Question | Key points to cover | Common weak-answer trap |
|---|---|---|
| Walk me through the six capability domains a GenAI engineer on Databricks needs to master. | Name all six with weights: Design (14%), Data Prep (14%), App Dev (30%), Deploy (22%), Governance (8%), Eval/Monitoring (12%). Explain why 30% of the surface area is application development — chains, agents, PyFunc, memory. | Candidate names the domains but cannot explain relative importance; treats governance and evaluation as afterthoughts because they carry lower weights |
| What is the difference between `groundedness` and `correctness` as evaluation metrics? | `groundedness`: no expected answer needed, checks faithfulness to retrieved context. `correctness`: requires `expected_response`, checks agreement with a reference. A response can be grounded but wrong (context was outdated). | Saying they are "basically the same thing" or conflating faithfulness-to-context with accuracy-against-ground-truth |
| When would you choose fine-tuning over RAG, and what signals drive that decision? | RAG: facts change frequently, data must stay outside model weights, latency budget allows retrieval, sources must be cited. Fine-tuning: new reasoning style or domain-specific output format, not just new facts, latency is critical and retrieval adds too much overhead. | Answering "fine-tuning is better because it's more accurate" without addressing the update frequency, data sovereignty, and latency tradeoffs |
| What does `databricks.agents.deploy()` do beyond creating a serving endpoint? | Creates endpoint + provisions short-lived auth credentials for downstream resources + enables Review App + enables MLflow real-time tracing + enables AI Gateway inference tables. | Saying it "just deploys the model" — missing the integrated governance and observability wiring |
| Explain the difference between a Delta Sync index and a Direct Access index in Databricks Vector Search. | Delta Sync: Databricks manages the embedding pipeline, auto-syncs on Delta table updates, `CONTINUOUS` or `TRIGGERED` refresh modes. Direct Access: caller computes embeddings externally and upserts via API. Embedding model mismatch is silent and catastrophic. | Saying the only difference is cost, or confusing sync mode with index type |

---

## Applied / Scenario Questions

**Q:** A financial services team is building a RAG chatbot over 200,000 internal policy documents stored in a Delta table. The documents are updated weekly. Compliance requires that no PII from user queries leaves the corporate perimeter. The team wants to measure whether the chatbot is actually answering from the policy documents and not hallucinating. Walk me through the architecture.

**Strong answer framework:**
- **Data layer:** Delta Sync index with `pipeline_type="TRIGGERED"` (weekly refresh aligns with update cadence; `CONTINUOUS` is unnecessary overhead). Embedding model selection: use an on-premises or Databricks-hosted embedding model to satisfy data perimeter requirement.
- **PII layer:** Pre-processing step using Presidio `AnalyzerEngine` + `AnonymizerEngine` to detect and mask PII (PERSON, EMAIL, PHONE_NUMBER entities) before the query reaches the LLM. Post-processing step to mask any PII that appears in the response.
- **Application layer:** LangGraph StateGraph: node 1 = PII masking, node 2 = Vector Search retrieval, node 3 = LLM generation with retrieved context in prompt, node 4 = PII unmask for authorised users. Compile and wrap as `mlflow.pyfunc.PythonModel` for deployment.
- **Evaluation layer:** `mlflow.evaluate()` with `model_type="databricks-agent"` and evaluation dataset including `retrieved_context` to activate `groundedness` judge. A `groundedness` score below 0.8 signals hallucination risk.
- **Show tradeoff awareness:** The PII masking step adds latency (~50–200 ms per request); the team must decide whether to apply it inline or async. `TRIGGERED` refresh means the index may be up to 7 days stale — acceptable for policy documents, not for real-time pricing.

---

**Q:** Your team's RAG agent is returning factually correct answers but `chunk_relevance` scores are consistently low (0.3 average). What does this tell you and what do you investigate first?

**Strong answer framework:**
- `chunk_relevance` measures whether the retrieved chunks were relevant to the user's query — a low score means the retriever is returning poor chunks even though the LLM somehow produced a correct answer.
- Investigate: (1) Embedding model mismatch between index and query — silent degradation. (2) Chunk size too large — large chunks score lower because they dilute the relevant signal with irrelevant content. (3) Missing metadata filters — chunks from wrong document categories are being returned. (4) Query embedding not normalised — for cosine-similarity indexes, un-normalised embeddings degrade ranking.
- The correct answer being produced despite low chunk relevance suggests the LLM is either using parametric knowledge (hallucinating correctly) or the chunks contain the right info buried in noise. Both are fragile; fix the retrieval step.

---

## System Design / Architecture Questions

**Q:** Design an evaluation and monitoring pipeline for a production RAG agent serving 10,000 daily queries. The team wants to detect quality regressions within 24 hours.

**Approach:**
1. **Clarify requirements:** Define "quality regression" — is it a drop in `groundedness`, `correctness`, or a customer-reported satisfaction signal? What is the acceptable false-positive rate for alerts?
2. **Propose structure:**
   - Enable AI Gateway inference table on the serving endpoint (`auto_capture_config` → Unity Catalog Delta table)
   - Schedule a nightly Databricks Job that: (a) samples 200 rows from yesterday's inference table, (b) runs `mlflow.evaluate()` with `model_type="databricks-agent"` on the sample, (c) logs results as a new MLflow run
   - Configure a Databricks Alert on the resulting metrics table: alert if `groundedness/rating/percentage` drops below 0.75 or `correctness/rating/percentage` drops below 0.70
   - Use MLflow tracing in real time to capture full reasoning chains for debug investigation when an alert fires
3. **Justify choices and name tradeoffs:**
   - Sampling 200 rows (vs all 10,000) reduces LLM judge cost by ~98% while maintaining statistical power for detecting a 5% quality drop
   - Nightly cadence meets the 24-hour detection requirement; hourly would detect faster but cost 24× more
   - AI Gateway inference table (not legacy) is the current Databricks recommendation — legacy tables are deprecated

---

## Vocabulary That Signals Expertise

Use these terms naturally — don't force them:
- **"weight-proportional study allocation"** — signals you approach exam prep as an optimisation problem, not anxiety-driven cramming
- **"faithfulness to retrieved context"** — precise language for `groundedness`, distinguishes it from correctness
- **"embedding model alignment"** — term for ensuring the same model is used at index creation and query time
- **"retrieval-stage vs generation-stage metric"** — clearly separates `chunk_relevance` from `relevance_to_query`
- **"PyFunc flavour"** — the MLflow term for a custom Python model; signals you know the MLflow model registry vocabulary
- **"conditional routing"** — the LangGraph construct (`add_conditional_edges`); signals graph-based agent understanding
- **"inference table schema"** — shows you know the specific columns (`databricks_request_id`, `execution_time_ms`, `request`, `response`)
- **"AI Gateway guardrail layer"** — positions PII filtering as a cross-cutting concern at the proxy, not just in application code

---

## Vocabulary That Signals Weakness

Avoid these — they signal outdated or shallow understanding:
- **"I'd just use LangChain for everything"** — fails to distinguish between LangChain (component library) and LangGraph (stateful graph orchestration); both have different strengths
- **"fine-tuning is always better than RAG"** — signals you haven't worked with systems where data changes frequently or must stay outside model weights
- **"the model serving API handles monitoring automatically"** — inference tables must be explicitly enabled; monitoring is not automatic
- **"Vector Search just does semantic similarity"** — misses hybrid keyword-similarity search (Reciprocal Rank Fusion), metadata filtering, and the HNSW algorithm detail that the exam tests
- **"MLflow just tracks experiments"** — undersells MLflow's role as the deployment registry, evaluation framework, tracing platform, and production monitoring substrate in the Databricks GenAI stack

---

## STAR Answer Frame

**Situation:** Our team deployed a RAG agent for customer support that was passing all unit tests but receiving low satisfaction scores from users in production (3.1/5 average). We suspected retrieval quality but had no quantitative evidence.

**Task:** Design and implement an evaluation pipeline to diagnose whether the issue was retrieval (wrong chunks), generation (hallucination), or a mismatch between the two.

**Action:** I enabled the AI Gateway inference table on the serving endpoint to capture all production request/response pairs. I then built a sampling job that extracted 300 representative queries from the previous week, retrieved the corresponding trace data (including retrieved chunks), and ran `mlflow.evaluate()` with `model_type="databricks-agent"` using those traces as the evaluation dataset. I focused on three metrics: `chunk_relevance/precision/average` (was the retriever returning good chunks?), `groundedness/rating/percentage` (was the LLM faithful to the chunks?), and `relevance_to_query/rating/percentage` (were the responses addressing the actual question?). Results showed `chunk_relevance` at 0.38 (low) and `groundedness` at 0.91 (high) — meaning the LLM was faithfully generating from bad chunks.

**Result:** Root cause was a chunk size mismatch (chunks were 2,000 tokens — too large for our embedding model's effective context window). Reducing chunk size to 512 tokens with 64-token overlap raised `chunk_relevance` to 0.74 within one week, and customer satisfaction recovered to 4.2/5 over the following fortnight. The inference table pipeline now runs nightly and alerts when `chunk_relevance` drops below 0.6.

---

## Red Flags Interviewers Watch For

- **Cannot distinguish which MLflow judge requires `expected_response` vs which runs without it** — signals superficial familiarity with the evaluation framework rather than hands-on use
- **Treats `agents.deploy()` and a plain Model Serving endpoint as interchangeable** — signals no hands-on production deployment experience on Databricks; missing the Review App, tracing, and auth provisioning is a production-readiness gap
- **Cannot name the three parts of a complete LangGraph node wiring** (node registration, edge, conditional edge with routing function) — signals they have read about LangGraph but never built a non-trivial graph
- **Describes governance as "just add PII filtering before the LLM"** — misses AI Gateway-level guardrails, Unity Catalog column masking, and the data-sovereignty implications of embedding model choice
- **Cannot explain why `chunk_relevance` and `relevance_to_query` are different things** — the most common exam confusable is also a real-world diagnostic gap: conflating these leads to fixing the wrong layer of the pipeline
