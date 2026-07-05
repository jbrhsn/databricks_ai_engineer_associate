# Common Pitfalls and Final Checklist

**Section:** Exam Prep | **Module:** Review and Practice | **Est. time:** 1.5 hrs
**Exam mapping:** Not directly scored — a consolidated trap list and final readiness checklist across all five domains

## Learning Objectives

By the end of this chapter, you will be able to:

- Recognize the recurring conceptual traps and near-miss distractors this exam uses
- Recall the renamed-product terminology that appears as deliberate distractors
- Recite the highest-yield decision rules from every section from memory
- Run a final knowledge checklist and a logistics checklist before test day

## Core Concepts

### The renamed-product trap (know both names)

Databricks renames products constantly, and the exam exploits this. An option using an old name might
be a trap — or might be correct, since the exam sometimes uses old names deliberately. **Know both
names and treat them as synonyms**; judge an option by the accuracy of its *claim*, not just its name.

| Current name (2026) | Former name(s) | Note |
|---|---|---|
| Databricks **AI Search** | (Mosaic AI) **Vector Search** | API/SQL identifiers still say `vector-search` / `vector_search()` |
| **Unity AI Gateway** | **Mosaic AI Gateway** | Beta in 2026 |
| **Declarative Automation Bundles** | Databricks **Asset Bundles** (DABs) | Same `databricks.yml`, same `databricks bundle` CLI |
| **Lakeflow Spark Declarative Pipelines** | Delta Live Tables (DLT) | |
| **MLflow model aliases** | MLflow **stages** | Stages are **gone** in Unity Catalog |

### The "which mechanism" traps

Many questions hinge on picking the right one of two similar mechanisms. These are the pairs most
likely to be tested:

- **PII MASK vs BLOCK vs service policy** — *masking* is an **endpoint guardrail** (PII behavior =
  MASK); *blocking* can be endpoint guardrail (BLOCK) or a `system.ai.block_pii` **service policy**;
  **service policies never mask, only ALLOW/DENY/ASK**.
- **`agents.deploy()` vs `mlflow.deployments`** — agents use `agents.deploy()` (adds Review App,
  tracing, inference tables, auth passthrough); plain custom models use `mlflow.deployments` /
  WorkspaceClient / REST / UI.
- **TRIGGERED vs CONTINUOUS sync** — TRIGGERED = on-demand, cheaper, only mode on storage-optimized;
  CONTINUOUS = seconds-fresh, always-on, standard endpoints only.
- **STANDARD vs STORAGE_OPTIMIZED endpoint** — standard supports both sync modes and dict filters;
  storage-optimized is TRIGGERED-only with SQL-string filters.
- **Knowledge Assistant vs custom LangGraph agent** — Knowledge Assistant = no-code doc Q&A only;
  custom agent = tools, multi-step logic, custom UI.
- **Pay-per-token vs provisioned throughput vs external models** — prototype/low-volume vs
  production/fine-tuned vs governed third-party proxy.
- **`Correctness` vs `RelevanceToQuery`** — Correctness needs ground-truth `expected_facts`;
  RelevanceToQuery needs none. RAG scorers (`RetrievalGroundedness`/`RetrievalRelevance`) need a trace.
- **Managed vs self-managed embeddings** — managed = Databricks embeds a text column; self-managed =
  your table already has a vector column.

### The "you forgot the prerequisite" traps

Questions often describe something that *looks* like it should work but fails on a missing prereq:

- **`resources` not declared at log time** → deployed agent fails auth at runtime ("works in notebook,
  fails on endpoint").
- **Change Data Feed not enabled** → Delta Sync index won't sync.
- **Row filter / column mask on a table** → you **can't** build an AI Search index from it.
- **Deploying from a Git folder** → real-time tracing doesn't work (use a non-Git experiment).
- **`ai_query()` on a Pro/Classic SQL warehouse** → unsupported; needs DBR 18.2+ serverless.
- **`databricks bundle deploy` alone** → doesn't restart an App; must `bundle run` + poll for `RUNNING`.

## Deep Dive / Advanced Topics

### High-yield decision rules to hold in memory

If you can recall these cold, you'll answer a large share of decision questions correctly:

**Foundations (01)**
- Lower temperature → more deterministic; use a structured-output/response-format schema for reliable JSON.
- Match model to task: small/fast models for routing/classification, larger models for complex reasoning.

**Data Preparation (02)**
- Chunk size trades recall vs precision; overlap preserves context across boundaries.
- Delta is the default format; medallion = bronze→silver→gold; UC Volumes need DBR 13.3 LTS+.
- Three embedding models: `databricks-gte-large-en` (8192t, not normalized), `databricks-bge-large-en`
  (512t, normalized), `databricks-qwen3-embedding-0-6b` (multilingual).

**Application Development (03)**
- LangGraph is the exam framework: `StateGraph`, `add_node`/`add_edge`/`add_conditional_edges`,
  `create_react_agent`, `interrupt()` for HITL, checkpointers for memory.
- `ResponsesAgent` is the deployment wrapper required for Playground/tracing/evaluation.
- `ChatDatabricks` connects LangGraph to Foundation Model API endpoints.

**Assembling & Deploying (04)**
- models-from-code: `python_model="agent.py"` (file path), `set_model()` in the file, `resources` for auth.
- UC: three-level names; **aliases not stages**; validate with `mlflow.models.predict()`.
- `agents.deploy()` for agents; re-deploy same model name + new version for zero-downtime.
- AI Search: endpoint vs index; L2/HNSW; hybrid = ANN + full-text via RRF; CDF required for Delta Sync.
- Prompt Registry: versions immutable, aliases mutable, double-brace `{{var}}`.

**Governance/Eval/Monitoring (05)**
- UC governs data *and* AI (models, functions, indexes, prompts, agents, MCP).
- Endpoint guardrails can MASK; service policies only DENY; jailbreak = request-only, hallucination = response-only.
- Tracing → evaluation → monitoring loop; `RetrievalGroundedness` = anti-hallucination (needs trace);
  production monitoring = register→start, `@scorer`-only, ≤20 scorers.

### Reasoning under uncertainty on the exam

When two options both look defensible, prefer the one that:
- Matches the **most recent** Databricks-recommended approach (e.g., `ResponsesAgent` over older
  `ChatAgent`; aliases over stages; ABAC policies over per-table masks) — the exam rewards current best
  practice.
- Respects the **explicit qualifier** in the stem (production vs prototype, no-code vs custom, which
  endpoint type).
- Uses the **native Databricks mechanism** over a generic one, when the question is Databricks-specific.

## Worked Examples & Practice

### A trap-spotting walkthrough

**Question:** "A team's agent works perfectly in their development notebook but returns authentication
errors on retrieval once deployed to a Model Serving endpoint. Which is the most likely cause?"
A. The AI Search index was deleted
B. The `resources` list omitted the vector index at log time
C. The endpoint has scale-to-zero enabled
D. The prompt template used single braces

**Walkthrough:** The tell is "works in notebook, fails on endpoint, on *retrieval*, with an *auth*
error." The notebook runs as the user (who has access); the endpoint relies on auth passthrough, which
only provisions credentials for resources declared at log time. Eliminate A (retrieval auth error, not
"index not found"), C (scale-to-zero causes latency, not auth errors), and D (brace style is a Prompt
Registry / LangChain concern, unrelated to serving auth). **Answer: B.** This is the single most
tested deployment gotcha — recognize the "works locally, fails on endpoint" signature instantly.

**Failure mode to observe:** a candidate who knows about `resources` but skims the stem might latch
onto "scale-to-zero" (C) because it's a real deployment setting they recently studied. Recency of study
makes wrong-but-familiar options tempting. The qualifier "authentication errors on retrieval" is
decisive and rules C out — always let the specific symptom in the stem drive the answer.

## Common Pitfalls & Misconceptions

- **Pitfall:** Picking an option because its terminology looks modern/familiar rather than because its claim is correct → **Why it happens:** recency bias from recent studying → **Fix:** evaluate the *claim*, not the buzzword. A correct claim using an old product name beats a wrong claim using the new name.

- **Pitfall:** Answering "move the model to the Production stage" for UC promotion → **Why it happens:** stages were the old registry mechanism → **Fix:** Unity Catalog has no stages; use **aliases** and environment catalogs.

- **Pitfall:** Choosing a service policy to *mask* PII → **Why it happens:** `system.ai.block_pii` sounds like a PII handler → **Fix:** service policies only DENY; masking is an **endpoint guardrail** (behavior = MASK).

- **Pitfall:** Forgetting that deploy ≠ restart for Databricks Apps, or that CDF is required for Delta Sync → **Why it happens:** the happy path hides the prereq → **Fix:** memorize the prerequisite/gotcha list above; these are exactly what "looks like it should work but doesn't" questions test.

## Key Definitions

| Term | Definition |
|---|---|
| Renamed-product distractor | An answer option using an old product name, used as a trap (or occasionally as the correct answer) |
| Prerequisite trap | A question describing a workflow that fails because a required setup step (CDF, `resources`, etc.) was skipped |
| Decision rule | A compact "if scenario, then choice" heuristic distilled from a chapter |
| Recency bias | The tendency to over-pick options that resemble recently studied material |

## Summary / Quick Recall

- **Know both product names:** AI Search=Vector Search, Unity AI Gateway=Mosaic AI Gateway, Declarative Automation Bundles=Asset Bundles, aliases replaced stages
- **Masking = endpoint guardrail; service policies only DENY**; `agents.deploy()` for agents vs `mlflow.deployments` for plain models
- **TRIGGERED vs CONTINUOUS**, **STANDARD vs STORAGE_OPTIMIZED**, **managed vs self-managed embeddings**, **pay-per-token/provisioned/external** — know each pairing's decision rule
- **Correctness needs ground truth; RelevanceToQuery doesn't; RAG scorers need a trace**
- **Prerequisite traps:** `resources` at log time, CDF for Delta Sync, no index on masked tables, non-Git experiment for tracing, `ai_query` needs serverless DBR 18.2+, App deploy ≠ restart
- Prefer the **current recommended** approach, respect the **stem's qualifier**, and choose the **native Databricks** mechanism
- Evaluate the **claim**, not the buzzword

## Final Checklist

### Knowledge readiness (tick before you book/sit the exam)

- [ ] I can map every exam domain to its repo section and my gap analysis shows no untouched weak spots
- [ ] I can explain LangGraph agent construction and why `ResponsesAgent` is the deployment wrapper
- [ ] I can describe the RAG data pipeline: chunking, embeddings, Delta, AI Search index creation/sync
- [ ] I know models-from-code logging, `resources` auth passthrough, UC registration, and aliases-not-stages
- [ ] I can pick the right serving/FM-API mode and query method for a scenario, including `ai_query()`
- [ ] I can distinguish endpoint guardrails (MASK) from service policies (DENY) and configure UC governance
- [ ] I understand tracing → evaluation → monitoring and which scorers need ground truth vs a trace
- [ ] I've reviewed the renamed-product table and the prerequisite-trap list
- [ ] I've taken at least one full timed mock and reviewed every miss by category
- [ ] I've re-checked the **official exam guide** for blueprint drift since 2026-07-05

### Logistics readiness (test day)

- [ ] Confirmed exam date/time, format (remote proctored vs test center), and time zone
- [ ] Valid government ID ready; name matches the registration exactly
- [ ] For remote: quiet private room, clear desk, working webcam/mic, stable internet, system check passed
- [ ] Pearson VUE / proctor software installed and tested in advance
- [ ] Know the rules: no notes, no second monitor, no phone; scratch policy per proctor
- [ ] Arrived/logged in early (15–30 min) to handle check-in
- [ ] Pacing plan in mind: ~90s/question, checkpoints at Q20/Q40, never leave a blank
- [ ] Rested — sleep beats a last-minute cram

## Self-Check Questions

1. An exam option reads: "Use Vector Search to create the retrieval index for the agent." The current product name is AI Search. Is this option automatically wrong?

<details>
<summary>Answer</summary>
No. "Vector Search" is the former name of AI Search, and the exam sometimes uses old names; the API even still uses `vector-search` identifiers. Judge the option by whether its *claim* is correct (creating a retrieval index for an agent is a valid use), not by the product name alone. Don't reject an option solely because it uses older terminology.
</details>

2. A question asks how to ensure a response containing a customer's email is returned with the email redacted but the rest intact. A candidate picks the `system.ai.block_pii` service policy. Correct?

<details>
<summary>Answer</summary>
No. Service policies (including `system.ai.block_pii`) only return ALLOW/DENY/ASK — they can't transform content, so `block_pii` would DENY the whole response, not redact the email. Redaction requires an **endpoint guardrail** with PII behavior = MASK. This is the classic MASK-vs-DENY trap.
</details>

3. You're unsure between two options: one uses MLflow "stages" to promote a model in Unity Catalog, the other uses an "alias." Which do you pick and why?

<details>
<summary>Answer</summary>
Pick the alias. Unity Catalog does not support stages — stages were the legacy Workspace Model Registry mechanism. In UC, deployment status is expressed with aliases (e.g., `@production`) and environments with separate catalogs. Any option relying on UC stages is wrong.
</details>

4. Give three "looks like it should work but fails" prerequisite traps and the missing step in each.

<details>
<summary>Answer</summary>
(1) Agent works in the notebook but fails auth on the endpoint — missing `resources` declaration at log time (auth passthrough). (2) A Delta Sync index won't sync — Change Data Feed not enabled on the source table. (3) A Databricks App still serves old code after a successful `bundle deploy` — you must `bundle run` to restart it and poll for `RUNNING`. (Others: can't index a table with row filters/column masks; `ai_query()` needs serverless DBR 18.2+; real-time tracing needs a non-Git experiment.)
</details>

## Further Reading

- [Databricks — Generative AI Engineer Associate certification](https://www.databricks.com/learn/certification/generative-ai-engineer-associate) — official exam guide and current logistics/policies
- `06/01/01-full-blueprint-review-and-gap-analysis/notes.md` — coverage map to remediate any unchecked knowledge item
- `06/01/02-mock-exam-and-timing-strategy/notes.md` — pacing and question technique to pair with this trap list
- Section 01–05 **Summary / Quick Recall** blocks — the fastest final-night review of every decision rule referenced here
