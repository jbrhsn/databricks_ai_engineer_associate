# Full Blueprint Review and Gap Analysis

**Section:** Exam Prep | **Module:** Review and Practice | **Est. time:** 1.5 hrs
**Exam mapping:** Not directly scored — a coverage map and self-assessment spanning all five exam domains

## Learning Objectives

By the end of this chapter, you will be able to:

- Recite the five exam domains, their weights, and the repo sections that cover each
- Map every major exam objective to specific chapters for targeted review
- Run an honest gap-analysis self-assessment and rank your weak areas
- Build a prioritized remediation plan that spends time where it earns the most points
- Decide when the official blueprint has drifted enough to warrant re-checking the source

## Core Concepts

### The blueprint at a glance

The Databricks Certified Generative AI Engineer Associate exam is **60 questions in 90 minutes**, pass
mark **~70%** (about 42 correct). Questions are single-best-answer and multi-select. Five domains,
weighted:

| # | Domain | Weight | ≈ Questions | Repo section |
|---|--------|--------|-------------|--------------|
| 1 | Design and build LLM applications | ~30% | ~18 | 03 Application Development |
| 2 | Data preparation and management | ~20% | ~12 | 02 Data Preparation |
| 3 | Assembling and deploying applications | ~20% | ~12 | 04 Assembling & Deploying |
| 4 | Governance, evaluation, and monitoring | ~15% | ~9 | 05 Governance/Eval/Monitoring |
| 5 | Foundations (GenAI, prompting, tools) | ~15% | ~9 | 01 Foundations |

The single most important takeaway: **Application Development (~30%) is the largest domain, and
Data Prep + Deployment together are another ~40%.** If you're time-constrained, the highest-leverage
review is Sections 03, 02, and 04 — roughly 70% of the exam.

### Objective-to-content coverage map

Use this to jump straight to the chapter that covers any weak objective.

**Domain 1 — Design and build LLM applications (~30%) → Section 03**
- Chains and RAG pipelines with LangGraph (StateGraph, nodes, edges, checkpointers) →
  `03/01-.../01-chains-langgraph-patterns-and-memory`
- Tools, ReAct agents, multi-agent/supervisor design, MCP tools →
  `03/01-.../02-agents-tools-and-multi-agent-design`
- Databricks-native agents (Knowledge Assistant, `ResponsesAgent`, Databricks Apps, Supervisor Agent) →
  `03/01-.../03-agent-bricks-and-databricks-native-agents`
- Embedding models, input/output guardrails, model selection for roles →
  `03/02-.../01-embedding-models-guardrails-and-safety`

**Domain 2 — Data preparation and management (~20%) → Section 02**
- Parsing, chunking strategies, deduplication, embedding-model choice →
  `02/01-.../01-extraction-and-chunking-strategies`
- Delta Lake (ACID, time travel, medallion), Auto Loader, UC Volumes, AI Search index basics →
  `02/01-.../02-delta-lake-and-unity-catalog-for-genai-data`

**Domain 3 — Assembling and deploying applications (~20%) → Section 04**
- MLflow 3 models-from-code logging, `resources` auth passthrough, UC registration, aliases →
  `04/01-.../01-pyfunc-mlflow-and-unity-catalog-registration`
- AI Search endpoints/indexes, sync modes, query types, reranking →
  `04/01-.../02-vector-search-indexing-and-config`
- Model Serving, `agents.deploy()`, FM API modes, `ai_query()`, MCP servers →
  `04/02-.../01-model-serving-foundation-model-apis-and-mcp`
- Prompt Registry, Declarative Automation Bundles/CI-CD, Databricks Apps →
  `04/02-.../02-prompt-registry-cicd-and-databricks-apps-uis`

**Domain 4 — Governance, evaluation, and monitoring (~15%) → Section 05**
- UC governance, guardrails vs service policies, PII BLOCK/MASK, Unity AI Gateway, DASF →
  `05/01-.../01-guardrails-masking-and-data-governance`
- MLflow tracing/spans, `mlflow.genai.evaluate()`, scorers, LLM judges, production monitoring →
  `05/02-.../01-mlflow3-tracing-scoring-and-ai-gateway`

**Domain 5 — Foundations (~15%) → Section 01**
- LLM/GenAI core concepts → `01/02-.../01-llm-and-genai-core-concepts`
- Prompt engineering and structured I/O → `01/02-.../02-prompt-engineering-and-structured-io`
- Matching models to tasks, problem decomposition → `01/02-.../03-matching-models-tasks-and-problem-decomposition`
- Prerequisites and workspace orientation (supporting content) → `01/01-prerequisites-and-setup`

### How to run a gap analysis

A gap analysis is a deliberate, honest inventory of what you *can't* reliably do yet. Do it in three
passes:

1. **Rate each objective 1–3** without opening the notes:
   - **3 = confident** — could teach it and answer application questions
   - **2 = shaky** — recognize it but couldn't reconstruct details or apply under pressure
   - **1 = weak** — vague or blank
2. **Weight by domain size.** A weak objective in Application Development (~30%) matters far more than
   a weak one in Foundations prerequisites (supporting content). Multiply your gap by the domain weight
   to get a *priority score*.
3. **Rank and schedule.** Attack the highest priority-score gaps first, using the coverage map above to
   go straight to the relevant chapter's **Summary / Quick Recall** and **Self-Check Questions**.

The trap here is comfort-studying — re-reading what you already know because it feels productive. The
whole value of a gap analysis is that it forces time onto the uncomfortable material.

## Deep Dive / Advanced Topics

### Cert-required vs depth/mastery knowledge

The repo (per `authoring-guidelines.md`) distinguishes **cert-required** knowledge (what you need to
pass) from **depth/mastery** knowledge (what makes you good at the job). For exam prep specifically,
prioritize cert-required material: the API names, the decision rules, the "which tool for which job"
mappings, and the renamed-product facts. Deep implementation nuance is valuable professionally but
lower-yield for a 60-question multiple-choice exam. When triaging a gap, ask "is this likely to be a
question?" — if it's a subtle edge case that even the docs bury, it's lower priority than a core
decision rule.

### Detecting blueprint drift

This repo targets the **2026-07-05** blueprint. Databricks moves fast, so before your exam, confirm
the official guide hasn't shifted. Signs of drift (from the roadmap):
- Official objectives that don't map to any chapter here
- New Databricks features widely referenced in docs but absent here
- A domain weight that has moved ≥5 percentage points

If you spot drift, trust the **official exam guide** over this repo, and note the delta. Product
renames are the most common drift you'll actually encounter on the exam — the Pitfalls chapter
(`06/01/03`) lists the ones known as of this repo's date.

## Worked Examples & Practice

### A filled-in gap-analysis example

Here's what one learner's prioritized gap analysis might look like after the first pass:

| Objective | Domain | Weight | Self-rating | Priority (gap × weight) | Action |
|---|---|---|---|---|---|
| MLflow models-from-code + `resources` | Deploy | 20% | 1 (weak) | **High** | Re-study `04/01/01`, redo self-checks |
| TRIGGERED vs CONTINUOUS sync | Data/Deploy | 20% | 2 (shaky) | Med-High | Re-read `04/01/02` summary |
| Guardrails BLOCK vs MASK | Governance | 15% | 1 (weak) | Med-High | Re-study `05/01/01` |
| LangGraph StateGraph basics | App Dev | 30% | 3 (confident) | Low | Skip |
| Prompt engineering patterns | Foundations | 15% | 3 (confident) | Low | Skip |
| Chunking strategies | Data | 20% | 2 (shaky) | Medium | Re-read `02/01/01` summary |

The learner should spend their limited time on models-from-code, guardrail behaviors, and sync modes —
not re-reading LangGraph basics they already know. **The gap analysis converted a vague "I should
review everything" into three concrete, high-yield targets.**

**Failure mode to observe:** a learner who rates everything "2" to avoid confronting real weaknesses
ends up with a flat priority list and defaults to re-reading front-to-back — spending 30% of their
time on Foundations they already know and running out of time before the heavily weighted Deployment
material. An honest 1/2/3 split is what makes the analysis actionable.

## Common Pitfalls & Misconceptions

- **Pitfall:** Studying every domain equally → **Why it happens:** it feels fair and thorough → **Fix:** weight by exam percentage. App Dev (30%) + Data (20%) + Deploy (20%) = 70% of the exam; that's where review time should concentrate.

- **Pitfall:** Comfort-studying material you already know → **Why it happens:** it produces a feeling of progress → **Fix:** the gap analysis exists to redirect time to weak, high-weight objectives. If a topic is already a "3," skip it.

- **Pitfall:** Trusting this repo's blueprint over the current official guide → **Why it happens:** the repo is convenient and self-contained → **Fix:** re-check the official exam guide before testing; on any conflict, the official guide wins. Databricks renames products often.

- **Pitfall:** Rating gaps without weighting them → **Why it happens:** a raw list of weak spots seems like enough → **Fix:** multiply each gap by its domain weight so you attack the objectives that actually move your score.

## Key Definitions

| Term | Definition |
|---|---|
| Exam blueprint | The official specification of domains, weights, and objectives the exam tests |
| Domain weight | The approximate share of exam questions drawn from a domain (e.g., App Dev ~30%) |
| Gap analysis | A structured, honest self-assessment of which objectives you can't yet reliably apply |
| Priority score | Gap severity multiplied by domain weight, used to rank what to study first |
| Cert-required knowledge | What you must know to pass, as opposed to deeper professional mastery |
| Blueprint drift | Divergence between a repo's targeted blueprint and the current official exam guide |

## Summary / Quick Recall

- **60 questions / 90 minutes / ~70% to pass**; five domains
- Weights: **App Dev ~30%**, Data ~20%, Deploy ~20%, Governance/Eval ~15%, Foundations ~15%
- App Dev + Data + Deploy ≈ **70% of the exam** — concentrate review there
- Gap analysis: rate objectives **1/2/3**, **weight by domain size**, then attack highest priority scores first
- Use the coverage map to jump straight to the relevant chapter's Summary + Self-Check
- Prioritize **cert-required** knowledge (decision rules, API names, renamed products) over deep edge cases
- **Re-check the official blueprint** before the exam; on conflict, the official guide wins

## Self-Check Questions

1. You have 6 hours of review time left and rate yourself weak on both "LLM tokenization details" (Foundations) and "MLflow models-from-code logging" (Deployment). Which do you prioritize and why?

<details>
<summary>Answer</summary>
Prioritize models-from-code logging. Deployment is ~20% of the exam and models-from-code is a core, frequently tested decision (`python_model` as a file path, `resources` for auth passthrough). Foundations is ~15% and tokenization details are lower-yield supporting knowledge. Weighting the gap by domain size sends your limited time to the higher-impact objective.
</details>

2. Which three repo sections should a time-constrained candidate review first, and roughly what share of the exam do they cover?

<details>
<summary>Answer</summary>
Sections 03 (Application Development, ~30%), 02 (Data Preparation, ~20%), and 04 (Assembling & Deploying, ~20%) — together roughly 70% of the exam. Concentrating review there maximizes expected score per hour.
</details>

3. Why is rating every objective "2" during a gap analysis self-defeating?

<details>
<summary>Answer</summary>
A flat rating produces no prioritization, so you fall back to reviewing everything front-to-back — wasting time on material you already know and likely running out before the heavily weighted domains. Honest 1/2/3 ratings (and weighting by domain) are what turn the analysis into an actionable, high-yield plan.
</details>

4. You find an objective in the official exam guide that doesn't appear anywhere in this repo. What does that tell you and what should you do?

<details>
<summary>Answer</summary>
It's a sign of blueprint drift — the official guide has likely added an objective since this repo's 2026-07-05 target date. Trust the official guide, study that objective from current Databricks docs, and treat the repo as incomplete for that item. Product renames and new features are the most common causes of such gaps.
</details>

## Further Reading

- [Databricks — Generative AI Engineer Associate certification](https://www.databricks.com/learn/certification/generative-ai-engineer-associate) — the authoritative, current exam guide and objectives
- `00-roadmap/learning-roadmap.md` — this repo's domain-weight table, logistics, and blueprint-drift warning
- `progress-tracker.md` — the full chapter checklist to confirm you've completed every objective's source material
