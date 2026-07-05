# Section 05 — Governance, Evaluation, and Monitoring

**Estimated time:** 8 hours
**Prerequisites:** Section 03 (agents, guardrails, Unity AI Gateway intro), Section 04 (Model Serving, `agents.deploy()`, MLflow logging, Unity Catalog registration)
**Exam mapping:** Databricks GenAI Engineer Associate — Governance, Evaluation, and Monitoring domain (~15% of exam)

## Overview

Section 04 shipped the agent to production; this section keeps it safe, correct, and observable
over time. You will learn how Databricks governs GenAI — controlling *who* can access data and
models (Unity Catalog), *what* content flows through the endpoints (guardrails and service
policies), and *how* the whole system is instrumented (MLflow tracing), *measured* (evaluation with
scorers and LLM judges), and *watched* in production (monitoring).

Key Databricks-specific context from live docs (2026-06 to 2026-07):
- **Unity Catalog is the single governance layer for data *and* AI.** Models, functions, AI Search
  indexes, prompts, agents, MCP servers, and connections are all UC **securables** governed with
  standard `GRANT`/`REVOKE`, plus fine-grained **row filters and column masks** (now increasingly
  via **ABAC policies**).
- **Unity AI Gateway** (formerly Mosaic AI Gateway, **Beta** in 2026) is the governance control plane
  for LLM traffic: guardrails, rate limits, spend caps, usage tracking, and payload logging.
- **Two generations of guardrails coexist:** the older **endpoint guardrails** (PII **BLOCK/MASK**
  via Presidio; safety via **Llama Guard**) and the newer **service policies** (`system.ai.block_*`,
  **ALLOW/DENY/ASK**, deny-only).
- **MLflow 3** provides **tracing** (spans), **`mlflow.genai.evaluate()`** with built-in and custom
  **scorers** and **LLM-as-a-judge**, and **production monitoring** (scheduled scorers on live traffic).
- **Responsible AI** on Databricks leans on lineage + audit logs for auditability, and the
  **DASF (Databricks AI Security Framework)** for a structured risk/control taxonomy.

## Learning Outcomes

By completing this section, you will be able to:

- Govern GenAI data and assets with Unity Catalog securables, grants, row filters, and column masks
- Configure input/output guardrails and distinguish endpoint guardrails from service policies
- Explain PII BLOCK vs MASK behavior and where masking is (and isn't) available
- Enable and configure Unity AI Gateway features: rate limits, spend caps, usage tracking, payload logging
- Instrument an agent with MLflow tracing and interpret spans (RETRIEVER, TOOL, CHAT_MODEL, etc.)
- Evaluate an agent with `mlflow.genai.evaluate()` using built-in and custom scorers
- Explain how LLM-as-a-judge works and how to align judges with human feedback
- Set up production monitoring with scheduled scorers and interpret quality trends
- Situate evaluation gates within the CI/CD promotion flow from Section 04

## Topics Covered

**Module 01 — Governance and Responsible AI**
- Unity Catalog as the governance layer for data and AI; securables and the three-level namespace
- Grants/revokes; row filters and column masks; ABAC policies; the vector-index masking limitation
- Endpoint guardrails: PII (Presidio, BLOCK/MASK/None), safety filter (Llama Guard), keywords, topics
- Service policies (Beta): `system.ai.block_pii/unsafe_content/jailbreak/hallucination`, ALLOW/DENY/ASK
- Unity AI Gateway: rate limits, spend caps/budgets, usage tracking, payload logging, fallbacks
- Responsible AI framing; DASF (12 components / 62 risks / 64 controls); lineage and audit logs
- Compliance framing: HIPAA-eligibility, data residency (Geos), secrets/credentials

**Module 02 — Evaluation and Monitoring**
- MLflow 3 tracing: traces vs spans, span types, automatic (autolog) and manual (`@mlflow.trace`)
- Real-time tracing for deployed agents; the Git-folder gotcha; production trace storage
- `mlflow.genai.evaluate()`: datasets, scorers, predict functions
- Built-in scorers (Correctness, RelevanceToQuery, RetrievalGroundedness, Safety, Guidelines, …)
- Custom scorers (`@scorer`); deterministic vs LLM-judge scorers
- LLM-as-a-judge: `make_judge`, judge alignment with human feedback, judge reliability
- Databricks Agent Evaluation and the Review App for human feedback
- Production monitoring: register/start scorers, sampling, dashboards, drift/hallucination detection
- Evaluation gates in CI/CD (deployment jobs)

## How This Section Fits

Section 03 built the agent and introduced guardrails and the AI Gateway at a conceptual level;
Section 04 deployed the agent and set up inference tables and real-time tracing via `agents.deploy()`.
This section is where those hooks pay off: you govern the data and assets the agent touches, enforce
content policies on its traffic, and close the quality loop with tracing → evaluation → monitoring.
It also connects back to the CI/CD deployment jobs from Section 04, which use `mlflow.genai.evaluate()`
as an automated promotion gate. Section 06 (Exam Prep) reviews all of this against the blueprint.

## Study Tips

- **Know the two guardrail generations cold.** Endpoint guardrails support PII **MASK**; service
  policies are **deny-only** (ALLOW/DENY/ASK). If a question asks about *masking* PII, that's the
  endpoint guardrail path, not service policies.
- **Unity Catalog governs everything** — data, models, functions, indexes, prompts, agents, MCP.
  Grants decide *whether* you can call; service policies decide *how* the interaction proceeds.
- **RetrievalGroundedness is the anti-hallucination scorer** and it needs a trace with a RETRIEVER
  span. Memorize which scorers need ground truth (`Correctness`) vs don't (`RelevanceToQuery`).
- **Tracing → evaluation → monitoring is one continuous loop.** Traces are the raw material; scorers
  measure them offline (evaluation) or online (monitoring); human feedback aligns the judges.
- **Watch the renames and Beta flags:** Unity AI Gateway (formerly Mosaic AI Gateway, Beta) and the
  service-policies mechanism (Beta) are recent — the exam may reference either the old or new terms.

## Chapters / Modules

- [ ] [01 Governance and Responsible AI](./01-governance-and-responsible-ai/) — 4 hrs
- [ ] [02 Evaluation and Monitoring](./02-evaluation-and-monitoring/) — 4 hrs
