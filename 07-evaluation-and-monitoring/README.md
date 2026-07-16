# Section 07 — Evaluation and Monitoring

**Estimated time:** 6 hrs | **Exam domain weight:** ~12%  | **Prerequisites:** Section 05 (Assembling & Deploying), Section 06 (Governance)

---

## Overview

This section covers the complete lifecycle of measuring and maintaining GenAI application quality — from selecting the right metrics at design time, through automated LLM-as-judge evaluation using MLflow, to continuous production monitoring via inference tables and agent traces. It maps to the **Evaluation and Monitoring** exam domain (12% of questions) and directly follows deployment, treating observability as a first-class engineering discipline rather than an afterthought. A well-deployed application without monitoring is an uncontrolled system; this section provides the tools to close that loop.

---

## Learning Outcomes

By completing this section you will be able to:

- Select appropriate evaluation metrics for RAG, summarization, and agent tasks, and explain why reference-based metrics alone are insufficient
- Run `mlflow.evaluate()` with Databricks built-in judges and custom scorers, and interpret per-row judge results
- Design an SME feedback loop that converts human ratings into reusable evaluation datasets
- Enable and query Databricks Model Serving inference tables to monitor production request/response traffic
- Instrument LangGraph agents with MLflow tracing to diagnose failures and attribute token costs

---

## Modules

| # | Module | Est. time | Chapters |
|---|---|---|---|
| 1 | Evaluation | 3 hrs | 3 chapters — Metrics & LLM Choice, MLflow Scoring/Judges/Scorers (LAB-20), SME Feedback Loops |
| 2 | Monitoring | 3 hrs | 2 chapters — Inference Tables & AI Gateway (LAB-21), Cost Control & Agent Monitoring (LAB-22) |

---

## Chapter Map

### Module 01 — Evaluation

| Chapter | Key topics | Lab |
|---|---|---|
| [01 — Evaluation Metrics and LLM Choice](01-evaluation/01-metrics-and-llm-choice/notes.md) | Reference-based vs LLM-as-judge; RAG metrics (faithfulness, answer relevance, context precision/recall); agent KPIs; LLM selection criteria; judge bias mitigations | — |
| [02 — MLflow Scoring, Judges, and Scorers](01-evaluation/02-mlflow-scoring-judges-scorers/notes.md) | `mlflow.evaluate()` API; Databricks built-in judges; `make_genai_metric()` custom scorers; MLflow GenAI Scorers (fast-evolving); evaluation datasets; judge model selection | [LAB-20](01-evaluation/02-mlflow-scoring-judges-scorers/LAB-20-mlflow-scoring-judges-scorers.md) |
| [03 — SME Feedback Loops](01-evaluation/03-sme-feedback-loops/notes.md) | Databricks Review App; annotation workflow design; inter-annotator agreement; converting thumbs-up/down to ground-truth eval sets; active learning for prioritization | — |

### Module 02 — Monitoring

| Chapter | Key topics | Lab |
|---|---|---|
| [01 — Inference Tables and AI Gateway Monitoring](02-monitoring/01-inference-tables-and-ai-gateway/notes.md) | Inference table schema; `auto_capture_config`; AI Gateway payload logging; SQL JSON parsing; Lakehouse Monitoring integration; connecting production traffic to `mlflow.evaluate()` | [LAB-21](02-monitoring/01-inference-tables-and-ai-gateway/LAB-21-inference-tables-and-ai-gateway.md) |
| [02 — Cost Control and Agent Monitoring](02-monitoring/02-cost-control-and-agent-monitoring/notes.md) | Token budgeting; AI Gateway rate limits and spend controls; MLflow tracing for LangGraph (`autolog()`); agent KPIs (latency, tool success rate, loop detection); trace-based failure diagnosis | [LAB-22](02-monitoring/02-cost-control-and-agent-monitoring/LAB-22-cost-control-and-agent-monitoring.md) |

---

## How This Section Fits

Section 06 (Governance) ensured that applications handle sensitive data safely and comply with policy guardrails at the edge. This section picks up after deployment to answer: "Is the application actually working in production?" It provides the feedback signal that makes the entire certification arc coherent — you design, build, deploy, govern, and then *measure*. The insights from monitoring (quality drift, cost overruns, SME-flagged failures) feed directly back into the improvement cycle covered by evaluation, and the datasets produced here power the iterative fine-tuning and prompt-engineering work that would start any next development iteration.

Section 08 (Exam Prep and Readiness) draws heavily on this section's metrics definitions and MLflow APIs for mock exam question reasoning.

---

## Exam Tips for This Domain (12%)

- **Know the five Databricks built-in judges by name**: `answer_correctness`, `answer_relevance`, `groundedness`, `chunk_relevance`, `safety`. Know which evaluate the LLM output vs which evaluate the retrieval layer.
- **`mlflow.evaluate()` parameters**: know `model_type="databricks-agent"`, `data=`, `extra_metrics=`, and what `EvaluationResult` contains.
- **Inference table schema**: know that `auto_capture_config` enables it, that records land in Unity Catalog as Delta, and that request/response are stored as nested JSON requiring `from_json()`.
- **Offline vs online evaluation distinction**: offline uses static datasets + judges; online samples live inference table traffic and routes to judges. Questions often test which approach is appropriate for a given scenario.
- **Judge bias types**: verbosity bias, position bias, self-preference — know at least one mitigation for each.
- **MLflow Tracing (fast-evolving)**: know that `mlflow.langchain.autolog()` captures LangGraph spans, and that traces are queryable via the MLflow UI or `mlflow.search_traces()`.

---

## Study Tips

1. **Lab sequence**: LAB-20 → LAB-21 → LAB-22 builds progressively — do them in order. LAB-20 introduces the eval API, LAB-21 connects production data to it, LAB-22 adds cost instrumentation.
2. **Prioritise MLflow `evaluate()` mechanics** — it appears in multiple exam domains (this one plus Application Development and Assembling & Deploying).
3. **Review the SME feedback chapter carefully** — human evaluation workflow questions are a common "soft" exam topic that surprises candidates who focused only on code.
4. **Fast-evolving caution**: AI Gateway rate limits, MLflow GenAI Scorers, and Lakehouse Monitoring for text data are all under active development. Re-verify against current docs before the exam.
5. **Prerequisites**: ensure you are comfortable with `databricks.agents.deploy()` (Section 05) and Unity Catalog governance (Section 06) — both appear in inference table and review app questions.
