# Evaluation and Monitoring

**Part of section:** Governance, Evaluation, and Monitoring
**Estimated time:** 4 hours
**Prerequisites:** Section 03 (LangGraph agents, `ResponsesAgent`), Section 04 (`agents.deploy()`, inference tables, deployment jobs)
**Exam mapping:** Governance, Evaluation, and Monitoring domain (~15%) — MLflow tracing, evaluation with scorers/judges, and production monitoring

## Overview

This module covers how you know your agent actually works — and keeps working. You will instrument
the agent with **MLflow 3 tracing**, measure quality with **`mlflow.genai.evaluate()`** using built-in
and custom **scorers** and **LLM-as-a-judge**, collect **human feedback** to align judges, and set up
**production monitoring** that runs scorers on live traffic and surfaces quality trends over time.

## Learning Outcomes

By completing this module, you will be able to:

- Instrument an agent with automatic and manual MLflow tracing and interpret spans
- Evaluate an agent with `mlflow.genai.evaluate()` using appropriate built-in and custom scorers
- Explain LLM-as-a-judge, align judges with human feedback, and reason about judge reliability
- Configure production monitoring with scheduled scorers and interpret drift/hallucination signals

## Topics Covered

- MLflow 3 tracing: traces vs spans, span types, autolog, `@mlflow.trace`, real-time tracing for agents
- `mlflow.genai.evaluate()`: datasets, predict functions, scorers
- Built-in scorers (Correctness, RelevanceToQuery, RetrievalGroundedness, Safety, Guidelines, …)
- Deterministic vs LLM-judge scorers; custom `@scorer` functions
- LLM-as-a-judge: `make_judge`, judge alignment with human feedback, reliability metrics
- Databricks Agent Evaluation and the Review App
- Production monitoring: register/start scorers, sampling, dashboards, drift/hallucination detection
- Evaluation gates in CI/CD (deployment jobs)

## How This Module Fits

This is the "measure and watch" half of Section 05, complementing Module 01's "control and secure."
It consumes the traces and inference tables produced by Section 04's `agents.deploy()`, and it feeds
back into Section 04's CI/CD — the same `mlflow.genai.evaluate()` you use here in development becomes
the automated promotion gate in a deployment job.

## Study Tips

- **Tracing → evaluation → monitoring is one loop.** Traces are the data; scorers measure it; human
  feedback aligns the judges; monitoring runs it continuously.
- Memorize which built-in scorers **need ground truth** (`Correctness`) vs **don't** (`RelevanceToQuery`,
  `Safety`), and that RAG scorers (`RetrievalGroundedness`, `RetrievalRelevance`) **need a trace**.
- Know the **production-monitoring constraints**: `@scorer`-decorator scorers only, defined in a
  notebook, self-contained; register then start; ≤20 scorers per experiment.

## Chapters

- [ ] [01 MLflow3 Tracing, Scoring, and AI Gateway](./01-mlflow3-tracing-scoring-and-ai-gateway/notes.md) — 4 hrs
