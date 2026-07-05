# Capstone — End-to-End RAG Agent on Databricks

**Estimated time:** Variable (12–20 hours is typical for a complete, deployed submission)
**Prerequisites:** Sections 01–05 complete. This project exercises every domain in the exam blueprint at once.
**Exam mapping:** Not a scored exam objective. This is an integrative project that consolidates all five domains (Foundations, Data Preparation, Application Development, Assembling and Deploying, Governance/Evaluation/Monitoring) into one working system.

## Overview

The chapters taught each stage of the GenAI engineering lifecycle in isolation. The capstone makes
you assemble them into a single, working, governed application: a **Retrieval-Augmented Generation
(RAG) agent** that ingests a document corpus, retrieves relevant context from a vector index, answers
questions grounded in that context, and runs as a deployed, evaluated, monitored service on Databricks.

The exam rewards people who understand how the pieces *connect* — which API belongs to which lifecycle
step, why a step exists, and what breaks downstream when you skip it. Building the whole thing end to
end is the single best way to internalize that flow. If you can build and deploy this system and explain
each decision, you are ready for the exam and, more importantly, for the job.

> **This is a learning artifact, not a graded assignment.** There is no autograder. The rubric below
> is a self-assessment tool. Score yourself honestly — the value is in identifying which lifecycle
> stage you're weakest on while you still have time to fix it before the exam.

## Learning Outcomes

By completing this capstone, you will be able to:

- Design and justify an end-to-end RAG architecture on Databricks, naming the product used at each stage
- Build a document ingestion + chunking + embedding pipeline landing in Delta and Unity Catalog
- Create and sync an AI Search (Vector Search) index from a Delta source table
- Implement a retrieval agent (LangGraph + `ChatDatabricks`) that grounds answers in retrieved context
- Package the agent as an MLflow 3 model (models-from-code) with the correct `resources` declared
- Register the model to Unity Catalog and deploy it with `agents.deploy()`
- Instrument the system with MLflow tracing, run `mlflow.genai.evaluate()`, and reason about the scores
- Apply guardrails and governance appropriate to the corpus and audience
- Articulate the trade-offs you made and the failure modes you observed

## The Project Brief

Build a RAG agent that answers questions over a document corpus **you choose**. Suggested corpora
(pick one — or bring your own):

- **This repository's own content** — a "study assistant" that answers Databricks GenAI exam questions
  from the section notes (nicely self-referential, and the corpus is already Markdown).
- A **public documentation set** (e.g., a product's docs, a set of research PDFs, a policy manual).
- An **internal knowledge base** you have legitimate access to (runbooks, FAQs, wiki exports).

Keep the corpus small enough to iterate quickly (tens to low hundreds of documents) but large enough
that retrieval actually matters (i.e., the answer isn't obvious from a single chunk).

### Functional requirements

1. **Ingest & prepare** the corpus: parse, chunk with a defensible strategy, attach metadata, and land
   the chunks in a **Delta table** in **Unity Catalog** (Section 02).
2. **Index**: create an **AI Search** endpoint and a **Delta Sync index** over the chunk table, with a
   sync mode (TRIGGERED vs CONTINUOUS) and embedding approach (managed vs self-managed) you can justify
   (Sections 02 & 04).
3. **Build the agent**: a **LangGraph** graph using `ChatDatabricks` that retrieves top-k chunks, injects
   them into a grounded prompt, and returns an answer **with citations** to the source chunks (Section 03).
4. **Add safety**: at minimum, a grounding/relevance check and appropriate input/output guardrails for
   your corpus and audience (Sections 03 & 05).
5. **Package & register**: log the agent as an **MLflow 3** model with the **models-from-code** pattern,
   declaring all `resources` (serving endpoint + vector index), and register it to **Unity Catalog** with
   an alias (Section 04).
6. **Deploy**: serve the agent via **`agents.deploy()`** and query the live endpoint successfully
   (Section 04).
7. **Evaluate & monitor**: build a small evaluation dataset (≥10 question/answer pairs), run
   **`mlflow.genai.evaluate()`** with relevant scorers (correctness, groundedness/faithfulness,
   relevance), enable **tracing**, and record the results (Section 05).

### Stretch goals (optional, for depth/mastery — not required to "pass" the capstone)

- A **Databricks Apps** chat UI bound to the serving endpoint.
- **CI/CD** with a Declarative Automation Bundle (`databricks.yml`) and a deployment job.
- **Hybrid search** (ANN + full-text with RRF) and/or a reranking step, with a before/after eval comparison.
- **Prompt Registry** versioning: register your grounded prompt and reference it by alias.
- A **multi-agent** or supervisor pattern (e.g., a router that chooses between retrieval and a direct answer).
- An **LLM-as-a-judge** custom scorer with alignment against a few human labels.

## Suggested Phased Approach

Work in phases and get each one working end to end before polishing — a deployed mediocre pipeline
teaches you more than a perfect ingestion step that never reaches serving.

```
Phase 1 — Data     : corpus → parse/chunk → Delta table in UC          (Section 02)
Phase 2 — Index    : AI Search endpoint + Delta Sync index             (Sections 02, 04)
Phase 3 — Agent    : LangGraph retrieval agent with grounded prompt    (Section 03)
Phase 4 — Package  : MLflow 3 models-from-code + UC registration       (Section 04)
Phase 5 — Deploy   : agents.deploy() → live endpoint → query it        (Section 04)
Phase 6 — Evaluate : eval dataset + mlflow.genai.evaluate() + tracing  (Section 05)
Phase 7 — Govern   : guardrails, grants, and a monitoring plan         (Section 05)
Phase 8 — Reflect  : fill in submission-template.md, score the rubric  (this section)
```

A common mistake is spending all your time on Phase 1 chunking tuning. Get a naive chunker working,
push through to a deployed, evaluated system, *then* come back and improve chunking with eval numbers
to prove the improvement.

## Self-Assessment Rubric

Score each dimension 0–3. This mirrors how the exam distributes weight across domains, so a low score
on any row points directly at the section you should revisit.

| Dimension (maps to exam domain) | 0 — Missing | 1 — Basic | 2 — Solid | 3 — Strong |
|---|---|---|---|---|
| **Data prep** (Sec 02, ~20%) | No pipeline | Chunks in Delta, no metadata/justification | Chunks + metadata in UC, defensible strategy | + dedup/quality handling, chunking justified with eval |
| **App development** (Sec 03, ~30%) | No agent | Retrieves + answers, no grounding/citations | Grounded answers with citations, guardrail present | + hybrid/rerank or multi-agent, semantic routing |
| **Assembling & deploying** (Sec 04, ~20%) | Runs only in notebook | Logged to MLflow, not deployed | Registered to UC + deployed via `agents.deploy()`, queried live | + CI/CD bundle and/or Apps UI, prompt versioned |
| **Governance/eval/monitoring** (Sec 05, ~15%) | No evaluation | Manual spot checks | `mlflow.genai.evaluate()` with scorers + tracing on | + custom judge w/ alignment, monitoring + guardrails plan |
| **Foundations & reasoning** (Sec 01, ~15%) | Can't explain choices | Explains what, not why | Justifies model/prompt/chunk choices | + articulates trade-offs and observed failure modes |

**Reading your score (out of 15):**
- **12–15** — Exam-ready across the board. You understand the full lifecycle.
- **8–11** — Solid. Revisit the section(s) where you scored ≤1 before sitting the exam.
- **≤7** — Rebuild the weakest phase end to end before scheduling the exam; the gaps here are the ones the exam will find.

## Deliverable

Complete [`submission-template.md`](./submission-template.md). It's a structured write-up that forces you
to state each architectural decision and its justification, paste the key code, record your evaluation
numbers, and reflect on failure modes. Filling it in honestly *is* the exam-prep value — if you can
complete every section from your own project, you understand the material.

## How This Section Fits

This is the terminal, integrative artifact of the repo. Sections 01–05 taught the stages; Section 06
prepared you to recognize them under exam conditions; the capstone proves you can wire them together.
Do it after Section 05 (you need evaluation and governance to complete it) and ideally alongside or
after Section 06's gap analysis, using whatever weak spots that analysis surfaced as your focus areas here.

## Study Tips

- **Build thin, then deep.** A deployed, mediocre-but-complete pipeline (all 8 phases) teaches more than
  a beautifully tuned Phase 1 that never ships. Get end-to-end working first.
- **Write the justification as you go, not at the end.** The submission template's "why" fields are where
  the learning lives. If you can't justify a choice in one sentence, that's a gap to close now.
- **Let evaluation drive iteration.** Once Phase 6 works, every improvement (better chunking, hybrid
  search, reranking) should be validated by a before/after eval number — that's exactly the practitioner
  discipline the exam is proxying for.
- **Watch the 2026 renames** while you build: AI Search (formerly Vector Search), Unity AI Gateway
  (formerly Mosaic AI Gateway), Declarative Automation Bundles (formerly Asset Bundles), MLflow aliases
  (not stages). Using the current names as you build cements them for the exam.

## Further Reading

- Databricks — Mosaic AI Agent Framework (build & deploy agents): https://docs.databricks.com/aws/en/generative-ai/agent-framework/
- Databricks — RAG / retrieval and data pipeline guide: https://docs.databricks.com/aws/en/generative-ai/tutorials/ai-cookbook/
- Databricks — AI Search (Vector Search): https://docs.databricks.com/aws/en/generative-ai/vector-search
- MLflow 3 — GenAI evaluation (`mlflow.genai.evaluate`): https://mlflow.org/docs/latest/genai/eval-monitor/
- LangGraph documentation: https://docs.langchain.com/oss/python/langgraph/
