# Module 02 — Embeddings, Guardrails, and Model Selection

**Part of section:** 03 Application Development
**Estimated time:** 5 hours
**Prerequisites:** Module 01 (LangGraph patterns, RAG chain); Section 02 (chunking, AI Search index)
**Exam mapping:** Application Development domain (~30%); embedding model integration in LangGraph, guardrail design, and model selection are exam-tested

## Overview

This module completes the application layer by addressing two critical production concerns:
embedding model integration (how chunked text becomes queryable vectors inside a LangGraph
retrieval node) and guardrails (how to prevent harmful, off-topic, or privacy-violating
content from flowing in or out of your agent).

Databricks provides native embedding models via Foundation Model API, and provides guardrails
through Unity AI Gateway service policies (Beta, 2026). This module covers both.

## Learning Outcomes

By completing this module, you will be able to:

- Use `DatabricksEmbeddings` in a LangGraph retrieval node to embed queries at query time
- Select the right Databricks embedding model based on domain, language, and token limit
- Explain the difference between embedding at indexing time (pipeline) and at query time (agent)
- Implement input guardrails that detect and block PII, prompt injection, and off-topic queries
- Implement output guardrails that validate schema compliance and filter harmful content
- Apply Unity AI Gateway service policies for guardrails at the governance layer
- Articulate the model selection trade-offs for each agent role: generator, retriever, router, evaluator

## Topics Covered

- Databricks embedding models: `databricks-bge-large-en`, `databricks-gte-large-en`, `databricks-qwen3-embedding-0-6b`
- `DatabricksEmbeddings` class from `databricks-langchain`
- Embedding at query time vs. embedding at index time — which model to use
- Input guardrail patterns: PII detection, prompt injection defense, topic classification
- Output guardrail patterns: schema validation (Pydantic), content moderation, length limits
- Unity AI Gateway service policies: built-in guardrails (PII, injection, unsafe content)
- Model selection framework for agent roles (generator, router, embedder, evaluator)

## How This Module Fits

Module 01 built the LangGraph graph structure (nodes, edges, state, tools). This module
completes the application by adding embedding-aware retrieval nodes and guardrail nodes.
The embeddings material connects back to Section 02 (where the index was built) and forward
to Section 04 (where the model serving endpoint for embeddings is configured). The guardrails
material connects forward to Section 05 (Governance), which covers MLflow AI Gateway and
production monitoring.

## Study Tips

- **There are two moments embeddings are used:** at pipeline time (when chunks are indexed —
  Section 02) and at query time (when the user's question is vectorized for similarity search).
  The same embedding model must be used at both times. This is a common exam question.
- **`DatabricksEmbeddings` is the query-time integration** — it calls the embedding endpoint
  inside a LangGraph retrieval node to vectorize the user's question before calling AI Search.
  When you use AI Search with managed embeddings (`embedding_source_column`), this step is
  handled automatically. When you use pre-computed embeddings, you must embed the query yourself.
- **Guardrails are not optional in production** — they are a scored exam topic and an expected
  production practice. Know at least two input guardrail patterns and two output guardrail patterns.
- **Unity AI Gateway service policies** are the Databricks-recommended governance-layer approach
  to guardrails (Beta 2026). Know what they protect against (PII, injection, unsafe content).

## Chapters

- [ ] [01 Embedding Models, Guardrails, and Safety](./01-embedding-models-guardrails-and-safety/notes.md) — 5 hrs
