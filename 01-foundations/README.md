# Foundations

**Estimated time:** ~15 hrs | **Exam domain weight:** N/A (foundational — underpins all six graded domains) | **Prerequisites:** None

## Overview

This section builds the mental model everything else in the course rests on: how LLMs actually work, how retrieval turns a frozen model into a system that answers from your own data, and how agents and orchestration frameworks turn a single model call into a controllable application. It is deliberately foundational — none of it is a graded exam domain on its own, but every graded domain (Design Applications, Data Preparation, Application Development, Assembling & Deploying, Governance, Evaluation & Monitoring) assumes you already own these concepts. Get this section right and the rest of the exam becomes vocabulary you already speak.

## Learning Outcomes

By completing this section you will be able to:
- Explain how tokenization, embeddings, and next-token prediction produce LLM behaviour, and reason about token-driven cost, latency, and context limits.
- Match an NLP task to the right model family and select a Databricks serving option (Foundation Model APIs, provisioned throughput, external models via AI Gateway).
- Engineer prompts using roles, few-shot examples, structured output, and decoding parameters — and decide between prompting, RAG, and fine-tuning.
- Describe the RAG pipeline end to end and choose RAG vs fine-tuning vs long-context for a given requirement.
- Explain embeddings, similarity metrics, and ANN retrieval, and configure a Mosaic AI Vector Search index (Delta Sync vs Direct Vector Access, managed vs self-managed embeddings).
- Distinguish an agent from a chain, apply the reason–act–observe loop, and choose between LangGraph, LangChain, and CrewAI for an orchestration problem.

## Modules

| # | Module | Est. time | Chapters |
|---|---|---|---|
| 1 | LLM & NLP Fundamentals | ~7 hrs | 3 |
| 2 | RAG & Retrieval Foundations | ~4.5 hrs | 2 |
| 3 | Agentic AI & Frameworks | ~5 hrs | 2 |

### Module 1 — LLM & NLP Fundamentals
- [01 · How LLMs Work: Tokens & Embeddings](01-llm-and-nlp-fundamentals/01-how-llms-work-tokens-embeddings/notes.md)
- [02 · Model Families & NLP Tasks](01-llm-and-nlp-fundamentals/02-model-families-and-nlp-tasks/notes.md)
- [03 · Prompt Engineering Fundamentals](01-llm-and-nlp-fundamentals/03-prompt-engineering-fundamentals/notes.md) · [LAB-01](01-llm-and-nlp-fundamentals/03-prompt-engineering-fundamentals/LAB-01-prompt-engineering-fundamentals.md)

### Module 2 — RAG & Retrieval Foundations
- [01 · What Is RAG and When to Use It](02-rag-and-retrieval-foundations/01-what-is-rag-and-when/notes.md)
- [02 · Embeddings & Vector Search Intro](02-rag-and-retrieval-foundations/02-embeddings-and-vector-search-intro/notes.md)

### Module 3 — Agentic AI & Frameworks
- [01 · Agentic AI Concepts](03-agentic-ai-and-frameworks/01-agentic-ai-concepts/notes.md)
- [02 · Framework Landscape: LangGraph, LangChain, CrewAI](03-agentic-ai-and-frameworks/02-framework-landscape-langgraph-etc/notes.md)

## How This Section Fits

This is the entry point of the course — it has no prerequisites and everything downstream depends on it. Module 1 gives you the primitives (tokens, embeddings, model families, prompts) that the *Design Applications* and *Data Preparation* domains assume. Module 2 introduces retrieval, which the *Application Development* domain expands into full RAG pipelines on Databricks. Module 3 introduces agents and orchestration frameworks, which *Application Development* and *Assembling & Deploying Applications* build on when you author and serve agents through the Mosaic AI Agent Framework. Finish here and you unlock the first graded domain.

## Study Tips

- **Set up a Databricks workspace early.** Enable Foundation Model APIs and confirm you can query a chat endpoint before LAB-01 — the labs from here on assume working serverless compute and endpoint access.
- **Do not skip embeddings.** They appear in three separate places (LLM internals, RAG, and Vector Search) and are the single most reused concept in the whole certification. If the ELI5 in chapter 1.1 doesn't click, re-read it before moving on.
- **Learn the decision frameworks, not just the definitions.** The exam rewards "prompt vs RAG vs fine-tune" and "chain vs agent" judgement far more than reciting what each one is — study the Worked Examples in chapters 1.3, 2.1, and 3.1 closely.
- **Watch for terminology drift.** Databricks is actively renaming and reshaping Foundation Model APIs, Mosaic AI Vector Search, and the Agent Framework. Each chapter flags fast-evolving surfaces; verify endpoint and product names against current official docs before relying on them in a lab.
- **Anchor frameworks correctly:** LangGraph is the primary orchestration layer, LangChain is the component library used inside it, CrewAI is comparison-only, and the Databricks Agent Framework is framework-agnostic. Keep those roles straight from the start.
