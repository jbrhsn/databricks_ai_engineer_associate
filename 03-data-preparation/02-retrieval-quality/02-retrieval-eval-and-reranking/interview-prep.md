# Retrieval Evaluation and Reranking — Interview Prep

**Section:** 03-Data Preparation | **Role target:** GenAI Engineer, Solutions Architect, Data Engineer

## Core Conceptual Questions

These test whether you understand the fundamentals.

| Question | Key points to cover | Common weak-answer trap |
|---|---|---|
| What is the difference between a bi-encoder and a cross-encoder? | Bi-encoders embed query/doc separately (fast, pre-computable). Cross-encoders process them together (slow, deep semantic match). | Saying cross-encoders are just "better models" without explaining the architectural difference (joint vs independent embedding). |
| Why is "correctness" not enough when evaluating a RAG pipeline? | Correctness just means the answer is factually true. Groundedness means the answer comes strictly from the retrieved context. | Failing to distinguish between the LLM using parametric memory vs retrieved context. |
| What is Reciprocal Rank Fusion (RRF)? | An algorithm that combines multiple search results (e.g., keyword and vector) by ranking them based on their position rather than raw score. | Claiming it normalizes cosine similarity and BM25 scores (it explicitly ignores raw scores). |

## Applied / Scenario Questions

**Q:** You've built a RAG pipeline that searches a 1-million document database. Users are complaining that the answers are generic and miss highly specific details requested in their queries. How do you fix this without breaking the sub-second latency requirement?

**Strong answer framework:**
- **Diagnose:** The bi-encoder (vector search) is prioritizing general semantic similarity over specific keyword or intent matching.
- **Propose Two-Stage Retrieval:** Increase the initial retrieval (`top_k`) to 50 to cast a wider net (high recall).
- **Implement Reranking:** Introduce a cross-encoder or managed reranker to score only those 50 documents, returning the absolute best 5 to the LLM. This provides high precision without running the slow model over 1M documents, preserving latency.

## System Design / Architecture Questions

**Q:** Design an evaluation pipeline for a new customer service RAG agent before it goes to production.

**Approach:**
1. **Clarify requirements:** Define the evaluation set (golden dataset of questions, expected responses, and source documents).
2. **Propose structure:** Use MLflow Agent Evaluation (or a similar LLM-as-a-judge framework). Capture MLflow traces of the agent execution.
3. **Justify choices and name tradeoffs explicitly:** Configure the LLM judge to evaluate two specific metrics: Groundedness (does the response match the retrieved context?) and Relevance (does the context actually answer the user's query?). The tradeoff is the cost/latency of running a judge LLM (like GPT-4) over thousands of test queries, which is why this is run offline in a CI/CD pipeline, not online per user request.

## Vocabulary That Signals Expertise

Use these terms naturally — don't force them:
- **Two-stage retrieval** — Signals you understand the standard enterprise architecture for search.
- **Groundedness vs Correctness** — Shows maturity in evaluating generative outputs.
- **Parametric memory** — The knowledge baked into the LLM's weights (what we want to avoid relying on in strict RAG).
- **O(1) vs O(N) inference** — Explaining exactly *why* cross-encoders can't scale to the whole corpus.

## Vocabulary That Signals Weakness

Avoid these — they signal outdated or shallow understanding:
- **"I'll just prompt the LLM to search better"** — Retrieval quality is an infrastructure/model problem, not a system prompt problem.
- **"We can use a cross-encoder on the database"** — Shows a fundamental misunderstanding of computational complexity; you cannot run a cross-encoder across a large database at query time.

## STAR Answer Frame

**Situation:** Our internal RAG search was returning irrelevant documents when queries contained highly specific jargon mixed with common terms.  
**Task:** I needed to improve retrieval precision from 60% to 90% without adding more than 500ms of latency.  
**Action:** I implemented a two-stage retrieval pipeline using LangChain's `ContextualCompressionRetriever`. I kept our existing Databricks Vector Search as the base retriever but increased the `top_k` to 40. Then, I added a HuggingFace cross-encoder as a reranking step to compress those 40 down to the top 3.  
**Result:** Retrieval precision jumped to 94%, and because the cross-encoder only processed 40 documents, the end-to-end latency only increased by 250ms, staying well within SLA.

## Red Flags Interviewers Watch For

- Candidates who think a vector database solves all search problems inherently.
- Candidates who don't know how to evaluate a pipeline other than "reading the outputs manually."
- Suggesting that fine-tuning the LLM is the solution to poor document retrieval.
