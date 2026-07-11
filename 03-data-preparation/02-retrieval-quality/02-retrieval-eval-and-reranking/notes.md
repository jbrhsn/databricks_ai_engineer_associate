# Retrieval Evaluation and Reranking

**Section:** 03-Data Preparation | **Module:** 02-Retrieval Quality | **Est. time:** 2 hrs | **Exam mapping:** Data Preparation (14%), Evaluation & Monitoring (12%)

---

## TL;DR

Reranking improves retrieval precision by taking a broad set of initially retrieved documents and reordering them using a more accurate, but computationally expensive, model like a cross-encoder. Evaluation of this retrieval quality is performed using MLflow Agent Evaluation, which employs LLMs-as-a-judge to measure metrics like groundedness and correctness. **The one thing to remember: Bi-encoders are fast but shallow for initial retrieval; cross-encoders are slow but deep for final precision.**

---

## ELI5 — Explain It Like I'm 5

Imagine looking for a specific recipe in a massive library. The library's search computer (the initial retriever) quickly gives you a list of 50 books that mention "chocolate" and "cake" in seconds. You can't read all 50 books, so you hand the list to an expert baker (the reranker). The baker carefully reads the actual recipe pages in those 50 books, comparing them exactly to what you asked for, and hands you the 3 absolute best matches. The baker is much slower than the computer, which is why you don't ask them to search the entire library, but they are much more accurate at picking the winner from a shortlist.

---

## Learning Objectives

By the end of this chapter you will be able to:
- [ ] Implement a two-stage retrieval pipeline using a cross-encoder for reranking.
- [ ] Contrast the architectural differences and performance trade-offs between bi-encoders and cross-encoders.
- [ ] Configure MLflow Agent Evaluation to measure RAG quality metrics such as groundedness and correctness.
- [ ] Diagnose poor retrieval results and apply Reciprocal Rank Fusion (RRF) to combine multiple search strategies.

---

## Visual Overview

### Two-Stage Retrieval Pipeline

```text
User Query ──► Bi-Encoder (Fast) ──► Vector Database Search
                                             │
                                             ▼
                                     Top 50 Documents
                                             │
                                             ▼
                              Cross-Encoder (Slow, Precise)
                                             │
                                             ▼
                                     Top 5 Documents ──► LLM Context
```

### MLflow Agent Evaluation Flow

```text
Input Request + Expected Response ──► Agent Execution (Trace recorded)
                                             │
                                             ▼
                                     Generated Response + Retrieved Context
                                             │
                                             ▼
                                     LLM Judge Evaluator
                                    ┌────────┴────────┐
                                    ▼                 ▼
                              Groundedness        Correctness
```

---

## Key Concepts

### Bi-Encoders vs. Cross-Encoders

**What is it?** Bi-encoders embed the query and document separately to compute similarity via a dot product. Cross-encoders process the query and document together through a transformer network to output a direct relevance score.
**How does it work mechanistically?** A bi-encoder pre-computes document embeddings offline and caches them in a vector database. At runtime, only the query is embedded, making the search extremely fast (O(1) inference per query). A cross-encoder concatenates the query and the document (`[CLS] Query [SEP] Document`) and passes the pair through all self-attention layers of the model. This allows the model to see how words in the query relate to words in the document, but it must be computed at runtime for every pair, making it O(N) where N is the number of documents to check.
**Where does it appear in the tool ecosystem?** In LangChain, bi-encoders are the standard `Embeddings` models used in `VectorStoreRetriever`. Cross-encoders are implemented as `DocumentCompressor` objects (e.g., Cohere Rerank or HuggingFace CrossEncoder) wrapped inside a `ContextualCompressionRetriever`.

### Reciprocal Rank Fusion (RRF)

**What is it?** An algorithm that combines the ranked results from multiple disparate search queries or retrieval methods (like vector search and keyword search) into a single, unified ranking.
**How does it work mechanistically?** RRF assigns a score to each document based on its rank in each retrieved list, rather than its raw similarity score. The formula is `Score = 1 / (k + rank)`, where `k` is a smoothing constant (typically 60). Documents that consistently appear near the top across multiple lists accumulate higher scores and float to the top of the final fused list. This prevents a document with an artificially high score from one retriever from dominating the final results.
**Where does it appear in the tool ecosystem?** RRF is often implemented as a custom post-processing node in LangGraph or through hybrid search configurations in vector databases like Databricks Vector Search when combining dense and sparse vectors.

### MLflow Agent Evaluation (LLM Judges)

**What is it?** A Databricks MLflow feature that uses state-of-the-art LLMs (like GPT-4 or Claude) to automatically score the quality of an agent's outputs based on specific criteria.
**How does it work mechanistically?** During an evaluation run (`mlflow.evaluate()`), MLflow extracts the user query, the agent's generated response, and the retrieved context from the MLflow trace. It then prompts a judge LLM with this data and specific grading rubrics (e.g., "Is the response fully supported by the context?"). The judge outputs a boolean score (pass/fail) or a numeric rating along with a written rationale explaining the decision.
**Where does it appear in the tool ecosystem?** Accessed via the `mlflow.evaluate()` API in Databricks, using `model_type="databricks-agent"`. The results are viewable in the MLflow UI under the evaluation tab.

### Key Parameters / Configuration Knobs

| Parameter | What it controls | Decision rule |
|---|---|---|
| `top_k` (Initial) | How many documents the bi-encoder retrieves before reranking. | Set to 20–50 to cast a wide net without overwhelming the cross-encoder's latency limits. |
| `top_n` (Reranked) | How many documents the cross-encoder returns to the final LLM. | Set to 3–7 depending on the LLM's context window and the specific task's need for precision. |
| `k` (in RRF) | The smoothing constant for Reciprocal Rank Fusion. | Keep at the standard default of 60 unless empirical evaluation proves a different value works better for your specific data distribution. |

### Worked Example: Requirement → Decision

**Given:** An internal company knowledge base where queries are highly specific (e.g., "What is the parental leave policy for part-time employees in the UK?"). Standard vector search often returns general "parental leave" documents for the US before the specific UK part-time document.
**Step 1 — Identify the goal:** Improve the precision of the retrieved documents so the final LLM receives the exact correct policy document.
**Step 2 — Define inputs:** User query, vector database of HR documents.
**Step 3 — Define outputs:** Top 3 highly relevant documents.
**Step 4 — Apply constraints:** We need sub-second latency for the final answer, so we cannot run a heavy model on all 10,000 HR documents.
**Step 5 — Select the approach:** Implement a two-stage retrieval pipeline. A fast vector search fetches the top 30 documents, and a cross-encoder reranks them to pick the top 3. *Rationale:* This provides the deep contextual matching needed for nuanced queries without the extreme latency of running a cross-encoder over the entire corpus.

---

## Implementation

```python
# Scenario: Adding a reranker to an existing vector search to improve precision
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# 1. Define the base retriever (fast, broad search)
base_retriever = vector_store.as_retriever(search_kwargs={"k": 30})

# 2. Define the reranker model (slow, precise)
model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
compressor = CrossEncoderReranker(model=model, top_n=5)

# 3. Combine them into a two-stage retrieval pipeline
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# The pipeline first gets 30 docs, then reranks them and returns the top 5
docs = compression_retriever.invoke("What is the UK parental leave policy?")
```

```python
# Anti-pattern: Attempting to score all documents with a cross-encoder

# Wrong: Iterating over all documents in a list to find the best match using a cross-encoder.
# This results in O(N) latency where N is the corpus size. If you have 100,000 docs, this will take hours per query.
def bad_retrieval(query, all_100k_documents, cross_encoder):
    scores = cross_encoder.predict([(query, doc.page_content) for doc in all_100k_documents])
    # ... sort and return ...

# Correct approach:
# Use a vector database (bi-encoder) to quickly filter down to the top 50 candidates,
# THEN apply the cross-encoder to only those 50 candidates.
```

---

## Common Pitfalls & Misconceptions

- **Overlooking Reranker Latency** — Beginners assume they can rerank 500 or 1,000 documents to guarantee they find the right one. Rerankers are computationally heavy; reranking more than ~50-100 documents will severely degrade the user experience (latency > 2-3 seconds).
- **Ignoring the Base Retriever** — Developers think a great reranker fixes a terrible initial search. If the target document isn't in the top 50 returned by the bi-encoder, the cross-encoder can never promote it. The base retriever must have high recall.
- **Confusing Groundedness with Correctness** — Teams evaluate RAG only for "correctness." A response can be factually correct (because the LLM knew the answer from its training data) but fail *groundedness* because the answer was not found in the retrieved context. In enterprise RAG, ungrounded answers are hallucinations, even if factually true.

---

## Key Definitions

| Term | Definition |
|---|---|
| **Bi-Encoder** | A model that embeds text into a vector space independently, allowing for fast similarity calculations via dot products. |
| **Cross-Encoder** | A model that processes two pieces of text (query and document) simultaneously, capturing deep semantic interactions to output a relevance score. |
| **Reranking** | The process of re-scoring and re-ordering a small subset of documents retrieved by an initial fast search method. |
| **Groundedness** | An evaluation metric measuring whether all facts in the LLM's generated response are strictly supported by the retrieved context. |
| **Reciprocal Rank Fusion (RRF)** | An algorithm that combines multiple ranked lists into a single ranking by assigning scores based on the inverse of a document's rank. |

---

## Summary / Quick Recall

- Two-stage retrieval uses bi-encoders for broad recall (fast) and cross-encoders for precise ranking (slow).
- Never run a cross-encoder across the entire document corpus due to latency constraints.
- RRF combines results from multiple search methods (e.g., keyword + semantic) without relying on raw, incompatible similarity scores.
- MLflow Agent Evaluation uses LLM judges to assess metrics like groundedness and correctness on MLflow traces.
- High recall is required from the base retriever; the reranker cannot rank what it doesn't receive.

---

## Self-Check Questions

1. What is the primary difference in how bi-encoders and cross-encoders process text?

   <details><summary>Answer</summary>
   
   Bi-encoders process the query and document separately to produce embeddings that are compared via dot product, enabling pre-computation and fast search. Cross-encoders concatenate the query and document and process them together through the attention layers, preventing pre-computation but allowing for deeper semantic matching. If you answered that bi-encoders are just "less accurate," you missed the structural reason *why*: independent vs. joint embedding.
   
   </details>

2. You have a corpus of 1 million documents. A user submits a query, and you want to use a cross-encoder to ensure the absolute best match is returned. Which workflow is correct?

   <details><summary>Answer</summary>
   
   You must first use a bi-encoder (vector search) to retrieve a small candidate set (e.g., top 50 documents). Then, you pass only those 50 documents to the cross-encoder to be scored and reordered, returning the final top 5. Using the cross-encoder on all 1 million documents is computationally infeasible for a real-time application.
   
   </details>

3. **Which TWO** of the following metrics are commonly evaluated using an LLM-as-a-judge in Databricks MLflow Agent Evaluation for a RAG pipeline?
   - A. Embedding dimension size
   - B. Groundedness
   - C. Hardware utilization (GPU memory)
   - D. Correctness
   - E. Number of nodes in the cluster

   <details><summary>Answer</summary>
   
   **B and D.** Groundedness and Correctness are qualitative aspects of the generated text that an LLM judge is uniquely suited to evaluate. Options A, C, and E are infrastructure or architectural metrics that do not require an LLM judge and cannot be determined from the chat payload.
   
   </details>

4. You are combining search results from a sparse keyword search (BM25) and a dense semantic vector search. The BM25 engine returns raw scores ranging from 5.0 to 12.5, while the vector engine returns cosine similarities from 0.7 to 0.9. How should you merge these lists?

   <details><summary>Answer</summary>
   
   You should use Reciprocal Rank Fusion (RRF). RRF ignores the raw, incompatible scores and instead assigns a new score based purely on the document's rank position in each respective list. Normalizing the raw scores and adding them (a linear combination) is often brittle and requires constant tuning of weights, making RRF the preferred robust choice.
   
   </details>

5. An MLflow evaluation run shows that your RAG agent scores perfectly on "Correctness" but fails completely on "Groundedness." What does this indicate?

   <details><summary>Answer</summary>
   
   This indicates that the LLM is answering the questions correctly based on its pre-trained parametric memory, but it is *not* using the retrieved documents to formulate those answers (or the retriever failed to find the right documents). In an enterprise context, this is dangerous because the agent is hallucinating answers outside of its provided context, even if it happens to be factually correct this time.
   
   </details>

---

## Further Reading

- [Agent Evaluation (MLflow 2) | Databricks](https://docs.databricks.com/en/generative-ai/agent-evaluation/index.html) — *verified 2026-07-11*
- [LangChain Contextual Compression Retriever](https://python.langchain.com/v0.2/docs/how_to/contextual_compression/) — *verified 2026-07-11*
