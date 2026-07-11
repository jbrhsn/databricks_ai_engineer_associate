# Retrieval Evaluation and Reranking

**Section:** 03-Data Preparation | **Module:** 02-Retrieval Quality | **Est. time:** 2 hrs | **Exam mapping:** Data Preparation (14%), Evaluation & Monitoring (12%)

---

## TL;DR

Reranking improves retrieval precision by taking a broad set of initially retrieved documents and reordering them using a more accurate, but computationally expensive, model like a cross-encoder. Evaluation of this retrieval quality is performed using MLflow Agent Evaluation, which employs LLMs-as-a-judge to measure metrics like groundedness and correctness. **The one thing to remember: Bi-encoders are fast but shallow for initial retrieval; cross-encoders are slow but deep for final precision.**

---

## ELI5 — Explain It Like I'm 5

Imagine looking for a specific recipe in a massive library. The library's search computer (the initial retriever) quickly gives you a list of 50 books that mention "chocolate" and "cake" in seconds. You can't read all 50 books, so you hand the list to an expert baker (the reranker). The baker carefully reads the actual recipe pages in those 50 books, comparing them exactly to what you asked for, and hands you the 3 absolute best matches. The baker is much slower than the computer, which is why you don't ask them to search the entire library, but they are much more accurate at picking the winner from a shortlist. A common misconception is that you can run the expert re-ranker (cross-encoder) on every document in the library — in reality this is computationally impossible at scale because cross-encoders process each (query, document) pair separately, so you must always use a fast bi-encoder first to narrow the candidate pool before applying the cross-encoder.

---

## Learning Objectives

By the end of this chapter you will be able to:
- [ ] Implement a two-stage retrieval pipeline using a cross-encoder for reranking.
- [ ] Contrast the architectural differences and performance trade-offs between bi-encoders and cross-encoders.
- [ ] Configure MLflow Agent Evaluation to measure RAG quality metrics such as groundedness and correctness.
- [ ] Diagnose poor retrieval results and apply Reciprocal Rank Fusion (RRF) to combine multiple search strategies.
- [ ] Construct an evaluation dataset (golden set) and run `mlflow.evaluate()` with `model_type="databricks-agent"`.

---

## Visual Overview

### Two-Stage Retrieval Pipeline

```
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

```
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

### Hybrid Search with RRF

```
User Query
    │
    ├──► BM25 Keyword Search ──► Ranked List A (scores: 5.0–12.5)
    │                                      │
    └──► Vector Semantic Search ──► Ranked List B (scores: 0.7–0.9)
                                           │
                                    ┌──────┴──────┐
                                    ▼             ▼
                             Rank positions from  Rank positions from
                               List A             List B
                                    └──────┬──────┘
                                           ▼
                              RRF Score = Σ 1/(k + rank)
                                           │
                                           ▼
                                  Unified Ranked List ──► Top N Documents
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

**Where does it appear in the tool ecosystem?** RRF is often implemented as a custom post-processing node in LangGraph or through hybrid search configurations in vector databases like Databricks Vector Search when combining dense and sparse vectors. When `query_type="HYBRID"` is set in the Databricks Vector Search Python SDK, BM25 and vector scores are merged using RRF automatically.

### BM25 Sparse Retrieval

**What is it?** A classical keyword-based ranking function that scores documents by term frequency and inverse document frequency (a TF-IDF variant), producing a sparse signal based purely on exact term matching.

**How does it work mechanistically?** BM25 scores a document by counting how many times query terms appear in that document, normalized by document length and penalized by how common those terms are across the entire corpus. The formula is `BM25(q,d) = Σ IDF(t) · (tf(t,d) · (k1+1)) / (tf(t,d) + k1·(1 - b + b·|d|/avgdl))`, where `k1` and `b` are tuning constants. Crucially, BM25 does not use embeddings — it operates on tokenized text — so it is extremely fast and perfectly interpretable: a document scores high only if the exact query words appear in it.

**Where does it appear in the tool ecosystem?** Databricks Vector Search supports hybrid search combining BM25 (keyword) with vector similarity. Setting `query_type="HYBRID"` in the Vector Search Python SDK enables this; the platform runs both BM25 and ANN internally and merges the two ranked lists using RRF before returning results.

### MLflow Agent Evaluation (LLM Judges)

**What is it?** A Databricks MLflow feature that uses state-of-the-art LLMs (like GPT-4 or Claude) to automatically score the quality of an agent's outputs based on specific criteria.

**How does it work mechanistically?** During an evaluation run (`mlflow.evaluate()`), MLflow extracts the user query, the agent's generated response, and the retrieved context from the MLflow trace. It then prompts a judge LLM with this data and specific grading rubrics (e.g., "Is the response fully supported by the context?"). The judge outputs a boolean score (pass/fail) or a numeric rating along with a written rationale explaining the decision.

**Where does it appear in the tool ecosystem?** Accessed via the `mlflow.evaluate()` API in Databricks, using `model_type="databricks-agent"`. The results are viewable in the MLflow UI under the evaluation tab and accessible programmatically via `results.metrics` in Python.

### MLflow Agent Evaluation Metrics

**What is it?** The quantitative scores produced by `mlflow.evaluate(model_type="databricks-agent")` that measure different dimensions of RAG quality using LLM-as-judge.

**How does it work mechanistically?** MLflow dispatches the evaluation payload — the user question, the generated response, and the retrieved context chunks — to a judge LLM model endpoint. The judge scores each dimension independently against a rubric: `groundedness` checks whether every factual claim in the response is traceable to the retrieved context; `answer_correctness` compares the generated response to the `expected_response` column in the eval dataset; `chunk_relevance` scores whether each retrieved chunk actually addresses the question; `document_recall` checks whether all expected source documents (specified via `retrieved_context`) were retrieved. Each metric returns a numeric score between 0 and 1 along with a free-text rationale.

**Where does it appear in the tool ecosystem?** Metrics appear in the MLflow UI under the experiment run's Evaluation tab. They are also returned in Python via `results.metrics` (aggregate) and `results.tables["eval_results_table"]` (per-row scores). This feature is fast-evolving — verify against current Databricks MLflow docs before assuming metric names are stable.

### Evaluation Set Construction

**What is it?** A structured dataset — called a golden set or eval set — used to measure retrieval and generation quality of a RAG pipeline before deploying to production.

**How does it work mechanistically?** An eval set is a Pandas DataFrame (or Delta table) with at minimum the columns `request` (the user question) and `expected_response` (the ground truth answer). An optional `retrieved_context` column holds pre-retrieved chunks as a list of dicts with `doc_uri` and `content` keys; when present, this isolates retrieval quality from generation quality by bypassing the live retriever. Databricks MLflow's `model_type="databricks-agent"` evaluator expects exactly this schema. The eval set can be assembled manually by subject-matter experts writing representative Q&A pairs, or auto-generated by prompting an LLM to synthesize questions from source documents (synthetic question generation — fast-evolving capability, verify current guidance in Databricks docs).

**Where does it appear in the tool ecosystem?** The eval set is passed as the `data=` argument to `mlflow.evaluate()`. In Databricks, eval sets are commonly stored as Delta tables and loaded with `spark.table()` or `pd.read_csv()`. The Databricks Agent Evaluation UI also provides a point-and-click interface for reviewing and annotating eval results row by row.

---

## Key Parameters / Configuration Knobs

| Parameter | What it controls | Decision rule |
|---|---|---|
| `top_k` (Initial retrieval) | How many documents the bi-encoder retrieves before reranking. | Set to 20–50 to cast a wide net without overwhelming the cross-encoder's latency limits. |
| `top_n` (Reranked output) | How many documents the cross-encoder returns to the final LLM. | Set to 3–7 depending on the LLM's context window and the specific task's need for precision. |
| `k` (RRF smoothing constant) | The smoothing constant for Reciprocal Rank Fusion. | Keep at the standard default of 60 unless empirical evaluation proves a different value works better for your specific data distribution. |
| `model_type` (`mlflow.evaluate`) | Selects which evaluator suite to run during `mlflow.evaluate()`. | Use `"databricks-agent"` for RAG/agent pipelines to get groundedness and chunk_relevance judges; do not use the default `"regressor"` or `"classifier"` types for generative evaluation — they have no concept of retrieved context. |
| `query_type` (Vector Search) | Controls whether retrieval uses vector similarity, keyword (BM25), or hybrid. | Use `"HYBRID"` when queries contain specific entity names, acronyms, or product codes that semantic search misses; use `"ANN"` (approximate nearest neighbor) for purely conceptual or paraphrasing-heavy queries. |

---

## Worked Example: Requirement → Decision

**Given:** An internal company knowledge base where queries are highly specific (e.g., "What is the parental leave policy for part-time employees in the UK?"). The corpus contains 50,000 HR document chunks. Standard vector search often returns general "parental leave" documents for the US before the specific UK part-time document. The production SLA requires a response within 2 seconds end-to-end.

**Step 1 — Identify the goal:** Improve the precision of the retrieved documents so the final LLM receives the exact correct policy document, not a semantically adjacent but wrong-jurisdiction document.

**Step 2 — Define inputs:** User query string; vector database of 50,000 HR document chunks pre-indexed with a bi-encoder.

**Step 3 — Define outputs:** Top 3 highly relevant, correctly scoped documents passed as context to the LLM.

**Step 4 — Apply constraints:** The cross-encoder runs in O(top_k) time per query; with top_k=100 and 10ms per pair, that is 1 second of reranking latency — acceptable for the 2-second SLA but means we cannot rerank more than 100 candidates without breaching it. Running the cross-encoder over all 50,000 chunks would require 500 seconds per query and is ruled out entirely. The bi-encoder initial retrieval must therefore achieve high recall within the top 100 results.

**Step 5 — Select the approach:** Implement a two-stage retrieval pipeline with `query_type="HYBRID"` on Databricks Vector Search to retrieve the top 100 candidates (BM25 + ANN merged by RRF), then apply a cross-encoder reranker to produce the final top 3. This is chosen over running the cross-encoder on all 50,000 chunks directly because that would be O(50,000) = 500 seconds per query, and over using only the bi-encoder because the bi-encoder alone misses lexically dissimilar but semantically relevant matches for jurisdiction-specific terminology like "UK part-time."

---

## Implementation

```python
# Scenario: Adding a reranker to an existing vector search to improve precision
# for a high-specificity HR query corpus where pure semantic search returns
# wrong-jurisdiction documents. Two-stage pipeline keeps latency under 1 second.
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# 1. Define the base retriever (fast, broad search — retrieves top 30 candidates)
base_retriever = vector_store.as_retriever(search_kwargs={"k": 30})

# 2. Define the reranker model (slow, precise — scores each (query, doc) pair jointly)
model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
compressor = CrossEncoderReranker(model=model, top_n=5)

# 3. Combine them into a two-stage retrieval pipeline
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# The pipeline first gets 30 docs via bi-encoder, then reranks and returns the top 5
docs = compression_retriever.invoke("What is the UK parental leave policy?")
```

```python
# Anti-pattern: Attempting to score all documents with a cross-encoder, destroying
# query latency. This approach is O(N) on corpus size — 100k docs = hours per query.

# Wrong: Iterating over all documents to find the best match using a cross-encoder.
def bad_retrieval(query, all_100k_documents, cross_encoder):
    scores = cross_encoder.predict([(query, doc.page_content) for doc in all_100k_documents])
    # ... sort and return ...

# Correct approach:
# Use a vector database (bi-encoder) to quickly filter down to the top 50 candidates,
# THEN apply the cross-encoder to only those 50 candidates.
def good_retrieval(query, vector_store, cross_encoder):
    # Step 1: Fast bi-encoder retrieval — sub-millisecond, pre-computed embeddings
    candidates = vector_store.similarity_search(query, k=50)
    # Step 2: Precise cross-encoder reranking — O(50) pairs, ~500ms
    scores = cross_encoder.predict([(query, doc.page_content) for doc in candidates])
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:5]]
```

```python
# Scenario: Evaluating retrieval quality of a RAG pipeline using MLflow Agent Evaluation
# to measure groundedness and chunk relevance before deploying to production.
# Uses model_type="databricks-agent" to activate Databricks-specific LLM judges.
import mlflow
import pandas as pd

# 1. Create an evaluation dataset (golden set: questions + expected answers + context)
eval_data = pd.DataFrame({
    "request": [
        "What is the capital of the Unity Catalog?",
        "How do I enable Change Data Feed?"
    ],
    "expected_response": [
        "Unity Catalog does not have a capital; it is a governance layer.",
        "Set delta.enableChangeDataFeed = true using ALTER TABLE or during table creation."
    ],
    "retrieved_context": [
        [{"doc_uri": "uc_docs.pdf", "content": "Unity Catalog is a unified governance layer..."}],
        [{"doc_uri": "delta_docs.pdf", "content": "Change Data Feed is enabled via TBLPROPERTIES..."}]
    ]
})

# 2. Define the RAG chain as a callable (here simplified as a lambda)
def rag_chain(inputs):
    # In production: call your LangChain chain or Databricks Model Serving endpoint
    return {"response": "placeholder"}

# 3. Run evaluation with Databricks agent evaluator
with mlflow.start_run():
    results = mlflow.evaluate(
        model=rag_chain,
        data=eval_data,
        model_type="databricks-agent"  # Enables Databricks-specific LLM judges
    )
    print(results.metrics)  # groundedness, chunk_relevance, document_recall
```

---

## Common Pitfalls & Misconceptions

- **Overlooking Reranker Latency** — Beginners assume they can rerank 500 or 1,000 documents to guarantee they find the right one, treating the reranker as a simple sort operation. Rerankers require a full forward pass through a transformer for every (query, document) pair; reranking more than ~50–100 documents will severely degrade user experience (latency > 2–3 seconds).
- **Ignoring the Base Retriever** — Developers think a great reranker can compensate for a weak initial retrieval, assuming the reranker will "find" the right answer. If the target document is not in the top N returned by the bi-encoder, the cross-encoder can never promote it — reranking only reorders what it receives, so the base retriever must have high recall.
- **Confusing Groundedness with Correctness** — Teams evaluate RAG only for "correctness" because that is the metric most intuitive to stakeholders. A response can be factually correct (because the LLM drew on training data) but fail groundedness because the answer was not derived from the retrieved context; in enterprise RAG, ungrounded answers are hallucinations even if the fact happens to be true, creating an audit and compliance risk.

---

## Key Definitions

| Term | Definition |
|---|---|
| **Bi-Encoder** | A model that embeds text into a vector space independently, allowing for fast similarity calculations via dot products and offline pre-computation of document embeddings. |
| **Cross-Encoder** | A model that processes two pieces of text (query and document) simultaneously through joint attention layers, capturing deep semantic interactions to output a relevance score; cannot pre-compute. |
| **Reranking** | The process of re-scoring and re-ordering a small candidate subset returned by an initial fast search method, using a slower but more precise model. |
| **Groundedness** | An MLflow Agent Evaluation metric that measures whether every factual claim in the LLM's generated response is strictly supported by the retrieved context, not by the model's parametric memory. |
| **Reciprocal Rank Fusion (RRF)** | An algorithm that merges multiple ranked lists into a single ranking by scoring each document as the sum of `1/(k + rank)` across all lists, making it robust to incompatible raw scores. |
| **BM25** | A classical sparse keyword-based ranking function that scores documents by term frequency normalized by document length and inverse document frequency; does not use embeddings and is therefore fast and exactly interpretable. |
| **Eval Set (Golden Set)** | A structured DataFrame with `request`, `expected_response`, and optionally `retrieved_context` columns used as ground truth input to `mlflow.evaluate()` to measure RAG quality before deployment. |
| **Chunk Relevance** | An MLflow Agent Evaluation metric that scores whether each retrieved chunk is actually relevant to the user's question, helping diagnose retrieval quality independently of generation quality. |

---

## Summary / Quick Recall

- Two-stage retrieval uses bi-encoders for broad recall (fast, O(1)) and cross-encoders for precise ranking (slow, O(top_k)) — never run a cross-encoder across the full corpus.
- BM25 sparse retrieval and dense vector search are complementary; Databricks Vector Search `query_type="HYBRID"` combines both using RRF automatically.
- RRF combines results from multiple search methods (keyword + semantic) by rank position, not raw scores — it is robust to incompatible score scales.
- MLflow Agent Evaluation uses LLM judges to score `groundedness`, `answer_correctness`, `chunk_relevance`, and `document_recall` on a golden eval set.
- An eval set requires at minimum `request` and `expected_response` columns; adding `retrieved_context` isolates retrieval quality from generation quality.
- High recall is required from the base retriever; the reranker cannot promote a document it never received.
- `model_type="databricks-agent"` in `mlflow.evaluate()` activates Databricks-specific judges; using `"regressor"` or `"classifier"` for a RAG pipeline is a misconfiguration.

---

## Self-Check Questions

1. What is the primary architectural difference between a bi-encoder and a cross-encoder that explains why one is fast and the other is precise?
   - A. Bi-encoders use larger transformer models with more parameters than cross-encoders.
   - B. Bi-encoders embed the query and document separately, enabling pre-computation; cross-encoders process the query-document pair jointly at runtime.
   - C. Cross-encoders store embeddings in a vector database while bi-encoders do not.
   - D. Bi-encoders use keyword matching; cross-encoders use semantic embeddings.

   <details><summary>Answer</summary>

   **B.** The speed advantage of bi-encoders comes entirely from the ability to pre-compute and cache document embeddings offline; at query time only the query needs encoding. Cross-encoders concatenate `[CLS] Query [SEP] Document` and run the full pair through all self-attention layers at runtime, which is why they are precise (every query word can attend to every document word) but cannot be pre-computed. Option A is wrong — cross-encoders are not necessarily larger; the cost comes from O(N) runtime pairs, not model size. Option C has it backwards. Option D conflates BM25 (keyword) with bi-encoders, which use embeddings.

   </details>

2. Your team has a corpus of 2 million documents. A product manager asks you to use a cross-encoder to "guarantee the best possible match is always returned" for user queries. Which of the following implementations is correct?
   - A. Run the cross-encoder over all 2 million documents for every query to ensure completeness.
   - B. Use only a BM25 keyword search to return the top 5 documents directly.
   - C. Use a bi-encoder to retrieve the top 50 candidate documents, then apply the cross-encoder to rerank those 50 and return the top 5.
   - D. Train a new bi-encoder on your corpus to replace the cross-encoder entirely.

   <details><summary>Answer</summary>

   **C.** This is the canonical two-stage retrieval pattern. The bi-encoder handles the O(1) broad search efficiently, and the cross-encoder applies deep joint attention only to the small candidate pool, keeping latency manageable. Option A is the most dangerous wrong answer — running a cross-encoder on 2 million documents per query would take thousands of seconds and is operationally impossible in real-time. Option B would miss semantically relevant documents that don't share exact keywords, and a cross-encoder is never applied. Option D does not address the problem — a bi-encoder alone does not provide the deep contextual matching the product manager is asking for.

   </details>

3. **Which TWO** of the following metrics are commonly evaluated using an LLM-as-a-judge in Databricks MLflow Agent Evaluation for a RAG pipeline?
   - A. Embedding dimension size
   - B. Groundedness
   - C. Hardware utilization (GPU memory)
   - D. Correctness
   - E. Number of nodes in the cluster

   <details><summary>Answer</summary>

   **B and D.** Groundedness (is the response supported by retrieved context?) and Correctness (does the response match the expected answer?) are qualitative judgments about text that an LLM judge is uniquely suited to evaluate by reading the response, context, and rubric. Options A, C, and E are infrastructure or architectural metrics that have nothing to do with the content of the model's response and cannot be determined from the chat payload; they require system monitoring tools, not LLM judges.

   </details>

4. You are building a search system that must handle queries containing exact product SKUs (e.g., "NX-4052-B") as well as broad conceptual queries (e.g., "enterprise security best practices"). Pure vector search returns poor results for SKU queries. What retrieval strategy and Databricks configuration should you use, and what is the trade-off of your choice?
   - A. Use `query_type="ANN"` only; vector search handles all query types equally well.
   - B. Use `query_type="HYBRID"` to combine BM25 keyword search and vector similarity via RRF, accepting that hybrid search adds a small amount of latency over pure ANN.
   - C. Use `query_type="HYBRID"` for SKU queries and `query_type="ANN"` for conceptual queries, switching at the application layer based on query content.
   - D. Use a cross-encoder directly on all documents to avoid needing a hybrid strategy.

   <details><summary>Answer</summary>

   **B.** `query_type="HYBRID"` in Databricks Vector Search runs both BM25 (which excels at exact token matching like SKUs) and ANN (which excels at conceptual similarity) and merges the results with RRF. This handles both query types without application-layer routing logic. The trade-off is a small latency increase from running two search paths and the RRF merge step. Option A is wrong because ANN/vector search encodes tokens into a continuous space, which can miss exact strings like product codes. Option C would work but is brittle — classifying queries as "SKU vs conceptual" is a fragile heuristic that requires ongoing maintenance. Option D re-introduces the O(N) cross-encoder problem on the full corpus, which is operationally infeasible.

   </details>

5. An MLflow evaluation run shows that your RAG agent scores 0.95 on `answer_correctness` but 0.10 on `groundedness`. What does this diagnostic pattern indicate, and what is the most likely root cause?
   - A. The retrieved context is high quality but the LLM is not reading it carefully enough.
   - B. The LLM is answering correctly from its pre-trained parametric memory rather than the retrieved documents, meaning the retriever may be failing or the LLM is ignoring context.
   - C. The eval set `expected_response` column contains errors, making correctness artificially high.
   - D. The `model_type` parameter was set incorrectly, causing the judge to apply the wrong rubric.

   <details><summary>Answer</summary>

   **B.** High correctness + low groundedness is the classic hallucination-from-memory signature: the LLM is producing factually accurate answers using knowledge baked into its weights during pretraining, not by reading the retrieved chunks. This is dangerous in enterprise settings because the same behavior on a query the LLM doesn't know will produce a confidently wrong, ungrounded answer. Option A is wrong because if the context were high quality and used, groundedness would be high. Option C is possible but would lower correctness, not inflate it alongside low groundedness. Option D would typically cause anomalous scores across all metrics uniformly, not a correctness/groundedness split.

   </details>

---

## Further Reading

- [Agent Evaluation (MLflow 2) | Databricks](https://docs.databricks.com/en/generative-ai/agent-evaluation/index.html) — *verified 2026-07-11*
- [Databricks Vector Search | Databricks Documentation](https://docs.databricks.com/en/generative-ai/vector-search.html) — *verified 2026-07-11*
- [LangChain Contextual Compression Retriever](https://python.langchain.com/v0.2/docs/how_to/contextual_compression/) — *verified 2026-07-11*
