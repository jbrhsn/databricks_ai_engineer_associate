# Embedding Models and Context Length for Vector Search

**Section:** 04 — Application Development | **Module:** 02 — Model and Embedding Selection | **Est. time:** 2 hrs | **Exam mapping:** Application Development — embedding model selection, retrieval quality, Databricks AI Search configuration

---

## TL;DR

An embedding model converts variable-length text into a fixed-size dense vector that encodes semantic meaning. Choosing the wrong model — or misaligning chunk size with the model's maximum sequence length — silently degrades retrieval quality because most frameworks truncate input without raising an error. Databricks AI Search (formerly Vector Search) uses L2 distance internally but exposes cosine-equivalent results via normalized vectors. **The one thing to remember: if your chunk size exceeds the embedding model's token limit, the text is silently truncated at index time, and the resulting embedding represents only part of your document — destroying retrieval precision with no warning.**

---

## ELI5 — Explain It Like I'm 5

Imagine you run a library where every book must be described by exactly a ten-word summary card placed in a physical filing cabinet. A librarian reads each book, picks the ten most important words, and writes them on the card. When a visitor asks a question, another librarian writes a ten-word card for the question, then searches the filing cabinet for the most similar-looking cards. The "summarising" job is exactly what an embedding model does — it compresses text of any length into a fixed-size list of numbers. Here is the critical twist most beginners miss: the librarian can only read up to a certain number of pages at a time. If a book is longer than that limit, the librarian stops reading and writes a card based only on the first portion. The resulting card will not match queries about the parts of the book that were never read — and nobody will ever tell you the card is wrong.

---

## Learning Objectives

By the end of this chapter you will be able to:
- [ ] Explain how a transformer encoder maps variable-length text to a fixed-dimension dense vector via pooling
- [ ] Identify the maximum sequence length of `databricks-gte-large-en`, `databricks-bge-large-en`, and `databricks-qwen3-embedding-0-6b` and describe the truncation risk when chunk sizes exceed those limits
- [ ] Compare cosine similarity, dot product, and L2 distance metrics and select the correct one given normalisation requirements
- [ ] Initialise `DatabricksEmbeddings` in LangChain and embed both documents and queries against a Databricks Model Serving endpoint
- [ ] Distinguish bi-encoder from cross-encoder and justify why bi-encoders are used at index time and cross-encoders only at reranking time

---

## Visual Overview

### Token-to-Vector Pipeline

```
Raw Text
   │
   ▼
┌─────────────┐
│  Tokeniser  │  (WordPiece / BPE / SentencePiece)
│             │  Maps words → integer token IDs
└──────┬──────┘
       │  Token IDs  [101, 2079, 2079, ...]
       ▼
┌─────────────────────────────┐
│  Transformer Encoder Stack  │  (BERT-family: bidirectional attention)
│  Layer 1 … Layer N          │  Each token attends to every other token
└──────────────┬──────────────┘
               │  Contextual token representations  [seq_len × hidden_dim]
               ▼
┌────────────────────────────────┐
│  Pooling  (CLS token or mean)  │  Collapses sequence dimension
└──────────────┬─────────────────┘
               │
               ▼
   Dense Vector  [1 × embedding_dim]   e.g. [0.12, -0.87, 0.04, … ]  (1024 dims)
```

### Chunk Size vs. Model Token Limit Alignment

```
                      Model Max Tokens
                            │
   ┌────────────────────────▼──────────────────────────────────┐
   │              Safe Zone (full chunk embedded)               │
   │  chunk_size  ◄────────────────────────────────────────    │
   └────────────────────────────────────────────────────────────┘

   ── chunk_size within limit ──►  FULL semantic representation ✓

   ┌────────────────────────────────────┬──────────────────────┐
   │         Embedded (truncated)        │  SILENTLY DROPPED ✗  │
   │                                    │  (no error raised)    │
   └────────────────────────────────────┴──────────────────────┘

   ── chunk_size exceeds limit ──►  PARTIAL embedding, retrieval degraded ✗

   Key rule:  chunk_size (tokens) ≤ embedding_model_max_tokens × 0.9
              (leave a 10% buffer for tokeniser overhead)
```

### Distance Metric Comparison

```
Metric        Formula                   Normalised?   Best use case
──────────────────────────────────────────────────────────────────────
Cosine        1 - (A·B)/(|A||B|)        Yes           Text retrieval,
similarity                                             length-invariant

Dot product   A·B                       No            When embeddings
                                                       are pre-normalised
                                                       (identical to cosine)

L2 (Euclidean) √Σ(Aᵢ-Bᵢ)²             No            Image/numeric
distance                                               embeddings with
                                                       magnitude meaning

Databricks AI Search internally uses L2 distance.
Set cosine: normalise embeddings before insertion (BGE does this by default).
```

---

## Key Concepts

### Embedding Model Architecture

An embedding model is a transformer encoder that converts variable-length text into a single fixed-dimension dense vector called an embedding. Unlike a generative LLM that produces sequences of tokens, an embedding model produces exactly one vector per input, regardless of input length.

The mechanism works as follows: input text is first tokenised into integer IDs. The encoder then runs a stack of bidirectional self-attention layers so every token representation is informed by all other tokens in the context window. Finally, a pooling step collapses the per-token representations into a single vector — either by taking the output of the special `[CLS]` token (BERT-style) or by averaging all token representations (mean pooling, used by GTE and most modern retrieval-focused encoders). The resulting vector is the embedding.

In the Databricks ecosystem, the primary embedding models available through Foundation Model APIs pay-per-token endpoints are `databricks-gte-large-en` (GTE Large; 1024 dims, 8192-token window), `databricks-bge-large-en` (BGE Large; 1024 dims, 512-token window, pre-normalised), and `databricks-qwen3-embedding-0-6b` (Qwen3 Embedding; up to 1024 configurable dims, ~32K-token window, multilingual). Custom models can be deployed via Model Serving and referenced by endpoint name in AI Search index configuration.

### Embedding Dimensionality

Dimensionality refers to the length of the output vector — the number of floating-point numbers used to represent a piece of text. Higher dimensionality gives the model more expressive capacity to encode semantic nuance, but increases storage cost per vector, increases memory pressure on the vector index, and slows approximate nearest-neighbour (ANN) search.

Under the hood, dimensionality is fixed at model training time and cannot be changed post-hoc without retraining. Some modern models (such as `databricks-qwen3-embedding-0-6b`) support *Matryoshka Representation Learning*, which allows truncating the output vector to a smaller size at inference time with a modest quality trade-off. The vector dimensions are learned features — there is no human-readable meaning attached to dimension 7 vs. dimension 412; meaning is encoded in relative distances between vectors across the full space.

In Databricks AI Search, dimensionality is specified when creating a Direct Vector Access Index via the `embedding_dimension` parameter, or it is inferred automatically from the first batch of embeddings when using a Delta Sync Index with a managed endpoint. Mismatching the `embedding_dimension` value with the actual model output dimension will cause index creation to fail or produce corrupted vectors.

### Max Sequence Length / Context Window of Embedding Models

The maximum sequence length (also called the context window or token limit) is the hard upper bound on how many tokens the encoder will process in a single forward pass. If an input exceeds this limit, the tokeniser silently truncates to the first N tokens — no exception is raised, no warning appears in logs.

The mechanism: tokenisers pad shorter sequences and truncate longer ones to match a fixed `max_length` parameter. When the HuggingFace `transformers` library's `tokenizer(text, truncation=True, max_length=512)` is called, text beyond token 512 is discarded before it ever reaches the model. The truncated representation is then encoded and the resulting embedding reflects only the first portion of the document.

This matters critically for chunking strategy: your `chunk_size` (measured in tokens, not characters) must be set to a value strictly less than the embedding model's token limit. Published limits for Databricks-hosted models verified 2026-07-11: **GTE Large En** = 8192 tokens; **BGE Large En** = 512 tokens; **Qwen3-Embedding-0.6B** ≈ 32K tokens. Character-to-token ratios vary by language; English text averages roughly 4 characters per token, but CJK languages can be 1–2 characters per token.

### Semantic Similarity and Distance Metrics

Semantic similarity measures how close two embeddings are in vector space as a proxy for how related their source texts are. Three distance functions are commonly used:

**Cosine similarity** measures the angle between two vectors, ignoring their magnitude. It is computed as the dot product of two unit-normalised vectors. This makes it robust to length variation: a short sentence and a long paragraph about the same topic will have a high cosine similarity even if their raw vector magnitudes differ. **Dot product** is numerically equivalent to cosine similarity when both vectors are pre-normalised to unit length, but diverges for non-normalised vectors — making it sensitive to embedding magnitude. **L2 (Euclidean) distance** measures the straight-line distance between two points in vector space; it is sensitive to both angle and magnitude.

Databricks AI Search uses the L2 distance algorithm (HNSW-based ANN) internally. If you want cosine-equivalent rankings, normalise embeddings before inserting them into the index — `databricks-bge-large-en` does this by default; `databricks-gte-large-en` does not (documented 2026-07-11). When normalised, L2 distance rankings are identical to cosine similarity rankings.

### Bi-encoder vs. Cross-encoder

A **bi-encoder** encodes the query and each document independently, producing two separate vectors. Similarity is computed cheaply as a single dot product or L2 distance. This architecture scales to billions of documents because documents are embedded once at index time and stored; at query time only the query is re-encoded. All Databricks AI Search indexes use bi-encoders for retrieval.

A **cross-encoder** concatenates the query and a candidate document and feeds both through the model simultaneously, producing a single relevance score. This gives far better accuracy because the model sees the full interaction between query and document tokens — but it cannot be precomputed and must run at query time for every (query, candidate) pair. Cross-encoders are therefore used only for *reranking* a short candidate set (typically top-20 results from a bi-encoder retrieval pass), not for initial retrieval across large indexes.

In Databricks AI Search, the `reranking` feature available in the `similarity_search` API uses a cross-encoder-style reranker. The bi-encoder stage determines the initial candidate pool; the cross-encoder stage reorders it.

### DatabricksEmbeddings LangChain Integration

`DatabricksEmbeddings` is the LangChain class (from the `databricks-langchain` package) that wraps any OpenAI-compatible embedding endpoint hosted on Databricks Model Serving. It exposes the standard `Embeddings` interface: `embed_documents(texts)` for batches of documents and `embed_query(text)` for a single query string. These methods directly feed into `DatabricksVectorSearch`, which handles indexing and similarity search.

Initialisation requires the `endpoint` parameter pointing to the serving endpoint name. Outside a Databricks workspace, credentials are passed via `DATABRICKS_HOST` and `DATABRICKS_TOKEN` environment variables. Inside a Databricks notebook, these are automatically resolved. The class supports async variants (`aembed_query`, `aembed_documents`) for high-throughput pipelines.

---

## Key Parameters / Configuration Knobs

| Parameter | What it controls | Decision rule |
|---|---|---|
| `embedding_model_endpoint_name` | Which model is called to embed source text and queries for a Delta Sync Index | Use `databricks-gte-large-en` for English-only workloads where chunks may reach 2000 tokens; use `databricks-bge-large-en` only when all chunks are guaranteed ≤ 512 tokens; use `databricks-qwen3-embedding-0-6b` for multilingual content or chunks up to ~32K tokens |
| `embedding_dimension` | Declared dimension of precomputed embedding vectors in a Direct Vector Access Index | Must match the model output dimension exactly (GTE = 1024, BGE = 1024, Qwen3 = up to 1024 configurable); mismatch causes index creation failure |
| Distance metric (normalisation) | Whether L2 distance rankings match cosine similarity rankings | Normalise vectors before insertion (or choose BGE which normalises by default) when cosine-equivalent ranking is required; skip normalisation only when using dot product on unnormalised vectors for specific recommendation-system workloads |
| `chunk_size` (in chunking pipeline, not AI Search itself) | Maximum number of tokens per chunk fed to the embedding model | Set `chunk_size` ≤ `model_max_tokens × 0.9` to prevent truncation; e.g. for BGE use `chunk_size=450`, for GTE use `chunk_size=7000`, for Qwen3 use `chunk_size=28000` |
| `DatabricksEmbeddings(endpoint=...)` | Which serving endpoint `embed_documents` / `embed_query` calls are routed to | Use the same endpoint name for both document embedding and query embedding unless you explicitly want asymmetric embedding (generally avoid) |

---


---

## Worked Example: Requirement → Decision

**Given:** A multilingual customer-support knowledge base contains 50,000 articles in English, Spanish, and Japanese. Article bodies average 1800 characters (≈ 600 English tokens; ≈ 900 CJK tokens). The team wants a Databricks AI Search Delta Sync Index with Databricks-managed embeddings. Budget is constrained.

**Step 1 — Identify the goal:**
Produce an embedding for every article chunk that captures full semantic content without truncation, supports all three languages, and can be queried with low latency.

**Step 2 — Define inputs:**
- Article text (multilingual, up to ~900 tokens when CJK-heavy)
- Databricks AI Search Delta Sync Index with `embedding_model_endpoint_name`
- Budget consideration: pay-per-token endpoint vs. provisioned throughput

**Step 3 — Define outputs:**
- Dense vector per chunk that accurately represents the full article, usable for cosine-equivalent similarity search
- `embedding_dimension` = 1024

**Step 4 — Apply constraints:**
- BGE Large En: max 512 tokens → Japanese articles at 900 tokens would be *severely* truncated. **Eliminated.**
- GTE Large En: max 8192 tokens, English-only model. Spanish/Japanese coverage uncertain; model name contains "en". **Risky for multilingual.**
- Qwen3-Embedding-0.6B (`databricks-qwen3-embedding-0-6b`): supports 100+ languages including Japanese and Spanish, ~32K token window, 1024 max dimensions. **Fits all constraints.**

**Step 5 — Select the approach:**
Use `databricks-qwen3-embedding-0-6b` as `embedding_model_endpoint_name` with `chunk_size=28000` tokens (safely below the 32K limit) and `embedding_dimension=1024`. This is preferable to GTE because it is explicitly multilingual, and preferable to BGE because the 512-token limit would truncate CJK content significantly. For production, use a provisioned-throughput endpoint to handle ingestion bursts.

---

## Implementation

```python
# Scenario: Embed a batch of knowledge-base article chunks for indexing into
# Databricks AI Search using DatabricksEmbeddings. The constraint is that
# the application runs outside a Databricks workspace, so credentials must
# be set explicitly. We need both document-level and query-level embeddings.

import os
from databricks_langchain import DatabricksEmbeddings

# Configure credentials when running outside Databricks
os.environ["DATABRICKS_HOST"] = "https://your-workspace.cloud.databricks.com"
os.environ["DATABRICKS_TOKEN"] = "dapi..."  # Use secrets manager in production

# Initialise the embedding model
# endpoint must be an OpenAI-compatible embedding endpoint on Model Serving
embeddings = DatabricksEmbeddings(
    endpoint="databricks-gte-large-en",  # 1024-dim, 8192 token window
)

# --- Document embedding (index time) ---
article_chunks = [
    "GTE Large supports a context window of 8192 tokens...",
    "Databricks AI Search uses HNSW for approximate nearest neighbour search...",
    "The L2 distance is used internally; normalise vectors for cosine equivalence.",
]

doc_vectors = embeddings.embed_documents(article_chunks)
print(f"Embedded {len(doc_vectors)} chunks, each {len(doc_vectors[0])} dims")
# Output: Embedded 3 chunks, each 1024 dims

# --- Query embedding (retrieval time) ---
query = "How does Databricks Vector Search calculate similarity?"
query_vector = embeddings.embed_query(query)
print(f"Query vector: {len(query_vector)} dims")
# Output: Query vector: 1024 dims
```

```python
# Anti-pattern: Using BGE Large (512-token limit) with 1000-token chunks.
# This causes SILENT truncation at index time: the embedding represents
# only tokens 1–512; tokens 513–1000 are permanently discarded without
# raising any error or warning. Queries about content in the latter half
# of chunks will retrieve the wrong documents or return no results.

# Anti-pattern: chunk_size=1000 with a 512-token-limit model
from langchain.text_splitter import RecursiveCharacterTextSplitter
from databricks_langchain import DatabricksEmbeddings

splitter_wrong = RecursiveCharacterTextSplitter(
    chunk_size=4000,   # ~1000 tokens — exceeds BGE's 512-token limit
    chunk_overlap=200,
)

embeddings_wrong = DatabricksEmbeddings(endpoint="databricks-bge-large-en")
# embed_documents below silently truncates every chunk at token 512
# No exception. No log warning. Retrieval quality silently degrades.

# ─────────────────────────────────────────────────
# Correct approach: align chunk_size to the model's token limit

splitter_correct = RecursiveCharacterTextSplitter(
    chunk_size=1800,   # ~450 tokens — safely below BGE's 512-token limit
    chunk_overlap=100,
)

# Alternatively, switch to a model with a larger context window
embeddings_correct = DatabricksEmbeddings(endpoint="databricks-gte-large-en")
splitter_correct_gte = RecursiveCharacterTextSplitter(
    chunk_size=28000,  # ~7000 tokens — well within GTE's 8192-token limit
    chunk_overlap=500,
)
# Now every character in every chunk is captured in the embedding.
```

```python
# Scenario: Create a Databricks AI Search Delta Sync Index using the Python SDK,
# specifying a managed embedding endpoint. Required when chunk content is stored
# in a Delta table and you want automatic incremental sync.

from databricks.ai_search.client import AISearchClient

client = AISearchClient()

index = client.create_delta_sync_index(
    endpoint_name="my_vector_search_endpoint",
    source_table_name="prod.knowledge_base.articles",
    index_name="prod.knowledge_base.articles_vs_index",
    pipeline_type="TRIGGERED",
    primary_key="article_id",
    embedding_source_column="chunk_text",       # text column to embed
    # Use qwen3 for multilingual content with long context
    embedding_model_endpoint_name="databricks-qwen3-embedding-0-6b",
)
print(index.describe())
```

---

## Common Pitfalls & Misconceptions

- **Silent truncation with small-context models** — Beginners assume that if code runs without errors, the embeddings are complete. The tokeniser's `truncation=True` default silently drops tokens beyond the model's limit; the embedding is created from partial content with no diagnostic output. Always verify `len(tokeniser.encode(chunk)) < model_max_tokens` before indexing.

- **Assuming "more dimensions = always better"** — Engineers often select the highest-dimensional model available without considering storage and query latency costs. Higher dimensions do not automatically improve retrieval if the training data or fine-tuning objective doesn't match the domain; a smaller model trained on domain-relevant data often outperforms a larger general model. Always benchmark on your actual query distribution before committing.

- **Confusing bi-encoder and cross-encoder use cases** — A common mistake is trying to use a cross-encoder (e.g., a reranker) to produce vectors for a vector index, or expecting a bi-encoder to produce relevance scores as accurate as a cross-encoder. Bi-encoders produce independent document and query vectors for scalable retrieval; cross-encoders produce a single relevance score from both inputs simultaneously and cannot be used to pre-compute document embeddings.

- **Using different embedding models for indexing and querying** — If you embed documents with model A and queries with model B, the vectors live in different semantic spaces and dot-product similarity becomes meaningless. Unless you deliberately configure `model_endpoint_name_for_query` to match a compatible model, always use the same endpoint for both operations.

- **Treating character count as a proxy for token count** — Beginners set `chunk_size=2000` thinking it represents 2000 tokens, but text splitters in LangChain default to measuring character count, not token count. Japanese text with 2000 characters may contain 1500+ tokens, blowing past BGE's 512-token limit. Use `RecursiveCharacterTextSplitter.from_tiktoken_encoder()` or `transformers.AutoTokenizer` to measure tokens accurately.

---

## Key Definitions

| Term | Definition |
|---|---|
| Embedding | A fixed-dimension dense vector representation of text that encodes semantic meaning; produced by an embedding model |
| Embedding model | A transformer encoder fine-tuned to map text to a vector space such that semantically similar texts have nearby vectors |
| Dimensionality | The number of floating-point values in an embedding vector; controls the capacity of the vector space and storage cost |
| Max sequence length | The maximum number of tokens an embedding model accepts in a single forward pass; input exceeding this is silently truncated |
| Token | The atomic unit of text processed by a language model; roughly 3–4 English characters on average, 1–2 characters for CJK scripts |
| Bi-encoder | An embedding architecture that independently encodes query and document; enables precomputed document embeddings for scalable retrieval |
| Cross-encoder | An architecture that jointly encodes query and document to produce a relevance score; used for reranking only, not for vector indexing |
| Cosine similarity | A distance metric measuring the angle between two vectors; normalisation makes it length-invariant |
| L2 (Euclidean) distance | The straight-line distance between two points in vector space; used internally by Databricks AI Search's HNSW algorithm |
| Mean pooling | A pooling strategy that averages all token-level representations to produce a single sequence-level embedding |
| CLS token pooling | A pooling strategy that uses the output of the special `[CLS]` token (prepended during tokenisation) as the sequence embedding |
| Databricks AI Search | Databricks' integrated vector search service (formerly Vector Search); stores and queries embedding-based indexes backed by Delta tables |
| DatabricksEmbeddings | The LangChain integration class (`databricks-langchain` package) for calling Databricks Model Serving embedding endpoints |
| HNSW | Hierarchical Navigable Small World; the approximate nearest-neighbour algorithm used by Databricks AI Search |

---

## Summary / Quick Recall

- Embedding models compress text of any length into a fixed-size dense vector via a transformer encoder + pooling.
- **BGE Large En** = 1024 dims, 512-token max, pre-normalised. **GTE Large En** = 1024 dims, 8192-token max, not pre-normalised. **Qwen3-Embedding-0.6B** = up to 1024 configurable dims, ~32K-token max, multilingual.
- Chunks longer than the model's token limit are silently truncated — no error, no warning, degraded retrieval.
- Set `chunk_size` to ≤ 90% of the model's token limit, measured in *tokens*, not characters.
- Databricks AI Search uses L2 distance internally; pre-normalise vectors (or use BGE) for cosine-equivalent results.
- Bi-encoders produce independent query/doc vectors for scalable indexing; cross-encoders are rerankers only.
- Use the same embedding model endpoint for both document ingestion and query-time embedding.

---

## Self-Check Questions

1. What is the maximum sequence length (in tokens) of `databricks-bge-large-en`, and what happens when a chunk exceeds that limit?

   <details><summary>Answer</summary>

   `databricks-bge-large-en` has a maximum sequence length of **512 tokens**. When a chunk exceeds this limit, the tokeniser silently truncates the input to the first 512 tokens before passing it to the model. The resulting embedding reflects only the first 512 tokens of the chunk; content beyond that is permanently discarded at index time. No exception is raised and no warning appears in logs, which is why this is one of the most dangerous misconfiguration patterns in RAG pipelines. The distractor "an exception is raised" is wrong — HuggingFace tokenisers default to `truncation=True`, which is a silent operation.

   </details>

2. A developer wants to index 10 million English-language product descriptions, each approximately 300–600 tokens long. They need the lowest possible per-query latency at retrieval time. Which Databricks-hosted embedding model is the best choice and why?

   <details><summary>Answer</summary>

   `databricks-gte-large-en` is the best choice. At 300–600 tokens, all chunks fall comfortably within its 8192-token window, so no truncation occurs. The 1024-dimensional output provides strong semantic representation for English text. The alternative, `databricks-bge-large-en`, has a 512-token window — chunks at the upper end of the range (600 tokens) would be truncated. `databricks-qwen3-embedding-0-6b` would also work but its multilingual capability is unnecessary overhead for an English-only corpus. Query latency is similar across all three since the retrieval step (ANN search on the index) is independent of model choice once vectors are ingested.

   </details>

3. **Which TWO** of the following statements about distance metrics in Databricks AI Search are correct?
   - A. Databricks AI Search uses cosine similarity as its internal distance algorithm
   - B. When embedding vectors are pre-normalised to unit length, L2 distance rankings are identical to cosine similarity rankings
   - C. Dot product and cosine similarity always produce identical rankings regardless of normalisation
   - D. Databricks AI Search uses the L2 distance metric with HNSW for approximate nearest-neighbour search
   - E. BGE Large En does not normalise its output embeddings, so L2 and cosine rankings will differ

   <details><summary>Answer</summary>

   **Correct answers: B and D.**

   **B** is correct: when vectors have unit length (magnitude = 1), the L2 distance formula simplifies algebraically to a monotonic function of the cosine similarity, so rankings are identical.

   **D** is correct: the Databricks AI Search documentation explicitly states it uses the HNSW algorithm with L2 distance for approximate nearest-neighbour search.

   **A** is wrong: AI Search uses L2 distance, not cosine similarity directly.

   **C** is wrong: dot product and cosine similarity only agree when both vectors are pre-normalised. For unnormalised vectors they diverge because dot product is sensitive to vector magnitude.

   **E** is wrong: `databricks-bge-large-en` *does* generate pre-normalised (unit-length) embeddings, as stated in the Databricks documentation; GTE Large does not.

   </details>

4. A retrieval pipeline uses `databricks-gte-large-en` to embed 10,000 product FAQs at index time, but a junior engineer later swaps the query-time embedding call to use `databricks-bge-large-en` to save cost. What will happen to retrieval quality and why?

   <details><summary>Answer</summary>

   Retrieval quality will degrade severely or become essentially random. GTE and BGE are trained with different objectives, architectures, and training data; their vector spaces are completely unrelated. The dot product between a GTE document vector and a BGE query vector measures nothing meaningful — there is no shared coordinate system. The index will return documents based on arbitrary numerical proximity rather than semantic similarity. The fix is to either use the same model for both indexing and querying, or to rebuild the index using BGE if a switch to BGE is desired. This failure mode is silent — the API returns results without errors, making it particularly insidious in production.

   </details>

5. A team is deciding between a bi-encoder and a cross-encoder for their RAG system. They have 5 million documents and need sub-200ms end-to-end latency. Analyse the trade-offs and justify which architecture should handle initial retrieval vs. reranking.

   <details><summary>Answer</summary>

   **Bi-encoder for initial retrieval; cross-encoder for reranking only.**

   A bi-encoder encodes documents once at index time, storing 5 million precomputed vectors. At query time only the query is encoded (one forward pass), and the ANN index performs retrieval in O(log n) time — easily sub-200ms even at this scale.

   A cross-encoder must jointly encode every (query, document) pair at query time. Running 5 million cross-encoder inferences per query would take minutes, not milliseconds, making it impossible to achieve sub-200ms latency.

   The correct architecture is to use the bi-encoder to retrieve the top-20 or top-50 candidates from the vector index (fast, scalable), then apply the cross-encoder only to those candidates for reranking (20–50 forward passes is tractable in milliseconds). This is precisely the architecture used by Databricks AI Search's `reranking` option in `similarity_search`. The trade-off is that the bi-encoder's approximation means the initial candidate set may not be perfectly optimal, but the cross-encoder corrects this for the final ranking.

   </details>

---

## Further Reading

- [Databricks AI Search overview](https://docs.databricks.com/en/generative-ai/vector-search.html) — *verified 2026-07-11* — Architecture, HNSW algorithm, distance metrics, hybrid search
- [Create AI Search endpoints and indexes](https://docs.databricks.com/en/ai-search/create-ai-search.html) — *verified 2026-07-11* — SDK parameters for `embedding_model_endpoint_name`, `embedding_dimension`, sync modes
- [Databricks-hosted foundation models](https://docs.databricks.com/en/machine-learning/foundation-model-apis/supported-models.html) — *verified 2026-07-11* — Token limits and dimensions for GTE Large, BGE Large, and Qwen3-Embedding-0.6B
- [Foundation Model APIs overview](https://docs.databricks.com/en/machine-learning/foundation-models/index.html) — *verified 2026-07-11* — Pay-per-token vs. provisioned throughput; embedding endpoint types
- [DatabricksEmbeddings LangChain integration](https://python.langchain.com/docs/integrations/text_embedding/databricks/) — *verified 2026-07-11* — `embed_documents`, `embed_query`, async usage, endpoint requirement
