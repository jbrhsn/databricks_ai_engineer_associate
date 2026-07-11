# Embedding Models and Context Length — Interview Prep

**Section:** 04 — Application Development | **Role target:** Senior ML Engineer, AI Platform Engineer, Solutions Architect (AI/ML)

---

## Core Conceptual Questions

These test whether you understand the fundamentals.

| Question | Key points to cover | Common weak-answer trap |
|---|---|---|
| What is an embedding model and how does it produce a fixed-size vector from variable-length text? | Transformer encoder; tokenisation → self-attention layers → pooling (mean or CLS); fixed output dimension regardless of input length | Saying "it compresses text" without describing the pooling mechanism, or conflating it with a generative LLM decoder |
| What is the maximum sequence length of an embedding model and what happens when it is exceeded? | Hard upper bound on token count; tokeniser silently truncates at `max_length`; no exception; embedding reflects only partial content; `truncation=True` is default in HuggingFace | Saying "an error is raised" or "the model refuses the input" — there is no error |
| What is the difference between cosine similarity and L2 distance, and how does Databricks AI Search handle this? | Cosine measures angle (length-invariant); L2 measures Euclidean distance (magnitude-sensitive); AI Search uses HNSW with L2 internally; pre-normalised vectors make L2 rankings identical to cosine rankings | Saying "AI Search uses cosine similarity" (it uses L2); or claiming normalisation doesn't matter |
| What is the difference between a bi-encoder and a cross-encoder? | Bi-encoder: independent query+doc encoding → vectors → scalable precomputation; cross-encoder: joint encoding → single relevance score → reranking only, cannot precompute | Claiming you can use a cross-encoder to build a vector index, or that bi-encoders are always less accurate (they are less accurate per pair but the only option at scale) |
| What are the key Databricks-hosted embedding models and their token limits? | `databricks-gte-large-en` (1024 dims, 8192 tokens, English, not pre-normalised); `databricks-bge-large-en` (1024 dims, 512 tokens, English, pre-normalised); `databricks-qwen3-embedding-0-6b` (up to 1024 configurable dims, ~32K tokens, multilingual) | Mixing up which model normalises output, or confusing token limits between models |

---

## Applied / Scenario Questions

**Q: A team reports that their RAG chatbot works well for short FAQ entries but fails to retrieve relevant context from multi-page technical manuals. The code runs without errors. What would you investigate first?**

**Strong answer framework:**
- Immediately suspect embedding token truncation: check the embedding model's maximum sequence length against the average chunk token count for multi-page manuals.
- Verify by running `len(tokeniser.encode(chunk))` on a sample of manual chunks and comparing to the model's `max_position_embeddings` from the model card.
- If truncation is confirmed: either switch to a model with a larger context window (`databricks-gte-large-en` at 8192 tokens or `databricks-qwen3-embedding-0-6b` at ~32K), or reduce `chunk_size` to fit within the current model's limit.
- Show trade-off awareness: smaller chunks lose cross-sentence context and increase index size and query latency; switching models requires a full index rebuild.
- Add a monitoring assertion: `assert max(token_counts) < model_max_tokens` as a pre-ingestion validation step.

---

**Q: Your team has a multilingual customer support knowledge base (English, Spanish, Japanese). An engineer proposes using `databricks-bge-large-en` because it appears on many RAG tutorial examples. How do you evaluate this proposal?**

**Strong answer framework:**
- Challenge on two dimensions simultaneously: language support and token limit.
- Language: BGE Large En is an English-only model. Japanese and Spanish content may not be well-represented in its training distribution; semantic similarity for those languages will be degraded.
- Token limit: BGE's 512-token limit is extremely tight. Japanese text is tokenised at 1–2 characters/token, so a 500-character Japanese article could exceed 512 tokens. Truncation would drop most of the content.
- Propose `databricks-qwen3-embedding-0-6b`: supports 100+ languages explicitly, ~32K token window, 1024 configurable dimensions.
- Acknowledge cost: Qwen3-Embedding-0.6B is a newer model — verify endpoint availability in your Databricks workspace region before committing.

---

## System Design / Architecture Questions

**Q: Design an embedding pipeline for a 10-million-document English knowledge base on Databricks. Walk through model selection, chunking, indexing, and query-time considerations.**

**Approach:**
1. **Clarify requirements:** Average document length (in tokens), language distribution, latency SLA, query volume, update frequency (batch vs. streaming), budget.
2. **Propose structure:**
   - Chunking: `RecursiveCharacterTextSplitter.from_tiktoken_encoder(encoding_name="cl100k_base", chunk_size=6000, chunk_overlap=400)` — token-aware, safely below GTE Large's 8192 limit.
   - Model: `databricks-gte-large-en` (1024 dims, 8192 token limit). For production, use provisioned throughput endpoint for ingestion bursts; pay-per-token for low-volume querying.
   - Index: Databricks AI Search Delta Sync Index with `embedding_model_endpoint_name="databricks-gte-large-en"`, `pipeline_type="TRIGGERED"` for batch refresh.
   - Pre-normalise embeddings before insertion (GTE does not auto-normalise) if you want cosine-equivalent rankings from the L2-based index.
   - Query: use the same endpoint name for query-time embedding via `DatabricksEmbeddings(endpoint="databricks-gte-large-en")`.
3. **Justify choices and name trade-offs explicitly:**
   - GTE vs. BGE: GTE's 8192-token window eliminates truncation risk for typical English documents; BGE's 512-token limit would require ~12x more aggressive chunking, dramatically increasing index size.
   - Provisioned throughput vs. pay-per-token: pay-per-token is simpler but has no throughput guarantees; provisioned throughput is required for ingestion jobs processing millions of documents.
   - Triggered vs. continuous sync: triggered is cheaper for batch update patterns; continuous is needed if sub-minute freshness is required.

---

## Vocabulary That Signals Expertise

Use these terms naturally — don't force them:

- **Mean pooling** — when describing how GTE-family models aggregate token representations (vs. CLS token pooling used in older BERT models)
- **Matryoshka Representation Learning (MRL)** — when discussing Qwen3's configurable dimensionality; signals awareness of modern embedding compression techniques
- **HNSW (Hierarchical Navigable Small World)** — when describing Databricks AI Search's ANN algorithm; shows you understand how retrieval speed is achieved
- **Approximate nearest neighbour (ANN)** — the correct term for what vector search does; "exact nearest neighbour" search is impractically slow at scale
- **Token budget** — framing chunk size as a resource with a hard limit, not just a tuning parameter
- **Index-time vs. query-time encoding** — the correct framing of when bi-encoder embeddings are produced
- **Semantic vector space** — the shared coordinate system that must be consistent between index-time and query-time embeddings

---

## Vocabulary That Signals Weakness

Avoid these — they signal outdated or shallow understanding:

- **"Word embeddings"** in the context of transformer-based models — word2vec and GloVe are static word-level embeddings; transformer encoders produce *contextual* embeddings where meaning depends on surrounding context
- **"The model converts text to numbers"** — too vague; demonstrates no understanding of the pooling step or why fixed dimensionality matters
- **"Cosine similarity is what the database uses"** — Databricks AI Search uses L2 distance; confidently stating otherwise signals you haven't read the docs
- **"More dimensions is always better"** — demonstrates no awareness of storage cost, query latency, or the diminishing returns of over-parameterised embedding spaces
- **"I'll use the top model from the MTEB leaderboard"** — without qualification this signals you pick models by benchmark ranking rather than domain fit, token limit, and language coverage
- **"The embedding model reads the document"** — embedding models do not "read" in a sequential sense; they process all tokens simultaneously in parallel via self-attention

---

## STAR Answer Frame

**Situation:** I was leading the RAG infrastructure for a customer-facing knowledge base chatbot serving support queries in English and Japanese. After launch, Japanese retrieval quality was significantly worse than English — users were getting irrelevant responses to Japanese questions even when the answer existed in the knowledge base.

**Task:** I needed to diagnose why Japanese retrieval was systematically worse, determine whether it was a data problem, a model problem, or a configuration problem, and fix it without rebuilding the entire index from scratch if possible.

**Action:** I started by measuring token counts on Japanese vs. English chunks. Using `transformers.AutoTokenizer` for our embedding model (`databricks-bge-large-en`), I found that Japanese chunks were averaging 490–650 tokens vs. 200–300 tokens for English — many Japanese chunks exceeded BGE's 512-token limit. I confirmed truncation was occurring by comparing `len(tokeniser.encode(text))` against `len(tokeniser.encode(text, truncation=True, max_length=512))` — they differed for 43% of Japanese chunks. I proposed two options: (1) reduce Japanese `chunk_size` from 2000 characters to 800 characters to stay under the 512-token limit, or (2) rebuild the index using `databricks-gte-large-en` with an 8192-token window. I presented the trade-off: option 1 was fast but would fragment Japanese articles into too-small chunks lacking cross-sentence context; option 2 required a full index rebuild but provided correct embeddings. We chose option 2.

**Result:** After rebuilding the index with GTE Large and aligned chunk sizes (4000 characters / ~1000 tokens), Japanese P@5 improved from 0.41 to 0.79 — a 93% relative improvement — with no change to the LLM or prompts. We also added a pre-ingestion assertion that aborts if any chunk exceeds 90% of the model's token limit, preventing recurrence.

---

## Red Flags Interviewers Watch For

- **Assuming `chunk_size` is a character count when the model imposes a token limit** — a senior engineer should know these are different units and should always verify which unit their text splitter uses
- **Not knowing that Databricks AI Search uses L2 distance, not cosine similarity** — this is in the official documentation; not knowing it suggests you haven't worked with the platform
- **Proposing to use a cross-encoder for initial retrieval** — shows a fundamental misunderstanding of the scalability constraints; cross-encoders cannot precompute document embeddings
- **Claiming that changing the query-time embedding model is safe without rebuilding the index** — a critical operational error; different models produce incompatible vector spaces
- **Evaluating embedding model quality only on MTEB without considering token limits or language coverage** — benchmark-first thinking that ignores operational constraints
- **Not being able to name the specific token limits of the Databricks-hosted embedding models** — on an exam or in a Databricks-focused interview, these numbers are expected knowledge
