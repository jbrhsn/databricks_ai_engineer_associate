# How LLMs Work: Tokens & Embeddings — Interview Prep

**Section:** Foundations | **Role target:** GenAI Engineer / ML Engineer

## Core Conceptual Questions

These test whether you understand the fundamentals.

| Question | Key points to cover | Common weak-answer trap |
|---|---|---|
| What is a token and how does BPE tokenization work? | Sub-word unit mapped to an integer ID; BPE greedily merges frequent character/byte pairs into a fixed vocabulary; common words = 1 token, rare/long/code/non-English = many; ~1 token ≈ ¾ word. | Saying "a token is a word" or "a character" — signals no grasp of cost/limit implications. |
| Why is token count, not word count, the right unit for cost and limits? | Billing and context windows are measured in tokens; `usage.total_tokens` is authoritative; word counts underestimate for technical text. | Claiming they're "basically the same" — under-provisions budgets in the real world. |
| What is an embedding and what property makes it useful? | Dense fixed-length vector encoding meaning; semantically similar text is geometrically near; enables cosine/L2 comparison. | Describing it as "a compressed version of the text" — misses the semantic-similarity property that powers search. |
| Token embeddings vs sentence/document embeddings? | Token embeddings: per-token, evolve with context, drive next-token prediction inside the LLM. Document embeddings: one pooled vector per span, for retrieval/clustering. Different endpoints. | Treating them as the same thing — leads to using a chat model where an embedding model was needed. |
| How does an LLM generate text? | Autoregressive next-token prediction; each step outputs a vocab-wide distribution via softmax; decoding (greedy/temperature/top_p) samples one token; append and repeat. | "It predicts the next word" without mentioning tokens, distributions, or the append loop. |
| What is a context window and why does it matter? | Per-request token cap for prompt + output; no cross-session memory; attention scales with length so longer = slower/costlier. | Calling it the model's "memory" — reveals the persistence misconception. |

## Applied / Scenario Questions

**Q:** You're building semantic search over a knowledge base on Databricks and paraphrased queries aren't matching formally-worded documents. How do you architect the retrieval path?

**Strong answer framework:**
- Chunk documents by tokens (single idea per chunk, well under the embedding model's token limit).
- Embed chunks and queries with the *same* embedding endpoint — e.g. `databricks-gte-large-en` (1024-dim) via `embeddings.create` or LangChain `DatabricksEmbeddings`.
- Store vectors in a Databricks AI Search index; retrieve top-k by L2 (HNSW).
- Consider hybrid keyword-similarity search when the corpus has exact identifiers (SKUs) that pure vector search misses.
- Show tradeoff awareness: bigger chunks improve recall of context but hurt passage precision; normalize vectors so L2 ranking matches cosine.

**Q:** A cost review shows your LLM spend is 30% over estimate. Where do you look first?

**Strong answer framework:**
- Confirm the estimate wasn't built from word counts; re-measure with `usage.total_tokens`.
- Check for token-heavy content (code, IDs, non-English) that BPE fragments.
- Audit `max_tokens` on generation and prompt bloat; trim retrieved context to what's needed.
- Tradeoff: shrinking context lowers cost/latency but risks dropping relevant evidence — tune against retrieval quality, not blindly.

## System Design / Architecture Questions (if applicable)

**Q:** Design a production RAG system on Databricks and explain how tokens and embeddings shape each decision.

**Approach:**
1. Clarify requirements: corpus size, query style (paraphrase vs exact), latency/cost budget, reproducibility/audit needs.
2. Propose structure: ingest → token-based chunking → embed (`databricks-gte-large-en`) → AI Search index (HNSW, L2) → retrieve top-k → chat endpoint with `temperature=0` for grounded, reproducible answers.
3. Justify choices and name tradeoffs: pin the embedding model as a schema decision (switching = full re-embed + rebuild); size chunks against the embedding token limit; measure cost in tokens; use hybrid search if exact identifiers matter; note the context window is a per-request cap, so retrieval (not "memory") supplies knowledge each call.

## Vocabulary That Signals Expertise

Use these terms naturally — don't force them:
- **Sub-word tokenization / BPE** — when explaining why token ≠ word and cost estimates drift.
- **`usage.total_tokens`** — when discussing cost/limit measurement; shows you read the response, not the docs only.
- **Embedding space / dimensionality** — when explaining why models can't be mixed in one index.
- **Approximate nearest neighbor (HNSW), L2 distance** — when describing how AI Search retrieves.
- **Autoregressive next-token prediction** — when describing generation precisely.
- **Normalization (unit vectors)** — when reconciling L2 vs cosine ranking.
- **Chunking strategy** — when connecting token limits to retrieval precision.

## Vocabulary That Signals Weakness

Avoid these — they signal outdated or shallow understanding:
- "The model remembers our previous chat" (about the context window) — reveals the persistence misconception.
- "Embeddings compress the text" — misses the semantic-similarity purpose.
- "One word is one token" — signals no grasp of cost mechanics.
- "Just use the chat model to make the vectors" — conflates token embeddings with document embeddings.
- "We'll swap embedding models later, no big deal" — ignores that it's a full re-embed migration.

## STAR Answer Frame

**Situation:** Our support RAG chatbot returned the right document but the wrong passage, and monthly LLM cost ran ~30% over the approved budget.  
**Task:** I owned retrieval quality and cost containment for the launch.  
**Action:** I switched cost/limit estimation from `str.split()` word counts to `usage.total_tokens`, re-chunked documents by tokens into single-idea passages (~500 tokens), standardized on one embedding endpoint (`databricks-gte-large-en`) and re-embedded the whole corpus into a single AI Search index, and added a normalization check so L2 ranking matched cosine.  
**Result:** Passage-level precision improved noticeably in eval, spend came back within budget, and we eliminated the silent-truncation incidents caused by oversized chunks.

## Red Flags Interviewers Watch For

- Cannot articulate why token count differs from word count — suggests they've never sized a real workload.
- Uses "embeddings" without distinguishing the retrieval vector from the model's internal token vectors.
- Thinks the context window is persistent memory — will design systems that skip retrieval.
- Proposes mixing or hot-swapping embedding models in a live index without a re-embed plan.
- Assumes L2 and cosine always rank identically, unaware normalization is the precondition.
- Reaches for a bigger model/context window as the first fix for every quality or cost problem.
