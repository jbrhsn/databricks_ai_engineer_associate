# How LLMs Work: Tokens & Embeddings

**Section:** Foundations | **Module:** LLM & NLP Fundamentals | **Est. time:** 2 hrs | **Exam mapping:** Foundational (supports Design & Data Prep domains)

---

## TL;DR

A large language model never sees your words — it sees *tokens* (sub-word integer IDs) that it maps to *embeddings* (dense vectors encoding meaning), and it works by repeatedly predicting the next token given every token so far. Tokens are the unit that drives cost, latency, and the context-window ceiling; embeddings are the unit that makes semantic search and RAG possible. On Databricks you meet both through Foundation Model APIs — chat endpoints consume tokens, and embedding endpoints like `databricks-gte-large-en` turn text into vectors for AI Search. **The one thing to remember: tokens are how the model reads and how you pay; embeddings are how the model (and your retriever) understands meaning — they are different objects for different jobs.**

---

## ELI5 — Explain It Like I'm 5

Imagine a librarian who cannot read letters, only numbered index cards. Before she can help you, every sentence you hand her is chopped into small familiar chunks — "un", "believ", "able" — and each chunk is swapped for the card number she has memorised. That numbering step is *tokenization*. Now, for every card she owns, she also keeps a pin on a giant map where cards about similar ideas are pinned close together — "cat" sits near "kitten", far from "invoice"; that map of positions is *embeddings*. When you ask a question, she finds the pins nearest your question's pin and reads those cards back to you — that is semantic search. The common mistake is thinking she counts *words* the way you do: she counts *cards*, and a single long or unusual word can cost her several cards, which is exactly why your bill and her memory limit are measured in tokens, not words.

---

## Learning Objectives

By the end of this chapter you will be able to:
- [ ] Explain how sub-word (BPE) tokenization converts text to token IDs and why token count ≠ word count.
- [ ] Diagnose context-window, cost, and latency problems by reasoning about token budgets.
- [ ] Compare token embeddings with sentence/document embeddings and choose the right one for a task.
- [ ] Implement an embedding call against a Databricks Foundation Model API endpoint and verify normalization.
- [ ] Design the tokenize → embed → retrieve path that powers semantic search and RAG on Databricks.

---

## Visual Overview

### Text-to-Prediction Pipeline

```
"unbelievable results"
        │
        ▼
  Tokenizer (BPE)          ──►  ["un","believ","able"," results"]  (4 tokens, not 2 words)
        │
        ▼
  Token IDs                ──►  [521, 8912, 481, 3629]
        │
        ▼
  Embedding lookup         ──►  each ID → vector (e.g. 1024 floats)
        │
        ▼
  Transformer layers       ──►  context-aware representation of every position
        │
        ▼
  Next-token distribution  ──►  P("!")=0.4  P(".")=0.3  P(" today")=0.1 ...
        │
        ▼
  Sample 1 token, append, repeat ──► streamed output
```

### Token Embedding vs Document Embedding

```
TOKEN EMBEDDING (inside the LLM)          DOCUMENT EMBEDDING (for retrieval)
┌──────────────────────────┐             ┌──────────────────────────────┐
│ one vector PER token      │             │ ONE vector for the whole text │
│ changes as context grows  │             │ fixed-size summary of meaning │
│ used to predict next token│             │ used to compare texts         │
└──────────────────────────┘             └──────────────────────────────┘
        │                                          │
        ▼                                          ▼
  drives generation                        drives semantic search / RAG
  (chat endpoint)                          (databricks-gte-large-en → AI Search)
```

### RAG Retrieval Flow on Databricks

```
User query ──► embed query (gte-large-en) ──► query vector
                                                   │
                              ┌────────────────────┘
                              ▼
                   AI Search index (HNSW, L2)
                              │
              top-k nearest document vectors
                              │
                              ▼
        chunks + query ──► chat endpoint (tokens in) ──► grounded answer
```

---

## Key Concepts

### Tokenization (BPE / sub-word)

**What is it?** Tokenization is the process of splitting raw text into tokens — the smallest units the model operates on — where a token is typically a sub-word fragment, not a whole word or a single character.

**How does it work under the hood?** Byte-Pair Encoding (BPE) starts from individual characters/bytes and greedily merges the most frequently co-occurring pairs into a fixed vocabulary (commonly 30k–200k entries). Common words become a single token; rare or novel strings (misspellings, code, long compound words, non-English text) fragment into several tokens. Each token maps to an integer ID via a lookup table the model was trained with, so the *same* tokenizer must be used at inference as at training. As a rough English rule of thumb, 1 token ≈ 4 characters ≈ ¾ of a word.

**Where does it appear in Databricks/the ecosystem?** Every Foundation Model API request is billed and limited by tokens, and the embeddings response returns a `usage.total_tokens` field. Provisioned-throughput endpoints let you register a model *with its own tokenizer* in Unity Catalog, which is why swapping tokenizers between train and serve is a supported, explicit step.

### Context Window

**What is it?** The context window is the maximum number of tokens (prompt + generated output) a model can attend to in a single request.

**How does it work under the hood?** The transformer's attention mechanism relates every token to every other token; the window is the hard cap on how many positions fit. Exceeding it forces truncation or an error, and because attention cost grows with sequence length, longer contexts mean higher latency and cost even when they fit. Databricks-hosted models publish explicit ceilings — e.g. `databricks-meta-llama-3-3-70b-instruct` has a 128,000-token context, `databricks-gpt-5` a 400K total window, and the `databricks-qwen3-embedding-0-6b` embedding model handles ~32K tokens.

**Where does it appear in Databricks/the ecosystem?** The [supported-models page](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/supported-models) lists each endpoint's context window; RAG chunking strategy and `max_tokens` settings are chosen against this number.

> ⚠️ Fast-evolving: verify against current official docs before relying on this. Model endpoint names, context windows, and the Foundation Model APIs surface change frequently.

### Embeddings (semantic vectors)

**What is it?** An embedding is a dense fixed-length vector of floating-point numbers that encodes the meaning of a token, sentence, or document, such that semantically similar inputs land near each other in vector space.

**How does it work under the hood?** An embedding model (a transformer encoder) reads the input tokens and pools their contextual representations into one vector — e.g. `databricks-gte-large-en` outputs a 1024-dimensional vector. Similarity is then a geometric operation: cosine similarity or L2 distance between vectors. Because the model was trained so that "refund policy" and "how do I get my money back" produce nearby vectors, meaning — not keyword overlap — governs closeness.

**Where does it appear in Databricks/the ecosystem?** You call an embedding endpoint (`databricks-gte-large-en`, `databricks-bge-large-en`, `databricks-qwen3-embedding-0-6b`) via the OpenAI-compatible `embeddings.create` API or LangChain's `DatabricksEmbeddings`. The resulting vectors are stored in a Databricks **AI Search** index (formerly Vector Search), which searches them with the HNSW approximate-nearest-neighbor algorithm using L2 distance.

### Token Embeddings vs Sentence/Document Embeddings

**What is it?** Token embeddings are the per-token vectors *inside* a generative LLM; sentence/document embeddings are a *single* vector summarising an entire span of text for comparison.

**How does it work under the hood?** During generation, each token has its own evolving vector that the model uses to predict the next token — there are as many vectors as tokens. An embedding model instead pools all those token representations (mean-pooling or a special `[CLS]`-style token) into one fixed-size vector regardless of input length, so two documents of different lengths still yield comparable vectors. The former is optimised for *prediction*; the latter for *retrieval*.

**Where does it appear in Databricks/the ecosystem?** Chat/completion endpoints operate on token embeddings internally (you never see them); embedding endpoints expose the single pooled vector in `response.data[0].embedding`. Using a chat model where you needed an embedding model is a category error the API will not stop you from making.

### Next-Token Prediction (how generation works)

**What is it?** Autoregressive generation is producing text one token at a time, each step predicting a probability distribution over the whole vocabulary for the next token.

**How does it work under the hood?** The transformer takes all prior tokens' embeddings, runs them through stacked self-attention + feed-forward layers, and outputs a score per vocabulary entry; a softmax turns those into probabilities. A decoding strategy (greedy, or sampling controlled by `temperature`/`top_p`) picks one token, which is appended and fed back in — so output length in tokens directly drives latency and cost.

**Where does it appear in Databricks/the ecosystem?** The chat/completions endpoints stream these tokens back; `max_tokens`, `temperature`, and `top_p` are the request parameters that shape this loop. (Note: some models, e.g. `databricks-claude-sonnet-5`, reject `temperature`/`top_p` — check the model card.)

### Key Parameters / Configuration Knobs

| Parameter | What it controls | Decision rule |
|---|---|---|
| `model` (endpoint name) | Which served model handles the request | For semantic search/RAG retrieval use an embedding endpoint (`databricks-gte-large-en`); for generation use a chat endpoint — never mix them. |
| `max_tokens` | Cap on generated output tokens | Set to the smallest value that fits your expected answer; every extra token adds latency and cost. |
| `temperature` | Randomness of next-token sampling | Use `0` for extraction/classification/RAG answers you want reproducible; raise toward `0.7` only for creative drafting. |
| `top_p` | Nucleus-sampling probability mass | Leave at default unless you specifically need to constrain diversity; tune `temperature` OR `top_p`, not both. |
| Embedding dimensionality (model choice) | Vector length stored in the index | Match the index's expected dimension to the model (e.g. gte-large-en = 1024); changing models means re-embedding the whole corpus. |
| Chunk size (tokens) for RAG | How much text each stored embedding covers | Keep chunks well under the embedding model's token limit (~32K for qwen3-embedding) and small enough that one chunk is one idea — usually 200–1000 tokens. |

### Worked Example: Requirement → Decision

**Given:** A support team wants a chatbot that answers policy questions from a 5,000-document knowledge base. Questions are paraphrased freely ("can I get my money back?"), documents use formal wording ("refund eligibility"), and answers must be reproducible for audit.

- **Step 1 — Identify the goal:** Retrieve the *meaning-relevant* documents for a paraphrased query, then generate a grounded, deterministic answer.
- **Step 2 — Define inputs:** The 5,000 documents (to be chunked and embedded) and the live user query text.
- **Step 3 — Define outputs:** (a) one document embedding vector per chunk stored in an index; (b) at query time, top-k nearest chunks; (c) a generated answer citing those chunks.
- **Step 4 — Apply constraints:** Keyword search fails on paraphrase; chunks must fit the embedding model's token limit; answers must be reproducible (audit); cost is measured in tokens.
- **Step 5 — Select the approach:** Embed chunks and queries with `databricks-gte-large-en` (1024-dim), store in a Databricks AI Search index (HNSW + L2), retrieve top-k, then call a chat endpoint with `temperature=0`. Rationale: an embedding model captures paraphrase where keyword search cannot, and `temperature=0` gives the reproducibility a pure-keyword or high-temperature approach would not.

---

## Implementation

```python
# Scenario: turn free-text support articles into vectors for semantic search on Databricks,
# and confirm the vectors are normalized so L2 ranking matches cosine ranking in AI Search.
from databricks_openai import DatabricksOpenAI
import numpy as np

client = DatabricksOpenAI()

def embed(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model="databricks-gte-large-en", input=texts)
    # usage.total_tokens is how you are billed — log it to track cost per batch
    print("tokens billed:", resp.usage.total_tokens)
    return [d.embedding for d in resp.data]

def is_normalized(vector, tol=1e-3) -> bool:
    return abs(np.linalg.norm(vector) - 1) < tol

vectors = embed(["How do I get a refund?", "Refund eligibility policy"])
print("dim:", len(vectors[0]), "normalized:", is_normalized(vectors[0]))
# AI Search uses L2 distance; if embeddings are normalized, L2 ranking == cosine ranking.
```

```python
# Anti-pattern: estimating cost/limits by counting words with str.split(), then sending
# the whole document as one embedding "chunk". Word count underestimates tokens (rare words
# and code fragment into many tokens), and one giant chunk buries many ideas in one vector,
# wrecking retrieval precision — and may silently exceed the model's token limit.
def bad_prepare(doc: str):
    word_count = len(doc.split())              # WRONG proxy for token count / cost
    if word_count < 5000:                      # false confidence about the limit
        return embed_single_giant_chunk(doc)   # one vague vector for the whole doc

# Correct approach: chunk by tokens into single-idea passages, embed each chunk, and rely on
# the API's token accounting for cost/limits — not word counts.
def good_prepare(doc: str, chunker):
    chunks = chunker.split(doc, target_tokens=500)   # ~1 idea per chunk, well under the limit
    resp = client.embeddings.create(model="databricks-gte-large-en", input=chunks)
    total = resp.usage.total_tokens                  # authoritative token/cost figure
    return [d.embedding for d in resp.data], total
# What breaks in the anti-pattern: words ≈ ¾ of tokens, so long/technical docs blow the budget
# you thought was safe, and coarse chunks return the right document but the wrong passage.
```

---

## Common Pitfalls & Misconceptions

- **Counting words instead of tokens** — Beginners equate "one word = one token" because that matches how humans read. In reality BPE splits rare/long/non-English strings and code into multiple sub-word tokens, so word counts underestimate cost and context usage; always reason in tokens and read `usage.total_tokens`.
- **Confusing token embeddings with document embeddings** — Learners hear "embeddings" as one thing and assume any model that "makes embeddings" fits retrieval. But token embeddings inside a chat model are per-token and prediction-oriented, whereas a document embedding is one pooled vector for comparison; use a dedicated embedding endpoint for search, not the chat model's internals.
- **Treating the context window as a memory/database** — People assume anything ever said stays "known" to the model. The window is a fixed per-request token cap with no persistence; once you exceed it text is truncated, which is precisely why RAG retrieves and re-injects relevant chunks each call.
- **Mixing embedding models across the corpus** — Beginners re-embed new documents with a different or upgraded model and expect the index to still work. Vectors from different models live in incompatible spaces and dimensions; changing the embedding model requires re-embedding the entire corpus and rebuilding the index.
- **Skipping normalization then using L2** — Newcomers assume L2 distance and cosine similarity always rank identically. They only match when vectors are normalized; if your model's output is not unit-length, normalize it before feeding AI Search or your rankings will drift.

---

## Key Definitions

| Term | Definition |
|---|---|
| Token | The smallest unit an LLM processes — usually a sub-word fragment mapped to an integer ID; the unit of billing, context limits, and latency. |
| BPE (Byte-Pair Encoding) | A tokenization algorithm that builds a fixed vocabulary by greedily merging the most frequent character/byte pairs. |
| Context window | The maximum number of tokens (prompt + output) a model can attend to in one request. |
| Embedding | A dense fixed-length numeric vector encoding the semantic meaning of text so similar meanings are geometrically near. |
| Token embedding | The per-token vector used inside a generative LLM to predict the next token; there is one per token and it evolves with context. |
| Document/sentence embedding | A single pooled vector summarising an entire text span for comparison, retrieval, and clustering. |
| Next-token prediction | Autoregressive generation: at each step the model outputs a probability distribution over the vocabulary and samples one token. |
| Foundation Model APIs | Databricks Model Serving feature exposing hosted open models (chat, embedding, vision) via an OpenAI-compatible API. |
| AI Search | Databricks' vector search solution (formerly Vector Search) that indexes embeddings and retrieves nearest neighbors via HNSW + L2. |

---

## Summary / Quick Recall

- The model reads tokens, not words — BPE fragments rare/long strings, so 1 token ≈ ¾ word.
- Tokens drive cost, latency, and the context-window ceiling; check `usage.total_tokens`.
- Embeddings are dense vectors placing similar meanings close together in space.
- Token embeddings (per-token, for prediction) ≠ document embeddings (one vector, for retrieval).
- Generation = repeatedly predicting and sampling the next token, then appending it.
- On Databricks: embed with `databricks-gte-large-en`/`bge`/`qwen3-embedding`, store in AI Search (HNSW + L2), generate with a chat endpoint.
- Changing embedding models forces a full re-embed; normalize vectors so L2 ranking matches cosine.

---

## Self-Check Questions

1. What is a token, and why does token count usually differ from word count for the same text?

   <details><summary>Answer</summary>

   A token is the smallest unit the model processes — typically a sub-word fragment mapped to an integer ID. Counts differ because BPE tokenization merges frequent character pairs into a fixed vocabulary: common words are one token, but rare words, long compounds, code, and non-English text fragment into several tokens. The tempting wrong answer — "a token is a word" — fails because it would make cost and context-limit estimates wildly optimistic for anything but simple English prose.

   </details>

2. You must build semantic search over 10,000 support articles so paraphrased queries match formal documents. Which Databricks component turns the text into something searchable by meaning, and what do you store?

   <details><summary>Answer</summary>

   Call an embedding endpoint such as `databricks-gte-large-en` to convert each chunk into a dense vector, and store those vectors in a Databricks AI Search index. Retrieval then finds nearest neighbors by L2 distance (HNSW). Answering "use a chat endpoint / keyword search" is wrong: a chat endpoint generates text and exposes no comparable document vector, and keyword search misses paraphrases like "get my money back" vs "refund eligibility".

   </details>

3. **Which TWO** of the following correctly describe the relationship between context window and cost/latency?
   - A. The context window is a per-request token cap covering prompt plus generated output.
   - B. Exceeding the context window makes the model remember earlier sessions instead.
   - C. Longer sequences generally increase latency and cost because attention scales with length.
   - D. The context window is unlimited on provisioned-throughput endpoints.
   - E. Output tokens are free; only input tokens count toward the window.

   <details><summary>Answer</summary>

   **A and C.** A is correct because the window caps prompt + output tokens together for a single request. C is correct because self-attention relates every token to every other, so more tokens cost more compute and time. B is the most tempting distractor — it wrongly treats the window as cross-session memory, but the window has no persistence, which is exactly why RAG re-injects context. D is false (provisioned throughput improves performance, not the window), and E is false (output tokens count toward both the window and the bill).

   </details>

4. A teammate re-embeds newly added documents with a newer embedding model than the one used for the existing 50,000 vectors, keeping them in the same index. Retrieval quality collapses. Why, and what is the correct fix?

   <details><summary>Answer</summary>

   Vectors from different embedding models occupy different, incompatible spaces (and often different dimensions), so nearest-neighbor distances between old and new vectors are meaningless — ranking collapses. The correct fix is to re-embed the entire corpus with a single model and rebuild the index. The tempting wrong answer — "just normalize the new vectors" — does not help, because normalization fixes L2-vs-cosine ranking within one model's space, not incompatibility across two different models.

   </details>

5. For a RAG answer that must be identical every run for audit purposes, you can either (a) send larger retrieved context with `temperature=0.7`, or (b) send tightly scoped retrieved context with `temperature=0`. Which do you choose and what is the trade-off?

   <details><summary>Answer</summary>

   Choose (b): `temperature=0` makes decoding effectively deterministic (greedy), giving the reproducibility an audit requires, and tightly scoped context keeps token cost and latency down. The trade-off is less answer diversity and dependence on retrieval precision — if the wrong chunk is retrieved, the answer is confidently wrong. Option (a) fails the hard requirement: `temperature=0.7` samples different tokens across runs, so identical audits are impossible, and larger context also raises cost and latency without improving determinism.

   </details>

---

## Further Reading

- [Databricks Foundation Model APIs](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/) — *verified 2026-07-11* — Overview of pay-per-token vs provisioned throughput and the OpenAI-compatible query surface.
- [Databricks-hosted foundation models available in Foundation Model APIs](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/supported-models) — *verified 2026-07-11* — Endpoint names, context windows, and embedding models (`gte-large-en`, `bge-large-en`, `qwen3-embedding-0-6b`).
- [Query an embedding model](https://docs.databricks.com/aws/en/machine-learning/model-serving/query-embedding-models) — *verified 2026-07-11* — Code for `embeddings.create`, `DatabricksEmbeddings`, and the `is_normalized` check.
- [Use foundation models](https://docs.databricks.com/aws/en/machine-learning/model-serving/score-foundation-models) — *verified 2026-07-11* — Task types (chat, embeddings, vision), client options, and `usage` token accounting.
- [Databricks AI Search](https://docs.databricks.com/aws/en/generative-ai/vector-search.html) — *verified 2026-07-11* — How embeddings are indexed and retrieved with HNSW + L2 for RAG and semantic search.
