# LLM and GenAI Core Concepts

**Section:** Foundations | **Module:** GenAI and Prompting Fundamentals | **Est. time:** 2 hrs
**Exam mapping:** Databricks GenAI Engineer Associate — Foundations domain (~15%); tokenization, context windows, and sampling parameters also appear as assumed knowledge across App Development and Governance domains

## Learning Objectives

By the end of this chapter, you will be able to:

- Explain at a mechanistic level how transformer-based LLMs process input and generate output
- Define tokens, describe the tokenization process, and estimate token counts for a given input
- Explain what embeddings are and why they enable semantic similarity search
- Describe the context window and its practical implications for application design
- Explain temperature, top-p, and top-k sampling and when to adjust each
- Use the Databricks Foundation Model API to call a hosted LLM programmatically

## Core Concepts

### What Is a Large Language Model?

A Large Language Model (LLM) is a neural network trained to predict the probability distribution
over the next token given a sequence of preceding tokens. That is the entire task it was trained
on. Everything else — answering questions, writing code, summarizing documents — is an emergent
capability that arises from doing that one task extremely well at enormous scale.

This framing matters because it clarifies two important truths:
1. **LLMs do not "understand" in the human sense** — they learn statistical patterns over text
2. **LLMs can be confidently wrong** — if the next-token prediction leads somewhere plausible
   but factually incorrect, the model has no mechanism to detect that (without additional tools)

### Tokens — The Unit of Processing

LLMs do not process characters or words — they process **tokens**. A token is a subword unit
produced by a tokenizer. Common rules of thumb:

- ~4 characters ≈ 1 token in English
- A 1,000-word document ≈ 750–1,000 tokens
- A word like "unbelievable" might tokenize as `["un", "believ", "able"]` — 3 tokens
- A code snippet with indentation often tokenizes less efficiently than prose

Why does this matter for you as an engineer?

- **Cost** — most LLM APIs charge per token (input + output)
- **Context window limits** — maximum input is measured in tokens, not characters or words
- **Chunking decisions** — when splitting documents for RAG, you must think in tokens, not words

**To count tokens locally (for OpenAI-compatible models):**
```python
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4o")
tokens = enc.encode("Hello, how are you?")
print(f"{len(tokens)} tokens")  # 5 tokens
```

For Databricks-hosted models (DBRX, Llama, Mistral), use the model provider's tokenizer or
estimate via the 4-char-per-token rule of thumb.

### The Transformer Architecture (Conceptual)

You do not need to implement a transformer for this exam, but you need to understand its key
properties:

**Attention mechanism:** Every token in the input can "attend to" (be influenced by) every other
token. This allows the model to resolve references ("it" → what does "it" refer to?) and capture
long-range dependencies across a document.

**Layers:** A transformer stacks many attention layers. Deeper layers capture increasingly abstract
patterns — early layers detect syntax, later layers capture semantics and world knowledge.

**Pre-training and fine-tuning:**
- **Pre-training:** The model is trained on a massive corpus to predict next tokens; this is where
  general knowledge is baked in
- **Instruction fine-tuning (IFT/SFT):** The model is further trained on instruction-response
  pairs to follow user directions well
- **RLHF (Reinforcement Learning from Human Feedback):** Human preference data is used to align
  the model toward helpful, accurate, and safe outputs

Most models you use via API (GPT-4o, Claude, DBRX) have all three stages applied.

### Embeddings

An **embedding** is a dense vector of floating-point numbers that represents a piece of text in a
high-dimensional space. Texts with similar meanings produce vectors that are close together by a
distance metric (typically cosine similarity).

Two types you will encounter:

| Type | What it represents | Use case |
|------|--------------------|---------|
| **Token embedding** | A single token in context | Internal to the model; not directly user-facing |
| **Sentence/document embedding** | An entire sentence or chunk of text | Semantic search, RAG retrieval, clustering |

The key property for GenAI engineering: embeddings let you find semantically similar text without
exact keyword matching. "What are the operating hours?" and "When are you open?" will have similar
embeddings even though they share no words. This is the foundation of vector search in RAG.

**Embedding models are separate from generation models.** You use:
- A generation model (GPT-4o, DBRX, Llama) to produce text
- An embedding model (`text-embedding-3-small`, `bge-m3`, etc.) to convert text to vectors

On Databricks, embedding models are available through Foundation Model APIs or Unity Catalog
registered models.

### Context Windows

The **context window** is the maximum number of tokens the model can process in a single forward
pass — it includes both your input (prompt + retrieved documents) and the model's output.

Common sizes (as of 2026):
- GPT-4o: 128,000 tokens
- Claude 3.x Sonnet: 200,000 tokens
- DBRX Instruct: 32,768 tokens
- Llama 3.1 70B: 128,000 tokens

**Practical implications for engineers:**
- If your prompt + context exceeds the context window, the call fails (or the oldest content is
  silently truncated, depending on the framework)
- Longer contexts cost more (most APIs charge per input token)
- Model quality may degrade on very long contexts — "lost in the middle" is a documented
  phenomenon where models underperform on information in the middle of a very long context
- For RAG pipelines: retrieved chunks must fit within the remaining context window after your
  system prompt and query are included

### Sampling Parameters — Temperature, Top-p, Top-k

After the model computes a probability distribution over the next token, it samples from that
distribution. The sampling parameters control how.

**Temperature:**
- Controls the "sharpness" of the probability distribution
- `temperature=0`: greedy decoding — always pick the highest-probability token; fully deterministic
- `temperature=1`: sample proportionally to the model's raw probabilities; natural variability
- `temperature>1`: flatten the distribution; more random, more "creative" (also more likely to
  be wrong or incoherent)
- `temperature<1` (but >0): sharpen the distribution; less variation than raw sampling

**Rule of thumb for practitioners:**
- Use `temperature=0` when you need deterministic, reproducible outputs (data extraction, classification, structured output)
- Use `temperature=0.2–0.7` for general chat and question-answering
- Use `temperature>0.7` for creative writing or ideation tasks

**Top-p (nucleus sampling):**
- Rather than sampling from the full vocabulary, limit sampling to the smallest set of tokens
  whose cumulative probability ≥ p
- `top_p=0.95`: sample only from tokens that together account for 95% of the probability mass
- Reduces the chance of very unlikely tokens without being fully greedy
- Usually used **instead of** temperature adjustment, not in addition to it

**Top-k:**
- Limit sampling to the top-k most probable tokens
- Less common in modern API usage; most models default to a sensible value internally

### Databricks Foundation Model API

Databricks provides **Foundation Model APIs** — OpenAI-compatible endpoints for curated hosted
models (DBRX Instruct, Llama 3.x, Mistral, etc.). You pay per token; no infrastructure to manage.

The API follows the OpenAI chat completions format, so any code that calls OpenAI can call
Databricks Foundation Model APIs with a URL and token swap.

```python
import os
from openai import OpenAI

# Databricks Foundation Model API — OpenAI-compatible
client = OpenAI(
    api_key=dbutils.secrets.get(scope="genai-keys", key="databricks-token"),
    base_url="https://<your-workspace-host>/serving-endpoints"
)

response = client.chat.completions.create(
    model="databricks-dbrx-instruct",  # or "databricks-meta-llama-3-1-70b-instruct"
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is a context window?"}
    ],
    temperature=0,
    max_tokens=256
)

print(response.choices[0].message.content)
```

Available models on Databricks Foundation Model APIs (check official docs for current list):
- `databricks-dbrx-instruct` — Databricks' own model, strong at reasoning and code
- `databricks-meta-llama-3-1-70b-instruct` — Llama 3.1 70B, strong general-purpose
- `databricks-meta-llama-3-1-8b-instruct` — Llama 3.1 8B, fast and cheap
- `databricks-mixtral-8x7b-instruct` — Mixtral, efficient mixture-of-experts model
- `databricks-bge-large-en` — BGE large embedding model for vector search

## Deep Dive / Advanced Topics

### Hallucinations — Why They Happen and What They Are Not

"Hallucination" is the term for when a model produces confident, fluent output that is factually
incorrect. Understanding the mechanism prevents a common misconception:

**Hallucinations are not random bugs** — they are the model doing exactly what it was trained to
do (predict plausible next tokens) in a case where plausibility and factual accuracy diverge.

Causes include:
- **Training data gaps:** The model has no (or sparse) accurate training data for this specific fact
- **Distributional shift:** The question is phrased in a way that pattern-matches to a different, wrong answer in training data
- **Long-range coherence failure:** In long outputs, the model may "drift" away from facts stated earlier

**Mitigation strategies (covered in depth in later sections):**
- RAG (Retrieval-Augmented Generation) — ground answers in retrieved factual documents
- Guardrails — validate model output against known-correct sources
- Asking the model to cite its sources and flag uncertainty

### The Difference Between a Base Model and a Chat/Instruct Model

| Type | Training | Behavior | Use case |
|------|----------|----------|---------|
| **Base model** | Next-token prediction on raw web text only | Completes text; does not "answer" questions | Fine-tuning, research |
| **Instruct/chat model** | Base + instruction fine-tuning + RLHF | Follows instructions; has system/user/assistant roles | Production applications |

When you call an API endpoint like `databricks-dbrx-instruct`, the `instruct` suffix tells you
this is the chat-tuned version. You should almost always use instruct/chat models for application
development.

### Tokens, Latency, and Cost — The Engineering Triangle

Every LLM call involves a trade-off:

```
                 Quality
                    |
         Cost ——————+—————— Latency
```

- Larger models → better quality, higher cost, more latency
- Shorter prompts → lower cost, lower latency, potentially lower quality if context is important
- `max_tokens` caps output length → limits cost and latency, but may truncate the response

For this exam, the key insight: **there is no universally best model or setting**. Good
engineering means choosing the right point on this triangle for each use case.

## Worked Examples & Practice

### Observing Temperature's Effect on Output

Run this in a Databricks notebook to see temperature's effect directly:

```python
from openai import OpenAI

client = OpenAI(
    api_key=dbutils.secrets.get(scope="genai-keys", key="databricks-token"),
    base_url="https://<your-workspace-host>/serving-endpoints"
)

prompt = "Complete this sentence in one word: The sky is"

for temp in [0.0, 0.0, 0.7, 1.5]:
    response = client.chat.completions.create(
        model="databricks-meta-llama-3-1-8b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=temp,
        max_tokens=5
    )
    output = response.choices[0].message.content.strip()
    print(f"temperature={temp:.1f} → '{output}'")
```

**Expected pattern:**
- `temperature=0.0` (run twice) → same answer both times ("blue")
- `temperature=0.7` → typically "blue" but may vary slightly
- `temperature=1.5` → unpredictable, may produce nonsense

**Failure mode to observe:** At very high temperature (>1.5), the model may produce grammatically
or semantically broken output. This demonstrates that high temperature is not "more creative"
in a useful sense — it is more random.

### Token Counting and Context Window Budgeting

```python
import tiktoken

def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    """Estimate token count for a string. Uses GPT-4 tokenizer as a proxy."""
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

system_prompt = "You are a helpful assistant that answers questions about Databricks."
user_message = "Explain the difference between a base model and an instruct model."

input_tokens = estimate_tokens(system_prompt) + estimate_tokens(user_message)
context_window = 32768  # DBRX Instruct
max_output_tokens = 512

remaining_for_context = context_window - input_tokens - max_output_tokens
print(f"Input tokens: {input_tokens}")
print(f"Available for retrieved context: {remaining_for_context} tokens")
print(f"Approx. text capacity: ~{remaining_for_context * 4} characters")
```

This is the core budget calculation for RAG pipelines — how much retrieved text can you fit
given your fixed system prompt and expected output size?

## Common Pitfalls & Misconceptions

- **Pitfall:** Treating `temperature=0` as "the model always gives the same answer" → **Why it happens:** Temperature 0 uses greedy decoding, which is deterministic; however, some models and APIs add a small random seed by default → **Fix:** For truly reproducible outputs, set both `temperature=0` and `seed=<fixed_integer>` if the API supports it; note that Databricks Foundation Model APIs may not expose `seed` for all models

- **Pitfall:** Confusing the embedding model with the generation model → **Why it happens:** Both process text; learners assume one model handles everything → **Fix:** Always use two separate models: a small, fast embedding model (e.g., `databricks-bge-large-en`) to vectorize text, and a large generation model (e.g., `databricks-dbrx-instruct`) to produce answers

- **Pitfall:** Ignoring token count when designing prompts and hitting context window limits in production → **Why it happens:** During development, prompts are short; in production, retrieved documents make the context explode → **Fix:** Budget tokens explicitly at design time: `context_window - system_prompt_tokens - query_tokens - max_output_tokens = available_for_retrieval`

- **Pitfall:** Thinking that a model with more parameters is always better → **Why it happens:** Larger is generally better at capability benchmarks, but "better" depends on task → **Fix:** Evaluate latency, cost, and task-specific performance; an 8B model is often sufficient for classification and extraction tasks at a fraction of the cost of 70B+

- **Pitfall:** Using `top_p` and `temperature` together at non-default values → **Why it happens:** Both control randomness; learners stack them → **Fix:** In practice, adjust one or the other, not both. If you need determinism, use `temperature=0`. If you want controlled randomness, use `temperature` OR `top_p`, not both

## Key Definitions

| Term | Definition |
|---|---|
| Token | The atomic unit of text processed by an LLM — typically a subword fragment; approximately 4 characters in English |
| Tokenizer | The algorithm that converts raw text into a sequence of integer token IDs for model input |
| Embedding | A dense floating-point vector representation of a piece of text in a high-dimensional semantic space; similar texts have vectors with high cosine similarity |
| Context window | The maximum number of tokens (input + output combined) an LLM can process in a single call |
| Temperature | A sampling parameter controlling output randomness; 0 = greedy/deterministic, 1 = proportional sampling, >1 = increasingly random |
| Top-p (nucleus sampling) | A sampling strategy that restricts token selection to the smallest set of most-probable tokens whose cumulative probability ≥ p |
| Hallucination | Confident, fluent model output that is factually incorrect; a natural consequence of next-token prediction in cases where plausibility and accuracy diverge |
| Base model | An LLM trained only on next-token prediction on raw text; does not follow instructions; used as a starting point for fine-tuning |
| Instruct/chat model | A base model further trained with instruction fine-tuning and RLHF to follow user directions and engage in multi-turn conversation |
| Foundation Model API | Databricks-hosted, pay-per-token endpoints for curated LLMs; OpenAI-compatible interface |

## Summary / Quick Recall

- LLMs predict the next token given preceding tokens — all capabilities emerge from this
- 1 token ≈ 4 English characters; estimate token cost and context window usage in tokens, not words
- Embeddings are dense vectors; similar texts → similar vectors → cosine similarity enables semantic search
- Context window = total input + output token budget; exceeding it causes failures or silent truncation
- `temperature=0` → deterministic; higher temperature → more random/creative; use 0 for extraction/classification
- Top-p restricts sampling to the minimum set of tokens covering p% of probability mass; use instead of (not in addition to) temperature adjustment
- Databricks Foundation Model APIs are OpenAI-compatible; swap `base_url` and `api_key` to use them with the `openai` Python library
- Embedding models and generation models are separate; use both in a RAG pipeline

## Self-Check Questions

1. A colleague sets `temperature=2.0` to "make the model more creative." What will actually happen, and what is a better approach for creative generation?

<details>
<summary>Answer</summary>
At temperature 2.0, the probability distribution is flattened to the point where low-probability tokens become nearly as likely as high-probability ones. The output will be incoherent or nonsensical rather than genuinely creative. A better approach: use `temperature=0.7–0.9` for creative tasks, which increases variability while keeping outputs coherent. True creativity in LLM applications typically comes from good prompting (asking for multiple options, specifying a creative style), not extreme temperature settings.
</details>

2. Your RAG pipeline uses DBRX Instruct (32,768-token context window). Your system prompt uses 500 tokens, the user query uses 50 tokens, and you expect the model to generate up to 1,000 tokens of response. How many tokens can you allocate to retrieved document chunks?

<details>
<summary>Answer</summary>
32,768 − 500 (system) − 50 (query) − 1,000 (max output) = 31,218 tokens available for retrieved context. At ~4 characters per token, that is roughly 125,000 characters of retrieved text — generous, but worth budgeting explicitly so you do not exceed the limit as prompt complexity grows.
</details>

3. What is the difference between a token embedding and a sentence/document embedding, and when do you use each?

<details>
<summary>Answer</summary>
A token embedding represents a single token within the context of the model's internal computation — it is not directly user-facing and changes based on surrounding tokens (contextual). A sentence or document embedding represents an entire piece of text as a single fixed vector, produced by an embedding model. You use sentence/document embeddings for external tasks: semantic search, RAG vector retrieval, clustering, and similarity comparisons. Token embeddings are internal to the LLM and not directly accessible via standard APIs.
</details>

4. Why can an LLM produce confident, fluent, and completely false output (a hallucination)?

<details>
<summary>Answer</summary>
LLMs are trained to predict the most plausible next token, not the most factually accurate one. When the model lacks accurate training data for a fact, or when the question's phrasing pattern-matches to a wrong answer in training data, the model generates text that is statistically plausible (fluent and confident-sounding) but factually incorrect. The model has no internal fact-checking mechanism — it cannot distinguish between what it "knows" and what it is guessing.
</details>

5. You want to call a Databricks Foundation Model API endpoint from Python code that currently calls OpenAI's GPT-4o. What two things do you need to change?

<details>
<summary>Answer</summary>
Two changes: (1) `base_url` — change from `https://api.openai.com/v1` to your Databricks workspace URL (`https://<workspace-host>/serving-endpoints`); (2) `api_key` — change from your OpenAI API key to a Databricks personal access token. The model name in the `model` parameter also changes (e.g., from `"gpt-4o"` to `"databricks-dbrx-instruct"`), but the SDK, method calls, and response format remain identical because both use the same OpenAI-compatible chat completions interface.
</details>

## Further Reading

- [Databricks — Foundation Model APIs](https://docs.databricks.com/en/machine-learning/foundation-models/index.html) — official documentation for hosted endpoints, available models, and pricing
- [OpenAI — Tokenizer](https://platform.openai.com/tokenizer) — interactive tool for visualizing how text is tokenized (uses tiktoken; good proxy for most GPT-family and many open models)
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) — the original transformer paper; the abstract and introduction are worth reading even without a deep math background
- [Andrej Karpathy — Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) — a 2-hour video walkthrough building a GPT from scratch in Python; the best intuition-builder for transformer internals available
