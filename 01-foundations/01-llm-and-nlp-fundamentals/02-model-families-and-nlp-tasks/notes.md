# Model Families & NLP Tasks

**Section:** Foundations | **Module:** LLM & NLP Fundamentals | **Est. time:** 2 hrs | **Exam mapping:** Foundational (supports Design Applications domain)

> ⚠️ Fast-evolving: verify against current official docs before relying on this. Foundation Model APIs, Unity AI Gateway, Model Serving, and the supported-model list (including whether DBRX/specific Llama versions are live) change frequently. This chapter was verified 2026-07-11 — the *concepts* are stable, the *model names* are not.

---

## TL;DR

Transformer models come in three architectural shapes — encoder-only (understanding), decoder-only (generation), and encoder-decoder (transform one sequence into another) — and each shape is good at a different family of NLP tasks. On Databricks you almost never build these yourself: you pick a **foundation model** (optionally fine-tuned or instruction-tuned), decide **proprietary vs open**, and serve it through the right Model Serving option (Foundation Model APIs pay-per-token, provisioned throughput, external models, or AI Functions for batch). The exam tests whether you can go from a task description to the correct model type to the correct serving option. **The one thing to remember: match the task to the architecture family first, then match the workload profile (latency, volume, governance) to the Databricks serving option — never reach for the biggest chat LLM by default.**

---

## ELI5 — Explain It Like I'm 5

Imagine a kitchen with three kinds of workers. The **reader** tastes a finished dish and tells you what's in it — sweet, salty, "this is Italian" — but never cooks anything; that's an encoder-only model that reads text and labels it. The **storyteller** hears the first half of a sentence and keeps talking, making up the rest one word at a time; that's a decoder-only model that generates text. The **translator** listens to a whole order in French, understands it completely, then says it back in English; that's an encoder-decoder model that turns one full sequence into another. The common mistake is thinking the storyteller is always the smartest worker so you should give it every job — but if all you need is to sort tickets into "complaint" or "praise," the reader does it faster and cheaper, and asking the storyteller to do it is like hiring a novelist to alphabetize a filing cabinet.

---

## Learning Objectives

By the end of this chapter you will be able to:
- [ ] Distinguish encoder-only, decoder-only, and encoder-decoder architectures by their mechanism and their natural task fit.
- [ ] Explain the difference between foundation, fine-tuned, and instruction-tuned models, and between proprietary and open models.
- [ ] Map a given NLP task (classification, NER, summarization, translation, Q&A, generation, embedding/retrieval) to the model type that suits it.
- [ ] Select the correct Databricks Model Serving option (Foundation Model APIs pay-per-token, provisioned throughput, external models, AI Functions batch) for a stated workload.
- [ ] Diagnose the cost/latency failure that occurs when a task is mapped to the wrong model type.

---

## Visual Overview

### Architecture Family Comparison

```
ENCODER-ONLY (e.g. BERT, GTE, BGE)
Input tokens ──► [bidirectional attention] ──► rich vector per token / [CLS] vector
Reads the WHOLE sequence at once. No text generation.
Best at: classification, NER, embeddings/retrieval

DECODER-ONLY (e.g. Llama, GPT, Claude, Gemma, Qwen)
prompt ──► [causal (left-to-right) attention] ──► next token ──► next token ──► ...
Predicts one token at a time, each seeing only the past. Pure generation.
Best at: chat, generation, summarization-as-generation, RAG answers, agents

ENCODER-DECODER (e.g. T5, BART)
source seq ──► [ENCODER: understand] ──► context ──► [DECODER: generate] ──► target seq
Understands a full input, then writes a full transformed output.
Best at: translation, abstractive summarization, seq-to-seq rewriting
```

### Task ──► Model Type ──► Databricks Serving Decision Tree

```
What is the task?
├── Label / sort / route text ──► encoder-only classifier OR small decoder LLM
│                                 └──► FMAPI batch (AI Functions) or a fine-tuned classifier endpoint
├── Turn text into a search vector ──► encoder-only EMBEDDING model (GTE, BGE, Qwen3-Embedding)
│                                      └──► FMAPI pay-per-token embedding endpoint + Vector Search
├── Translate / rewrite whole passage ──► encoder-decoder OR instruction-tuned decoder
│                                          └──► FMAPI pay-per-token, or provisioned throughput at scale
└── Generate / chat / answer / agent ──► decoder-only instruction-tuned LLM
                                          ├── prototype / low volume ──► FMAPI pay-per-token
                                          ├── production / high volume / SLA ──► provisioned throughput
                                          └── must use OpenAI/Anthropic/Bedrock ──► external model endpoint
```

### Foundation → Fine-tuned → Instruction-tuned

```
Foundation (base) model
   trained on broad web-scale text, predicts next token, NOT chat-friendly
        │
        ├── + domain data (fine-tune) ──► Fine-tuned model (knows your jargon/task)
        │
        └── + instruction/chat data (RLHF/SFT) ──► Instruction-tuned model
                                                    (follows prompts, safe, "-Instruct" / chat suffix)
```

---

## Key Concepts

### The three transformer architecture families

**What it is.** Every modern NLP model is a transformer, but the transformer can be wired three ways: encoder-only, decoder-only, or encoder-decoder. The wiring determines which direction tokens can "see" and therefore what the model is naturally good at.

**How it works under the hood.** An **encoder-only** model uses *bidirectional* self-attention: every token attends to every other token (left and right), producing a context-rich vector for each token — ideal for understanding but it has no mechanism to emit new text. A **decoder-only** model uses *causal* (masked) attention: each token can only attend to tokens before it, so it can be trained to predict the next token and generate text autoregressively one token at a time. An **encoder-decoder** model runs an encoder to build a full representation of the source, then a decoder that attends both to its own generated tokens *and* (via cross-attention) to the encoder output — this is why it excels at mapping one full sequence to a different full sequence.

**Where it appears in Databricks.** The open models Databricks hosts through Foundation Model APIs are overwhelmingly decoder-only generative LLMs (`databricks-meta-llama-3-3-70b-instruct`, `databricks-gpt-oss-120b`, `databricks-gemma-3-12b`, `databricks-claude-sonnet-5`). The encoder-only family shows up as **embedding** endpoints (`databricks-gte-large-en`, `databricks-qwen3-embedding-0-6b`, BGE), which you pair with Databricks Vector Search. Encoder-decoder models like T5/BART are typically brought in as custom models logged with the MLflow `transformers` flavor and served via provisioned throughput or a custom Model Serving endpoint.

### Foundation vs fine-tuned vs instruction-tuned

**What it is.** A **foundation model** is a large network pre-trained on broad data to learn general patterns; a **fine-tuned** model is a foundation model further trained on task- or domain-specific data; an **instruction-tuned** model is fine-tuned specifically to follow natural-language instructions and behave as a helpful, safe assistant.

**How it works under the hood.** Pre-training optimizes next-token prediction over web-scale text — the result is knowledgeable but not aligned to *follow directions* (a raw base model may just continue your prompt). Fine-tuning continues training on a narrower labeled dataset so weights shift toward your task. Instruction tuning is a specific fine-tune using prompt→response pairs plus preference optimization (SFT + RLHF-style alignment), which is why chat models carry an `-Instruct` or chat suffix.

**Where it appears in Databricks.** Databricks' hosted pay-per-token endpoints are the *instruction-tuned* variants (note the `-instruct` suffix on Llama endpoints). Provisioned throughput explicitly supports **fine-tuned variants of base models** and **fully custom weights using a supported base architecture** — you register the fine-tuned model in Unity Catalog (from Databricks Marketplace, Hugging Face, or your own training run) and deploy it on an optimized endpoint.

### Proprietary vs open models

**What it is.** **Proprietary** models (OpenAI GPT, Anthropic Claude, Google Gemini) are closed-weight and consumed as a service; **open** models (Meta Llama, Google Gemma, OpenAI GPT-OSS, Alibaba Qwen) have downloadable weights you can host, fine-tune, and inspect.

**How it works under the hood.** With a proprietary model you send tokens to the vendor's API and get tokens back — you never hold the weights, so you cannot fine-tune the base weights (only prompt or, where offered, adapter-tune) and your data-residency depends on the provider. With an open model you can load the weights onto your own serving hardware, apply full or parameter-efficient fine-tuning, and keep inference inside your security perimeter.

**Where it appears in Databricks.** Open models are served *inside the Databricks security perimeter* through Foundation Model APIs (both pay-per-token and provisioned throughput). Proprietary models are reached two ways: some (Claude, GPT, Gemini) are Databricks-hosted FMAPI endpoints, and any third-party provider (OpenAI, Anthropic, Cohere, Amazon Bedrock, Google Vertex AI, or a custom OpenAI-compatible URL) can be wired in as an **external model** endpoint with centralized credential management, so all traffic flows through one governed interface.

### NLP tasks and the model type that fits each

**What it is.** The core NLP tasks are classification, named-entity recognition (NER), summarization, translation, question answering (Q&A), open-ended generation, and embedding/retrieval — each has a natural architecture fit.

**How it works under the hood.** Classification and NER are *understanding* tasks that output labels/spans → encoder-only is the classic fit (though a small instruction-tuned decoder LLM can do it zero-shot). Embedding/retrieval needs a fixed-length semantic vector → encoder-only embedding models. Translation and abstractive summarization are *sequence transformation* → encoder-decoder, or an instruction-tuned decoder LLM in practice. Open generation, chat, RAG answering, and agentic reasoning are *autoregressive generation* → decoder-only. Q&A splits: extractive Q&A (find the span) leans encoder-only; generative/RAG Q&A leans decoder-only.

**Where it appears in Databricks.** Task type is an explicit field. External model endpoints declare `task` as `llm/v1/chat`, `llm/v1/completions`, or `llm/v1/embeddings`. The FMAPI is OpenAI-compatible, so chat/generation hits the chat completions route while retrieval hits the embeddings route feeding Vector Search. High-volume label/extract/summarize workloads run through **AI Functions** batch inference (`ai_query` and friends) rather than interactive endpoints.

### Databricks serving options as the deployment layer

**What it is.** Once you have chosen a model type, Databricks Model Serving offers four ways to run it: FMAPI **pay-per-token**, FMAPI **provisioned throughput**, **external models**, and **AI Functions** for batch.

**How it works under the hood.** Pay-per-token routes to a shared pre-configured endpoint billed per token — zero infra to manage, best for prototyping and moderate production. Provisioned throughput reserves optimized serverless capacity for performance guarantees, custom/fine-tuned weights, and compliance (e.g. HIPAA). External models proxy to a third-party API behind a Databricks endpoint with stored credentials. AI Functions run a supported model over a table at scale for batch. All four route through **Unity AI Gateway** (Beta) for rate limits, budgets, usage tracking, and guardrails.

**Where it appears in Databricks.** In the workspace **Serving** tab: pre-configured FMAPI endpoints sit at the top; you create provisioned-throughput and external endpoints there or via the MLflow Deployments SDK / REST API. You query any of them with the OpenAI client, the Python SDK, or REST — pointing `base_url` at `.../serving-endpoints` and `model` at the endpoint name.

### Key Parameters / Configuration Knobs

| Parameter | What it controls | Decision rule |
|---|---|---|
| Serving mode (pay-per-token vs provisioned throughput) | Whether you use a shared per-token endpoint or reserved optimized capacity | Prototype / low-to-moderate volume → pay-per-token; production with SLA, high throughput, fine-tuned weights, or compliance (HIPAA) → provisioned throughput |
| `task` (`llm/v1/chat` \| `llm/v1/completions` \| `llm/v1/embeddings`) | The interaction contract of an external/serving endpoint | Set `chat` for conversational/RAG generation, `completions` for single-turn text-in/text-out, `embeddings` for retrieval vectors — mismatched task returns a 4xx |
| `provider` (external model) | Which third-party API the endpoint proxies | Use `openai`/`anthropic`/`cohere`/`amazon-bedrock`/`google-cloud-vertex-ai` for named vendors; `custom` for any OpenAI-compatible URL |
| Open vs proprietary model choice | Weight access, fine-tunability, data residency | Need to fine-tune base weights, keep data in-perimeter, or optimize cost → open (Llama/Gemma/GPT-OSS/Qwen); need top frontier reasoning fast → proprietary (Claude/GPT/Gemini) |
| Model size / variant (e.g. 8B vs 70B, mini/nano) | Latency, cost, and quality trade-off | Route simple classification/extraction to the smallest capable model; reserve large models for hard reasoning |

### Worked Example: Requirement → Decision

**Given:** A support team has two needs on Databricks. (A) Every incoming ticket must be routed into one of 12 categories — ~500k tickets/day, latency not critical, cost sensitive. (B) An agent-assist panel must summarize a ticket thread into 3 sentences in real time while the rep reads it.

- **Step 1 — Identify the goal.** (A) is high-volume *classification*; (B) is interactive *abstractive summarization/generation*.
- **Step 2 — Define inputs.** (A) raw ticket text rows in a Delta table; (B) a single conversation thread on demand.
- **Step 3 — Define outputs.** (A) one category label per row; (B) a short generated summary with sub-second-ish latency.
- **Step 4 — Apply constraints.** (A) must be cheap and scale to batch; interactive latency irrelevant. (B) needs low latency and good fluency but modest volume.
- **Step 5 — Select the approach.** (A) → an *understanding* task, so use a small model (encoder-only classifier or a small instruction-tuned decoder like Llama 3.1 8B) run through **AI Functions batch inference** over the table — not an interactive chat endpoint. (B) → a *generation* task, so use an instruction-tuned decoder LLM on **FMAPI pay-per-token** (or provisioned throughput if the panel goes high-traffic). Rationale vs alternatives: routing 500k rows/day through a large chat LLM interactive endpoint (the tempting default) is both slower and far more expensive than batch classification; conversely serving the live summary from a nightly batch job cannot meet the real-time requirement.

---

## Implementation

```python
# Scenario: Interactive RAG answer generation. Ticket-summary panel needs a fluent,
# instruction-following response in real time, and we want zero infra to manage while
# prototyping, so we call an open decoder-only chat model via FMAPI pay-per-token.
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.environ["DATABRICKS_TOKEN"],
    base_url="https://<workspace-host>/serving-endpoints",  # OpenAI-compatible route
)

resp = client.chat.completions.create(
    model="databricks-meta-llama-3-3-70b-instruct",  # instruction-tuned decoder-only
    messages=[
        {"role": "system", "content": "Summarize the support thread in 3 sentences."},
        {"role": "user", "content": thread_text},
    ],
    temperature=0.2,
)
print(resp.choices[0].message.content)
```

```python
# Scenario: Semantic retrieval for RAG. We need to turn documents into search vectors.
# Correct model TYPE = encoder-only EMBEDDING model, queried on the embeddings route,
# not a chat model.
embeds = client.embeddings.create(
    model="databricks-gte-large-en",   # encoder-only embedding endpoint
    input=["Reset instructions for the router", "Billing dispute policy"],
)
vectors = [d.embedding for d in embeds.data]  # feed these into Databricks Vector Search
```

```python
# Anti-pattern: routing 500k tickets/day through a large chat LLM interactive endpoint
# to get a category label. This is slow, expensive (billed per generated token), and
# fragile (free-text output must be parsed back into a label).
for ticket in all_500k_tickets:                      # 500k synchronous chat calls!
    r = client.chat.completions.create(
        model="databricks-claude-sonnet-5",          # frontier model for a label
        messages=[{"role": "user", "content": f"Category of: {ticket}"}],
    )
    label = parse_label(r.choices[0].message.content)  # brittle parsing

# Correct approach: batch classification with AI Functions over the Delta table using a
# small model. Understanding task -> small model + batch, not interactive frontier LLM.
# (Databricks SQL / notebook)
# SELECT ticket_id,
#        ai_classify(ticket_text, ARRAY('billing','network','account', ... )) AS category
# FROM support_tickets;
# Runs in-engine at scale, one pass, orders of magnitude cheaper and faster.
```

The anti-pattern breaks on three axes the exam loves: **cost** (per-token billing on a frontier model × 500k rows), **latency/throughput** (synchronous interactive calls can't keep up with batch volume), and **reliability** (parsing a label out of free text is brittle vs a structured classify call). The fix keeps the *task type* (classification) matched to the right *model size* and the right *serving mode* (batch).

---

## Common Pitfalls & Misconceptions

- **"Bigger/newer model is always better."** Beginners equate model size with correctness, so they default to the largest chat LLM for everything. The correct mental model: match architecture family and *smallest capable* model to the task — a small classifier or embedding model beats a frontier LLM on cost and latency for understanding tasks.
- **"Decoder-only can't do classification."** Because BERT is the textbook classifier, beginners assume decoder LLMs can't classify. In reality instruction-tuned decoders classify zero-shot fine; the real question is whether that's the *cheapest, most reliable* option at your volume — often it isn't.
- **"Foundation, fine-tuned, and instruction-tuned are the same thing."** The words blur together, so learners use them interchangeably. Correct model: foundation = broad pre-training; fine-tuned = further trained on your data; instruction-tuned = a specific fine-tune to follow prompts — a raw foundation model won't reliably obey chat instructions.
- **"Open vs proprietary is only about price."** Beginners think open just means free. The real axes are weight access, fine-tunability, and data residency: open models can be hosted in your perimeter and fully fine-tuned; proprietary models trade that control for frontier quality.
- **"Pay-per-token is only for toys; provisioned throughput is only for prod."** People treat the modes as a strict toy/prod split. Correct model: pay-per-token can serve production too; you move to provisioned throughput when you need performance guarantees, high throughput, custom/fine-tuned weights, or compliance — it's a workload-profile decision, not a maturity badge.
- **"An embedding model and a chat model are interchangeable."** Because both are "LLMs," learners call a chat endpoint for retrieval. Correct model: retrieval needs a fixed semantic vector from an encoder-only embedding endpoint (`llm/v1/embeddings`); a chat endpoint returns text, not vectors, and will 4xx or waste tokens.

---

## Key Definitions

| Term | Definition |
|---|---|
| Encoder-only model | Transformer with bidirectional attention that produces contextual vectors for understanding tasks (classification, NER, embeddings); does not generate text. |
| Decoder-only model | Transformer with causal attention that generates text autoregressively one token at a time; the shape behind most chat/generation LLMs. |
| Encoder-decoder model | Transformer that encodes a full source sequence then decodes a full target sequence via cross-attention; suited to translation and abstractive summarization. |
| Foundation model | A large model pre-trained on broad data to learn general patterns, usable as-is or as a base for further training. |
| Fine-tuned model | A foundation model further trained on task- or domain-specific data to specialize its behavior. |
| Instruction-tuned model | A model fine-tuned (SFT + preference alignment) to follow natural-language instructions and act as a helpful, safe assistant. |
| Open model | A model whose weights are downloadable, allowing self-hosting, inspection, and full fine-tuning (e.g. Llama, Gemma, GPT-OSS, Qwen). |
| Proprietary model | A closed-weight model consumed only as a hosted API (e.g. Claude, GPT, Gemini). |
| Foundation Model APIs (FMAPI) | Databricks Model Serving feature exposing hosted open/proprietary models via pay-per-token, provisioned throughput, and AI Functions modes. |
| External model | A Databricks serving endpoint that proxies to a third-party provider (OpenAI, Anthropic, Bedrock, Vertex AI, custom) with centralized credentials. |
| Provisioned throughput | FMAPI mode giving reserved optimized capacity with performance guarantees, support for fine-tuned/custom weights, and compliance certifications. |
| MLflow flavor | A convention in the MLmodel file (e.g. `transformers`, `sentence_transformers`, `python_function`) that tells deployment tools how to load and run a model. |
| Unity AI Gateway | Databricks governance layer routing model/external traffic with rate limits, budgets, usage tracking, and guardrails (Beta). |

---

## Summary / Quick Recall

- Three shapes: encoder-only = understand, decoder-only = generate, encoder-decoder = transform sequence-to-sequence.
- Classification/NER/embeddings → encoder-only; translation/abstractive summary → encoder-decoder; chat/generation/RAG/agents → decoder-only.
- Foundation → fine-tuned (your data) → instruction-tuned (follows prompts); the `-Instruct` suffix means chat-ready.
- Open = downloadable weights, self-host, fine-tune, in-perimeter; proprietary = frontier quality via API only.
- Databricks serving: pay-per-token (prototype/moderate prod), provisioned throughput (SLA/high-volume/custom weights/compliance), external models (third-party APIs), AI Functions (batch over tables).
- Endpoints declare a `task`: `llm/v1/chat`, `llm/v1/completions`, or `llm/v1/embeddings`; the FMAPI is OpenAI-compatible.
- Default sin to avoid: using a frontier chat LLM for high-volume understanding tasks — use the smallest capable model + batch instead.

---

## Self-Check Questions

1. Which transformer architecture family uses bidirectional attention and is the natural fit for producing embeddings and classifying text, without generating new text?

   <details><summary>Answer</summary>

   **Encoder-only** (e.g. BERT, GTE, BGE). Its bidirectional self-attention lets every token see the full context, producing rich vectors for understanding tasks. The tempting wrong answer is decoder-only — but decoder-only uses *causal* (left-to-right) attention built for generating text one token at a time, and encoder-decoder adds a generation stage that classification/embedding tasks don't need.

   </details>

2. You must turn 2 million product descriptions into vectors so users can do semantic search over them on Databricks. Which model type and serving route do you choose, and why?

   <details><summary>Answer</summary>

   Choose an **encoder-only embedding model** (e.g. `databricks-gte-large-en`) via the **`llm/v1/embeddings`** route, then load the vectors into Databricks Vector Search. Embeddings are fixed-length semantic vectors — exactly what retrieval needs. A common wrong choice is a chat/completions LLM, which returns text not vectors and will either 4xx on the embeddings contract or waste tokens generating prose you can't index.

   </details>

3. **Which TWO** situations justify moving a decoder-only chat model from Foundation Model APIs pay-per-token to **provisioned throughput**?
   - A. You are building a quick proof-of-concept and want the least setup.
   - B. You need a performance guarantee (SLA) for high, spiky production traffic.
   - C. You want to deploy a fine-tuned/custom-weight variant of a supported base architecture.
   - D. You want to call OpenAI's hosted GPT model directly.
   - E. You want to lower cost for a single occasional developer query.

   <details><summary>Answer</summary>

   **B and C.** Provisioned throughput exists for performance guarantees / high throughput (B) and supports fine-tuned and custom-weight variants of a base architecture (C). A and E describe exactly what pay-per-token is *for* (minimal setup, low/occasional volume), so they're wrong. D is a distractor: calling OpenAI's hosted model is an **external model** endpoint, a different mechanism entirely — not a provisioned-throughput decision.

   </details>

4. A team routes 500k tickets/day through a frontier chat LLM interactive endpoint just to assign each a category label, then regex-parses the label out of the reply. Which is the strongest architectural objection, and what's the fix?

   <details><summary>Answer</summary>

   The strongest objection is a **task/serving mismatch**: classification is an *understanding* task at *batch* volume, but they used a *generation* model on an *interactive* endpoint — driving up per-token cost, capping throughput, and forcing brittle text parsing. The fix is a small model (encoder-only classifier or small instruction-tuned decoder) run through **AI Functions batch inference** (e.g. `ai_classify`) over the Delta table, returning structured labels in one scalable pass. Choosing a slightly smaller chat model is a weak fix — it treats the symptom (model size) not the cause (wrong serving mode and output contract).

   </details>

5. Your workload needs frontier-level reasoning quality, must call a specific proprietary vendor model, and all traffic must be centrally governed with rate limits and stored credentials. Compare using a Databricks-hosted FMAPI proprietary endpoint vs an external model endpoint, and pick one.

   <details><summary>Answer</summary>

   Use an **external model endpoint** when the requirement is to call a *specific third-party vendor's* model (e.g. your contracted OpenAI/Anthropic/Bedrock deployment) — it proxies to that provider, stores credentials centrally, and (with Unity AI Gateway) applies rate limits, budgets, and guardrails. A Databricks-hosted FMAPI proprietary endpoint is the better pick when you just need frontier quality and are happy with a Databricks-hosted, in-perimeter version — less credential management, but you don't control the upstream vendor account. The deciding factor is *whose* model/account you must use; both can be governed by AI Gateway, so governance alone doesn't decide it. Picking pay-per-token open models here would be wrong because it fails the "specific proprietary vendor" requirement.

   </details>

---

## Further Reading

- [Databricks Foundation Model APIs](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/) — *verified 2026-07-11* — overview of the three FMAPI modes (pay-per-token, provisioned throughput, AI Functions) and when to use each.
- [Databricks-hosted foundation models available in Foundation Model APIs](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/supported-models) — *verified 2026-07-11* — current list of hosted open/proprietary and embedding model endpoints (Llama, GPT-OSS, Gemma, Qwen, Claude, GTE).
- [Supported foundation models on Model Serving](https://docs.databricks.com/aws/en/machine-learning/model-serving/foundation-model-overview) — *verified 2026-07-11* — the four hosting options and per-region model/feature availability.
- [External models in Model Serving](https://docs.databricks.com/aws/en/generative-ai/external-models/) — *verified 2026-07-11* — providers, the `task` types (`llm/v1/chat|completions|embeddings`), and endpoint configuration for third-party models.
- [AI governance with Unity AI Gateway](https://docs.databricks.com/aws/en/ai-gateway/) — *verified 2026-07-11* — routing, rate limits, budgets, and guardrails across model and external endpoints.
- [MLflow Models](https://mlflow.org/docs/latest/ml/model/) — *verified 2026-07-11* — model flavors (`transformers`, `sentence_transformers`, `python_function`) and packaging for serving.
