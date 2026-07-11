# What Is RAG and When to Use It

**Section:** Foundations | **Module:** RAG & Retrieval Foundations | **Est. time:** 2 hrs | **Exam mapping:** Supports Application Development (30%) & Design Applications (14%)

---

## TL;DR

Retrieval-Augmented Generation (RAG) bolts a search step onto a language model: before the model answers, you fetch relevant documents from an external knowledge base and paste them into the prompt. This lets an LLM answer questions about proprietary, frequently-changing, or domain-specific data that it was never trained on — and cite its sources so a human can verify the answer. On Databricks the pieces are Databricks AI Search (formerly Vector Search) for retrieval, Foundation Model APIs for generation, the Agent Framework to wire them together, Unity Catalog to govern the source data, and MLflow to evaluate quality. **The one thing to remember: RAG changes what the model *reads at answer time*, not what the model *knows from training* — it is retrieval plumbing, not model training.**

---

## ELI5 — Explain It Like I'm 5

Imagine a very smart student who has read millions of books but finished reading two years ago and is not allowed to look anything up during a test. That is a plain LLM: it answers from memory, so it invents plausible-sounding facts when the answer is not in its head, and it never knows what happened after it stopped studying. RAG turns that closed-book exam into an open-book exam: right before each question, a librarian rushes over, finds the exact three pages that matter, and slides them onto the student's desk to read before answering. The student is exactly as smart as before — we did not send them back to school — we just handed them the right pages at the right moment, and now they can point to the page they used. The common mistake is thinking RAG "teaches" or "retrains" the model; it does not touch the model's weights at all, it only controls which pages land on the desk.

---

## Learning Objectives

By the end of this chapter you will be able to:
- [ ] Explain the four problems RAG solves (knowledge cutoff, hallucination, proprietary data, freshness) and why they are retrieval problems, not training problems.
- [ ] Describe the RAG pipeline end to end: ingest → chunk → embed → index → retrieve → augment → generate.
- [ ] Compare RAG, fine-tuning, long-context, and prompt engineering, and select the right one under given constraints.
- [ ] Map each RAG stage to its Databricks component (AI Search, Foundation Model APIs, Agent Framework, Unity Catalog, MLflow).
- [ ] Diagnose why retrieval quality — not the LLM — is usually the bottleneck in a failing RAG system.

---

## Visual Overview

### RAG Pipeline Flow

```
INGEST-TIME (offline, batch)                     QUERY-TIME (online, per request)
────────────────────────────                     ────────────────────────────────

Source docs        User question
(PDF, wiki,             │
 Delta table)           ▼
    │             ┌─────────────┐
    ▼             │  Embed the  │
┌────────┐        │   query     │
│ Chunk  │        └─────┬───────┘
└───┬────┘              ▼
    ▼             ┌─────────────┐
┌────────┐        │  RETRIEVE   │  top_k most similar chunks
│ Embed  │        │ (AI Search) │◄──────────────┐
└───┬────┘        └─────┬───────┘                │
    ▼                   ▼                        │
┌────────┐        ┌─────────────┐                │
│ INDEX  │───────►│  AUGMENT    │  stuff chunks  │
│(vector │        │  the prompt │  into template │
│ store) │        └─────┬───────┘                │
└────────┘              ▼                        │
                  ┌─────────────┐                │
                  │  GENERATE   │  LLM answers   │
                  │ (FM API)    │  + citations ──┘
                  └─────┬───────┘
                        ▼
                  Grounded answer to user
```

### Decision Tree — RAG vs Fine-tune vs Long-context vs Prompt

```
Does the answer depend on facts the base model does not reliably know?
├── No ──► Prompt engineering (system prompt / few-shot examples). Cheapest. Start here.
└── Yes ─► Do those facts change often, or must answers cite sources?
          ├── Yes ──► RAG (retrieve fresh facts at query time; update the index, not the model)
          └── No  ─► Is it a small, static set of facts that fits in the context window?
                     ├── Yes ──► Long-context (paste the whole corpus into the prompt)
                     └── No  ─► Is the gap FORMAT/STYLE/BEHAVIOUR rather than facts?
                                ├── Yes ──► Fine-tuning (teach tone, format, task pattern)
                                └── No  ─► RAG (large, changing knowledge base)
```

### Where the Databricks Pieces Sit

```
                        ┌──────────────── Unity Catalog (governs ALL of it) ────────────────┐
                        │                                                                    │
 Delta tables / files ──┼──► AI Search index ──► retrieve ──► Agent Framework ──► FM API ──┼──► user
 (source data)          │      (Vector Search)                (LangGraph node)   (LLM)      │
                        │                                          │                         │
                        └──────────── MLflow Tracing + Agent Evaluation (quality) ───────────┘
```

---

## Key Concepts

### The four problems RAG solves

**What it is.** RAG exists to close four specific gaps between what an LLM knows and what a business question needs: (1) **knowledge cutoff** — the model's training froze at a date; (2) **hallucination** — the model fills gaps with confident fabrication; (3) **proprietary data** — internal memos, contracts, and tickets were never in the training set; (4) **freshness** — prices, policies, and inventory change daily.

**How it works under the hood.** Every one of these is a *retrieval* problem, not a *knowledge* problem. Instead of encoding facts into the model's frozen weights, RAG keeps the facts in an external store and injects the relevant ones into the prompt at answer time. Because the model is now reading the fact rather than recalling it, it can ground its answer in the retrieved text and cite the source document, which is what lets a human verify accuracy.

**Where in Databricks.** The Databricks RAG guide frames these as RAG's benefits verbatim: "proprietary knowledge," "up-to-date information," "citing sources," and "data security and ACL." The source data lives in Delta Lake and is governed by Unity Catalog; the retrieval step can be designed to selectively return only rows a user's credentials permit.

### The RAG pipeline: ingest → chunk → embed → index → retrieve → augment → generate

**What it is.** A two-phase pipeline. The **offline (ingest-time)** phase prepares the knowledge base; the **online (query-time)** phase answers each request. Databricks splits this into three stages: a **data pipeline**, a **RAG chain** (retrieval, augmentation, generation), and **evaluation/monitoring**, all under **governance/LLMOps**.

**How it works under the hood.** Offline: raw documents are split into *chunks* small enough to embed and retrieve precisely; each chunk is passed through an embedding model to produce a vector; the vectors plus metadata are written to a vector index. Online: the user's question is embedded with the *same* embedding model, the index returns the `top_k` nearest chunks by vector similarity, those chunks are formatted into a prompt template ("augmentation"), and the LLM generates the final answer from that augmented prompt.

**Where in Databricks.** Ingest runs on Delta Lake + Lakeflow pipelines. Indexing and retrieval are Databricks AI Search, which builds an index from a Delta table and can auto-sync when the table changes. Generation calls a Foundation Model API endpoint. The chain is authored with the Agent Framework.

### Grounding and citations

**What it is.** *Grounding* means the answer is derived from the retrieved passages rather than from the model's parametric memory; *citations* are the document references the model attaches so a human can check the source.

**How it works under the hood.** Because the retrieved chunks carry metadata (a `doc_uri`, title, or URL), the augmentation step can pass that metadata alongside the text and instruct the model to attribute each claim to its chunk. Evaluation then measures *groundedness* — whether every claim in the answer is actually supported by a retrieved chunk — which is only possible because you know exactly which passages were in the prompt.

**Where in Databricks.** Databricks AI Search returns the associated documents and their metadata for each hit. MLflow Agent Evaluation includes a `groundedness` LLM judge that reads the `retrieved_context` and flags claims not supported by it.

### Cost, latency, and freshness trade-offs

**What it is.** RAG buys freshness and verifiability but adds a retrieval hop and extra prompt tokens, which cost money and time. Every design choice trades among these three.

**How it works under the hood.** Retrieving more chunks (higher `top_k`) raises answer quality up to a point, then adds latency and token cost while diluting the prompt with marginally-relevant text. Freshness comes from how often the index syncs with the source table — continuous sync is fresher but more expensive than scheduled sync. Fine-tuning, by contrast, front-loads a large one-time training cost and produces stale answers the moment the underlying facts change.

**Where in Databricks.** AI Search index sync mode (continuous vs triggered) sets the freshness/cost balance. MLflow Agent Evaluation reports quality, cost, and latency together so you tune `top_k` and sync mode against measured numbers, not guesses.

### Retrieval quality as the bottleneck

**What it is.** In most failing RAG systems the LLM is fine; the retriever handed it the wrong or incomplete passages. "Garbage retrieved in, garbage generated out."

**How it works under the hood.** If the relevant chunk is never returned in the `top_k`, no amount of prompt engineering or a bigger model can recover the missing fact — the model simply cannot cite what it never saw. This is why Databricks stresses evaluating individual components (retrieval separately from generation), because a data-formatting or chunking change silently shifts which chunks get retrieved.

**Where in Databricks.** AI Search offers hybrid keyword-similarity search (combining vector similarity with BM25 keyword matching via Reciprocal Rank Fusion) and reranking specifically to lift retrieval quality when pure vector search misses exact identifiers like SKUs. MLflow judges retrieval with `chunk_relevance` / context-recall style metrics.

### Key Parameters / Configuration Knobs

| Parameter | What it controls | Decision rule |
|---|---|---|
| `top_k` (number of chunks retrieved) | How many passages get stuffed into the prompt | Start at 3–5; raise only if evaluation shows relevant facts are being missed (low recall), lower it if answers get diluted or latency/token cost climbs. |
| Retrieval mode (vector vs hybrid) | Whether retrieval uses pure embedding similarity or also BM25 keyword matching | Use hybrid when the corpus contains exact identifiers (SKUs, error codes, part numbers) that pure semantic search misses; pure vector is enough for prose Q&A. |
| Index sync mode (continuous vs triggered) | How fresh the index is relative to the source Delta table | Use continuous sync when stale answers are unacceptable (pricing, inventory); use triggered/scheduled sync for docs that change daily or less to save cost. |
| Context budget (prompt token limit) | Total tokens available for retrieved chunks + instructions | Reserve headroom for the answer; if `top_k × chunk_size` exceeds the budget, reduce `top_k` or chunk size rather than truncating mid-chunk. |
| Reranking (on/off) | Whether a second model reorders the retrieved candidates by relevance | Turn on when `top_k` must stay small but first-pass recall is noisy; skip it to shave latency when first-pass precision is already high. |

### Worked Example: Requirement → Decision

**Given:** A SaaS company wants a customer-support assistant that answers from a product knowledge base of ~4,000 help articles. The docs are edited several times a week, agents must be able to click through to the source article, and the team asks whether they should fine-tune a model on the articles instead.

- **Step 1 — Identify the goal:** Answer support questions accurately from the current help content, with clickable source citations.
- **Step 2 — Define inputs:** 4,000 markdown help articles in a Delta table, edited multiple times weekly; each article has a stable URL.
- **Step 3 — Define outputs:** A grounded natural-language answer plus links to the specific articles it used.
- **Step 4 — Apply constraints:** Content changes weekly (freshness matters); citations are mandatory (verifiability); 4,000 articles far exceed a single context window; no tolerance for confidently-wrong answers.
- **Step 5 — Select the approach:** **RAG**, not fine-tuning. Fine-tuning would bake the articles into weights, so every weekly edit would require retraining and still could not produce a source link; long-context cannot hold 4,000 articles at once; prompt engineering alone has no facts to draw on. RAG indexes the Delta table in AI Search (triggered sync after each edit batch), retrieves the top articles at query time, and cites their URLs — updating the *index* on each edit instead of retraining the *model*.

---

## Implementation

```python
# Scenario: Answer an employee's HR-policy question from the current policy
# docs, with citations, without retraining any model — a minimal retrieve-then-
# generate loop on Databricks (Vector Search client + Foundation Model API).
from databricks.vector_search.client import VectorSearchClient
from openai import OpenAI
import os

vsc = VectorSearchClient()
index = vsc.get_index(
    endpoint_name="hr_vs_endpoint",
    index_name="main.hr.policy_chunks_index",  # governed by Unity Catalog
)

question = "How many days of parental leave do I get?"

# RETRIEVE: top_k most relevant chunks + their source metadata for citations
hits = index.similarity_search(
    query_text=question,
    columns=["chunk_text", "doc_uri"],
    num_results=3,  # top_k — start small, tune against evaluation
)
rows = hits["result"]["data_array"]
context = "\n\n".join(f"[{uri}] {text}" for text, uri in rows)

# AUGMENT + GENERATE: call a Foundation Model API endpoint (OpenAI-compatible)
client = OpenAI(
    api_key=os.environ["DATABRICKS_TOKEN"],
    base_url=f"{os.environ['DATABRICKS_HOST']}/serving-endpoints",
)
resp = client.chat.completions.create(
    model="databricks-meta-llama-3-3-70b-instruct",
    messages=[
        {"role": "system", "content": "Answer ONLY from the context. Cite the [source] for each fact. If the context does not contain the answer, say so."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ],
)
print(resp.choices[0].message.content)
```

```python
# Anti-pattern: "just make the model know it." Two flavours of the same mistake —
# (a) stuff the ENTIRE corpus into every prompt, and (b) fine-tune on volatile facts.
# (a) blows the context budget, spikes cost/latency, and buries the relevant passage
#     in noise; (b) freezes the facts into weights so tomorrow's policy change makes
#     the model confidently wrong, and it can never emit a source citation.
with open("all_50000_policy_pages.txt") as f:
    everything = f.read()                       # 50k pages -> way over context limit
prompt = f"{everything}\n\nQuestion: {question}"  # truncated, expensive, ungrounded
# ...OR: fine_tune(model, policy_docs)          # stale the moment a policy is edited

# Correct approach: retrieve only what THIS question needs, keep facts in the index,
# and update the index (not the model) when policies change.
hits = index.similarity_search(
    query_text=question,
    columns=["chunk_text", "doc_uri"],
    num_results=3,            # only the relevant chunks enter the prompt
)
# Edited a policy? Re-sync the AI Search index from its Delta source table.
# No retraining, no redeploy, and every answer still carries a verifiable citation.
```

---

## Common Pitfalls & Misconceptions

- **"RAG trains the model on my documents."** — Beginners assume adding data means learning, because in classic ML more data means retraining. In RAG the model weights never change; documents live in an external index and are injected into the prompt at query time, so you update the *index*, not the *model*.
- **"A bigger or better LLM will fix wrong answers."** — When answers are wrong people reach for a stronger model, since the LLM is the visible component. But if the retriever never returned the relevant chunk, the model had nothing correct to work from — fix retrieval (chunking, `top_k`, hybrid search) first; the LLM is rarely the bottleneck.
- **"Long context makes RAG obsolete."** — Large context windows tempt people to paste the whole corpus and skip retrieval. That works only for a small, static corpus; for a large, changing knowledge base it is slow, expensive, dilutes attention, and still cannot cite which passage was used — retrieval remains the scalable, verifiable choice.
- **"RAG guarantees no hallucination."** — People treat retrieval as a truth switch. RAG only grounds answers *if* the right passage is retrieved and the prompt instructs the model to answer from it; a bad retrieval or a permissive prompt still yields fabrication, which is why groundedness must be evaluated.
- **"Citations are just cosmetic."** — Teams skip citations to save effort, thinking they are decoration. Citations are the mechanism that makes an answer *verifiable* and auditable, and they are what evaluation uses to measure groundedness — they are load-bearing, not decorative.

---

## Key Definitions

| Term | Definition |
|---|---|
| Retrieval-Augmented Generation (RAG) | A technique that queries an external knowledge base for relevant data and injects it into an LLM's prompt at answer time, so the model generates from retrieved facts rather than only from training memory. |
| Knowledge cutoff | The date after which a model has no training knowledge; facts newer than the cutoff must be supplied via retrieval, not recalled. |
| Grounding | The property that an answer's claims are derived from and supported by the retrieved passages rather than the model's parametric memory. |
| Chunk | A passage-sized slice of a source document, small enough to embed and retrieve precisely as a unit. |
| `top_k` | The number of most-similar chunks the retriever returns for a query and stuffs into the prompt. |
| Foundation Model API | The Databricks endpoint that serves LLMs (pay-per-token, provisioned throughput, or external) for the generation step. |
| Databricks AI Search | Databricks' vector search service (formerly Vector Search) that indexes embeddings from a Delta table and returns the most similar documents for retrieval. |
| Groundedness | An evaluation metric measuring whether every claim in the answer is supported by the retrieved context. |

---

## Summary / Quick Recall

- RAG = Retrieve → Augment → Generate; it changes what the model *reads at answer time*, not what it *knows from training*.
- It solves four retrieval problems: knowledge cutoff, hallucination, proprietary data, and freshness.
- Pipeline: offline (ingest → chunk → embed → index) then online (embed query → retrieve → augment → generate).
- Choose RAG for large/changing/citable knowledge; fine-tuning for style/format/behaviour; long-context for small static corpora; prompt engineering when no external facts are needed.
- Retrieval quality — not the LLM — is the usual bottleneck; the model can't cite what it never retrieved.
- Databricks stack: AI Search (retrieve), Foundation Model APIs (generate), Agent Framework (orchestrate, framework-agnostic), Unity Catalog (govern), MLflow (evaluate).
- Update the index when facts change; never retrain the model for volatile data.

---

## Self-Check Questions

1. What does the "augmentation" step in RAG actually do?

   <details><summary>Answer</summary>

   Augmentation combines the retrieved supporting passages with the user's request — usually via a prompt template with instructions — to build the prompt that is then sent to the LLM. It is correct because Databricks defines RAG's middle step as merging retrieved data with the query into a prompt. The tempting wrong answer is "it retrains or updates the model with the new data" — that is false; augmentation only assembles the prompt and never touches model weights.

   </details>

2. Your team runs a chatbot over a legal-contracts database that is updated several times a day and requires every answer to link to the exact clause. Which approach fits best, and why?

   <details><summary>Answer</summary>

   RAG. The data changes daily (freshness) and answers must cite the source clause (verifiability), both of which RAG delivers by retrieving current chunks with metadata at query time and updating the index rather than the model. Fine-tuning is the tempting distractor — "train it on the contracts" — but it bakes facts into frozen weights, goes stale on every update, and cannot emit a clause-level citation.

   </details>

3. A RAG support bot returns answers that are fluent but frequently miss facts that definitely exist in the knowledge base. You have already tried a larger LLM with no improvement. Which change is most likely to help?

   <details><summary>Answer</summary>

   Improve retrieval — e.g. raise `top_k`, fix chunking, or switch to hybrid keyword-similarity search so the relevant chunk actually appears in the prompt. Because the facts exist but aren't reaching the model, the retriever is the bottleneck: the model can't use what it never received. "Use an even bigger LLM" is the tempting wrong move — you already proved model size isn't the issue, and no model can cite a passage it never saw.

   </details>

4. **Which TWO** of the following are legitimate reasons to choose RAG over fine-tuning for a given use case?
   - A. The knowledge base changes frequently and answers must stay current.
   - B. You need the model to adopt a specific brand tone and output format.
   - C. Answers must cite the exact source document for verification.
   - D. You want to permanently reduce the model's inference latency.
   - E. You want to teach the model a brand-new reasoning skill it lacks.

   <details><summary>Answer</summary>

   **A and C.** RAG shines when facts change (A) — you re-sync the index instead of retraining — and when answers must be verifiable with source citations (C), because retrieved chunks carry metadata the model can attribute. B and E describe *behaviour/skill/style* gaps, which are the classic fine-tuning use cases, not RAG. D is the most tempting distractor: RAG *adds* a retrieval hop and extra prompt tokens, so it generally *increases* latency rather than reducing it.

   </details>

5. A manager proposes skipping retrieval entirely by pasting the company's full 30,000-page knowledge base into every prompt using a long-context model, arguing it is "simpler than building a RAG pipeline." Evaluate this trade-off.

   <details><summary>Answer</summary>

   It is a poor trade-off at this scale. Stuffing 30,000 pages per request blows the token budget or forces truncation, spikes cost and latency on every call, dilutes the model's attention across mostly-irrelevant text, and still cannot report *which* passage grounded the answer. Long-context is only reasonable for a small, static corpus that comfortably fits the window; for a large, changing base, RAG's selective `top_k` retrieval is cheaper, faster, fresher (index syncs), and verifiable (citations). The "simpler" argument ignores that the simplicity is paid for in recurring cost, latency, and loss of grounding.

   </details>

---

## Further Reading

- [RAG (Retrieval Augmented Generation) on Databricks](https://docs.databricks.com/aws/en/generative-ai/retrieval-augmented-generation) — *verified 2026-07-11* — Official overview of what RAG is, its benefits (proprietary knowledge, freshness, citations, ACL), its components, and the Databricks stack.
- [Databricks AI Search](https://docs.databricks.com/aws/en/ai-search/ai-search) — *verified 2026-07-11* — The vector-search service (formerly Vector Search) that powers retrieval; covers HNSW ANN, hybrid keyword-similarity search, reranking, filtering, and Delta index sync.
- [Use agents on Databricks](https://docs.databricks.com/aws/en/generative-ai/agent-framework/build-genai-apps) — *verified 2026-07-11* — The Agent Framework for building RAG apps, explicitly framework-agnostic (LangGraph, LangChain, OpenAI, LlamaIndex) with MLflow Tracing.
- [Agent Evaluation](https://docs.databricks.com/aws/en/agents/agent-evaluation/) — *verified 2026-07-11* — How MLflow LLM judges assess RAG quality (correctness, groundedness), cost, and latency using evaluation sets.
- [Supported foundation models on Model Serving](https://docs.databricks.com/aws/en/machine-learning/model-serving/foundation-model-overview) — *verified 2026-07-11* — The Foundation Model APIs (pay-per-token, provisioned throughput, external models) that serve the generation step.

> ⚠️ Fast-evolving: verify against current official docs before relying on this. Databricks AI Search (formerly Vector Search) and the Agent Framework change frequently — feature names, model IDs, and APIs drift.
