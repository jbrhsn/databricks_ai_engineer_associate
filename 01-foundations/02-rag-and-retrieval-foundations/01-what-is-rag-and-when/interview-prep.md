# What Is RAG and When to Use It — Interview Prep

**Section:** Foundations | **Role target:** GenAI Engineer / ML Engineer

## Core Conceptual Questions

These test whether you understand the fundamentals.

| Question | Key points to cover | Common weak-answer trap |
|---|---|---|
| What is RAG, in one sentence? | Retrieve relevant docs from an external store, augment the prompt with them, then generate; it changes what the model reads at answer time, not its training. | Saying RAG "trains" or "teaches" the model on your data — it never touches weights. |
| What problems does RAG solve? | Knowledge cutoff, hallucination, proprietary/private data, and freshness — all retrieval problems, plus source citation and ACL-based access. | Listing only "hallucination" and missing freshness/proprietary-data/citations. |
| Walk me through the RAG pipeline. | Offline: ingest → chunk → embed → index. Online: embed query → retrieve top_k → augment prompt → generate. | Merging offline and online phases, or forgetting the query is embedded with the *same* model as the chunks. |
| RAG vs fine-tuning — when each? | RAG for facts that change or must be cited; fine-tuning for style/format/behaviour. They compose. | Framing them as competitors and claiming one is "better" in general. |
| Why are citations important beyond UX? | They make answers verifiable and auditable, and they are what groundedness evaluation measures. | Calling citations cosmetic or optional. |
| Where is the usual bottleneck in a failing RAG system? | Retrieval — if the relevant chunk isn't in top_k, no model can use it. Fix chunking, top_k, hybrid search first. | Blaming the LLM and reaching for a bigger model reflexively. |

## Applied / Scenario Questions

**Q:** A support bot over frequently-edited product docs gives fluent answers that miss facts you know are in the knowledge base. You've already tried a larger LLM. What do you do?

**Strong answer framework:**
- Diagnose retrieval first: open the trace and check whether the relevant chunk was even returned in `top_k`.
- Concrete levers: raise `top_k`, revisit chunking (are tables/lists being split?), switch to hybrid keyword-similarity search if the misses involve exact identifiers (SKUs, error codes).
- Confirm the augmentation prompt actually instructs the model to answer *only* from context and to abstain when the context lacks the answer.
- **Tradeoff awareness:** note that raising `top_k` adds latency and token cost, so tune it against measured retrieval metrics rather than maximizing it; mention re-syncing the index on doc edits rather than retraining.

## System Design / Architecture Questions (if applicable)

**Q:** Design a RAG system on Databricks for an internal HR assistant answering from policy documents, with per-user access control and mandatory citations.

**Approach:**
1. **Clarify requirements:** corpus size and edit frequency? latency SLA? who can see which documents (ACL)? are citations mandatory? expected query volume (for pay-per-token vs provisioned throughput)?
2. **Propose structure:**
   - **Ingest/govern:** land policy docs in Delta Lake; register in Unity Catalog for lineage and access control.
   - **Index:** chunk → embed → build a Databricks AI Search index from the Delta table; choose sync mode (triggered after edit batches for weekly-changing docs). Apply ACL filtering so retrieval returns only rows the user may see.
   - **Chain:** author with the Agent Framework — which is framework-agnostic and supports LangGraph (primary choice for a stateful, graph-based flow), LangChain, LlamaIndex, or OpenAI. Retrieve top_k, augment with a template that demands citations and abstention, generate via a Foundation Model API endpoint.
   - **Evaluate/monitor:** MLflow Tracing for observability; Agent Evaluation with correctness and groundedness judges on an evaluation set; carry the same judges into production monitoring.
3. **Justify choices and name tradeoffs:** RAG over fine-tuning because policies change and must be cited; triggered sync over continuous to save cost since edits are infrequent; hybrid search only if policies contain exact codes; `top_k` kept small (3–5) to bound latency/cost, tuned against evaluation. ACL filtering at retrieval time is what keeps a governed system compliant.

## Vocabulary That Signals Expertise

Use these terms naturally — don't force them:
- **Grounding / groundedness** — say "we evaluate groundedness" to show you know answers must be traceable to retrieved context.
- **top_k and retrieval recall** — frames retrieval quality as measurable, not vibes.
- **Hybrid keyword-similarity search / reranking** — signals you know pure vector search misses exact identifiers.
- **Index sync mode (triggered vs continuous)** — shows you think about the freshness/cost tradeoff.
- **Framework-agnostic Agent Framework (LangGraph, LangChain, LlamaIndex, OpenAI)** — shows current Databricks knowledge.
- **Unity Catalog governance / ACL filtering at retrieval** — signals production and compliance maturity.

## Vocabulary That Signals Weakness

Avoid these — they signal outdated or shallow understanding:
- **"RAG trains the model on our data"** — factually wrong; RAG never updates weights.
- **"Just use a bigger model to fix accuracy"** — ignores that retrieval is the usual bottleneck.
- **"Vector Search"** as the current product name without noting it is now **Databricks AI Search** — minor, but signals stale docs.
- **"RAG eliminates hallucination"** — overclaims; it only grounds answers when the right chunk is retrieved and the prompt enforces it.
- **"Citations are just for the UI"** — misses that they are the basis of evaluation and audit.

## STAR Answer Frame

**Situation:** Our internal knowledge bot returned fluent but frequently incorrect answers, and the team had already spent a sprint swapping in progressively larger LLMs with no improvement.
**Task:** I owned answer quality and had to raise correctness without ballooning inference cost.
**Action:** I stopped touching the model and instrumented retrieval with MLflow tracing and Agent Evaluation. The traces showed the relevant chunk was often absent from the `top_k` because chunking split multi-column tables mid-row and pure vector search missed exact part numbers. I re-chunked on semantic boundaries, raised `top_k` from 2 to 5, and switched AI Search to hybrid keyword-similarity search; I also tightened the prompt to force abstention when context was insufficient.
**Result:** Groundedness and correctness scores rose sharply on the evaluation set, we reverted to the smaller/cheaper model, and per-query cost dropped while accuracy improved — proving the bottleneck had been retrieval all along.

## Red Flags Interviewers Watch For

- Conflating RAG with fine-tuning, or claiming RAG modifies model weights.
- Reaching for a bigger LLM before inspecting the retrieved context.
- No mention of *evaluating* retrieval separately from generation.
- Ignoring governance/ACL when the scenario clearly involves private data.
- Treating long-context as a universal replacement for retrieval without naming the cost, latency, and grounding tradeoffs.
- Being unable to name a single retrieval knob (`top_k`, chunking, hybrid search, sync mode) when asked how to improve quality.
