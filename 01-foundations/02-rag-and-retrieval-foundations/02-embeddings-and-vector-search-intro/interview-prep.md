# Embeddings & Vector Search Intro — Interview Prep

**Section:** Foundations | **Role target:** GenAI Engineer / ML Engineer

## Core Conceptual Questions

These test whether you understand the fundamentals.

| Question | Key points to cover | Common weak-answer trap |
|---|---|---|
| What is an embedding and why does similarity work? | Fixed-length dense float vector encoding meaning; similar meaning → nearby vectors; distance is a proxy for semantic difference; dimension is fixed per model. | Saying "it's a numerical representation" and stopping — no mention of *why nearness ≈ meaning*. |
| Cosine vs dot product vs L2 — which does Databricks use? | Cosine = direction only; dot = direction + magnitude; L2 = geometric distance. Databricks uses **L2 + HNSW**; normalize vectors to get cosine ranking. | Asserting "cosine, obviously" — the default is L2; cosine only via normalization. |
| What is HNSW and how does ANN differ from exact search? | HNSW = layered navigable graph; ANN visits a fraction of vectors, trading recall for speed; exact scans everything. | Describing ANN as "just faster search" with no mention that it's *approximate* (recall < 100%). |
| Delta Sync Index vs Direct Vector Access Index? | Delta Sync tracks a source Delta table, updates incrementally (continuous/triggered); Direct Vector Access has no source table, you upsert/delete vectors yourself. | Thinking one retrieves "better" — the difference is ingestion/freshness, not quality. |
| Managed vs self-managed embeddings? | Managed: Databricks embeds from a text column at ingest + query. Self-managed: you precompute and store `array<float>`, can pass raw `query_vector`. | Claiming self-managed is "always faster/better" without noting the model-and-dimension lock-in burden. |
| What is hybrid search and when do you use it? | ANN + BM25 keyword fused via Reciprocal Rank Fusion (RRF, param 60); best for mixed conceptual + keyword workloads (SKUs, codes); ~2× ANN cost. | Confusing hybrid with reranking, or thinking hybrid is always on. |

## Applied / Scenario Questions

**Q:** Your RAG chatbot retrieves plausible-looking but irrelevant chunks. Walk me through how you'd diagnose it.

**Strong answer framework:**
- Start upstream, not at the index: inspect the actual retrieved chunks for a set of failing queries.
- Check **chunking** first — oversized chunks average multiple topics into a blurry vector; verify coherent, overlapping passages.
- Confirm the **ingest and query embedding models match** (same model, same dimension) — a mismatch returns noise.
- Verify the **metric assumption** — if the team expected cosine, confirm vectors are normalized (Databricks is L2).
- Only then consider retrieval strategy: try **hybrid** if queries carry keywords/IDs, add a **reranker** for precision.
- How to show tradeoff awareness: note that raising `num_results` is *not* a fix — it costs latency/QPS without helping relevance.

**Q:** You have a 5M-row Delta table updated hourly and no embedding infrastructure. Design the index.

**Strong answer framework:**
- **Delta Sync Index** (data is already in Delta) with **managed embeddings** (`embedding_source_column`, no infra to run).
- **Sync mode**: hourly cadence → **TRIGGERED** fired by the ETL job, not CONTINUOUS (avoid a 24/7 streaming cluster).
- **Endpoint**: ~5M vectors is comfortable on **Standard** for low latency.
- Sync only PK + embedding + fields you filter/display (`columns_to_sync`) to trim cost and latency.
- Tradeoff called out: TRIGGERED trades sub-minute freshness for cost — acceptable at hourly cadence.

## System Design / Architecture Questions (if applicable)

**Q:** Design retrieval for a support RAG system over 100M documents that must serve 100+ QPS with sub-second latency and stay cost-aware.

**Approach:**
1. **Clarify requirements:** freshness (real-time vs batch), query mix (semantic vs keyword-heavy), latency SLO, budget, governance (Unity Catalog).
2. **Propose structure:** Delta table (one coherent chunk per row) → Delta Sync Index → Vector Search endpoint → retriever node in a LangGraph agent (LangChain retriever as the component inside the node); CrewAI would be a comparison-only alternative for multi-agent orchestration.
3. **Justify choices and name tradeoffs:** 100M vectors + cost-awareness leans **Storage Optimized** (up to ~1B vectors, ~7× cheaper) but accept ~300–500 ms and TRIGGERED-only sync; if the sub-second SLO is tight, consider Standard with **high-QPS target** or **replicate the index across endpoints** for linear QPS gains. Use **hybrid** for the keyword-heavy support queries, **reranker** only if the latency budget allows the ~10% quality-for-latency trade. Use **service principals + OAuth** (not PATs) for lower per-query latency, and reuse the index object rather than calling `get_index` per request.

## Vocabulary That Signals Expertise

Use these terms naturally — don't force them:
- **Recall-vs-latency trade-off** — shows you understand ANN is a tunable dial, not free accuracy.
- **HNSW / graph traversal** — names the actual algorithm rather than hand-waving "the vector DB."
- **L2 ranking equals cosine under normalization** — signals you know Databricks' real metric behaviour.
- **Reciprocal Rank Fusion (RRF)** — the specific hybrid-merge mechanism.
- **Incremental sync (triggered vs continuous)** — shows you think about freshness *and* cost.
- **Managed vs self-managed embeddings** — the correct Databricks framing of who owns the vectors.
- **Model/dimension lock-in between ingest and query** — a subtle, senior-level correctness point.

## Vocabulary That Signals Weakness

Avoid these — they signal outdated or shallow understanding:
- "We'll just pick the best vector database" — treats index/vendor choice as the quality lever.
- "It uses cosine similarity" (stated as the Databricks default) — it's L2 unless you normalize.
- "ANN finds the exact nearest neighbours" — it's approximate by definition.
- "Bigger `num_results` means better answers" — it mostly buys latency.
- "Continuous sync is always better" — ignores the 24/7 cost for batch data.
- "Embeddings are just a hash / lookup" — misses that meaning is encoded as position.

## STAR Answer Frame

**Situation:** A RAG support assistant on Databricks was returning irrelevant passages, and the team was mid-way through a vector-database migration proposal to "fix retrieval."
**Task:** I owned retrieval quality and had to decide whether the migration was warranted before we spent a sprint on it.
**Action:** I pulled ~30 failing queries and inspected the retrieved chunks instead of the index. The chunks were 2,000-token blobs averaging several topics; I also found the query path used a different embedding model than ingest. I re-chunked into coherent overlapping passages, pinned a single managed embedding model end-to-end on a Delta Sync Index, and switched the query to hybrid for the keyword-heavy tickets — leaving the index type unchanged.
**Result:** Top-k relevance improved sharply on our eval set, p50 latency stayed flat because we kept `num_results` at 20, and the vector-database migration was cancelled, saving the planned sprint.

## Red Flags Interviewers Watch For

- Jumps to comparing vector databases before mentioning embeddings, chunking, or metric — the classic "plumbing over water" tell.
- Can't state which metric Databricks uses or how to get cosine behaviour.
- Describes ANN as exact, or can't explain what recall means.
- Confuses index type (freshness/ingestion) with retrieval quality.
- Proposes CONTINUOUS sync or self-managed embeddings with no cost/operational justification.
- Uses PATs in a production design or re-instantiates the index client per query (latency red flags from the performance guide).
