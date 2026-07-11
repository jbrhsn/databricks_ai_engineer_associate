# Embeddings & Vector Search Intro — Thought Leadership

**Section:** Foundations | **Target audience:** senior data/ML engineers and tech leads standing up RAG on Databricks | **Target publication:** LinkedIn / personal blog

## Hook / Opening Thesis

Almost every "our RAG is bad" post-mortem I've sat through ends with the team debating vector databases — and almost none of them should have. The index type you pick on Databricks decides how your data gets *in and stays fresh*; it has nearly nothing to do with whether retrieval is any good. We keep tuning the plumbing and ignoring the water.

## Key Claims (3–5)

1. **Index type is an ingestion decision, not a quality decision.** Delta Sync vs Direct Vector Access changes freshness and operational ownership, not relevance.
2. **The three levers that actually move retrieval quality are the embedding model, the similarity metric, and the chunking** — and most teams under-invest in all three.
3. **"We need cosine similarity" is usually a non-issue on Databricks** because it uses L2 with HNSW; normalize your vectors and the ranking is identical.
4. **ANN recall is a dial you are already turning, whether you know it or not** — and defaulting `num_results` upward is the most common way engineers silently pay latency for recall they don't need.
5. **Managed embeddings are the right default for most teams**, and reaching for self-managed too early is premature optimization dressed up as control.

## Supporting Evidence & Examples

The pattern repeats. A team migrates from a bolt-on vector DB to Mosaic AI Vector Search, wires up a Delta Sync Index with managed embeddings, and retrieval is still mediocre. When we actually look, the culprit is 2,000-token chunks that blur three unrelated topics into one averaged vector — no index on earth retrieves a blurry vector well. Fix the chunking to coherent, overlapping passages and precision jumps, with zero change to the index.

On the cosine debate: the docs are explicit that Vector Search uses L2 distance, and that when vectors are normalized, L2 ranking equals cosine ranking. I've watched an engineer burn two days trying to "switch the metric to cosine." The one-line fix was normalizing embeddings at ingest and query. Same ordering, problem gone.

On `num_results`: the performance guide is blunt — raising it 10× can double latency and cut QPS roughly threefold. I've seen `num_results=200` set "to be safe" turn a 40 ms endpoint into a 400 ms bottleneck that improved answers not at all, because the LLM only ever needed the top handful of chunks.

## The Original Angle

I've built retrieval on both self-hosted vector stores and Databricks Vector Search (now "AI Search"), and the thing nobody says out loud is this: the vendor conversation is a distraction from the data conversation. The most senior thing you can do in a RAG project is refuse to touch the index until you've proven your embeddings and chunking are sound. I can say this because the times I *did* jump to index tuning first, I wasted the most time — and the cheapest wins always came from upstream in the Delta table, not from the endpoint.

## Counterarguments to Address

A skeptic will say: "Index choice absolutely matters — Storage Optimized vs Standard changes latency, QPS, and cost by large factors." True, and I'd never dismiss SKU sizing. But that's a *serving* decision — how fast and how cheap — not a *relevance* decision. It determines whether you can afford and sustain retrieval at scale, not whether the right documents come back. Another push-back: "Self-managed embeddings give lower latency, so they're strictly better." Also true on latency — but only if you already run a matched embedding model and can keep ingest and query dimensions locked together. For most teams that operational burden costs more than the milliseconds it saves.

## Practical Takeaways for the Reader

- Before touching your index, audit chunking and confirm your ingest and query embedding models are the *same model and dimension*.
- If someone says "we need cosine," normalize your vectors and move on.
- Start with managed embeddings and hybrid search; only go self-managed or ANN-only when you have measured a reason.
- Treat `num_results` as a latency budget, not a recall safety net — keep it 10–100 and rerank if you need precision.

## Call to Action

Next time your RAG under-performs, resist opening the vector-database comparison spreadsheet. Instead, pull ten failed queries, inspect the retrieved chunks, and ask whether the *chunks* — not the *index* — were ever capable of answering. What's the last retrieval bug you chased that turned out to be upstream of the index? I'd genuinely like to hear it.

## Further Reading / References

- [Databricks AI Search (overview)](https://docs.databricks.com/aws/en/ai-search/ai-search) — grounds the L2/HNSW and cosine-via-normalization claim.
- [Query an AI Search index](https://docs.databricks.com/aws/en/ai-search/query-ai-search) — hybrid, reranking, and `query_type` behaviour.
- [AI Search performance guide](https://docs.databricks.com/aws/en/ai-search/best-practices) — the `num_results`, dimensionality, and SKU trade-off numbers.
