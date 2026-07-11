# What Is RAG and When to Use It — Thought Leadership

**Section:** Foundations | **Target audience:** senior engineers, ML leads, and architects evaluating GenAI approaches | **Target publication:** personal blog / LinkedIn

## Hook / Opening Thesis

Most teams reach for RAG because they heard fine-tuning is expensive — and most teams then build a retrieval system that fails for reasons that have nothing to do with the model they picked. RAG is not an LLM technique. It is a search problem wearing an LLM costume, and the sooner you treat it that way, the sooner your answers stop being confidently wrong.

## Key Claims (3–5)

1. RAG's value is almost entirely in the retrieval layer, not the generation layer — yet that is where teams invest the least.
2. The "RAG vs fine-tuning" debate is usually a category error: they solve different problems (facts vs behaviour), so the real question is rarely "which one" but "what is actually broken."
3. Long-context windows did not kill RAG; they changed *where* the boundary sits, and moving that boundary without measuring cost and grounding is a downgrade disguised as a simplification.
4. Citations are not a UX nicety — they are the only reason RAG can be evaluated, audited, and trusted in a regulated setting.
5. On a platform like Databricks, the winning move is to treat the vector index as governed data, not as a model artifact.

## Supporting Evidence & Examples

I have watched more than one team respond to a RAG bot that "misses obvious facts" by swapping in a bigger, pricier model — and get the same wrong answers, now slower and more expensive. The fix, every time, was upstream: chunks that split a table mid-row, a `top_k` of 2 on a corpus that needed 6, or pure vector search that could not match an exact error code like `SQLSTATE 42000`. Databricks documents this directly: AI Search added hybrid keyword-similarity search (vector + BM25, fused with Reciprocal Rank Fusion) precisely because semantic search alone misses SKUs and identifiers. The bottleneck was never the LLM.

The fine-tuning confusion is just as common. A team fine-tuned a model on a policy handbook, shipped it, and two weeks later the policy changed — the model now cited the old rule with total confidence and no source link. RAG would have needed a single index re-sync. The Databricks RAG guide lists "up-to-date information" and "citing sources" as core RAG benefits for exactly this reason; neither is achievable by baking facts into frozen weights.

## The Original Angle

The framing I have not seen said plainly enough: **RAG is a data engineering discipline first and a GenAI discipline second.** The interesting work — chunking strategy, embedding choice, sync freshness, access-control filtering, retrieval evaluation — all sits before the model is ever called. That is why Databricks positions RAG on top of Delta Lake and Unity Catalog rather than as a bolt-on to Model Serving: the index is *governed data with lineage*, and whoever owns your data quality already owns your RAG quality. Engineers who came up through data pipelines have an edge here that pure prompt-engineers do not, and the industry keeps miscasting RAG as the latter.

## Counterarguments to Address

A skeptic will say: "Context windows are exploding — soon you'll just paste everything and retrieval disappears." Partly true, and I use long-context freely for small, static corpora. But for a large, changing knowledge base, stuffing everything is slower, costs tokens on every call, dilutes attention, and — the killer — cannot tell you *which* passage grounded the answer. Retrieval is not just a context-size workaround; it is what makes an answer selective, fresh, and auditable. A second objection: "Fine-tuning gives better domain fluency." Also true, and it composes — fine-tune for *how* the model speaks, RAG for *what* it knows. They are complementary, not rivals.

## Practical Takeaways for the Reader

- Before you touch model selection, instrument retrieval: measure whether the relevant chunk is even in the `top_k`. Databricks' Agent Evaluation gives you groundedness and retrieval judges for this.
- Decide the axis of your problem first: facts that change → RAG; behaviour/format/tone → fine-tuning; small static corpus → long-context; no external facts → prompt engineering.
- Treat your vector index as a governed Delta-backed data asset with a sync policy, not as a model you retrain.
- Make citations mandatory from day one — they are your evaluation harness and your audit trail, not decoration.

## Call to Action

Next time a RAG answer is wrong, resist the urge to upgrade the model. Open the trace, look at the retrieved chunks, and ask: *was the right passage even on the desk?* Tell me the last time a retrieval fix — not a model swap — rescued your RAG system; I collect these stories because they keep proving the same point.

## Further Reading / References

- [RAG (Retrieval Augmented Generation) on Databricks](https://docs.databricks.com/aws/en/generative-ai/retrieval-augmented-generation) — establishes RAG's benefits (freshness, citations, ACL) and component model that this argument builds on.
- [Databricks AI Search](https://docs.databricks.com/aws/en/ai-search/ai-search) — documents hybrid keyword-similarity search and reranking, the retrieval-layer levers central to the thesis.
- [Agent Evaluation](https://docs.databricks.com/aws/en/agents/agent-evaluation/) — the groundedness and quality judges that let you prove retrieval, not the model, is the bottleneck.
