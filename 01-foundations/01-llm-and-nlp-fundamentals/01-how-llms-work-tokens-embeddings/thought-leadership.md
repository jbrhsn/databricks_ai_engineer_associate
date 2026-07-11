# How LLMs Work: Tokens & Embeddings — Thought Leadership

**Section:** Foundations | **Target audience:** Engineers and tech leads shipping their first RAG system | **Target publication:** LinkedIn / personal blog

## Hook / Opening Thesis

Most RAG systems that "hallucinate" or return junk aren't failing at the LLM layer at all — they're failing at the two least-glamorous primitives nobody profiles: how text becomes tokens, and how text becomes vectors. I've watched more launches slip on a mismatched embedding model than on a bad prompt.

## Key Claims (3–5)

1. Token illiteracy is a budgeting bug, not a trivia gap. Teams that estimate cost and context limits in *words* systematically under-provision, because sub-word tokenization inflates rare and technical text.
2. "Embeddings" is an overloaded word that quietly causes architecture mistakes: token embeddings inside a chat model and document embeddings for retrieval are different objects for different jobs.
3. The embedding model is a schema decision, not a hyperparameter. Swapping it after you've indexed a corpus is a migration, not a tweak.
4. On Databricks the abstraction is clean enough that the *only* thing left to get wrong is conceptual — the API won't stop you from embedding with the wrong model or mixing spaces.

## Supporting Evidence & Examples

On one project my team sized a summarization workload by running `str.split()` over sample documents and multiplying by the per-word price we'd back-calculated. The real corpus was full of SKUs, code snippets, and German product names — all of which BPE shatters into many tokens. Actual usage came in roughly 35% over the "safe" estimate, and two documents silently truncated at the context limit, dropping the exact clauses the summaries were supposed to cover. The fix wasn't a better model; it was reading `usage.total_tokens` off the response and chunking by tokens instead of words.

The embedding-model trap is even more common. A teammate upgraded from one embedding endpoint to a newer one for freshly ingested documents and left the old 50,000 vectors in place in the same index. Retrieval quality didn't degrade gracefully — it collapsed, because vectors from two models live in different geometric spaces with different dimensions. Databricks AI Search happily indexed both; the incompatibility is conceptual, and the platform can't catch it for you. The remediation was a full re-embed with `databricks-gte-large-en` (1024-dim) and an index rebuild.

I've also seen the "context window as memory" fallacy waste weeks: an engineer assumed the model would "remember" earlier documents across calls and skipped retrieval entirely. The window is a per-request token cap with zero persistence — which is the whole reason RAG exists.

## The Original Angle

The industry conversation is stuck at the top of the stack — prompt engineering, agent frameworks, model leaderboards. I want to drag attention back down to tokens and embeddings, because that's where most production incidents actually originate and where the fixes are cheapest. I've shipped these systems on Databricks Foundation Model APIs and AI Search, and the pattern is consistent: the failures are boring, upstream, and preventable with a correct mental model, not a bigger model.

## Counterarguments to Address

A skeptic will say: "Tokenizers and embeddings are abstracted away — I call `embeddings.create` and move on; why should I care?" Two reasons. First, the abstraction hides the *cost and limit* mechanics, not the consequences — you still get truncated inputs and surprise bills if you reason in words. Second, the abstraction is exactly what makes the conceptual errors invisible: because the API accepts a chat endpoint where you meant an embedding endpoint, or two incompatible vector spaces in one index, the only defense is understanding what's underneath. Another objection: "Bigger context windows make chunking obsolete." Larger windows help, but attention cost scales with length, so stuffing everything in trades correctness for latency and price — retrieval is still the economical path.

## Practical Takeaways for the Reader

- Measure everything in tokens. Log `usage.total_tokens`, chunk by tokens, and never size a budget with `str.split()`.
- Say which embedding you mean. "Document embedding for retrieval" and "the model's internal token embeddings" are different sentences with different endpoints.
- Treat the embedding model as a schema. Pin it, version it, and plan a re-embed + index rebuild before you ever switch it.
- Normalize before you rank. If your vectors aren't unit-length, L2 (what AI Search uses) won't match cosine — check with a two-line `is_normalized`.
- Retrieve, don't remember. The context window is a cap, not a cache; re-inject relevant chunks each call.

## Call to Action

Before your next RAG sprint, run one experiment: tokenize ten of your *hardest* real documents and compare the token count to the word count. If the gap surprises you, you've just found the budget bug that would have surfaced in production. What's the largest word-to-token gap you've measured in your own corpus?

## Further Reading / References

- [Databricks Foundation Model APIs](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/) — grounds the pay-per-token cost model the argument depends on.
- [Query an embedding model](https://docs.databricks.com/aws/en/machine-learning/model-serving/query-embedding-models) — shows the exact `embeddings.create` and normalization checks referenced above.
- [Databricks AI Search](https://docs.databricks.com/aws/en/generative-ai/vector-search.html) — documents the HNSW + L2 retrieval that makes the embedding-space compatibility point concrete.
