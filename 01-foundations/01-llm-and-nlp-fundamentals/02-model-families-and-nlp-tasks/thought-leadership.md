# Model Families & NLP Tasks — Thought Leadership

**Section:** Foundations | **Target audience:** ML engineers, tech leads, AI platform decision-makers | **Target publication:** LinkedIn / personal blog

## Hook / Opening Thesis

Most of the AI budget I see burned in production isn't burned on training — it's burned on serving the wrong-shaped model for the job. The single most expensive habit in enterprise GenAI right now is reaching for a frontier chat LLM to do work that a 600-million-parameter encoder was built for.

## Key Claims (3–5)

1. **Architecture shape, not model size, should be the first decision.** Encoder-only, decoder-only, and encoder-decoder are not trivia — they map directly to task fit and to your cloud bill.
2. **"Which serving mode" is a more consequential choice than "which model."** On Databricks, moving a classification workload from an interactive chat endpoint to AI Functions batch inference changes cost and latency by orders of magnitude — same model quality, different economics.
3. **The open-vs-proprietary debate is mostly a data-residency and fine-tunability debate in disguise.** Price is the least interesting axis.
4. **Instruction-tuning is why your app works, and most teams can't articulate it.** The gap between a base foundation model and its `-Instruct` sibling is the difference between a demo and a product.

## Supporting Evidence & Examples

I keep a running tally of the same failure. A team wires Claude Sonnet or a 70B Llama into a ticket-routing pipeline, sends 500k rows/day through the interactive chat endpoint, and then regex-parses a category out of free text. Every one of those calls is billed per generated token, throttled by interactive concurrency, and fragile at the parsing step. The fix on Databricks is a one-line `ai_classify` over the Delta table — a small model, one batch pass, structured output. Same accuracy target, a fraction of the spend.

The reverse mistake is just as common: calling a chat endpoint to "get embeddings" for retrieval. A chat model returns prose; retrieval needs a fixed-length vector from an encoder-only embedding model like `databricks-gte-large-en` on the `llm/v1/embeddings` route, feeding Databricks Vector Search. The tell is when someone's RAG latency is dominated by an LLM call that was never supposed to be there.

On the serving side, Databricks makes the trade-off explicit and people still miss it. Pay-per-token is zero-infra and fine for prototypes *and* moderate production. You graduate to provisioned throughput for exactly three reasons — performance guarantees, custom/fine-tuned weights, or compliance like HIPAA — not because you've "gone to prod." And external model endpoints exist specifically so you can front OpenAI, Anthropic, Bedrock, or Vertex behind one governed, credential-managed interface routed through Unity AI Gateway. Three different mechanisms, three different questions.

## The Original Angle

The industry talks about model *selection* as a leaderboard sport — who's top of the benchmark this week. I think that's the wrong frame for engineers. The valuable skill isn't knowing which model is #1; it's the decision tree: task → architecture family → workload profile → serving mode. I've watched senior engineers who can recite transformer internals still deploy a generation model for an understanding task, because they never internalized that the *shape* of the model is a first-class design decision, not an implementation detail. My angle: treat "what shape and how do I serve it" as the architecture review question, before anyone names a model.

## Counterarguments to Address

*"Just use one big model for everything — it's simpler to maintain."* Fair, and for a low-volume internal tool it's the right call; operational simplicity has real value. But that argument collapses at scale: the moment volume or latency SLAs enter, the one-big-model pattern is the most expensive way to be slow. The discipline is knowing *when* the simplicity premium is worth paying.

*"Small/encoder models are legacy — everything is decoder-only now."* Decoder LLMs can indeed classify zero-shot, and that's genuinely useful for prototyping. But "can" isn't "should at 500k rows/day." Embedding models are still overwhelmingly encoder-only for a reason: retrieval needs vectors, not generation. The families didn't merge; the defaults just got lazy.

## Practical Takeaways for the Reader

- Before naming a model, write one sentence: "This is an [understanding | generation | sequence-transformation] task at [interactive | batch] volume." The model type falls out of that sentence.
- For any high-volume label/extract/summarize job, default to batch inference (AI Functions) with the smallest capable model — not an interactive frontier endpoint.
- Make "open vs proprietary" a data-residency and fine-tunability decision, and "pay-per-token vs provisioned throughput" a performance/compliance decision. Stop conflating them.
- If your RAG stack calls a chat model to "embed," that's a bug, not a design.

## Call to Action

Go audit one GenAI pipeline you own this week. Find the most expensive model call in it and ask: is this task actually generation, or did we default to a chat LLM out of habit? I'd genuinely like to hear what you find — the "we replaced a chat call with a batch classify and cut cost 30x" stories are the ones worth sharing.

## Further Reading / References

- [Databricks Foundation Model APIs](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/) — the canonical breakdown of pay-per-token vs provisioned throughput vs AI Functions.
- [Supported foundation models on Model Serving](https://docs.databricks.com/aws/en/machine-learning/model-serving/foundation-model-overview) — shows the four hosting options side by side.
- [External models in Model Serving](https://docs.databricks.com/aws/en/generative-ai/external-models/) — the `task` types and provider config behind governed third-party access.
- [AI governance with Unity AI Gateway](https://docs.databricks.com/aws/en/ai-gateway/) — why serving-mode choices are also governance choices.
