# The Contract Nobody Reads: Why RAG Model Signatures Are the Most Under-Engineered Layer in Enterprise AI

**Section:** 05-assembling-and-deploying-applications | **Module:** 01-chains-and-rag-assembly | **Type:** Thought Leadership

---

## Hook / Opening Thesis

Most RAG failures in production are not retrieval quality problems. They are interface contract problems. Teams spend months tuning chunk sizes, embedding models, and reranking strategies — then ship a chain with no `ModelSignature`, and discover at integration time that the consuming application has been guessing at the output format since day one. The MLflow model signature is the schema that makes a RAG chain a deplorable service rather than a science experiment, and it is the single most skipped step in enterprise AI projects.

---

## Key Claims (3–5)

1. **A RAG chain without a `ModelSignature` is not production software** — it is a prototype that happens to be running. Without an explicit schema declaration, every downstream consumer builds on an undocumented assumption about the output format. When that assumption changes, silent failures are guaranteed.

2. **The embedding model is a shared schema, not a component choice** — teams treat the embedding model as interchangeable mid-project, not realising that swapping models (even to an ostensibly "better" one) invalidates every vector in the index and breaks similarity scores. The embedding model is a data contract, and changing it requires a full re-index.

3. **Databricks AI Search's move from Vector Search to AI Search signals that the retrieval layer is converging with search infrastructure** — the renaming is not cosmetic; hybrid keyword-vector search using BM25 + RRF is now the default retrieval mode. RAG systems that use pure vector similarity are leaving measurable recall gains on the table for any corpus with proper nouns, codes, or identifiers.

4. **Most RAG "hallucination" complaints are actually context assembly failures** — the LLM answered from its training data because the retrieved chunks were truncated past the context window boundary without the developer noticing. Logging context token counts as a chain metric — and alerting on truncation — is a hygiene practice almost nobody follows.

5. **The `resources=` parameter in `mlflow.langchain.log_model()` is the most consequential single line of code in a RAG deployment** — omitting it does not cause a test failure; it causes a production 500 error that appears after deployment when the serving environment tries to call the vector index and has no credentials.

---

## Supporting Evidence & Examples

**On signatures and silent failures:** A model logged without a signature generates no OpenAPI spec for Model Serving. The Databricks AI Playground and the external review app both require a valid schema to render the input/output fields. Without it, teams manually wrap every API call in custom payload-shaping code, then maintain that code indefinitely as the chain evolves.

**On embedding model drift:** The Databricks AI Search documentation confirms that if you use managed embeddings (embedding model configured in the index definition), the same model is used at both index and query time automatically. This is the correct architecture. The failure mode appears when developers use a different embedding client at query time "because it's faster" or "because the endpoint changed" — neither the index nor the vector similarity function reports an error; the scores simply become meaningless numbers between 0 and 1.

**On hybrid search:** Databricks AI Search uses Reciprocal Rank Fusion (RRF) to combine ANN cosine/L2 scores with Okapi BM25 keyword scores. For corpora containing unique identifiers (product SKUs, document IDs, regulatory codes), pure vector search frequently misses exact matches that BM25 catches trivially. The documentation explicitly calls this out: *"This method is particularly useful in RAG applications where source data has unique keywords such as SKUs or identifiers."*

**On context truncation:** Most LLMs truncate from the left (dropping the oldest tokens). In a RAG prompt where the context section is placed before the question, the documents are dropped first. A `k=10` retriever that returns 2,000 tokens of context for an LLM with a 4,096-token budget, after accounting for system prompt (~200 tokens) and expected answer (~300 tokens), has only ~3,596 tokens left — barely enough for k=5 at typical chunk sizes. Nobody catches this until a user reports that an answer contradicts a document they know was uploaded.

---

## The Original Angle

The conversation in the AI engineering community is almost entirely about retrieval quality — chunk size, overlap, embedding model benchmarks, reranking. The layer nobody systematically designs is the output contract: what exactly does the chain promise to return, in what format, with what guarantees about provenance fields? In traditional software engineering this is API design 101. In AI engineering it is treated as an afterthought. The `ModelSignature` + `resources=` pair in MLflow is the direct equivalent of an API contract + service dependency declaration. Treating these as bureaucratic overhead is the source of most RAG production incidents that are incorrectly attributed to model quality.

---

## Counterarguments to Address

**"Signatures slow down iteration."** This is true in week one and false in week two. During iteration you can use `mlflow.models.infer_signature()` with a sample input/output — it takes four lines of code and automatically generates the schema from your actual chain output. The argument that explicit signatures slow down development conflates "fast prototyping" with "production engineering," which is a category error.

**"We can always add the signature later."** You can, but only by creating a new model version. MLflow model artifacts are immutable after logging. Adding the signature later means re-logging, re-registering, and re-deploying — more work than including it the first time. The "add it later" path is also risk-prone because the chain's output format often changes between prototype and production, and the signature you add retroactively may not reflect the deployed chain's actual behaviour.

**"Our team controls both sides of the API so we don't need a schema."** This works until you don't — until a new team member, a third-party integration, or an automated evaluation pipeline starts consuming the endpoint. At that point you discover that the "shared knowledge" about the output format was actually tribal knowledge held by two people, one of whom has since left.

---

## Practical Takeaways for the Reader

1. **Always log with a signature.** Use `infer_signature(input_example, output_example)` if you prefer automation over manual `ColSpec` construction. Four lines of code eliminates an entire class of deployment failures.

2. **Lock your embedding model into the index definition.** Use Databricks AI Search's managed embeddings so the same model is called at index time and query time without any manual coordination. Treat the embedding model name as a constant in your configuration, not a variable.

3. **Switch to `query_type="HYBRID"` by default.** Unless your corpus is entirely natural-language prose with no identifiers or codes, hybrid search outperforms pure vector search at negligible additional cost.

4. **Declare `resources=` even in development.** Logging with resource declarations from the start means the model is deployable without modifications when you're ready to promote to production.

5. **Add context token count as a chain metric.** Log `len(context_tokens)` alongside each inference in development. Alert if it exceeds 80% of the context budget. This surfaces truncation bugs before they become user-facing hallucination complaints.

6. **Design `source_documents` into the output schema from day one.** Retrofitting provenance into a deployed chain requires a new model version and a consumer-side change. Start with `ColSpec(name="source_documents", type=DataType.string, required=False)` and populate it from the first iteration.

---

## Call to Action

The next RAG chain your team ships: open the MLflow run, look at the `MLmodel` file, and check whether `signature:` is present. If it is not, you have a prototype in production. Add the signature, declare the resources, and re-register the model before the next sprint ends. The work takes less than an hour and turns your chain from an undocumented experiment into a governed, auditable service.

---

## Further Reading / References

- [RAG on Databricks — official guide](https://docs.databricks.com/aws/en/generative-ai/retrieval-augmented-generation) — *verified 2026-07-11*
- [Databricks AI Search — hybrid search and HNSW details](https://docs.databricks.com/aws/en/ai-search/ai-search) — *verified 2026-07-11*
- [MLflow Model Signatures — how to log models with signatures](https://mlflow.org/docs/latest/model/signatures) — *verified 2026-07-11*
- [Log and register AI agents](https://docs.databricks.com/aws/en/agents/agent-framework/log-agent) — *verified 2026-07-11*
