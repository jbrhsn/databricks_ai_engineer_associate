# Interview Prep: RAG Building Blocks and MLflow Model Signatures

**Section:** 05-assembling-and-deploying-applications | **Module:** 01-chains-and-rag-assembly | **Type:** Interview Prep

---

## Core Conceptual Questions

| Question | Key points to cover | Common weak-answer trap |
|---|---|---|
| What are the four stages of a RAG pipeline and what happens at each stage? | (1) Query encoding: embedding model converts query to vector; (2) Retrieval: ANN search over vector index returns top-k chunks; (3) Context assembly: chunks + query + system prompt formatted into final prompt; (4) Generation: LLM produces response from assembled prompt. LLM does NOT retrieve — retrieval is a pre-LLM step. | Saying "the LLM looks up documents" or conflating retrieval with generation. Weak answers treat RAG as a single black-box step. |
| What is an MLflow `ModelSignature` and why is it required for Databricks Model Serving? | A `ModelSignature` declares the input schema (`Schema([ColSpec(...)])`) and output schema of a model. Model Serving uses it to validate requests, generate an OpenAPI spec, and enable downstream integrations like the AI Playground. Without it the endpoint cannot validate inputs, has no discoverable schema, and will fail integrations. | Saying it is "optional documentation." The signature is a deployment contract, not metadata. Missing it causes deployment failures, not warnings. |
| Explain the difference between a Delta Sync Index and a Direct Vector Access Index in Databricks AI Search. | Delta Sync: auto-syncs embeddings from a source Delta table; supports managed embeddings (Databricks calls the embedding model); ideal for documents that change on a schedule. Direct Vector Access: caller upserts vectors directly via API; used when embeddings are computed externally (e.g. custom GPU pipeline, proprietary model). | Describing both as "ways to store vectors." The key distinction is who computes the embeddings and how the index stays fresh. |
| Why must the same embedding model be used at both index time and query time in RAG? | Embedding models project text into a specific vector space. Two different models produce vectors in incompatible spaces. Similarity scores computed between vectors from different models are meaningless — the cosine distance no longer correlates with semantic similarity. | Saying "it's a best practice" without explaining why. The mechanism (incompatible vector spaces) must be stated. |
| What is the purpose of the `resources=` parameter in `mlflow.langchain.log_model()`? | Declares Databricks-managed resources (e.g. `DatabricksVectorSearchIndex`, `DatabricksServingEndpoint`) needed at serving time. This enables automatic authentication passthrough so the deployed endpoint can call back to those resources using the caller's identity, without embedding credentials. | Saying it is "for documentation" or "optional." Omitting it causes 500 errors at serving time when the endpoint has no credentials to call Databricks APIs. |

---

## Applied / Scenario Questions

**Q: A user asks why their RAG chain returns correct answers in the notebook but wrong answers after deployment to Model Serving. What would you investigate first?**

Strong answer framework:
1. **Check the signature** — run `mlflow.models.get_model_info(model_uri).signature` and confirm inputs/outputs match what the production caller is sending. A type mismatch could silently coerce or drop the query field.
2. **Check resource declarations** — verify `resources=` was included at log time. If the deployed chain is hitting a different AI Search index (e.g. a dev index) or a different embedding endpoint, retrieval results will differ.
3. **Check embedding model consistency** — confirm the embedding model used by the deployed retriever matches the one used when the index was built. A model name environment variable that resolves differently in serving vs. notebook could cause this.
4. **Check context truncation** — log the number of tokens in the assembled prompt. If the serving environment has a different `max_tokens` setting or the index returned more verbose documents, context may be truncated.
5. **Check the chain serialisation** — if the chain uses stateful components (FAISS in-memory index, custom Python objects), verify they were serialised and reconstructed correctly; use the "Models from Code" logging approach to avoid serialisation issues.

---

**Q: You need to add citation metadata (`source_documents`) to an already-deployed RAG chain that currently returns only a `response` string. Walk through the changes required.**

Strong answer framework:
1. **Update the chain output** — modify the chain logic to return a dict `{"response": str, "source_documents": List[dict]}` instead of a plain string. In LangChain this means removing `StrOutputParser()` and replacing it with a custom output parser that retains the docs.
2. **Update the signature** — add `ColSpec(name="source_documents", type=DataType.string, required=False)` to the output schema. Use `required=False` to avoid breaking callers before they update their parsers.
3. **Re-log the chain** — call `mlflow.langchain.log_model()` with the new chain and updated signature. MLflow creates a new model version; the existing version is unaffected.
4. **Re-register** — call `mlflow.register_model()` to create a new version in the Unity Catalog model registry.
5. **Update the serving endpoint** — promote the new version using the Databricks SDK or UI. Plan a brief cutover window and verify the new endpoint schema with a test call before full traffic cutover.
6. **Update consumers** — notify downstream callers that a new optional field is now available in the response.

---

## System Design / Architecture Questions

**Q: Design a production RAG system for a legal document corpus (100,000 PDFs, updated daily) on Databricks. Specify each component, how it stays fresh, and how it is logged for serving.**

Approach:
1. **Clarify:** Confirm latency SLA (interactive vs. batch), accuracy requirements (citations mandatory?), user base size, and whether documents are confidential (ACL filtering needed?).
2. **Propose:**
   - **Data pipeline:** Delta Live Tables (Lakeflow) ingests PDFs, extracts text, chunks at 512 tokens with 50-token overlap, writes to a Delta table `legal.docs.chunks`.
   - **Index:** Delta Sync Index with managed embeddings (`databricks-bge-large-en`), `continuous` sync mode. Hybrid search enabled. Filtered on `document_type` and `access_group` metadata columns for ACL.
   - **Chain:** LangGraph with a `retrieve` node (`DatabricksVectorSearch.as_retriever(search_kwargs={"k": 7, "query_type": "HYBRID"})`) and a `generate` node (`ChatDatabricks` with `databricks-meta-llama-3-3-70b-instruct`). Output: `{"response": str, "source_documents": List[dict]}`.
   - **Signature:** `ModelSignature(inputs=Schema([ColSpec("query", DataType.string)]), outputs=Schema([ColSpec("response", DataType.string), ColSpec("source_documents", DataType.string, required=False)]))`.
   - **Logging:** `mlflow.langchain.log_model(lc_model="agent.py", resources=[DatabricksVectorSearchIndex("legal.docs.index"), DatabricksServingEndpoint("databricks-meta-llama-3-3-70b-instruct")])`.
   - **Registry:** Register to `prod.legal.rag_chain` in Unity Catalog. Deploy to a Model Serving endpoint with ACL passthrough.
3. **Justify tradeoffs:**
   - Delta Sync over Direct Vector Access: documents are owned by Databricks pipelines, not external GPU clusters.
   - Hybrid over pure vector: legal documents contain statute numbers, case citations, and defined terms that BM25 handles better than dense vectors.
   - `required=False` on `source_documents`: allows gradual rollout of citation UI without a breaking change.

---

## Vocabulary That Signals Expertise

| Term | When/why to use it |
|---|---|
| `ModelSignature` + `ColSpec` | When discussing MLflow deployment contracts — signals you know the exact API, not just the concept |
| Reciprocal Rank Fusion (RRF) | When explaining hybrid search — shows you understand how vector and keyword scores are combined, not just that "hybrid search exists" |
| HNSW (Hierarchical Navigable Small World) | When describing how Databricks AI Search performs ANN — signals awareness of the underlying algorithm and its latency/accuracy tradeoffs |
| Automatic authentication passthrough | When discussing `resources=` — signals you understand the security model, not just the syntax |
| Delta Sync Index vs Direct Vector Access Index | When discussing index freshness strategies — signals you understand the two architectural patterns and their tradeoffs |
| `infer_signature()` | When discussing practical signature construction — signals you know the efficient path, not just the manual `ColSpec` approach |
| Context window budget | When discussing chunk size and `k` selection — signals you think about token arithmetic, not just semantic relevance |

---

## Vocabulary That Signals Weakness

| Term/phrase | Why it's a red flag |
|---|---|
| "The LLM retrieves from its training data" | Fundamentally misunderstands what RAG is — retrieval happens before the LLM, not inside it |
| "The signature is just metadata/documentation" | Reveals that you have never deployed a chain to Model Serving and encountered a validation failure |
| "We'll add the signature later" | Signals you don't know that MLflow model artifacts are immutable; "adding it later" requires a new model version |
| "Vector search" (without qualification) | Using the old product name without noting the rename to "AI Search" signals you are behind on the current platform state |
| "We can use any embedding model at query time" | Reveals a gap in understanding vector space compatibility — a dangerous misconception in production systems |
| "It works in the notebook" | Not addressing the gap between notebook and serving environment is a red flag for production experience |

---

## STAR Answer Frame

**Situation:** I was on a team building a customer support RAG chatbot on Databricks. After three months of development and tuning, the chain worked well in the notebook but failed to integrate with the company's support ticketing portal.

**Task:** Diagnose and fix the integration failure and establish practices to prevent recurrence.

**Action:** I audited the MLflow run and found the logged model had no `ModelSignature` — the team had never included one. The portal engineering team had been reverse-engineering the output format from ad-hoc test calls, and when the chain output format changed (we added `source_documents` in a late sprint), their parsing code broke silently. I re-logged the chain using `mlflow.langchain.log_model()` with an explicit `ModelSignature` that declared `query` as the string input and `response` + `source_documents` as string outputs. I also added `resources=[DatabricksVectorSearchIndex(...), DatabricksServingEndpoint(...)]` which resolved a separate 500 error from the serving environment not having credentials to call the index. I introduced a code review checklist item: "Does the MLflow log call include `signature=` and `resources=`?" and added a CI step that loads the logged model and validates the signature is present.

**Result:** The portal integration was completed in two days. The next two chain updates were integrated by the portal team without any engineering involvement from my side — they consumed the OpenAPI spec generated from the signature and updated their client code against it. The authentication 500 errors disappeared entirely.

---

## Red Flags Interviewers Watch For

- **Cannot explain why retrieval precedes the LLM** — a fundamental conceptual gap that calls into question all downstream design decisions.
- **Treats chunk size and embedding model as independent choices** — in reality they interact: larger chunks require more context budget, and the embedding model determines the resolution at which semantic similarity is measured.
- **Does not know the difference between `mlflow.langchain.log_model()` and `mlflow.pyfunc.log_model()`** — signals they have only used one path and may make the wrong choice when frameworks mix.
- **Says "just use `infer_signature()`" without knowing when it fails** — `infer_signature()` fails if the output is a Pydantic model, a streaming generator, or a non-serialisable Python object; knowing the fallback path (manual `ColSpec` construction) signals production depth.
- **Cannot articulate what `resources=` does** — means they have only run chains in notebooks with ambient credentials, not deployed them to isolated serving environments.
- **Conflates context window size with model quality** — teams that see retrieval working but answers being wrong often blame the LLM when the real problem is that retrieved context was truncated past the token budget.
