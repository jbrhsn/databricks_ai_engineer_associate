# RAG Building Blocks and MLflow Model Signatures

**Section:** 05-assembling-and-deploying-applications | **Module:** 01-chains-and-rag-assembly | **Est. time:** 3 hrs | **Exam mapping:** Application Development (30%) — RAG assembly and model deployment contracts

---

## TL;DR

Retrieval-Augmented Generation (RAG) is a pipeline that routes a user query through an embedding model, a vector retriever, a context-assembly step, and finally an LLM to produce a grounded response. On Databricks, Databricks AI Search (formerly Vector Search) serves as the retriever, and MLflow model signatures formally define the input/output contract so Model Serving knows exactly what to expect. Logging a RAG chain with `mlflow.langchain.log_model()` or `mlflow.pyfunc.log_model()` with a `ModelSignature` is what makes it deployable and auditable.

**The one thing to remember: a RAG chain without a `ModelSignature` will fail schema validation at Model Serving time — the signature is not optional decoration, it is the deployment contract.**

---

## ELI5 — Explain It Like I'm 5

Imagine you work at a library reference desk. When a customer asks a question, you do not answer from memory; instead you search the shelves for the most relevant books, photocopy the relevant pages, clip them to the question, and hand the whole packet to an expert who writes a final answer. RAG works exactly like this: the query is the customer question, the embedding model converts it into a shelf-location code, Databricks AI Search is the shelf-finding robot, the retrieved document chunks are the photocopied pages, and the LLM is the expert who reads those pages before writing a response. The most common misconception is that the LLM itself is "retrieving" from its training data — it is not; retrieval happens before the LLM ever sees the question.

---

## Learning Objectives

By the end of this chapter you will be able to:
- [ ] Trace the full RAG pipeline from a user query through embedding, retrieval, context assembly, and LLM generation
- [ ] Configure Databricks AI Search as the retriever component in a RAG chain
- [ ] Construct an `mlflow.models.ModelSignature` using `ColSpec` for RAG input/output schemas
- [ ] Log a RAG chain with `mlflow.langchain.log_model()` or `mlflow.pyfunc.log_model()` using an explicit signature
- [ ] Explain why a missing or incorrect `ModelSignature` breaks Model Serving compatibility

---

## Visual Overview

### Full RAG Pipeline Flow

```
User Query (string)
       │
       ▼
Embedding Model (e.g. DBRX, BGE, text-embedding-3-small)
  converts query → dense vector
       │
       ▼
Databricks AI Search (Vector Index)
  ANN similarity search ──► top-k document chunks + metadata
       │
       ▼
Context Assembly / Prompt Template
  [system prompt] + [retrieved chunks] + [user query]
       │
       ▼
LLM (Foundation Model API / DBRX / Llama)
  generates grounded response
       │
       ▼
Response + source_documents (output)
```

### MLflow Model Signature Structure

```
ModelSignature
├── inputs  (Schema)
│   └── ColSpec(name="query", type=DataType.string)
└── outputs (Schema)
    ├── ColSpec(name="response",         type=DataType.string)
    └── ColSpec(name="source_documents", type=DataType.string, required=False)
```

### Databricks AI Search Index Types

```
Delta Table (source of truth)
       │
       ▼
AI Search Index
├── Delta Sync Index  ──► auto-syncs when Delta table changes
│                         uses Managed Embeddings OR pre-computed vectors
└── Direct Vector Access Index
                          ──► caller upserts vectors directly via API
                              used when embeddings are computed externally
```

### RAG Chain Logging and Deployment Path

```
Author chain (LangGraph / LangChain / PyFunc)
       │
       ▼
mlflow.langchain.log_model()  OR  mlflow.pyfunc.log_model()
  ├── python_model / lc_model  (code or object)
  ├── signature=ModelSignature(inputs, outputs)
  └── resources=[DatabricksVectorSearchIndex(...), DatabricksServingEndpoint(...)]
       │
       ▼
MLflow Run  ──► Unity Catalog Model Registry
       │
       ▼
Databricks Model Serving endpoint
  validates incoming requests against signature ──► invoke chain ──► return response
```

---

## Key Concepts

### The RAG Pipeline: Four Stages

**What is it?** RAG is a three-step augmentation pattern — Retrieval, Augmentation, Generation — that supplies an LLM with real-time external context before it generates a response. In practice the pipeline has four distinct stages: (1) query encoding, (2) retrieval, (3) context assembly, and (4) generation.

**How does it work under the hood?** At runtime, the user's query string is passed to an embedding model which converts it into a dense floating-point vector. That vector is sent to a vector index for approximate nearest-neighbor (ANN) search, which returns the top-k most semantically similar document chunks. Those chunks are formatted into a prompt template alongside the original query and any system instructions, then the assembled prompt is sent to an LLM. The LLM never "looks up" data itself — it only reads what was assembled for it.

**Where does it appear in Databricks?** The full pipeline is orchestrated using LangGraph (primary) or LangChain. The retriever step uses the `databricks-langchain` `DatabricksVectorSearch` retriever or the Databricks AI Search REST/Python SDK. The LLM step calls a Foundation Model API endpoint such as `databricks-meta-llama-3-3-70b-instruct`. The chain is logged to an MLflow Run and deployed via Model Serving.

---

### Embedding Models for Query Encoding

**What is it?** An embedding model maps text (or other data) to a dense vector in a high-dimensional space where semantically similar texts are geometrically close. In RAG, the same embedding model must be applied to both the document corpus at index time and to each user query at inference time.

**How does it work under the hood?** The model tokenises input text and produces a fixed-length vector (commonly 768 or 1536 dimensions). Cosine similarity or L2 distance between vectors approximates semantic similarity. If the query and the indexed documents were embedded using different models, the vectors live in incompatible spaces and similarity scores are meaningless — this is the most dangerous silent failure in RAG.

**Where does it appear in Databricks?** Databricks AI Search supports managed embeddings (you configure the embedding model name in the index definition and Databricks handles inference) as well as pre-computed embeddings (you supply vectors directly). Supported models include BGE, GTE, and any Foundation Model API endpoint. In LangChain code this appears as `DatabricksEmbeddings(endpoint="databricks-bge-large-en")`.

---

### Databricks AI Search as the Retriever Component

> ⚠️ **Fast-evolving:** Databricks Vector Search was renamed Databricks AI Search (verified 2026-06-25). The Python SDK namespace `databricks.vector_search` still exists but `databricks.ai_search` is the current name. Re-verify API surface before production use.

**What is it?** Databricks AI Search (formerly Vector Search) is a managed similarity-search service built into the Databricks Data Intelligence Platform. It stores embeddings alongside document metadata in an index that auto-syncs with a backing Delta table.

**How does it work under the hood?** The index uses the Hierarchical Navigable Small World (HNSW) algorithm for ANN search with L2 distance. Hybrid search combines vector similarity (cosine/L2) with keyword relevance (Okapi BM25) using Reciprocal Rank Fusion (RRF). Filtering is applied as a pre- or post-ANN step on metadata columns. When `Delta Sync` mode is active, inserts/updates to the backing Delta table are automatically reflected in the index with a configurable latency.

**Where does it appear in Databricks?** Create an index from the Catalog Explorer UI or via Python SDK: `client = VectorSearchClient(); client.create_delta_sync_index(...)`. Query with `index.similarity_search(query_vector=..., num_results=5)`. In LangChain use `DatabricksVectorSearch(index_name="catalog.schema.index", embedding=...).as_retriever(search_kwargs={"k": 5})`.

---

### Context Window Assembly and Prompt Template Injection

**What is it?** Context assembly is the step between retrieval and generation that formats the retrieved document chunks, the original query, and any system instructions into a single prompt that stays within the LLM's context window.

**How does it work under the hood?** Retrieved chunks are concatenated (typically with chunk separators) and truncated or ranked if they exceed the available token budget. A prompt template then wraps the context: `{system_prompt}\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:`. LangChain's `ChatPromptTemplate` and LangGraph's state graph handle this step declaratively. Exceeding the context window silently truncates the prompt from the left (oldest tokens) in most LLMs — chunks at the end are dropped without error.

**Where does it appear in Databricks?** In LangGraph this is a node that reads `state["query"]` and `state["docs"]`, formats the prompt, and writes to `state["prompt"]`. In LangChain it is a `ChatPromptTemplate` within a `RunnableSequence`. The context size budget is controlled by the LLM endpoint's `max_tokens` parameter.

---

### MLflow `ModelSignature` for RAG Chains

**What is it?** A `ModelSignature` is a formal schema declaration attached to an MLflow model that specifies the names and types of every input and output column. For RAG chains the signature defines: input `query` as `DataType.string`, and outputs `response` (string) and optionally `source_documents` (string, often serialised JSON).

**How does it work under the hood?** At log time, `ModelSignature(inputs=Schema([ColSpec(...)]), outputs=Schema([ColSpec(...)]))` is serialised into the `MLmodel` YAML file. When the model is loaded by Model Serving, the serving framework validates each incoming request payload against the input schema and each response against the output schema before returning it to the caller. A `ColSpec` holds a `name` and a `DataType` enum value (`string`, `long`, `double`, `boolean`, `binary`, `datetime`). `Schema` is simply a list of `ColSpec` (or `TensorSpec`) objects.

**Where does it appear in Databricks?** The signature is passed as the `signature=` parameter to `mlflow.langchain.log_model()` or `mlflow.pyfunc.log_model()`. It can be constructed manually with `ModelSignature(inputs=..., outputs=...)` or inferred automatically with `mlflow.models.infer_signature(input_example, prediction)`. Model Serving exposes the schema in the endpoint's OpenAPI spec.

---

### Logging a RAG Chain with MLflow

**What is it?** Logging is the act of persisting a trained or assembled chain as a versioned MLflow artifact alongside its signature, dependencies, and resource declarations. The two primary flavours for RAG are `mlflow.langchain.log_model()` (for LangChain/LangGraph chains) and `mlflow.pyfunc.log_model()` (for custom Python wrappers).

**How does it work under the hood?** Both functions wrap the chain in a flavour-specific serialisation layer and write to the active MLflow Run's artifact store. The `resources=` parameter declares Databricks-managed resources (vector search index, serving endpoint) needed at serving time, enabling automatic authentication passthrough so the deployed endpoint can call back into Databricks APIs without embedding credentials. Databricks recommends the "Models from Code" approach: pass a Python file path as `lc_model=` or `python_model=` and use `mlflow.models.set_model()` inside that file; the file is replayed in the serving environment to reconstruct the chain.

**Where does it appear in Databricks?** Inside a `with mlflow.start_run():` block. After logging, `mlflow.register_model()` moves the artifact to the Unity Catalog Model Registry. From there, the registered model is deployed as a Model Serving endpoint using `mlflow.deployments` or the Databricks SDK.

---

## Key Parameters / Configuration Knobs

| Parameter | What it controls | Decision rule |
|---|---|---|
| `num_results` / `k` in retriever | Number of document chunks returned per query | Start with `k=5`; increase to `10–20` if answer completeness is low; decrease if latency is high or context window is tight |
| `score_threshold` in `similarity_search` | Minimum similarity score for a chunk to be included | Set if you see irrelevant chunks in responses; a threshold of 0.6–0.7 (normalised) filters noise without over-restricting recall |
| `query_type` in AI Search (`ANN` vs `HYBRID`) | Whether to use vector-only or vector+keyword search | Use `HYBRID` when the corpus contains unique identifiers (SKUs, codes) that pure vector search misses |
| `ColSpec(required=...)` in `ModelSignature` | Whether the field must be present in every request/response | Mark output fields like `source_documents` as `required=False` to allow chains that omit provenance in some modes |
| `resources=` in `log_model()` | Declares Databricks resources for automatic auth passthrough | Always specify; omitting it causes the deployed endpoint to fail on any call back to a Databricks API |
| `model_config=` in `log_model()` | YAML config passed to the agent code at serving time | Use to externalise index names, endpoint names, and chunk sizes so the same chain code can serve multiple environments |

---

## Worked Example: Requirement → Decision

**Given:** A team is building an internal knowledge assistant that answers employee HR policy questions. Documents are stored in a Delta table `hr.docs.policies` which is updated weekly. Answers must cite the source document page. The model must be deployable via Databricks Model Serving and callable from an external HR portal.

**Step 1 — Identify the goal:** Deploy a RAG chain that accepts a plain-text question string and returns a grounded answer plus the page references used to generate it.

**Step 2 — Define inputs:** A single string field `query` representing the employee's question. The RAG chain needs the query to encode it and to populate the prompt template.

**Step 3 — Define outputs:** Two fields: `response` (the generated answer as a string) and `source_documents` (a JSON-serialised list of `{page_id, excerpt}` dicts, as a string column). Both must be schema-validated so the HR portal can parse the response reliably.

**Step 4 — Apply constraints:**
- Delta table updates weekly → use a Delta Sync index in continuous-sync mode so documents are always fresh.
- Answers must cite sources → chain must return `source_documents`; the MLflow signature must declare this field.
- External portal calls → Model Serving requires a valid `ModelSignature` to generate an OpenAPI spec.
- Authentication → the deployed endpoint needs access to the AI Search index; declare it in `resources=`.

**Step 5 — Select the approach:** Build a LangGraph chain with a retriever node (using `DatabricksVectorSearch`) and a generation node (using `ChatDatabricks`), log it with `mlflow.langchain.log_model()` passing an explicit `ModelSignature(inputs=Schema([ColSpec("query", DataType.string)]), outputs=Schema([ColSpec("response", DataType.string), ColSpec("source_documents", DataType.string, required=False)]))` and `resources=[DatabricksVectorSearchIndex("hr.docs.policies_index"), DatabricksServingEndpoint("databricks-meta-llama-3-3-70b-instruct")]`. Rationale vs alternatives: `mlflow.pyfunc.log_model()` would work but requires a manual `predict()` wrapper; `mlflow.langchain.log_model()` with a code path is preferred because the serving environment automatically reconstructs the chain, and LangGraph gives explicit state management for multi-hop retrieval if requirements expand.

---

## Implementation

```python
# Scenario: Build and log a production-ready HR policy RAG chain with an explicit
# MLflow signature so Databricks Model Serving can validate requests and generate
# an OpenAPI spec for the external HR portal.

import mlflow
from mlflow.models import ModelSignature
from mlflow.types.schema import ColSpec, DataType, Schema
from mlflow.models.resources import DatabricksVectorSearchIndex, DatabricksServingEndpoint

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from databricks_langchain import ChatDatabricks, DatabricksVectorSearch, DatabricksEmbeddings

# --- RAG chain components ---
embedding = DatabricksEmbeddings(endpoint="databricks-bge-large-en")

retriever = DatabricksVectorSearch(
    index_name="hr.docs.policies_index",
    embedding=embedding,
    text_column="page_content",
).as_retriever(search_kwargs={"k": 5})

prompt = ChatPromptTemplate.from_template(
    "You are an HR assistant. Use ONLY the context below to answer.\n\n"
    "Context:\n{context}\n\nQuestion: {query}\nAnswer:"
)

llm = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct", max_tokens=512)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# LangChain LCEL chain (used as a node in a LangGraph state graph in production)
rag_chain = (
    {"context": retriever | format_docs, "query": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- Explicit ModelSignature ---
signature = ModelSignature(
    inputs=Schema([ColSpec(name="query", type=DataType.string)]),
    outputs=Schema([
        ColSpec(name="response", type=DataType.string),
        ColSpec(name="source_documents", type=DataType.string, required=False),
    ]),
)

# --- Log to MLflow ---
with mlflow.start_run():
    model_info = mlflow.langchain.log_model(
        lc_model=rag_chain,
        name="hr_rag_chain",
        signature=signature,
        input_example={"query": "What is the parental leave policy?"},
        resources=[
            DatabricksVectorSearchIndex(index_name="hr.docs.policies_index"),
            DatabricksServingEndpoint(endpoint_name="databricks-meta-llama-3-3-70b-instruct"),
        ],
    )

print(f"Model URI: {model_info.model_uri}")

# --- Register to Unity Catalog ---
mlflow.set_registry_uri("databricks-uc")
uc_info = mlflow.register_model(
    model_uri=model_info.model_uri,
    name="prod.hr_assistant.policy_rag",
)
```

```python
# Scenario: Construct a ModelSignature manually with ColSpec when infer_signature
# is not available (e.g. during offline testing or for a custom pyfunc RAG wrapper).

from mlflow.models import ModelSignature
from mlflow.types.schema import ColSpec, DataType, Schema
import mlflow

# Explicit construction — always know what your chain accepts and returns
input_schema = Schema([
    ColSpec(name="query", type=DataType.string),
])

output_schema = Schema([
    ColSpec(name="response",         type=DataType.string),
    ColSpec(name="source_documents", type=DataType.string, required=False),
])

signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Inspect what was built — useful for debugging serving issues
print(signature)
# inputs:
#   ['query': string (required)]
# outputs:
#   ['response': string (required), 'source_documents': string (optional)]
```

```python
# Anti-pattern: Logging a RAG chain with no signature and assuming Model Serving
# will accept arbitrary dict inputs.

# WRONG — no signature means no schema validation; the endpoint's OpenAPI spec
# has no type information, external callers cannot discover the contract, and any
# type mismatch produces a cryptic 500 error instead of a clear 400 with field names.

with mlflow.start_run():
    mlflow.langchain.log_model(
        lc_model=rag_chain,
        name="hr_rag_chain_bad",
        # signature= is omitted entirely
    )

# Correct approach: always supply an explicit signature or use infer_signature()
# with a representative input_example.

from mlflow.models import infer_signature

sample_input  = {"query": "What is the remote work policy?"}
sample_output = {"response": "Employees may work remotely up to 3 days per week.",
                 "source_documents": '[{"page_id": "P-42", "excerpt": "..."}]'}

signature = infer_signature(sample_input, sample_output)

with mlflow.start_run():
    mlflow.langchain.log_model(
        lc_model=rag_chain,
        name="hr_rag_chain_good",
        signature=signature,
    )
# Now Model Serving knows: input is {"query": string}, output is {"response": string, ...}
```

---

## Common Pitfalls & Misconceptions

- **Mismatched embedding models at index and query time** — Beginners embed documents with one model (e.g. BGE) during index creation but forget to use the same model when encoding the query at inference time, because the pipeline is written in two separate places. The correct mental model: the embedding model is a shared, immutable component of the index definition — treat it like a foreign key; the index and the retriever code must always reference the same endpoint name.

- **Treating `ModelSignature` as optional documentation** — Teams skip the signature to "iterate faster" during development, then discover the chain cannot be deployed to Model Serving or integrated with the AI Playground review app because both require a valid schema. The correct mental model: the signature is the deployment contract, not documentation — without it the chain does not satisfy the Databricks Agent Framework's interface requirements.

- **Assuming `source_documents` is automatically captured** — Beginners expect LangChain/LangGraph to automatically append retrieved chunks to the response object. The correct mental model: you must explicitly design the chain to surface `source_documents` in its output dict; a `StrOutputParser` discards all context and returns only the answer string.

- **Using the wrong log_model flavour** — Teams use `mlflow.sklearn.log_model()` or `mlflow.pyfunc.log_model()` for LangChain chains and encounter broken serialisation or missing flavour metadata. The correct mental model: LangChain/LangGraph chains must be logged with `mlflow.langchain.log_model()` which knows how to serialise LCEL runnables; `mlflow.pyfunc.log_model()` is correct for custom Python class wrappers that do not use LangChain natively.

- **Omitting `resources=` when logging** — The deployed endpoint fails with authentication errors when it tries to call back to the AI Search index or Foundation Model API, because the serving environment has no credentials to pass. The correct mental model: `resources=` at log time enables Databricks' automatic authentication passthrough at serving time; it is not a hint but a required declaration.

---

## Key Definitions

| Term | Definition |
|---|---|
| RAG (Retrieval-Augmented Generation) | A pattern that augments an LLM's prompt with documents retrieved from an external knowledge base at inference time, grounding the response without retraining the model |
| Embedding model | A model that maps text to a dense floating-point vector; the same model must be used at both index time and query time for similarity scores to be meaningful |
| Databricks AI Search | Managed vector search service built into Databricks; uses HNSW + BM25 + RRF for hybrid retrieval; formerly called Databricks Vector Search |
| Delta Sync Index | An AI Search index that automatically syncs with a backing Delta table when rows are added, updated, or deleted |
| Direct Vector Access Index | An AI Search index where callers upsert vectors directly via API; used when embeddings are computed outside Databricks |
| Context window | The maximum number of tokens an LLM can process in a single call; retrieved chunks must fit within the remaining budget after the system prompt |
| `ModelSignature` | An MLflow object (`mlflow.models.ModelSignature`) that formally declares the input and output schema of a model using `Schema` and `ColSpec` objects |
| `ColSpec` | A column specification (`mlflow.types.ColSpec`) that defines a named field with a `DataType` enum value; the building block of a `Schema` |
| `Schema` | An ordered list of `ColSpec` or `TensorSpec` objects that describes all columns in an MLflow model's input or output |
| `mlflow.langchain.log_model()` | MLflow flavour function that serialises a LangChain or LangGraph chain, records its dependencies, signature, and resources, and writes the bundle to the active MLflow Run |
| `mlflow.pyfunc.log_model()` | MLflow function for logging custom Python model wrappers; used when the chain is not natively a LangChain object |
| Automatic authentication passthrough | Databricks mechanism that forwards the calling user's identity to downstream Databricks APIs when `resources=` is declared at log time |

---

## Summary / Quick Recall

- RAG pipeline = encode query → retrieve top-k chunks → assemble prompt → generate response. The LLM does not retrieve; retrieval is a separate step.
- Databricks AI Search (formerly Vector Search) is the managed retriever; use Delta Sync for live-updating indexes.
- The same embedding model **must** be used at both index time and query time — a mismatch is a silent accuracy killer.
- `ModelSignature` = `ModelSignature(inputs=Schema([ColSpec(...)]), outputs=Schema([ColSpec(...)]))` — always declare it before logging.
- Log LangChain/LangGraph chains with `mlflow.langchain.log_model()`; log custom wrappers with `mlflow.pyfunc.log_model()`.
- Always include `resources=[DatabricksVectorSearchIndex(...), DatabricksServingEndpoint(...)]` to enable auth passthrough in the deployed endpoint.
- A missing signature causes a failed deployment, not a warning; Model Serving requires a schema to validate requests.

---

## Self-Check Questions

1. What does `ColSpec(name="query", type=DataType.string)` represent in an MLflow `ModelSignature`?

   <details><summary>Answer</summary>

   It defines a single named input column called `query` whose data type is `string`. In the `ModelSignature`, this `ColSpec` would be placed inside a `Schema` object that is passed as the `inputs=` parameter. The `ColSpec` is the atomic unit of a column-based MLflow schema; `DataType.string` is an enum value from `mlflow.types.DataType` that maps to Python `str` at runtime. The main distractor is confusing `ColSpec` with `TensorSpec` — `TensorSpec` is for numpy array inputs (e.g. image tensors), not for text fields in a RAG chain.

   </details>

2. You build a RAG chain using LangChain and log it with `mlflow.langchain.log_model()`. When you deploy to Model Serving, the endpoint returns a 500 error every time it tries to call the Databricks AI Search index. What is the most likely root cause and how do you fix it?

   <details><summary>Answer</summary>

   The most likely root cause is a missing `resources=` parameter in the `log_model()` call. Without it, Databricks' automatic authentication passthrough is not configured, and the serving environment has no credentials to authenticate to the AI Search index. The fix is to re-log the chain with `resources=[DatabricksVectorSearchIndex(index_name="catalog.schema.my_index")]` (and any other Databricks-managed resources such as LLM serving endpoints). An incorrect distractor answer is "the signature is wrong" — a wrong signature causes a 400 validation error, not a 500 on downstream calls.

   </details>

3. **Which TWO** of the following are valid reasons to use a Delta Sync Index (rather than a Direct Vector Access Index) in Databricks AI Search?

   - A. You want documents in the index to automatically update when the backing Delta table changes.
   - B. You compute embeddings in a custom GPU pipeline external to Databricks and need to upsert them directly.
   - C. You want Databricks to manage the embedding model inference during index creation.
   - D. You need to store vectors that were generated by a proprietary on-premises model.
   - E. You want to minimise the code required to keep the index fresh.

   <details><summary>Answer</summary>

   **A and C** (and implicitly E follows from A) are correct.

   **A** — Delta Sync Indexes auto-sync with the source Delta table; when rows change, the index stays current without any extra code. **C** — Delta Sync with managed embeddings lets Databricks call the specified embedding endpoint at sync time, so the developer does not write embedding inference code. **E** is a consequence of A, not a distinct reason, but acceptable.

   **B and D** are wrong: both describe situations where embeddings come from outside Databricks, which is the use case for *Direct Vector Access Index*, not Delta Sync.

   </details>

4. A data scientist uses `mlflow.models.infer_signature(input_example, output_example)` to build the signature, but the `output_example` dict only contains the `response` key and omits `source_documents`. The chain was later updated to also return `source_documents`. What is the consequence and how should it be handled?

   <details><summary>Answer</summary>

   The inferred signature will not include `source_documents` in the output schema. When the updated chain is deployed to Model Serving, the schema validation step will either silently drop the extra field or raise a validation error depending on the serving framework version — either way, the portal consuming the endpoint cannot rely on `source_documents` being schema-declared. The correct approach is to re-log the chain with a new signature inferred from (or manually declaring) the complete output including `source_documents` as `ColSpec(name="source_documents", type=DataType.string, required=False)`. Model versions are immutable — the fix requires creating a new model version with the corrected signature.

   </details>

5. Your team debates whether to use `mlflow.langchain.log_model()` or `mlflow.pyfunc.log_model()` for a RAG chain built entirely with LangGraph. What is the trade-off and which should you choose?

   <details><summary>Answer</summary>

   **Use `mlflow.langchain.log_model()`** for a LangGraph chain. The LangChain flavour natively serialises LCEL runnables and LangGraph `CompiledStateGraph` objects, records LangChain-specific metadata (input/output keys), and supports the `loader_fn` / `persist_dir` pattern for non-serialisable components like vector stores. `mlflow.pyfunc.log_model()` would require wrapping the graph in a `PythonModel` subclass that implements `predict()`, which is more boilerplate and forgoes the native flavour advantages. The trade-off: `pyfunc` gives maximum control (custom predict logic, arbitrary I/O format) but at the cost of more code and the loss of LangChain-specific serving optimisations. Choose `pyfunc` only when the chain is not a LangChain/LangGraph object or when you need a custom `predict()` signature that the LangChain flavour cannot express.

   </details>

---

## Further Reading

- [RAG (Retrieval-Augmented Generation) on Databricks](https://docs.databricks.com/aws/en/generative-ai/retrieval-augmented-generation) — *verified 2026-07-11*
- [Databricks AI Search (formerly Vector Search)](https://docs.databricks.com/aws/en/ai-search/ai-search) — *verified 2026-07-11*
- [Log and register AI agents (Model Serving)](https://docs.databricks.com/aws/en/agents/agent-framework/log-agent) — *verified 2026-07-11*
- [mlflow.langchain.log_model() API reference](https://mlflow.org/docs/latest/python_api/mlflow.langchain.html#mlflow.langchain.log_model) — *verified 2026-07-11*
- [mlflow.models.ModelSignature API reference](https://mlflow.org/docs/latest/python_api/mlflow.models.html#mlflow.models.ModelSignature) — *verified 2026-07-11*
- [mlflow.types.ColSpec and Schema API reference](https://mlflow.org/docs/latest/python_api/mlflow.types.html#mlflow.types.ColSpec) — *verified 2026-07-11*
