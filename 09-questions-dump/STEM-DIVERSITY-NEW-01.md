# Databricks Generative AI Engineer Associate — Practice Questions (Stem Diversity: Negative Framing, Direct Technical, Non-Persona)

*Focus: Exam-realistic stem formats—negative framing, direct technical questions, non-persona comparisons. Difficulty: medium-to-hard. 15 questions, all single-select.*

---

## Pattern 1: Negative Framing (5 questions)

**Q1: Which of the following will NOT reduce query latency in a Vector Search index?**

   A. Increasing the number of probes during approximate nearest neighbor search
   B. Reducing the dimensionality of embeddings through quantization
   C. Increasing the overlap between chunks in the source documents before indexing
   D. Adding more hardware resources (compute) to the Vector Search cluster

A: **C**. Increasing chunk overlap does NOT affect query latency—it is an ingestion-time parameter that affects how many chunks exist, not how fast queries execute against an already-built index. Latency is determined by search algorithm efficiency (A: more probes = slower, fewer probes = faster), embedding size (B: smaller embeddings search faster), and infrastructure (D: more compute resources reduce contention). Chunk overlap affects downstream quality (more overlapping chunks may improve retrieval context) but has no direct relationship to query speed in an already-indexed system.

---

**Q2: Which approach will NOT help prevent an agent from entering an infinite loop?**

   A. Adding a maximum step limit to the agent's execution
   B. Implementing state tracking to detect repeated tool calls
   C. Increasing the temperature parameter of the LLM generating tool decisions
   D. Designing tools that return clear success/failure signals to break cycles

A: **C**. Increasing temperature does NOT prevent infinite loops—it actually makes them *more* likely by introducing randomness into the LLM's decisions, potentially making outputs less deterministic and harder to control. Temperature affects output diversity, not loop prevention. Effective loop prevention requires: maximum step limits (A: hard stops after N steps), state tracking (B: detect when the same tool is called repeatedly), and clear tool feedback (D: explicit signals telling the agent when it has succeeded or failed). Randomness via temperature is orthogonal to loop logic.

---

**Q3: Which metric will NOT improve if you fix a chunking boundary misalignment issue?**

   A. Retrieval recall (the proportion of relevant chunks actually retrieved)
   B. Answer faithfulness (whether generated answers are grounded in retrieved context)
   C. The size of the embedding model (measured in parameters)
   D. Context relevance (whether retrieved chunks directly support the final answer)

A: **C**. Embedding model size is a characteristic of the model itself and is completely orthogonal to chunking strategy. Fixing chunk boundaries does NOT change the embedding model's size; it is already fixed before chunking happens. Conversely, better chunking directly improves: recall (A: proper boundaries ensure relevant chunks are found), faithfulness (B: better chunks reduce hallucination risk), and context relevance (D: well-aligned chunks are more on-topic). Model size is an independent design decision unaffected by chunking adjustments.

---

**Q4: Which step is NOT required when deploying a LangChain model via `mlflow.langchain.log_model()`?**

   A. Specifying the model's input and output schema (signature) for serving endpoints to validate requests
   B. Retraining the model with additional data
   C. Logging the model's dependencies (packages, versions) so serving environments can install them
   D. Specifying the model artifact path or path argument so MLflow knows where the model is stored

A: **B**. Retraining is NOT required during deployment—`mlflow.langchain.log_model()` logs an *already-trained* model as an artifact; deployment uses the existing trained weights without modification. All other steps are mandatory: you must specify the schema (A: so the serving endpoint validates inputs/outputs), log dependencies (C: so the serving environment can be reconstructed), and reference the artifact location (D: so MLflow can find the model to serve). Deployment is about packaging and serving the existing model, not retraining it.

---

**Q5: Which judge in MLflow Eval will NOT provide useful feedback if your retrieval quality degrades?**

   A. `answer_relevance` alone, without cross-referencing with a groundedness judge
   B. `faithfulness` (groundedness), which checks whether answers are supported by retrieved context
   C. `retrieval_precision`, which measures the proportion of retrieved chunks that are relevant
   D. All three judges above will provide diagnostic value

A: **A**. `answer_relevance` alone is insufficient for diagnosing retrieval degradation—it only measures whether an answer is topically relevant to the question, not whether it is grounded in the retrieved context. If retrieval fails, an LLM may still generate a topically relevant answer via hallucination, and `answer_relevance` will score it as high. You need `faithfulness` (B: detects when answers are hallucinated vs. grounded) and `retrieval_precision` (C: directly measures whether retrieved chunks are on-topic). Together, faithfulness + precision pinpoint retrieval problems; `answer_relevance` alone masks them.

---

## Pattern 2: Direct Technical (5 questions)

**Q6: What does `RunnableParallel(a=runnable_a, b=runnable_b).invoke(x)` return?**

   A. A list of two results: `[runnable_a(x), runnable_b(x)]`
   B. A dictionary with keys 'a' and 'b' and values `runnable_a(x)` and `runnable_b(x)` respectively
   C. A single merged output from both runnables combined in sequence
   D. An error, because `RunnableParallel` does not support the `invoke()` method

A: **B**. `RunnableParallel` with named runnables returns a dict with the same key names and values as each runnable's output. When you call `.invoke(x)` on `RunnableParallel(a=runnable_a, b=runnable_b)`, it executes both runnables in parallel (or concurrently) and returns `{'a': runnable_a(x), 'b': runnable_b(x)}`. A is wrong because the output is a dict, not a list. C is wrong because parallel composition does not merge or combine results sequentially; it invokes both independently. D is wrong because `RunnableParallel` fully supports `invoke()`.

---

**Q7: What parameter of `tiktoken.encoding_for_model()` specifies the exact model whose tokenization you want?**

   A. The `model_name` keyword argument
   B. The first positional argument (string) passed to the function
   C. The `encoding_type` keyword argument
   D. You cannot specify the model directly; `tiktoken` always infers it from your environment

A: **B**. The first positional argument to `tiktoken.encoding_for_model()` is the model name (a string like `"gpt-4"` or `"text-davinci-003"`), which uniquely determines which tokenizer is returned. For example: `tiktoken.encoding_for_model("gpt-4")` returns GPT-4's tokenizer. A is wrong because `tiktoken.encoding_for_model()` does not accept a `model_name` keyword argument—the model is a positional argument. C is wrong because `encoding_type` is not a parameter of this function. D is wrong because you *must* explicitly specify the model as an argument.

---

**Q8: What is the type of a LangGraph state when defined as `class State(TypedDict): messages: Annotated[list, operator.add]`?**

   A. A regular Python dictionary with no special update semantics
   B. A list of messages, where updates to the state *concatenate* new messages to the existing list via the `operator.add` reducer
   C. A tuple, because `Annotated` enforces immutability
   D. An error; LangGraph states cannot use `Annotated` with reducers

A: **B**. When you annotate a field with `Annotated[list, operator.add]`, LangGraph treats that field as a list with a *reducer* that concatenates updates using `operator.add`. Each time the state is updated with a new value for `messages`, the new value is added to (concatenated with) the existing list instead of replacing it. This is the core mechanism of stateful graph nodes—they accumulate messages over multiple steps. A is wrong because `Annotated` adds special semantics beyond a plain dict. C is wrong because `Annotated` does not enforce immutability; it specifies a reducer. D is wrong because `Annotated` with reducers is a standard LangGraph pattern.

---

**Q9: What does `SELECT COUNT(*) FROM vector_search_index_name` return in a Databricks SQL query?**

   A. An error, because Vector Search indexes cannot be queried directly via SQL
   B. The number of embeddings (vectors) stored in the index
   C. A nested structure representing the index metadata and all vectors
   D. NULL, because Vector Search indexes are not exposed as SQL tables

A: **B**. In Databricks, a Vector Search index can be queried as a Delta table via SQL. `SELECT COUNT(*) FROM vector_search_index_name` returns an integer count of the total number of rows (embeddings) in the index. A is wrong because Vector Search indexes *are* queryable as SQL tables (they are backed by Delta tables). C is wrong because `COUNT(*)` returns a single integer, not a nested structure. D is wrong because Vector Search indexes *are* exposed as SQL tables in Databricks and can be accessed like any other Delta table.

---

**Q10: What does `infer_signature()` return when called on a model's predictions?**

   A. A string describing the model name and version
   B. A `ModelSignature` object that captures the input schema, output schema, and their types
   C. A dictionary mapping input names to output values
   D. A boolean indicating whether the signature was successfully inferred

A: **B**. `mlflow.models.infer_signature()` returns a `ModelSignature` object—a structured representation of the model's input and output schemas. For example, if you call `infer_signature(X, y)` on features X and predictions y, it returns a signature that MLflow uses for validation when serving the model. A is wrong because `infer_signature()` does not return a string identifier. C is wrong because it returns a schema object, not a mapping of values. D is wrong because it returns an object, not a boolean flag.

---

## Pattern 3: Non-Persona Comparison (5 questions)

**Q11: When should fine-tuning be preferred over RAG for a knowledge-intensive task?**

   A. When the knowledge domain is constantly changing and you want to avoid frequent retraining
   B. When the knowledge domain is stable, small, and benefits from learned reasoning patterns that blend domain knowledge with task-specific behavior
   C. When you want to minimize latency by avoiding vector search lookups
   D. When you want maximum flexibility to swap out knowledge sources without model changes

A: **B**. Fine-tuning is preferable when the knowledge domain is *stable and compact* and the task requires learned reasoning—patterns that cannot be simply retrieved and inserted into a prompt. Examples: specialized coding tasks where reasoning about *how* to apply knowledge matters as much as the knowledge itself, or domain-specific linguistic styles where the model must learn the register alongside the facts. RAG is better for large, volatile knowledge bases; fine-tuning is better for compact, stable domains where the model learns to reason with the knowledge. A is wrong because constant changes favor RAG (easier to update a vector store than retrain). C is wrong because fine-tuning does not necessarily reduce latency—it may actually increase it due to increased model size. D is wrong because fine-tuning binds knowledge into model weights, making it hard to swap sources; RAG is far more flexible for this.

---

**Q12: What is the PRIMARY distinction between a smaller embedding model and a larger one in terms of retrieval quality?**

   A. Smaller models are always faster but always less accurate; larger models are always slower but always more accurate
   B. Larger models encode more semantic nuance and are more accurate on nuanced queries, but have higher latency and memory cost; smaller models are faster and cheaper but may miss semantic subtleties
   C. Embedding model size is unrelated to retrieval quality; only the vector dimension matters
   D. Smaller models are more accurate because they are simpler; larger models overfit and perform worse on novel queries

A: **B**. Larger embedding models capture finer semantic distinctions (more "understanding"), leading to better retrieval on complex, nuanced queries. Smaller models are faster and cheaper but may conflate semantically distinct concepts. There is a clear trade-off: accuracy vs. speed/cost. For a high-stakes task (e.g., legal retrieval), a larger model justifies the latency. For high-volume, speed-critical retrieval, a smaller model may be preferable even if accuracy is slightly lower. A is wrong because the relationship is not absolute—context matters; sometimes smaller models are sufficient. C is wrong because model size and dimension are related (larger models typically use higher-dimensional embeddings), and larger models *do* improve quality. D is wrong because larger models generalize better, not worse; overfitting is managed by training practices, not model size alone.

---

**Q13: When is `ConversationSummaryMemory` preferable to `ConversationBufferMemory` in a multi-turn agent?**

   A. When you want to preserve every single message for legal compliance auditing
   B. When the conversation is short (a few turns) and context window is not a constraint
   C. When conversation history is long and token budgets are tight, so you need to compress history while retaining key facts
   D. When the agent has no access to an LLM (summary memory does not require LLM calls)

A: **C**. `ConversationSummaryMemory` periodically summarizes old messages using an LLM, compressing history while retaining key facts. This is ideal for long conversations where storing every message would exhaust the context window. `ConversationBufferMemory` keeps all messages verbatim, which is simple but expensive for long histories. A is wrong because legal compliance typically requires *verbatim* buffers, not summaries (summaries lose information). B is wrong because with short conversations, regular buffer memory is fine and simpler; summaries add unnecessary complexity. D is wrong because `ConversationSummaryMemory` *requires* LLM calls to generate summaries—that is its core mechanism.

---

**Q14: Which Vector Search index type (Delta Sync vs. Direct Vector Access) is MOST appropriate when you want to manage embeddings outside Databricks?**

   A. Delta Sync, which syncs embeddings from a Databricks Delta table to the index
   B. Direct Vector Access, which allows you to upload embeddings from external systems directly to the index without requiring a Delta table as a source
   C. Both are equally suitable; the choice does not affect where embeddings come from
   D. Neither; Databricks Vector Search requires embeddings to be generated *within* Databricks

A: **B**. Direct Vector Access allows you to ingest embeddings (vectors) from external systems—you compute embeddings elsewhere and push them directly into the Vector Search index without requiring a Databricks-managed Delta table as the authoritative source. Delta Sync, by contrast, synchronizes embeddings from a Delta table in Databricks, which means the source of truth for documents *and* embeddings is Databricks. A is wrong because Delta Sync binds you to a Databricks Delta table as the source. C is wrong because the two types have fundamentally different data-flow patterns. D is wrong because Direct Vector Access explicitly supports external embedding sources.

---

**Q15: When is inference table sampling BETTER than continuous full monitoring for production RAG quality?**

   A. When you want to catch every single quality regression, no matter how small
   B. When you have unlimited budget and want to log every inference for real-time diagnostics
   C. When application traffic is very high and you want to reduce monitoring cost while still detecting meaningful quality issues through statistically representative samples
   D. When you do not care about quality and just want to minimize overhead

A: **C**. Inference table sampling (periodically capturing a representative subset of inferences) is cost-effective for high-volume systems—you can detect quality trends and regressions without logging and analyzing *every* inference. This works well when sample size is large enough to be statistically meaningful. Continuous full monitoring is necessary only when regressions must be caught immediately (e.g., critical systems) or when the traffic volume is low enough that logging everything is affordable. A is wrong because sampling may miss very small, gradual regressions (though in practice a well-designed sample catches meaningful problems). B is wrong because full continuous monitoring is expensive and unnecessary for many use cases. D is wrong because it's a cynical frame—sampling is about balancing cost and detection quality, not about not caring.

---
