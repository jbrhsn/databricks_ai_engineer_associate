# RECALL-03: Data Preparation & Application Development – Technical Foundations

These 10 questions test knowledge of data engineering, embedding, and framework-specific APIs essential to Domains 2 (Data Preparation) and 3 (Application Development).

---

**Q21: What is chunking in the context of RAG?**

   A. A method for compressing embeddings
   B. The process of splitting large documents into smaller, overlapping segments to fit within an LLM's context window and embedding model limits
   C. A Databricks job scheduling feature
   D. A compression algorithm for Delta tables

A: **B**. Chunking breaks documents into manageable pieces (e.g., 512–1024 tokens) with optional overlap to preserve context. Chunks are then individually embedded and stored, allowing the RAG pipeline to retrieve the most relevant segments rather than entire documents.

---

**Q22: What is semantic search?**

   A. Search that only works on English text
   B. Search based on meaning and similarity rather than keyword matching, using embeddings
   C. A full-text search feature in SQL
   D. A Databricks-exclusive technology

A: **B**. Semantic search uses embeddings to find documents by meaning. The query is embedded and compared (via vector similarity) to document embeddings to find contextually similar results, even if keywords don't match. This differs from keyword/lexical search.

---

**Q23: What is a token?**

   A. A security credential in Databricks
   B. The basic unit used by LLMs to represent text; one token ≈ 4 characters on average
   C. A type of Delta Lake transaction
   D. An embedding vector

A: **B**. A token is the atomic unit that LLMs process. A token roughly corresponds to 4 characters of English text (e.g., "hello" ≈ 1 token, "Databricks" ≈ 2 tokens). Understanding tokenization is critical for managing context window limits and costs.

---

**Q24: What does an embedding model return?**

   A. A classification label (e.g., "spam" or "not spam")
   B. A text summary of the input
   C. A dense vector representation capturing the semantic meaning of the input text
   D. A SQL table

A: **C**. An embedding model (e.g., sentence-transformers, OpenAI's text-embedding-3-small) takes text as input and returns a vector of fixed dimensionality (e.g., 768 or 3072 dimensions) that encodes its meaning. Vectors for semantically similar texts are close in vector space.

---

**Q25: What is LCEL in LangChain?**

   A. A type of SQL query language
   B. LangChain Expression Language—a declarative way to compose chains and runnables using pipes and methods
   C. A database indexing technique
   D. An acronym for a Databricks feature

A: **B**. LangChain Expression Language (LCEL) is a syntax for building composed workflows: `chain = prompt | model | output_parser`. It enables easy chaining of runnables and is the modern, recommended way to build LangChain applications.

---

**Q26: What is a LangGraph node?**

   A. A vertex in the computation graph representing a discrete step (task or decision)
   B. A type of embeddings database
   C. A Delta Lake table optimization
   D. A Databricks SQL query

A: **A**. In LangGraph, a node is a step in the graph where computation happens. Nodes are functions that take state and produce output; edges connect nodes and define transitions. Graphs enable complex, multi-step agentic workflows with branching and loops.

---

**Q27: What is RunnableLambda in LangChain?**

   A. A way to run code on cloud GPUs
   B. A wrapper that converts a plain Python function into a LangChain Runnable, making it composable with chains and LCEL
   C. An optimization technique for embeddings
   D. A Databricks feature for parallel processing

A: **B**. `RunnableLambda` wraps a Python function (e.g., `lambda x: x.upper()`) as a LangChain Runnable, allowing it to be used in LCEL chains with pipes and other runnables. This bridges custom logic and LangChain's composition model.

---

**Q28: What does `mlflow.log_model()` do?**

   A. Sends a model's logs to stdout
   B. Registers a model in MLflow Model Registry for versioning, deployment, and tracking
   C. Compresses a model for storage
   D. Logs hyperparameters during training

A: **B**. `mlflow.log_model()` saves a trained model artifact to MLflow (into the run's artifacts folder or Model Registry), associating it with experiment metadata, parameters, and metrics. This enables reproducibility, versioning, and deployment.

---

**Q29: What is a Runnable in LangChain?**

   A. An executable Python script
   B. A composable unit in LangChain (LLM, retriever, chain, custom function) that implements a standard interface (.invoke(), .stream(), .batch())
   C. A scheduled job in Databricks
   D. A type of Delta Lake format

A: **B**. A Runnable is LangChain's abstraction for any callable component (language models, retrievers, chains, custom functions). All Runnables support `.invoke()`, `.stream()`, `.batch()`, and can be composed with pipes and methods in LCEL.

---

**Q30: What is semantic chunking?**

   A. Splitting documents randomly
   B. Splitting documents based on meaning (e.g., sentences, topics) rather than fixed token counts, improving retrieval quality
   C. Compressing embeddings
   D. A Databricks Delta Lake feature

A: **B**. Semantic chunking uses NLP or model-based approaches to split documents at natural boundaries (sentence ends, paragraph breaks, topic shifts) rather than fixed-size windows. This preserves context better than naive chunking and often improves RAG retrieval quality.

---
