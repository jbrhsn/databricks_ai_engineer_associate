# RECALL-01: Foundations – Core Concepts & Products

These 10 questions test basic knowledge of foundational GenAI concepts and Databricks products essential to all subsequent domains.

---

**Q1: Which Databricks product provides a serverless, managed vector database for semantic search?**

   A. Delta Lake
   B. Databricks Vector Search
   C. Databricks SQL
   D. MLflow Model Registry

A: **B**. Databricks Vector Search is Databricks' serverless, managed vector database that stores and indexes embeddings for semantic search operations. Delta Lake is the underlying storage format; Databricks SQL is for traditional SQL queries; and MLflow Model Registry is for versioning models.

---

**Q2: What does RAG stand for in the context of generative AI?**

   A. Retrieval-Augmented Generation
   B. Recurrent Attention Gradient
   C. Real-time Analytics Grid
   D. Regression-Aligned Grouping

A: **A**. Retrieval-Augmented Generation (RAG) is a technique that combines document retrieval with LLM generation to produce answers grounded in external data. This approach reduces hallucination and enables applications to answer questions about proprietary or recent information.

---

**Q3: What is a foundation model?**

   A. A model that can only perform one specific task
   B. A large, pre-trained model on broad data that can be adapted to multiple downstream tasks
   C. A model stored in MLflow Model Registry
   D. A model that requires manual labeling for every use case

A: **B**. A foundation model is a large-scale, pre-trained neural network (such as GPT-4 or Claude) trained on diverse, broad data that can be adapted via prompt engineering, fine-tuning, or RAG for many different applications. Task-specific models are typically smaller and trained for one purpose.

---

**Q4: What is the primary purpose of Delta Lake in Databricks?**

   A. To store vector embeddings only
   B. To provide a serverless SQL interface
   C. To provide ACID transactions, schema enforcement, and unified data governance on cloud object storage
   D. To replace MLflow entirely

A: **C**. Delta Lake is an open-source storage format built on Parquet that adds ACID transaction guarantees, schema enforcement, data governance, and time-travel capabilities to cloud storage (S3, ADLS, GCS). It is the foundation for Databricks' Lakehouse architecture, not specific to vectors.

---

**Q5: What is LangChain in the context of GenAI application development?**

   A. A blockchain framework for secure AI
   B. An open-source framework providing components (chains, agents, retrievers) to build applications with language models
   C. A Databricks-exclusive product
   D. A method for chaining multiple databases together

A: **B**. LangChain is an open-source Python/JavaScript framework that provides modular components (LLMs, retrievers, chains, agents, memory) to simplify building applications that use language models. It is not Databricks-exclusive and is widely used across the industry.

---

**Q6: What does LLM stand for?**

   A. Large Language Model
   B. Localized Learning Module
   C. Language Logic Machine
   D. Linear Logical Mapping

A: **A**. LLM stands for Large Language Model—a neural network trained on massive amounts of text data that can perform a wide variety of language understanding and generation tasks. Examples include GPT-4, Claude, Llama, and Mistral.

---

**Q7: What is Databricks AI Gateway?**

   A. A single interface to manage access and governance for multiple foundation model providers
   B. A tool for storing Delta tables
   C. A replacement for Vector Search
   D. An on-premises only product

A: **A**. Databricks AI Gateway (part of the Databricks GenAI Stack) provides a unified API and governance layer for accessing multiple foundation model providers (OpenAI, Anthropic, etc.) with cost tracking, rate limiting, and audit logging. It is cloud-based and reduces vendor lock-in.

---

**Q8: What is MLflow?**

   A. A tool for writing machine learning code
   B. An open-source platform for managing the ML lifecycle (tracking experiments, packaging models, and deploying them)
   C. A Databricks-exclusive database
   D. A prompt engineering library

A: **B**. MLflow is an open-source platform that provides tools for tracking experiments (MLflow Tracking), managing model versions (Model Registry), and deploying models (Model Serving). It covers the full ML lifecycle and is framework-agnostic.

---

**Q9: What is Unity Catalog in Databricks?**

   A. A vector storage system for embeddings
   B. A unified data governance and metadata management layer for all Databricks assets (tables, models, functions)
   C. A SQL query optimizer
   D. An alternative to Delta Lake

A: **B**. Unity Catalog is Databricks' centralized metadata and governance layer that provides a three-level namespace (catalog, schema, table/model), fine-grained access controls, lineage tracking, and audit logging across all data and models in the workspace.

---

**Q10: What is an embedding?**

   A. A document stored in Delta Lake
   B. A vector representation of text or images that captures semantic meaning for use in search and similarity tasks
   C. A type of SQL query
   D. A pre-trained foundation model

A: **B**. An embedding is a dense vector (typically 256–4096 dimensions) that represents the semantic meaning of a text passage, word, or image. Embeddings enable semantic search, clustering, and similarity comparisons because nearby vectors have similar meanings.

---
