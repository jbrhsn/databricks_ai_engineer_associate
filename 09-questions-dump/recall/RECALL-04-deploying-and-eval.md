# RECALL-04: Deployment & Evaluation – Production Operations

These 10 questions test knowledge of model deployment, monitoring, and evaluation essential to Domains 4 (Assembling & Deploying), 5 (Governance), and 6 (Evaluation & Monitoring).

---

**Q31: What is Databricks Model Serving?**

   A. A process for choosing between multiple LLMs
   B. A managed service for deploying and scaling models (LLMs, classifiers, RAG applications) with REST endpoints and automatic scaling
   C. A tool for storing model artifacts
   D. An alternative to Delta Lake for model storage

A: **B**. Databricks Model Serving is a fully managed endpoint service that deploys MLflow models or external models, provides scalable REST API access, handles batching and autoscaling, and integrates with Databricks AI Gateway for cost tracking and governance.

---

**Q32: What is provisioned throughput in Databricks Model Serving?**

   A. The maximum file size that can be uploaded
   B. A reserved capacity option that guarantees consistent inference latency and throughput for production workloads
   C. A method for caching embeddings
   D. The bandwidth available for downloading models

A: **B**. Provisioned throughput reserves dedicated compute capacity for a Model Serving endpoint, ensuring predictable latency and throughput for critical workloads. This contrasts with on-demand endpoints that scale up based on traffic but may have variable latency.

---

**Q33: What is an inference table?**

   A. A SQL table that stores customer invoices
   B. A table that automatically logs predictions, inputs, and outcomes from a Model Serving endpoint for monitoring and evaluation
   C. A type of embedding index
   D. A Databricks workspace permission table

A: **B**. An inference table is a Delta table that captures every prediction made by a Model Serving endpoint along with inputs, outputs, timestamps, and actual labels (when available). It enables monitoring for data drift, performance degradation, and model debugging.

---

**Q34: What is MLflow Tracing?**

   A. A debugging tool for SQL queries
   B. A feature that records the step-by-step execution of LLM calls, tool invocations, and agent decisions for debugging and auditing
   C. A method for compressing models
   D. A feature of Delta Lake

A: **B**. MLflow Tracing (part of MLflow 2.8+) captures granular execution traces: each LLM call, tool invocation, agent decision, and output is logged with inputs, outputs, and latency, enabling visibility into complex agentic workflows and RAG pipelines.

---

**Q35: What does `mlflow.evaluate()` do?**

   A. Runs a syntax check on Python code
   B. Computes evaluation metrics (accuracy, F1, custom metrics) and LLM-as-judge scores, comparing a model against ground truth or multiple models
   C. Deploys a model to production
   D. Uploads logs to the cloud

A: **B**. `mlflow.evaluate()` is a function that computes built-in metrics (accuracy, NDCG, etc.) or custom metrics against a model and dataset, and can use LLM-as-judge evaluators (e.g., relevance, toxicity) to assess model output quality without ground truth.

---

**Q36: What is a judge in MLflow evaluation?**

   A. A person who manually reviews model outputs
   B. An LLM-based evaluator that scores model outputs on criteria like relevance, faithfulness, or toxicity
   C. A permission role in Databricks
   D. A tool for code review

A: **B**. A judge is an LLM-based evaluator (e.g., Claude, GPT-4) configured to score model outputs on specified criteria (relevance, hallucination, bias). Judges enable scalable, automated evaluation without ground truth labels and are commonly used in RAG evaluation.

---

**Q37: What is groundedness in RAG evaluation?**

   A. A metric for measuring how grounded the data center is
   B. A measure of whether the model's generated answer is supported by the retrieved documents (no hallucination)
   C. A database consistency check
   D. A type of embedding metric

A: **B**. Groundedness (or faithfulness) evaluates whether the LLM's answer is supported by the retrieved context. High groundedness means the answer reflects the documents; low groundedness indicates hallucination or off-topic generation.

---

**Q38: What is context recall in RAG evaluation?**

   A. The number of retrievals performed per second
   B. A metric measuring whether all relevant documents needed to answer the question were successfully retrieved
   C. The time it takes to recall information from cache
   D. A measure of how well the embedding model works

A: **B**. Context recall measures how many of the ground-truth relevant documents for a question were actually retrieved by the RAG system. High context recall ensures the retriever is not filtering out critical information, enabling the LLM to answer accurately.

---

**Q39: What is the purpose of Delta Live Tables?**

   A. To store and serve pre-computed embeddings
   B. To define and orchestrate data pipelines that auto-refresh, with built-in quality assurance and schema enforcement
   C. To monitor model serving endpoints
   D. To store vector embeddings in Delta format

A: **B**. Delta Live Tables (DLT) is Databricks' declarative framework for building, testing, and maintaining data pipelines. Pipelines auto-refresh on new data, enforce quality (data expectations), track lineage, and use Delta for ACID guarantees.

---

**Q40: What is the primary purpose of Lakehouse Monitoring?**

   A. To track how many users access the Lakehouse
   B. To detect data quality issues, drift, and schema changes in production tables
   C. To monitor user permissions in Unity Catalog
   D. To track MLflow experiment runs

A: **B**. Lakehouse Monitoring (integrated with Unity Catalog) automatically monitors production tables for data drift, missing values, schema changes, and anomalies using statistical baselines and rules. It alerts data teams to quality issues in near real-time.

---
