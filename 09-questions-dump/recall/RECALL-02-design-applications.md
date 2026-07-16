# RECALL-02: Design Applications – Techniques & Patterns

These 10 questions test knowledge of prompting strategies, application design patterns, and architectural concepts essential to Domain 1 (Design Applications).

---

**Q11: What is the primary purpose of a system prompt?**

   A. To define the model's personality, role, constraints, and response format
   B. To store database connection strings
   C. To tune model parameters like temperature
   D. To create inference tables for monitoring

A: **A**. A system prompt (or system message) defines the model's behavior: its role (e.g., "You are a technical support agent"), personality, constraints (e.g., "Do not provide medical advice"), and output format. It shapes every response the model generates and is a core component of prompt engineering.

---

**Q12: What is prompt engineering?**

   A. The process of optimizing SQL queries in Delta Lake
   B. The practice of designing, testing, and refining prompts to elicit desired outputs from language models
   C. The setup of LLM infrastructure on Kubernetes
   D. The process of retraining a foundation model on custom data

A: **B**. Prompt engineering is the discipline of crafting effective prompts through iterative design, testing different phrasings, adding examples, using techniques like chain-of-thought, and refining instructions to maximize model output quality. It does not involve retraining weights or SQL optimization.

---

**Q13: What is few-shot learning in the context of language models?**

   A. Training a model with only a few GPU accelerators
   B. Providing a small number of input–output examples in the prompt to guide the model's behavior on similar tasks
   C. Using a model that has been trained on less than 1 billion parameters
   D. Fine-tuning a model on fewer than 1,000 samples

A: **B**. Few-shot learning is an in-context learning technique: you include a small number of labeled examples (typically 2–5) in the prompt to demonstrate the desired output format and reasoning pattern. The model learns to mimic that pattern without updating its weights. This differs from fine-tuning and is distinct from model size.

---

**Q14: What is chain-of-thought prompting?**

   A. A technique for connecting multiple LLM calls in sequence
   B. A method that instructs the model to show step-by-step reasoning before answering, improving accuracy on complex tasks
   C. A way to chain Delta Lake tables together for joins
   D. A feature of the Databricks SQL IDE

A: **B**. Chain-of-thought (CoT) prompting asks the model to "think step by step" or "show your reasoning," breaking complex reasoning into intermediate steps. This technique significantly improves model accuracy on tasks requiring math, logic, or multi-hop reasoning.

---

**Q15: What is an agentic application?**

   A. A centralized system for managing user permissions
   B. An application that uses an LLM as a reasoning engine to decide which tools to call and when
   C. A tool for versioning Delta Lake schema changes
   D. A monitoring solution for model performance

A: **B**. An agentic application uses an LLM in a loop: the LLM reasons about the user's request, decides which tools (APIs, databases, calculators) to use, executes them, and refines its approach based on results. This enables dynamic, multi-step workflows beyond simple prompt-response pairs.

---

**Q16: What is the key difference between RAG and fine-tuning?**

   A. RAG is free; fine-tuning costs money
   B. RAG retrieves external documents at runtime; fine-tuning updates model weights offline to incorporate domain knowledge
   C. RAG only works with text; fine-tuning works with images
   D. They are identical approaches to the same problem

A: **B**. RAG (Retrieval-Augmented Generation) retrieves relevant documents at inference time and passes them to the LLM as context, keeping the model frozen. Fine-tuning updates the model's weights on domain-specific data during training, baking knowledge into the model. RAG is faster to set up and update; fine-tuning is more permanent but requires training infrastructure.

---

**Q17: What is a retrieval pipeline in the context of RAG?**

   A. A set of database joins used to fetch documents
   B. The end-to-end process of storing documents, embedding them, indexing them, and retrieving relevant ones based on a user query
   C. A tool for orchestrating Databricks jobs
   D. A feature of MLflow for versioning datasets

A: **B**. A retrieval pipeline encompasses: document ingestion and preprocessing, chunking, embedding each chunk, storing vectors in a database (like Vector Search), and at query time—embedding the user query and retrieving the top-k most similar vectors. Together, these steps form a complete RAG pipeline.

---

**Q18: What is a tool in the context of agents?**

   A. A piece of software for code editing
   B. A function or API that an agent can call to take action (e.g., search the web, query a database, send an email)
   C. A UI component in Databricks Workspace
   D. A unit of GPU memory

A: **B**. In agentic frameworks (LangGraph, LangChain), a tool is an action an agent can invoke—a function, API endpoint, or database query. The LLM decides when and how to call tools based on the user's request, creating dynamic workflows.

---

**Q19: What is context window?**

   A. The maximum number of tokens an LLM can process in a single request (input + output)
   B. The time it takes to load a model into GPU memory
   C. A Databricks UI panel for viewing logs
   D. The size of a chunk in a chunking strategy

A: **A**. Context window (or maximum sequence length) is the upper limit on the total number of tokens the model can accept in one request, including both the input prompt and generated output. GPT-4 has a 128K token context window; older models like GPT-3.5 have 4K–16K.

---

**Q20: What is model hallucination?**

   A. A security breach in Databricks
   B. When a model generates plausible-sounding but factually incorrect or made-up information
   C. A visual artifact in image models
   D. A type of overfitting during training

A: **B**. Hallucination occurs when an LLM generates false or fabricated information that it presents confidently as fact. It is a common limitation of LLMs and is one key reason RAG is valuable—grounding answers in retrieved documents reduces hallucination.

---
