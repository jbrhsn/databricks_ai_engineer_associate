# Why Groundedness is the Only RAG Metric That Actually Matters (and How to Fix It With Reranking) — Thought Leadership

**Section:** 03-Data Preparation | **Target audience:** AI Engineering Leads, Solutions Architects | **Target publication:** LinkedIn, Engineering Blog

## Hook / Opening Thesis

Your enterprise RAG pipeline might be scoring 95% on correctness, but if its groundedness score is failing, you've just built a hallucination engine with a database attached. Factually correct answers that don't stem from the retrieved context are the most dangerous failure mode in modern AI applications.

## Key Claims

1. **Correctness is a deceptive metric in RAG.** If an LLM answers a question correctly using its pre-trained weights instead of the provided context, the system has fundamentally failed its architectural purpose.
2. **Bi-encoders create a false sense of security.** Returning documents with 0.85 cosine similarity does not mean the documents actually answer the user's specific query; it only means they share semantic neighborhood.
3. **Two-stage retrieval is non-negotiable for production.** Relying entirely on a fast vector search for final context selection guarantees that nuanced queries will fail, leaving the LLM to guess.

## Supporting Evidence & Examples

We recently audited an internal HR agent that passed 98% of its unit tests for "correctness." When we ran Databricks MLflow Agent Evaluation to measure groundedness, the score plummeted to 42%.

What was happening? Employees were asking complex questions like, "What is the parental leave policy for part-time contractors in the UK?" The vector search (bi-encoder) was quickly retrieving general "parental leave" documents, heavily biased toward full-time US employees. Because the underlying LLM (GPT-4) knew general facts about UK employment law, it formulated a legally plausible—and sometimes factually correct—answer. But it was entirely ungrounded from the company's actual policy documents. The agent was confidently guessing.

## The Original Angle

Most engineering teams treat RAG as a search problem: "How quickly can I find similar text?" But production RAG is an *extraction* problem. 

The industry obsession with scaling vector databases obscures the reality that similarity is not relevance. A bi-encoder compresses entire paragraphs into a single coordinate. It is structurally impossible for a bi-encoder to understand the difference between "UK parental leave for contractors" and "US contractor leave policies." Only by implementing a cross-encoder—where the attention heads can interactively compare the query tokens with the document tokens—can you bridge the gap between semantic similarity and actual relevance.

## Counterarguments to Address

*“Cross-encoders are too slow and expensive for real-time applications.”*

This is only true if you design the architecture poorly. You do not use a cross-encoder to search your entire corpus. You use your fast, cheap vector search to return the top 50 candidates (O(1) latency), and then apply the cross-encoder *only* to those 50 documents to select the final 3 for the LLM. The added latency is typically ~200-400ms, which is completely eclipsed by the time the LLM takes to generate the final response.

## Practical Takeaways for the Reader

- **Evaluate Groundedness First:** Use an LLM-as-a-judge framework like MLflow Agent Evaluation to measure whether the final answer can be traced directly back to the retrieved context. If it can't, treat it as a critical bug, even if the answer is "right."
- **Implement a Two-Stage Pipeline:** Stop trying to tune your vector search to be perfect. Let it be broad and fast (high recall), and add a reranking step (high precision) right before the LLM.
- **Stop Chasing Similarity Scores:** A cosine similarity of 0.9 means nothing in isolation. What matters is the document's rank relative to the user's explicit intent.

## Call to Action

Run a groundedness evaluation on your top 100 most common RAG queries today. If your score is under 90%, it's time to add a reranker. Have you implemented two-stage retrieval yet? What latency impact did you see?

## Further Reading / References

- [Agent Evaluation (MLflow 2) | Databricks](https://docs.databricks.com/en/generative-ai/agent-evaluation/index.html) — *verified 2026-07-11* — The framework used to measure groundedness natively in the platform.
