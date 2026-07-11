# Chunking Strategies for RAG — Interview Prep

**Section:** 01-ingestion-and-chunking | **Role target:** GenAI Engineer, Data Engineer

## Core Conceptual Questions

These test whether you understand the fundamentals.

| Question | Key points to cover | Common weak-answer trap |
|---|---|---|
| Why do we chunk documents for RAG? | LLM context limits; embedding model token limits; improved retrieval precision by isolating specific concepts. | "To make the database smaller." (It actually increases row count). |
| What is chunk overlap and why is it necessary? | Repeating text at chunk boundaries ensures context is not lost if a concept or sentence is split across two chunks. | Defining it, but failing to explain *why* losing context degrades LLM generation. |
| Compare `CharacterTextSplitter` and `RecursiveCharacterTextSplitter`. | Fixed vs. prioritized separators. Recursive respects paragraphs/sentences before words, preserving meaning better. | Claiming they are the same, or that Recursive is an AI model. |

## Applied / Scenario Questions

**Q:** Your RAG system is built on a knowledge base of highly structured Markdown documents containing many code snippets. Users are complaining that the retrieved code is often broken in half and won't compile. How do you fix this?

**Strong answer framework:**
- Identify the current flaw: The system is likely using a naive character splitter that ignores Markdown syntax and code block backticks.
- Propose the structural fix: Implement a `MarkdownHeaderTextSplitter` or a custom regex splitter to ensure text inside ` ``` ` blocks is never severed.
- Propose the fallback: Apply `RecursiveCharacterTextSplitter` only to the natural language sections outside the code blocks if they exceed the size limits.

## System Design / Architecture Questions (if applicable)

**Q:** You need to ingest 10 million PDF documents into Databricks Vector Search for a legal research assistant. How do you design the chunking pipeline?

**Approach:**
1. **Clarify requirements:** Are users searching for broad case summaries or specific legal clauses? (Assume specific clauses).
2. **Propose structure:** 
   - Use a PDF parsing library to extract text and layout (identifying headers).
   - Use a hierarchical chunking strategy: smaller chunks (e.g., 256 tokens) for embedding and vector search to ensure high precision on specific clauses.
   - Implement a Parent Document Retriever pattern: store the larger section (e.g., 1024 tokens) in a Delta table. When the vector search hits the small chunk, retrieve the parent chunk to feed to the LLM for full legal context.
3. **Justify choices and name tradeoffs explicitly:** Trade-off is increased storage and pipeline complexity, justified by the critical need for context in legal use cases.

## Vocabulary That Signals Expertise

Use these terms naturally — don't force them:
- **Semantic Boundary** — Using it to describe where a chunk *should* naturally end (e.g., end of a paragraph).
- **Parent Document Retriever** — When discussing the trade-off between small chunks for search and large chunks for context.
- **RRF (Reciprocal Rank Fusion)** — When discussing how chunk metadata can be combined with semantic similarity in Databricks AI Search.

## Vocabulary That Signals Weakness

Avoid these — they signal outdated or shallow understanding:
- **"I just split by 1000 characters."** — Shows a lack of awareness of structural and recursive chunking techniques.
- **"Chunking doesn't matter if the LLM is smart enough."** — A fundamental misunderstanding of RAG; the LLM cannot reason about text it never receives.

## STAR Answer Frame

**Situation:** In a previous RAG project, our retrieval precision for technical manuals was very low, resulting in frequent LLM hallucinations.  
**Task:** I needed to diagnose the ingestion pipeline and improve the quality of the retrieved context.  
**Action:** I analyzed the vector database payloads and discovered we were using a fixed character splitter with zero overlap, which was slicing tables and lists in half. I rewrote the ingestion job to use LangChain's `RecursiveCharacterTextSplitter` with a 15% overlap, and added metadata tagging for document sections.  
**Result:** Retrieval accuracy metrics improved by 35%, and user complaints about "missing information" dropped to near zero because the LLM was finally receiving complete, unbroken thoughts.

## Red Flags Interviewers Watch For

- **Ignoring the embedding model constraints:** Proposing a chunk size of 2,000 tokens when standard embedding models (like BGE or older OpenAI models) cap at 512 tokens.
- **Treating all data as flat text:** Failing to ask about the structure of the source documents (PDF vs JSON vs Markdown) before proposing a chunking strategy.
