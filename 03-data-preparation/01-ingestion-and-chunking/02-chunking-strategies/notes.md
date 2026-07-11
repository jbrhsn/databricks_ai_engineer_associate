# Chunking Strategies for RAG

**Section:** 01-ingestion-and-chunking | **Module:** 03-data-preparation | **Est. time:** 1.5 hrs | **Exam mapping:** Data Preparation (14%)

---

## TL;DR

Chunking is the process of breaking large documents into smaller, semantically meaningful pieces before generating embeddings for retrieval. The choice of chunking strategy directly determines the quality of your RAG system's context window and the relevance of the retrieved results. **The one thing to remember: Always prefer recursive or structure-aware chunking over fixed-size character chunking to preserve semantic boundaries.**

---

## ELI5 — Explain It Like I'm 5

Imagine you have a giant textbook and you want to create flashcards for studying. If you just take a pair of scissors and cut the book into perfectly equal 3x5 inch squares (fixed chunking), you will inevitably slice right through the middle of sentences and diagrams, making many flashcards useless. Instead, a better approach is to cut along natural boundaries—separating chapters, then sections, then paragraphs, only cutting sentences if a paragraph is too big for a single flashcard (recursive chunking). This ensures each flashcard contains a complete, understandable thought, correcting the misconception that all chunks need to be exactly the same length.

---

## Learning Objectives

By the end of this chapter you will be able to:
- [ ] Explain the trade-offs between chunk size and retrieval granularity.
- [ ] Implement `RecursiveCharacterTextSplitter` to preserve paragraph and sentence boundaries.
- [ ] Design an overlap strategy to maintain context across chunk boundaries.
- [ ] Compare fixed-size, recursive, and semantic chunking approaches.
- [ ] Diagnose retrieval failures caused by poor chunking decisions.

---

## Visual Overview

### The Problem with Zero Overlap

```
Source Text: "Databricks AI Search uses HNSW. It supports hybrid search."
Fixed Split 1: "Databricks AI Search uses H"
Fixed Split 2: "NSW. It supports hybrid sea"
Fixed Split 3: "rch."
Result ──► Meaning is completely destroyed.
```

### Recursive Character Splitting Flow

```
Input Document
│
├── Can we split by "\n\n" (paragraphs)?
│   ├── Yes ──► Are chunks under chunk_size?
│   │           ├── Yes ──► Keep chunks
│   │           └── No  ──► Pass oversized chunks to next separator
│   │
│   └── No  ──► Can we split by "\n" (lines)?
│               ├── Yes ──► (Repeat size check)
│               └── No  ──► Can we split by " " (words)?
│                           └── ...
```

---

## Key Concepts

### Fixed-Size Chunking (with Overlap)

Fixed-size chunking breaks text into blocks of a specific length, either measured in characters or tokens. Mechanistically, it scans the document and splits it the moment the counter reaches the threshold, regardless of where it is in the text. To mitigate splitting a word or thought in half, a "sliding window" or overlap is used so the end of one chunk is repeated at the start of the next. In the LangChain ecosystem, this appears as `CharacterTextSplitter` or `TokenTextSplitter`.

### Recursive Character Chunking

Recursive character chunking attempts to split text using a prioritized list of natural separators (e.g., double newline, single newline, space, empty string). Mechanistically, it tries to split on the highest-priority separator (like paragraph breaks); if a resulting chunk is still too large, it recursively applies the next separator to just that oversized chunk until all chunks are within the size limit. This is the default recommended approach for natural language. In the LangChain ecosystem, this manifests as `RecursiveCharacterTextSplitter`.

### Document-Structure and Semantic Chunking

Semantic chunking breaks text based on its logical structure or meaning rather than arbitrary length. Mechanistically, it looks for structural markers (like markdown headers, HTML tags) or uses an embedding model to detect shifts in topic, creating chunks that encompass a complete semantic unit. In the LangChain ecosystem, this is implemented using `MarkdownHeaderTextSplitter` or the `SemanticChunker` (note: SemanticChunker using embeddings is compute-heavy and fast-evolving).

### Key Parameters / Configuration Knobs

| Parameter | What it controls | Decision rule |
|---|---|---|
| `chunk_size` | The maximum size of a chunk (in characters or tokens) | Set to match the expected length of a complete thought or answer; smaller (256-512) for granular facts, larger (1024-2048) for broad context. |
| `chunk_overlap` | The number of characters/tokens shared between adjacent chunks | Set to 10-20% of `chunk_size` to ensure continuity; increase if retrieved context often feels cut off abruptly. |
| `separators` | The list of strings used to break the text (in priority order) | Leave as default `["\n\n", "\n", " ", ""]` for standard text; customize if parsing specific formats like code or logs. |

### Worked Example: Requirement → Decision

**Given:** A collection of highly structured technical documentation files written in Markdown. Users frequently ask questions about specific API endpoints detailed under specific headers.

- **Step 1 — Identify the goal:** Chunk the documentation so that each API endpoint's description is retrieved as a complete, unified block.
- **Step 2 — Define inputs:** Markdown documents with `#`, `##`, and `###` headers.
- **Step 3 — Define outputs:** Chunks that contain the header metadata (so the context of the chunk is known) and the associated body text.
- **Step 4 — Apply constraints:** The text contains code blocks that should not be split in half, and the headers provide critical context for the paragraphs beneath them.
- **Step 5 — Select the approach:** Use `MarkdownHeaderTextSplitter` to group text by its headers, followed by a `RecursiveCharacterTextSplitter` if any single section is too large. This ensures the structural metadata is preserved and injected into the chunk's metadata for more accurate Databricks Vector Search filtering.

---

## Implementation

```python
# Scenario: Splitting standard natural language documents (e.g., PDF text) 
# while preserving paragraphs and sentence structures.
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

chunks = text_splitter.split_text(document_text)
# Each chunk will prioritize staying within paragraph boundaries.
```

```python
# Anti-pattern: Using a naive character splitter with zero overlap on text.
# This will blindly cut words and sentences in half, destroying the semantic 
# meaning of the embedding at the boundary.
from langchain_text_splitters import CharacterTextSplitter

bad_splitter = CharacterTextSplitter(
    separator="",
    chunk_size=500,
    chunk_overlap=0
)
bad_chunks = bad_splitter.split_text(document_text)

# Correct approach: Always use an overlap to maintain context, and prefer 
# recursive splitting for natural language.
good_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
good_chunks = good_splitter.split_text(document_text)
```

---

## Common Pitfalls & Misconceptions

- **Over-optimizing for small chunks** — Beginners think smaller chunks mean more precise retrieval. In reality, chunks that are too small lose the surrounding context needed by the LLM to formulate a complete answer.
- **Ignoring the embedding model's token limit** — Beginners set `chunk_size` in characters to 8000, not realizing their embedding model has a 512-token limit. Always ensure your maximum chunk size (in tokens) fits within the context window of your chosen embedding model.
- **Failing to attach metadata** — Beginners split a document and lose track of where the chunk came from. Always retain source metadata (document ID, header, page number) in the chunk so it can be filtered in Databricks AI Search.

---

## Key Definitions

| Term | Definition |
|---|---|
| Chunking | The process of dividing large texts into smaller, manageable segments for embedding and retrieval. |
| Chunk Overlap | The portion of text duplicated at the end of one chunk and the beginning of the next to preserve continuous context. |
| RRF (Reciprocal Rank Fusion) | A technique used by Databricks AI Search to combine the scores from hybrid (keyword + similarity) search results. |

---

## Summary / Quick Recall

- Chunking determines what the LLM "sees" during retrieval; garbage chunks lead to garbage answers.
- `RecursiveCharacterTextSplitter` is the industry standard for natural language.
- Always configure an overlap (typically 10-20%) to prevent loss of context at boundaries.
- Match chunk size to the embedding model's maximum token limit.
- Structure-aware splitters (like Markdown/HTML) provide superior semantic boundaries.

---

## Self-Check Questions

1. What is the primary purpose of adding `chunk_overlap` when splitting documents?

   <details><summary>Answer</summary>
   
   **To preserve context and continuous thoughts across chunk boundaries.** Without overlap, a critical sentence or entity name might be sliced in half, causing the embedding model to fail to recognize its meaning. The main distractor—"to increase the total number of chunks"—is wrong because increasing chunk count is a side effect, not the goal.
   
   </details>

2. You are processing a large JSON file where each object represents a product review. Which chunking strategy is most appropriate?

   <details><summary>Answer</summary>
   
   **A structure-aware splitter (e.g., splitting by the JSON object boundaries).** For structured data, you want to keep the logical unit (the review) intact rather than arbitrarily splitting by character count. Recursive character splitting is for unstructured text and would likely destroy the JSON formatting.
   
   </details>

3. **Which TWO** of the following statements about `RecursiveCharacterTextSplitter` are true?
   - A. It always results in chunks of exactly `chunk_size` characters.
   - B. It attempts to split by the first separator in the list, falling back to the next only if chunks are too large.
   - C. It is generally preferred over `CharacterTextSplitter` for natural language processing.
   - D. It requires an LLM to determine the semantic meaning of the text to find split points.
   - E. It can only be used with Databricks foundational models.

   <details><summary>Answer</summary>
   
   **B and C.** It uses a prioritized list of separators (B) which makes it ideal for natural language (C). A is wrong because it produces chunks *up to* the maximum size, rarely exactly the size. D describes an advanced Semantic Chunker, not the recursive character splitter. E is wrong because LangChain splitters are model-agnostic.
   
   </details>

4. An engineering team notices their RAG application frequently retrieves chunks that start with "He then went to..." or "Therefore, the result is...", lacking the subject of the sentence. What is the most likely misconfiguration?

   <details><summary>Answer</summary>
   
   **The `chunk_overlap` is set to 0 or is too small.** When overlap is insufficient, pronouns and transitional phrases get orphaned in a new chunk without the preceding context that defines them. Changing the embedding model will not fix this structural data issue.
   
   </details>

5. You are designing a RAG system for legal contracts. Users need to retrieve exact clauses, but also understand the broader section the clause belongs to. What trade-off must you consider regarding `chunk_size`?

   <details><summary>Answer</summary>
   
   **Smaller chunks provide precise retrieval of specific clauses, but larger chunks provide the necessary surrounding legal context.** You must balance granularity against context. A common solution is to use smaller chunks for the vector search but retrieve the "parent" larger chunk (Parent Document Retriever pattern) to feed to the LLM.
   
   </details>

---

## Further Reading

- [LangChain Text Splitters](https://python.langchain.com/docs/how_to/recursive_text_splitter/) — *verified 2026-07-11* — Official documentation for LangChain text splitting techniques.
- [Databricks AI Search](https://docs.databricks.com/en/generative-ai/vector-search.html) — *verified 2026-07-11* — Core documentation for Databricks Vector Search capabilities.
