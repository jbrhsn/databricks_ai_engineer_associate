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
- [ ] Compare fixed-size, token-based, recursive, semantic, and structure-aware chunking approaches.
- [ ] Diagnose retrieval failures caused by poor chunking decisions.
- [ ] Apply the Parent Document Retriever pattern to balance granular indexing with rich LLM context.

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

### Chunking Strategy Decision Tree

```
Is the document structured (Markdown / HTML headers)?
├── Yes ──► Use MarkdownHeaderTextSplitter
│           └── Is any section still too large?
│               └── Yes ──► Re-split with RecursiveCharacterTextSplitter
│
└── No  ──► Is semantic coherence critical AND compute budget available?
            ├── Yes ──► Use SemanticChunker (embedding-based)
            │
            └── No  ──► Use RecursiveCharacterTextSplitter
                        └── Is chunk still too large?
                            └── Yes ──► Re-split with next separator
                                        (e.g., "\n" ──► " " ──► "")
```

---

## Key Concepts

### Token-Based Chunking (TokenTextSplitter)

Token-based chunking splits text by token count rather than character count, ensuring chunks respect the hard token limits imposed by embedding models. Mechanistically, it passes each portion of text through a tokenizer (such as tiktoken's BPE encoder) and counts the resulting tokens rather than raw characters; the same 500-character passage may yield 80 tokens for simple prose but 200 tokens for code or technical jargon because BPE merges common substrings aggressively. A split is inserted when the token count reaches the configured limit, not when the character count does. In the LangChain ecosystem this appears as `langchain_text_splitters.TokenTextSplitter`; it is the required choice when targeting models with strict token limits such as `text-embedding-3-small` (8191 tokens) where character-based estimates are unreliable.

### Fixed-Size Chunking (with Overlap)

Fixed-size chunking breaks text into blocks of a specific length, either measured in characters or tokens. Mechanistically, it scans the document and splits it the moment the counter reaches the threshold, regardless of where it is in the text. To mitigate splitting a word or thought in half, a "sliding window" or overlap is used so the end of one chunk is repeated at the start of the next. In the LangChain ecosystem, this appears as `CharacterTextSplitter`.

### Recursive Character Chunking

Recursive character chunking attempts to split text using a prioritized list of natural separators (e.g., double newline, single newline, space, empty string). Mechanistically, it tries to split on the highest-priority separator (like paragraph breaks); if a resulting chunk is still too large, it recursively applies the next separator to just that oversized chunk until all chunks are within the size limit. This is the default recommended approach for natural language. In the LangChain ecosystem, this manifests as `RecursiveCharacterTextSplitter`.

### Structure-Aware Chunking (MarkdownHeaderTextSplitter)

Structure-aware chunking splits documents by their declared logical hierarchy rather than by character or token count. Mechanistically, it scans the raw text for `#`, `##`, and `###` markers (or HTML `<h1>`–`<h6>` tags) and creates one chunk per section; crucially, the full header path (e.g., `"API Reference > Authentication > Token Endpoint"`) is injected into the chunk's metadata, not into its text body, so downstream filters can scope retrieval to a specific section without polluting the embedding. Where: `langchain_text_splitters.MarkdownHeaderTextSplitter` for Markdown source and `HTMLHeaderTextSplitter` for HTML source; the metadata fields `Header 1`, `Header 2`, etc. are queryable as Databricks Vector Search metadata filters.

### Semantic Chunking (Embedding-Based)

Semantic chunking splits text at the points where the underlying topic changes, rather than at fixed lengths or structural markers. Mechanistically, it encodes a sliding window of consecutive sentences into dense embeddings, then computes the cosine similarity between adjacent windows; when similarity drops below a configured threshold, the algorithm inserts a chunk boundary at that position. This means chunk sizes are variable and can span just two sentences or several paragraphs depending on topical coherence. Where: `langchain_experimental.text_splitter.SemanticChunker` requires a live embedding model at split time (e.g., a Databricks-hosted BGE or OpenAI embedding endpoint).

> ⚠️ **Fast-evolving:** `SemanticChunker` is part of `langchain_experimental` and its API surface changes frequently. Verify the current import path and `breakpoint_threshold_type` options against the official LangChain changelog before use in production.

### Parent Document Retriever Pattern

The Parent Document Retriever pattern decouples the chunk size used for vector indexing from the chunk size returned to the LLM, solving the classic precision-vs-context trade-off. Mechanistically, documents are split twice: once into small "child" chunks (e.g., 256 tokens) that are embedded and stored in the vector store for precise retrieval, and once into large "parent" chunks (e.g., 1024 tokens) that are stored in a separate docstore keyed by the child chunk's ID. At query time, the vector store returns the most relevant child chunk IDs, and the retriever then fetches the corresponding parent chunks from the docstore to deliver rich context to the LLM. Where: `langchain.retrievers.ParentDocumentRetriever` orchestrates both stores; the child index lives in Databricks Vector Search; the parent store can be backed by `InMemoryStore` for prototyping or a Delta table / MLflow artifact store for production.

---

## Key Parameters / Configuration Knobs

| Parameter | What it controls | Decision rule |
|---|---|---|
| `chunk_size` | The maximum size of a chunk (in characters or tokens) | Set to match the expected length of a complete thought or answer; smaller (256–512) for granular facts, larger (1024–2048) for broad context. |
| `chunk_overlap` | The number of characters/tokens shared between adjacent chunks | Set to 10–20% of `chunk_size` to ensure continuity; increase if retrieved context often feels cut off abruptly. |
| `separators` | The list of strings used to break the text (in priority order) | Leave as default `["\n\n", "\n", " ", ""]` for standard text; customize if parsing specific formats like code or logs. |
| `length_function` | Whether chunk size is measured in characters or tokens | Use `tiktoken.encoding_for_model("text-embedding-3-small").encode` when the downstream embedding model has a strict *token* limit (not character limit); use `len` only for rough size estimation. |
| `add_start_index` | Whether the character offset of the chunk within the source document is added to metadata | Set to `True` when you need precise source attribution for citation (e.g., for linking retrieved chunks back to a page number). |

---

## Worked Example: Requirement → Decision

**Given:** A collection of highly structured technical documentation files written in Markdown. Users frequently ask questions about specific API endpoints detailed under specific headers.

- **Step 1 — Identify the goal:** Chunk the documentation so that each API endpoint's description is retrieved as a complete, unified block.
- **Step 2 — Define inputs:** Markdown documents with `#`, `##`, and `###` headers.
- **Step 3 — Define outputs:** Chunks that contain the header metadata (so the context of the chunk is known) and the associated body text.
- **Step 4 — Apply constraints:** The text contains code blocks that should not be split in half, and the headers provide critical context for the paragraphs beneath them.
- **Step 5 — Select the approach:** Use `MarkdownHeaderTextSplitter` to group text by its headers, followed by a `RecursiveCharacterTextSplitter` if any single section is too large. This ensures the structural metadata is preserved and injected into the chunk's metadata for more accurate Databricks Vector Search filtering, which is not achievable with a plain `RecursiveCharacterTextSplitter` alone.

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
# Scenario: Splitting Markdown API documentation so that each section header
# is captured as metadata, enabling header-scoped filtering in Vector Search.
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
header_chunks = md_splitter.split_text(markdown_document)

# Re-split any sections that exceed the embedding model's token limit.
char_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
final_chunks = char_splitter.split_documents(header_chunks)
# Each chunk retains {"Header 1": "...", "Header 2": "..."} in its metadata.
```

```python
# Scenario: Indexing a large contract corpus where exact clause retrieval is
# needed but the LLM requires surrounding paragraph context to reason correctly.
# Constraint: chunk size for search must be small; context for generation must be large.
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DatabricksVectorSearch

child_splitter = RecursiveCharacterTextSplitter(chunk_size=256)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1024)

vectorstore = DatabricksVectorSearch(...)  # Databricks Vector Search index
docstore = InMemoryStore()                 # Replace with Delta-backed store for production

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
retriever.add_documents(documents)
# Query returns parent (1024-token) chunks even though child (256-token) chunks were matched.
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
from langchain_text_splitters import RecursiveCharacterTextSplitter

good_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
good_chunks = good_splitter.split_text(document_text)
```

---

## Common Pitfalls & Misconceptions

- **Over-optimizing for Small Chunks** — Beginners think smaller chunks mean more precise retrieval, so they set `chunk_size` as low as possible. In reality, chunks that are too small lose the surrounding context needed by the LLM to formulate a complete answer; the correct mental model is that retrieval precision and generation quality are in tension, and a 256–512 token chunk is usually the minimum viable floor.
- **Ignoring the Embedding Model's Token Limit** — Beginners set `chunk_size` in characters to 8000, not realizing their embedding model has a 512-token limit, because they conflate character count with token count. Always configure `length_function` to use the same tokenizer as the downstream embedding model so that `chunk_size` is measured in tokens, not characters.
- **Failing to Attach Metadata** — Beginners split a document and lose track of where the chunk came from because they treat the chunk text as self-contained. Always retain source metadata (document ID, header path, page number, `add_start_index=True`) in the chunk so it can be filtered in Databricks Vector Search and surfaced as citations.

---

## Key Definitions

| Term | Definition |
|---|---|
| Chunking | The process of dividing large texts into smaller, manageable segments for embedding and retrieval. |
| Chunk Overlap | The portion of text duplicated at the end of one chunk and the beginning of the next to preserve continuous context. |
| RRF (Reciprocal Rank Fusion) | A technique used by Databricks AI Search to combine the scores from hybrid (keyword + similarity) search results. |
| `length_function` | A callable passed to a text splitter that measures the size of a text segment; determines whether `chunk_size` is interpreted as characters (`len`) or tokens (a tokenizer encode function). |
| `add_start_index` | A boolean parameter on LangChain text splitters; when `True`, each chunk's metadata includes the character offset of its start position within the original document, enabling precise source citation. |

---

## Summary / Quick Recall

- Chunking determines what the LLM "sees" during retrieval; garbage chunks lead to garbage answers.
- `RecursiveCharacterTextSplitter` is the industry standard for unstructured natural language.
- `TokenTextSplitter` is required when the embedding model enforces a hard token limit (e.g., 8191 tokens for `text-embedding-3-small`).
- Always configure an overlap (typically 10–20%) to prevent loss of context at boundaries.
- Match chunk size to the embedding model's maximum token limit, measured with the correct tokenizer.
- Structure-aware splitters (MarkdownHeaderTextSplitter, HTMLHeaderTextSplitter) inject header paths into metadata, enabling filtered retrieval.
- The Parent Document Retriever pattern solves the precision-vs-context trade-off by indexing small chunks but retrieving large ones.

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
- [LangChain SemanticChunker](https://python.langchain.com/docs/how_to/semantic-chunker/) — *verified 2026-07-11* — Official documentation for embedding-based semantic chunking.
- [LangChain ParentDocumentRetriever](https://python.langchain.com/docs/how_to/parent_document_retriever/) — *verified 2026-07-11* — Official documentation for the Parent Document Retriever pattern.
- [Databricks AI Search](https://docs.databricks.com/en/generative-ai/vector-search.html) — *verified 2026-07-11* — Core documentation for Databricks Vector Search capabilities.
