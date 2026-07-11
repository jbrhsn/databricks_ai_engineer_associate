# Source Document Quality and Selection

**Section:** 03-data-preparation | **Module:** 02-retrieval-quality | **Est. time:** 1 hr | **Exam mapping:** Data Preparation

---

## TL;DR

The foundation of any robust RAG application lies in the quality and composition of its source data. Relying on raw, uncurated documents leads to hallucination and poor retrieval accuracy, making data preprocessing, deduplication, and filtering essential first steps. **The one thing to remember: The best embedding model in the world cannot fix a fundamentally flawed or duplicate-heavy data corpus.**

---

## ELI5 — Explain It Like I'm 5

Imagine you are a chef making a complex soup. The ingredients are your data corpus. If you throw unwashed vegetables, duplicate spices, and irrelevant items (like a shoe) straight into the pot, the soup will be ruined, no matter how good your stove (the LLM) is. Data quality and selection act as the washing, chopping, and sorting of your ingredients before they ever touch the heat. A common misconception is that retrieval errors are mostly caused by bad embedding models; in reality, they are usually caused by feeding the model poorly parsed, noisy, or redundant documents.

---

## Learning Objectives

By the end of this chapter you will be able to:
- [ ] Evaluate an unstructured data corpus for composition and quality issues.
- [ ] Implement data parsing and metadata enrichment pipelines to improve chunk context.
- [ ] Implement MinHash for Jaccard distance to deduplicate unstructured text.
- [ ] Design filtering logic to remove toxic, irrelevant, or sensitive (PII) information before indexing.

---

## Visual Overview

### Data Preprocessing Pipeline

```text
Raw Corpus ──► Parsing (Text/HTML/OCR) ──► Enrichment (Metadata) ──► Deduplication (MinHash) ──► Filtering (PII/Toxicity) ──► Cleaned Corpus
```

### Deduplication Mechanism (MinHash)

```text
Document A ──► Featurize (n-grams) ──► Hash Vectors ──┐
                                                      ├─► Similarity Join ──► Keep Best Version
Document B ──► Featurize (n-grams) ──► Hash Vectors ──┘
```

---

## Key Concepts

### Corpus Composition and Ingestion

Selecting and ingesting the right data sources tailored to the RAG application's specific goals. It involves engaging domain experts to identify high-value documents (FAQs, manuals) and ingesting them incrementally into a target table to preserve traceability and enable auditing. This is typically handled via standard connectors in Databricks Lakeflow Connect, storing raw source data in Delta tables for immutable tracking.

### Data Preprocessing and Parsing

Transforming raw, unstructured files (PDFs, HTML, Images) into clean, consistent text formats while removing noise like headers and footers. The pipeline uses libraries (e.g., `unstructured`, `BeautifulSoup`, `Tesseract` for OCR) to extract text, stripping out non-essential characters or malformed information that could confuse the RAG chain. In Databricks, this is often implemented in Spark Notebooks or Delta Live Tables using Python UDFs.

### Metadata Enrichment

Supplementing raw text with additional context (e.g., document title, section header, author, update timestamp). Parsers or LLMs automatically extract structural and contextual metadata, which is stored alongside the chunk text. At retrieval time, this enables hybrid search and keyword-based filtering. This metadata is generated using LangChain/LlamaIndex document loaders, or custom logic, and is saved as distinct columns in the vector-backed Delta table.

### Deduplication (MinHash)

Identifying and eliminating exact or near-duplicate documents from the corpus to prevent redundant chunks from degrading retrieval performance. The mechanism converts documents into n-gram feature vectors, hashes them using locality-sensitive hashing (MinHash), and performs a similarity join. If the similarity exceeds a threshold, one copy is dropped based on a rule (e.g., most recent). This is executed scalably using `MinHashLSH` from the Spark MLlib `pyspark.ml.feature` module in a Databricks environment.

### Key Parameters / Configuration Knobs

| Parameter | What it controls | Decision rule |
|---|---|---|
| `numHashTables` (MinHashLSH) | Number of hash tables used in LSH OR-amplification. | Increase to improve recall (find more duplicates) at the cost of longer computation time. |
| Metadata fields included | Which extracted metadata fields are appended to chunks in the index. | Include fields that users are highly likely to filter by (e.g., `last_updated`, `doc_category`). |

### Worked Example: Requirement → Decision

Given: A company wants a RAG bot for internal HR policies. The shared drive contains many duplicate drafts and obsolete versions of the employee handbook.
Step 1 — Identify the goal: Ensure the RAG index only contains the single most authoritative, up-to-date version of each policy document.
Step 2 — Define inputs: Raw PDF and Word documents extracted from the HR SharePoint drive, containing multiple versions of the same texts.
Step 3 — Define outputs: A cleaned, deduplicated Delta table of policy documents ready for chunking.
Step 4 — Apply constraints: The process must run across millions of documents scalably and catch "near-duplicates" where only a date or minor typo was changed.
Step 5 — Select the approach: Use Spark MLlib's `MinHashLSH` on n-gram featurized text to find near-duplicates, then filter out the older versions using the `last_modified` metadata timestamp. This is chosen over exact-string matching because it gracefully handles minor draft variations and scales across a Databricks cluster.

---

## Implementation

```python
# Anti-pattern: Loading raw documents directly without parsing, deduplication, or metadata enrichment.
# This results in noisy chunks containing HTML tags and duplicate information.

from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = DirectoryLoader('/Workspace/raw_docs', glob="**/*.html")
docs = loader.load() # Loads raw HTML, including navbars and footers

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)
# The index will now be bloated with duplicate navbars and raw HTML tags.
```

```python
# Correct approach:
# Scenario: Parsing HTML files to extract clean text and attaching source metadata before chunking.
from bs4 import BeautifulSoup
import os
from langchain_core.documents import Document

clean_docs = []
for file_name in os.listdir('/Workspace/raw_docs'):
    if file_name.endswith('.html'):
        with open(os.path.join('/Workspace/raw_docs', file_name), 'r') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            # Extract only the main article text, dropping navbars/footers
            main_content = soup.find('article').get_text(separator=' ', strip=True)
            
            # Enrich with contextual metadata
            metadata = {
                "source": file_name,
                "category": "HR_Policy",
                "length": len(main_content)
            }
            clean_docs.append(Document(page_content=main_content, metadata=metadata))
```

---

## Common Pitfalls & Misconceptions

- **Ignoring document deduplication** — Beginners assume the vector database will naturally filter out duplicate information. In reality, multiple identical chunks will crowd out diverse results in the top-K retrieval, degrading LLM generation quality and wasting context space.
- **Over-relying on LLMs for parsing** — Developers try to use LLMs to extract text from raw messy HTML or PDFs. While powerful, this is extremely slow and expensive; use traditional libraries like `BeautifulSoup` or `unstructured` for raw data parsing at scale.
- **Discarding metadata too early** — Teams strip text to bare strings before chunking. Always retain and attach document-level metadata (like source and date) to chunks to enable hybrid search filtering and accurate LLM citation.

---

## Key Definitions

| Term | Definition |
|---|---|
| MinHash | A locality-sensitive hashing technique used to quickly estimate the Jaccard similarity between two sets, highly effective for finding near-duplicate documents in large corpora. |
| Data Poisoning Attack | An exploit where malicious or conflicting data is intentionally introduced into the RAG corpus to manipulate the LLM's answers. |
| Hybrid Search | A retrieval strategy combining semantic vector search with keyword or metadata filtering to improve precision. |

---

## Summary / Quick Recall

- Poor retrieval quality is often an upstream data pipeline issue, not an embedding model issue.
- Always parse raw files to remove noise (headers, HTML tags) before processing.
- Enrich documents with structural and contextual metadata to enable hybrid search filtering.
- Use MinHash (via Spark MLlib) to deduplicate near-identical documents at scale.
- Filter out toxic, irrelevant, or sensitive (PII) data prior to chunking and indexing.

---

## Self-Check Questions

1. What is the primary purpose of using MinHash in a RAG data preparation pipeline?
   - A. To convert text chunks into dense vector embeddings.
   - B. To efficiently identify and remove near-duplicate documents from the corpus.
   - C. To parse text from scanned PDF images.
   - D. To mask Personally Identifiable Information (PII).

   <details><summary>Answer</summary>
   
   B. MinHash is a locality-sensitive hashing algorithm designed to estimate Jaccard similarity, making it highly efficient for finding and removing near-duplicates. It does not generate dense embeddings (A), perform OCR (C), or mask PII (D).
   
   </details>

2. An engineering team notices their RAG application frequently hallucinates conflicting answers about the company leave policy. Upon inspection, they see the top-K retrieved chunks contain older, deprecated versions of the policy alongside the current one. What is the most effective data preprocessing step to fix this?
   - A. Increase the chunk overlap size.
   - B. Switch to a larger embedding model.
   - C. Implement metadata enrichment with `last_updated` dates and filter/deduplicate based on it.
   - D. Prompt the LLM to ignore older dates in the context.

   <details><summary>Answer</summary>
   
   C. Implementing metadata enrichment allows the pipeline to identify document versions and deduplicate or filter out obsolete information before it enters the vector index. Upgrading the embedding model (B) or prompting the LLM (D) are downstream bandaids for an upstream data quality issue. Adjusting chunk overlap (A) has no impact on filtering deprecated versions.
   
   </details>

3. **Which TWO** of the following represent structural or contextual metadata that should be extracted during parsing to improve retrieval quality?
   - A. The raw HTML navigation bar content.
   - B. The document's original publication date.
   - C. The dense vector embedding of the document.
   - D. The section header under which the text appeared.
   - E. The randomly assigned UUID of the vector database row.

   <details><summary>Answer</summary>
   
   B and D. Publication date (contextual) and section headers (structural) provide critical context that can be used for pre-filtering or providing better context to the LLM. Navbars (A) are noise that should be parsed out. Embeddings (C) and DB UUIDs (E) are not metadata extracted during the upstream parsing step.
   
   </details>

4. You are ingesting millions of unstructured text documents into a Databricks Lakehouse. You want to execute deduplication efficiently. Which approach represents the most scalable Databricks-native method for this?
   - A. Iterate through all documents in a Python `for` loop and compare exact strings.
   - B. Use LangChain's `RecursiveCharacterTextSplitter` to drop duplicate characters.
   - C. Use Spark MLlib's `MinHashLSH` to perform a similarity join across the corpus.
   - D. Ask a Databricks hosted LLM endpoint to read pairs of documents and output whether they match.

   <details><summary>Answer</summary>
   
   C. Spark MLlib's `MinHashLSH` provides a distributed, scalable way to find near-duplicates via similarity joins. A Python loop (A) will not scale to millions of documents. LangChain splitters (B) do not deduplicate documents. Using an LLM (D) would be prohibitively slow and computationally expensive.
   
   </details>

5. A RAG pipeline processes web pages by downloading the raw HTML, splitting it by a fixed character count, and embedding the chunks. Users complain that retrieved answers contain irrelevant CSS code and footer menus. What is the most robust way to solve this?
   - A. Use a semantic chunking strategy instead of fixed-size.
   - B. Introduce a parsing step using a library like `BeautifulSoup` to extract only the main article text before chunking.
   - C. Change the embedding model to one trained specifically on code and HTML.
   - D. Increase the `chunk_size` so the LLM gets enough context to ignore the CSS.

   <details><summary>Answer</summary>
   
   B. A dedicated parsing step is required to clean the raw data and remove noise like CSS and footers before chunking. Semantic chunking (A) and larger chunks (D) will still index the garbage text, leading to wasted token space and potential LLM confusion. Changing the embedding model (C) does not remove the underlying noise.
   
   </details>

---

## Further Reading

- [Build an unstructured data pipeline for RAG](https://docs.databricks.com/aws/en/agents/tutorials/ai-cookbook/quality-data-pipeline-rag) — *verified 2026-07-11* — Databricks guide on parsing, deduplication, and filtering.
- [Improve RAG application quality](https://docs.databricks.com/aws/en/agents/tutorials/ai-cookbook/quality-overview) — *verified 2026-07-11* — Overview of retrieval and generation quality considerations.
