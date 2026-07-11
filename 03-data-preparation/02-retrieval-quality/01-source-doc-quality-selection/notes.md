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

```
┌───────────────┐    ┌──────────────────┐    ┌───────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Raw Corpus   │──►│  Parsing          │──►│  Metadata         │──►│  Deduplication  │──►│  Filtered       │
│  (PDF/HTML/  │    │  (Unstructured/  │    │  Enrichment       │    │  (MinHashLSH)   │    │  Clean Corpus   │
│   Images)    │    │   OCR/BS4)       │    │  (title, date...) │    │                 │    │                 │
└───────────────┘    └──────────────────┘    └───────────────────┘    └─────────────────┘    └─────────────────┘
```

### Deduplication Mechanism (MinHash)

```
Document A ──► Featurize (n-grams) ──► Hash Vectors ──┐
                                                      ├─► Similarity Join ──► Keep Best Version
Document B ──► Featurize (n-grams) ──► Hash Vectors ──┘
```

### PII and Toxicity Filtering Decision Path

```
Text Chunk
    │
    ▼
Run PII Detector (NER / regex)
    │
    ▼
Confidence > threshold?
    ├── Yes ──► Redact span / Drop document ──► Clean Chunk
    └── No  ──► Pass Through
                    │
                    ▼
             Run Toxicity Classifier
                    │
                    ▼
             Score > threshold?
                    ├── Yes ──► Quarantine to Delta Table
                    └── No  ──► Clean Chunk to Index
```

---

## Key Concepts

### Corpus Composition and Ingestion

**What:** Selecting and ingesting the right data sources tailored to the RAG application's specific goals. It involves engaging domain experts to identify high-value documents (FAQs, manuals, policies) and ingesting them incrementally into a Delta Table to preserve traceability and enable auditing.

**How:** Under the hood, Databricks Lakeflow Connect (formerly backed by Delta Live Tables connectors) establishes a source connector against a SaaS application or file store (e.g., SharePoint, Salesforce, Google Drive). It then schedules incremental pulls on a configurable cadence, reading only records that have changed since the last run using a watermark or cursor column for change data capture (CDC) tracking. Each batch is written transactionally to a Delta Table, providing an auditable, immutable record of every ingestion event.

**Where in Databricks:** Lakeflow Connect is configured in the Databricks UI under **Data Engineering → Lakeflow Connect**, or via Declarative Asset Bundles in the Databricks CLI. The resulting pipeline writes to streaming Delta Tables governed by Unity Catalog. See: [Managed connectors in Lakeflow Connect](https://docs.databricks.com/en/ingestion/lakeflow-connect/index.html) — *verified 2026-07-11*.

### Data Preprocessing and Parsing

**What:** Transforming raw, unstructured files (PDFs, HTML, Images) into clean, consistent text formats while removing noise like headers and footers.

**How:** The pipeline uses libraries such as `unstructured`, `BeautifulSoup`, and `pytesseract` (for OCR on scanned images) to extract text. Each library targets a different input format: `unstructured` handles heterogeneous formats including PDFs and DOCX; `BeautifulSoup` parses the DOM of HTML pages to extract only the main article body; `pytesseract` invokes the Tesseract OCR engine to convert image pixels to text. Non-essential content (navbars, CSS, page numbers) is stripped using CSS selectors or bounding-box heuristics.

**Where in Databricks:** Implemented in Spark Notebooks or Delta Live Tables using Python UDFs that apply the parsing library per row. The output is a Delta Table with a clean `text_content` column.

### Metadata Enrichment

**What:** Supplementing raw text with additional context such as document title, section header, author, and update timestamp.

**How:** During or after parsing, extractors inspect document structure (HTML headings, PDF bookmarks, file-system properties) to derive metadata fields. These fields are appended as additional columns in the output Delta Table alongside the `text_content`. At query time, this metadata enables pre-filtering and hybrid search — the retriever narrows the candidate set before embedding similarity is calculated.

**Where in Databricks:** Metadata is generated using LangChain or LlamaIndex document loaders, or custom Python logic, and saved as distinct columns in the vector-backed Delta Table. Vector Search indexes in Databricks support filtering on these metadata columns directly via the `filters` parameter in the search API.

### Deduplication (MinHash)

**What:** Identifying and eliminating exact or near-duplicate documents from the corpus to prevent redundant chunks from degrading retrieval performance.

**How:** Documents are tokenized into n-gram sets and converted into sparse feature vectors via `HashingTF`. A `MinHashLSH` model applies locality-sensitive hashing: each document is hashed into multiple hash tables such that similar documents are more likely to collide in the same bucket. An `approxSimilarityJoin` operation then compares all document pairs that share a bucket, computing their Jaccard distance. Pairs whose Jaccard similarity exceeds the configured threshold are considered near-duplicates; the pipeline retains the most recent version (by `last_modified` metadata) and drops the rest.

**Where in Databricks:** Executed at scale using `MinHashLSH` from `pyspark.ml.feature` in a Databricks Spark cluster. The deduplication step is typically implemented as a Delta Live Tables pipeline stage that reads from the raw ingestion table and writes to a deduplicated silver table.

### PII and Sensitive Data Filtering

**What:** The process of detecting and removing personally identifiable information (PII) before data enters the vector index, preventing the RAG system from surfacing sensitive data in responses.

**How mechanistically:** PII detection uses named-entity-recognition (NER) models or regex patterns to identify spans (names, email addresses, SSNs, phone numbers, dates of birth). Once identified, detected spans are replaced with a typed placeholder (e.g., `[REDACTED_EMAIL]`, `[REDACTED_NAME]`) or the document is dropped entirely when the PII density is too high to safely redact. The confidence threshold controls the trade-off between false positives (legitimate text wrongly redacted) and false negatives (PII missed).

**Where in Databricks:** Microsoft Presidio can be run inside a Pandas UDF on a Databricks cluster, applying entity recognition to each document row at scale. Databricks AI Functions (`ai_extract`) can also be called in SQL within a Delta Live Tables pipeline to extract and flag PII spans without leaving the Lakehouse. Redacted documents are written to a governed Delta Table; dropped documents are written to a quarantine table for compliance review.

### Toxicity and Harm Filtering

**What:** Screening content for harmful, offensive, or adversarial text before it is indexed, preventing the RAG system from retrieving and surfacing such content in answers.

**How:** A classifier — such as Unitary's `detoxify` library or a custom fine-tuned model hosted on a Databricks Model Serving endpoint — assigns a toxicity probability score to each text chunk. Chunks whose score exceeds the configured threshold are flagged. Flagged chunks are either excluded from the index entirely or routed to a quarantine Delta Table for human review, preserving an audit trail rather than silently deleting content.

**Where in Databricks:** Implemented as a Spark UDF applied in a Delta Live Tables pipeline filter step. The quarantine table is governed by Unity Catalog with restricted read permissions, ensuring only compliance reviewers can access flagged content.

---

## Key Parameters / Configuration Knobs

| Parameter | What it controls | Decision rule |
|---|---|---|
| `numHashTables` (MinHashLSH) | Number of hash tables used in LSH OR-amplification. | Increase to improve recall (find more duplicates) at the cost of longer computation time. Start at 5; raise to 10+ if duplicate detection is missing known similar documents. |
| `similarityThreshold` (MinHashLSH `approxSimilarityJoin`) | The Jaccard similarity cutoff above which two documents are considered near-duplicates. | Set between 0.8–0.9 for strict deduplication (minor edits only); lower to 0.6–0.7 to catch substantially similar but differently worded documents. Must be set explicitly — there is no default. |
| `numFeatures` (HashingTF) | The size of the feature vector used for n-gram hashing before MinHash. | Use `1 << 18` (262,144) for large corpora to minimize hash collisions; smaller values reduce memory but increase false-positive duplicate detection. |
| `PII detection confidence threshold` | The minimum confidence score above which a PII span is redacted. | Set high (0.9+) in regulated industries to avoid false positives; lower only if recall (catching all PII) is more important than precision. |
| Metadata fields included | Which extracted metadata fields are appended to chunks in the index. | Include fields that users are highly likely to filter by (e.g., `last_updated`, `doc_category`). Exclude fields with high cardinality and no filtering value (e.g., paragraph-level byte offsets). |

---

## Worked Example: Requirement → Decision

**Given:** A company wants a RAG bot for internal HR policies. The shared drive contains many duplicate drafts and obsolete versions of the employee handbook, and the documents may contain employee names and ID numbers.

**Step 1 — Identify the goal:** Ensure the RAG index only contains the single most authoritative, up-to-date version of each policy document, with all PII removed.

**Step 2 — Define inputs:** Raw PDF and Word documents extracted from the HR SharePoint drive via Lakeflow Connect, containing multiple versions of the same texts and embedded employee PII.

**Step 3 — Define outputs:** A cleaned, deduplicated, PII-redacted Delta Table of policy documents ready for chunking and vector indexing.

**Step 4 — Apply constraints:** The process must run across millions of documents scalably and catch "near-duplicates" where only a date or minor typo was changed. PII redaction must meet regulatory requirements, meaning false negatives (missed PII) are more costly than false positives.

**Step 5 — Select the approach:** Use Spark MLlib's `MinHashLSH` on n-gram featurized text to find near-duplicates (threshold 0.85), then filter out the older versions using the `last_modified` metadata timestamp. Apply a Presidio-based Pandas UDF with a high confidence threshold (0.92) for PII redaction before writing to the silver table. This is chosen over exact-string matching because MinHash handles minor draft variations and scales across a Databricks cluster; Presidio is preferred over regex-only approaches because it combines NER with rule-based patterns for higher recall.

---

## Implementation

```python
# Anti-pattern: Loading raw documents directly without parsing, deduplication, or metadata enrichment.
# This results in noisy chunks containing HTML tags and duplicate information that pollutes the index.

from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = DirectoryLoader('/Workspace/raw_docs', glob="**/*.html")
docs = loader.load() # Loads raw HTML, including navbars and footers

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)
# The index will now be bloated with duplicate navbars and raw HTML tags.
```

```python
# Scenario: Parsing HTML files to extract clean text and attaching source metadata before chunking,
# to prevent CSS/navbar noise from entering the vector index and degrading retrieval precision.
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

```python
# Scenario: Finding and removing near-duplicate policy documents using Spark MLlib MinHashLSH
# to prevent redundant chunks from crowding out diverse results in top-K retrieval.
from pyspark.ml.feature import HashingTF, MinHashLSH
from pyspark.sql.functions import col, split

# 1. Tokenize documents into n-grams (words as tokens for simplicity)
tokenized = df_docs.withColumn("tokens", split(col("text_content"), " "))

# 2. Create feature vectors — use 1<<18 to minimize hash collisions on a large corpus
hashing_tf = HashingTF(inputCol="tokens", outputCol="features", numFeatures=1<<18)
df_featured = hashing_tf.transform(tokenized)

# 3. Apply MinHash LSH with 5 hash tables for balanced recall/performance
mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=5)
model = mh.fit(df_featured)

# 4. Find approximate duplicates (Jaccard similarity > 0.85)
duplicates = model.approxSimilarityJoin(
    df_featured, df_featured, threshold=0.85, distCol="jaccard_dist"
)

# 5. Filter self-matches and keep only the most recent version of each pair
dupes_filtered = duplicates.filter(col("datasetA.doc_id") != col("datasetB.doc_id"))
# Downstream: join back to df_docs and drop rows where doc_id appears as the older duplicate
```

---

## Common Pitfalls & Misconceptions

- **Ignoring document deduplication** — Beginners assume the vector database will naturally filter out duplicate information. In reality, multiple identical chunks will crowd out diverse results in the top-K retrieval, degrading LLM generation quality and wasting context space.
- **Over-relying on LLMs for parsing** — Developers try to use LLMs to extract text from raw messy HTML or PDFs because LLMs seem capable of understanding any format, but this is extremely slow and expensive at scale. The correct mental model is: use traditional parsing libraries (`unstructured`, `BeautifulSoup`, `pytesseract`) for all structured text extraction at scale, reserving LLMs exclusively for semantic tasks like classification or summarization that cannot be achieved with deterministic tools.
- **Discarding metadata too early** — Teams strip text to bare strings before chunking, believing leaner data is always better. Always retain and attach document-level metadata (like source and date) to chunks to enable hybrid search filtering and accurate LLM citation.

---

## Key Definitions

| Term | Definition |
|---|---|
| MinHash | A locality-sensitive hashing technique used to quickly estimate the Jaccard similarity between two sets, highly effective for finding near-duplicate documents in large corpora. |
| Data Poisoning Attack | An exploit where malicious or conflicting data is intentionally introduced into the RAG corpus to manipulate the LLM's answers. |
| Hybrid Search | A retrieval strategy combining semantic vector search with keyword or metadata filtering to improve precision. |
| PII (Personally Identifiable Information) | Any data that could be used on its own or in combination to identify, contact, or locate an individual — including names, email addresses, SSNs, and phone numbers. In a RAG context, PII must be detected and redacted before documents are indexed to prevent the system from surfacing sensitive data in responses. |
| Toxicity Filtering | The process of scoring text chunks with a classifier to detect harmful, offensive, or adversarial content, and excluding or quarantining chunks that exceed a configured probability threshold before they are added to the vector index. |
| Lakeflow Connect | The Databricks-managed data ingestion service (formerly backed by Delta Live Tables connectors) that provides pre-built connectors to SaaS applications (Salesforce, Jira, Workday), file stores (SharePoint, Google Drive), and databases (MySQL, PostgreSQL) with incremental CDC-based ingestion into Unity Catalog-governed Delta Tables. |

---

## Summary / Quick Recall

- Poor retrieval quality is often an upstream data pipeline issue, not an embedding model issue.
- Always parse raw files to remove noise (headers, HTML tags) before processing; use `unstructured`, `BeautifulSoup`, or `pytesseract` — not LLMs — for this step.
- Enrich documents with structural and contextual metadata to enable hybrid search filtering.
- Use MinHash (via Spark MLlib `MinHashLSH`) to deduplicate near-identical documents at scale; set `similarityThreshold` explicitly between 0.8–0.9 for strict deduplication.
- Filter out PII using NER-based tools (Presidio) and toxic content using a classifier before chunking and indexing.
- Lakeflow Connect handles incremental ingestion from SaaS and file sources into Delta Tables with CDC watermarking.

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
- [Managed connectors in Lakeflow Connect](https://docs.databricks.com/en/ingestion/lakeflow-connect/index.html) — *verified 2026-07-11* — Reference for SaaS, file, database, and streaming connectors with incremental CDC ingestion into Delta Tables.
