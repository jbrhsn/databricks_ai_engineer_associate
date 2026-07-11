# Source Document Quality and Selection — Interview Prep

**Section:** 03-data-preparation | **Role target:** Data Engineer, Senior GenAI Engineer

## Core Conceptual Questions

These test whether you understand the fundamentals.

| Question | Key points to cover | Common weak-answer trap |
|---|---|---|
| Why can't we just load raw HTML/PDF files directly into a vector database? | Noise (headers, CSS, footers) degrades chunk quality; degrades top-K semantic matching; wastes LLM token context. | "The vector database only accepts text files." (Databases accept any strings; the issue is retrieval quality, not format). |
| How do you handle duplicate documents in a RAG corpus? | Apply locality-sensitive hashing (MinHash) via Spark MLlib for near-duplicates; filter out older versions using metadata timestamps. | "I would use a Python set or exact string matching." (Fails to catch near-duplicates and doesn't scale to millions of docs). |
| What is the role of metadata in document selection? | Enables hybrid search (vector similarity + exact keyword/date filtering); helps LLMs cite sources; allows hard exclusions (e.g., dropping toxic or PII data). | Assuming vector embeddings alone are sufficient for all retrieval tasks. |

## Applied / Scenario Questions

**Q:** You are building a RAG bot for internal company policies. Users are complaining that the bot is giving them answers based on policies that were retired three years ago. How do you fix this?

**Strong answer framework:**
- Diagnose the root cause: The data corpus contains obsolete documents that semantically match user queries perfectly.
- Immediate fix: Enrich the ingestion pipeline to parse out `last_updated` and `status` metadata from the source files.
- Long-term fix: Implement hybrid search to explicitly filter for `status == 'active'` during retrieval, or use a deduplication pipeline to completely purge retired documents from the index before chunking.

## System Design / Architecture Questions (if applicable)

**Q:** Design an unstructured data pipeline to ingest millions of messy PDF reports into a Databricks-backed RAG application.

**Approach:**
1. Clarify requirements: Define what text matters, acceptable latency for ingestion, and data scale.
2. Propose structure:
   - Ingestion: Databricks Lakeflow Connect to land raw PDFs into a Delta table.
   - Parsing: UDFs leveraging libraries like `unstructured` or `PyPDF2` to strip footers and extract raw text.
   - Enrichment & Deduplication: Spark MLlib `MinHashLSH` to group near-duplicates and keep only the latest version based on timestamp metadata.
   - Chunking & Indexing: Split text, generate embeddings, and load into Databricks AI Search.
3. Justify choices and name tradeoffs explicitly: Spark is chosen over Pandas for scalability across millions of PDFs. MinHash is chosen over exact matching for resilience to typos and minor document variations.

## Vocabulary That Signals Expertise

Use these terms naturally — don't force them:
- **MinHash / Locality-Sensitive Hashing (LSH)** — When discussing scalable deduplication of near-identical text.
- **Hybrid Search** — When solving retrieval problems that require semantic meaning plus hard constraints (like date bounds).
- **Data Poisoning** — When discussing the security risks of blindly ingesting unverified document sources.

## Vocabulary That Signals Weakness

Avoid these — they signal outdated or shallow understanding:
- **"Just put everything in LangChain"** — Signals a lack of understanding of distributed processing (Spark) needed for enterprise-scale data preparation.
- **"We will prompt the LLM to ignore bad data"** — Relying the LLM to fix upstream data quality issues is an expensive, high-latency, and unreliable anti-pattern.

## STAR Answer Frame

**Situation:** The internal knowledge bot was hallucinating procedures because the shared drive contained 10 years of overlapping, redundant project manuals.
**Task:** Clean the RAG data corpus to ensure only the single source of truth for each procedure was indexed.
**Action:** I built a PySpark pipeline that extracted text using `unstructured`, applied `MinHashLSH` to find documents with >90% similarity, and retained only the most recently modified file based on SharePoint metadata.
**Result:** Vector index size dropped by 40%, and user thumbs-down ratings for hallucinated policies decreased by 85%.

## Red Flags Interviewers Watch For

Candidates who skip data preparation entirely and jump straight to discussing embedding models and chunk sizes. A mature AI engineer knows that garbage-in, garbage-out dictates RAG performance.
