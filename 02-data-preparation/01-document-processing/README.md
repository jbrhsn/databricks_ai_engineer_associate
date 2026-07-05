# Module 01 — Document Processing

**Part of section:** 02 Data Preparation
**Estimated time:** 9 hours
**Prerequisites:** Section 01 — LLM and GenAI Core Concepts (embeddings, context windows); Databricks Workspace Orientation (Unity Catalog Volumes, Delta Lake basics)
**Exam mapping:** Data Preparation domain (~20% of exam) — both chapters are directly exam-relevant

## Overview

This module is the complete hands-on guide to building the data pipeline that feeds a RAG
application. It covers the full journey: raw files arrive in Unity Catalog Volumes → they are
parsed, cleaned, and enriched → chunked into semantically coherent units → embedded into
vectors → stored in a Delta table → indexed in Databricks AI Search for low-latency retrieval.

The Databricks documentation (updated 2026-06-30) identifies this as the "unstructured data
pipeline for RAG" and calls out five key stages: corpus ingestion, data preprocessing (parsing +
enrichment), chunking, embedding, and indexing. Both chapters in this module cover these stages
in depth.

## Learning Outcomes

By completing this module, you will be able to:

- Parse PDFs, HTML, and scanned images into clean text using appropriate libraries
- Extract and attach document-level, content-based, and structural metadata to each chunk
- Deduplicate a corpus using MinHash locality-sensitive hashing in Spark
- Apply relevance and PII filtering to a document corpus
- Choose and implement one of four chunking strategies for a given document type
- Calculate token-level chunk sizes and overlap values for a target context window budget
- Describe the role of Delta Lake in a GenAI data pipeline (ACID, schema enforcement, time travel)
- Store unstructured files in Unity Catalog Volumes (managed and external)
- Create a Databricks AI Search index from a Delta table
- Use Auto Loader for incremental, scalable document ingestion
- Explain how hybrid keyword-similarity search and Reciprocal Rank Fusion (RRF) work in AI Search

## Topics Covered

**Chapter 01 — Extraction and Chunking Strategies**
- Corpus composition and source selection
- Parsing: `unstructured`, `PyPDF2`, BeautifulSoup, Tesseract OCR, cloud OCR APIs
- Metadata extraction: document-level, content-based, structural, contextual
- Deduplication: MinHash LSH with Spark ML
- Filtering: relevance, toxicity classifiers, PII detection
- Chunking strategies: fixed-size, paragraph-based, format-specific, semantic
- Chunk size, overlap, and token budget calculations
- Embedding model selection for RAG (MTEB leaderboard, token limits, domain fit)

**Chapter 02 — Delta Lake and Unity Catalog for GenAI Data**
- Delta Lake: ACID transactions, schema enforcement, schema evolution, time travel
- Medallion architecture applied to GenAI: bronze (raw) → silver (parsed/chunked) → gold (embedded/indexed)
- Auto Loader: incremental file ingestion from cloud storage
- Unity Catalog Volumes: managed vs. external, path format, access control, compute requirements
- Databricks AI Search (formerly Vector Search): HNSW algorithm, hybrid search, RRF scoring
- Creating and syncing an AI Search index from a Delta table
- Lakeflow Spark Declarative Pipelines: overview and role in GenAI pipelines

## How This Module Fits

This module is the bridge between "I understand embeddings conceptually" (Section 01) and "I can
call a retriever in my RAG chain" (Section 03). Without a properly built data pipeline, retrieval
quality is poor regardless of how good your prompts or generation model are. In production, data
pipeline quality is usually the first place to look when RAG answers are wrong or irrelevant.

## Study Tips

- **Chunking strategy is the most exam-tested concept in this module.** Know all four strategies,
  their trade-offs, and which document types they suit. The exam often presents a scenario and asks
  which strategy is most appropriate.
- **"Databricks AI Search" is the current name** for what the exam guide may call "Vector Search."
  Know both names and that they refer to the same product.
- **Practice the token budget calculation.** Given a context window, system prompt size, and query
  size, compute how many chunks of a given size fit. This arithmetic shows up in exam scenarios.
- **Delta Lake is the default** — you do not specify it explicitly. Every Databricks table is Delta
  unless you opt out. This is a common exam distractor.
- **Medallion architecture** (bronze/silver/gold) is a Databricks-idiomatic pattern that the exam
  assumes you know. Apply it to GenAI pipelines: raw docs → parsed/chunked → embedded/indexed.

## Chapters

- [ ] [01 Extraction and Chunking Strategies](./01-extraction-and-chunking-strategies/notes.md) — 4.5 hrs
- [ ] [02 Delta Lake and Unity Catalog for GenAI Data](./02-delta-lake-and-unity-catalog-for-genai-data/notes.md) — 4.5 hrs
