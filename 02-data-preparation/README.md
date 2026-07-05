# Section 02 — Data Preparation

**Estimated time:** 9 hours
**Prerequisites:** Section 01 (Foundations) — especially Chapter 01-03 (LLM core concepts, embeddings, context windows)
**Exam mapping:** Databricks GenAI Engineer Associate — Data Preparation domain (~20% of exam)

## Overview

Every RAG application is only as good as its data pipeline. This section covers the complete
lifecycle of turning raw, unstructured documents into vector-indexed, retrieval-ready data —
from parsing PDFs and web pages, through chunking and embedding strategies, to storing and
governing everything in Delta Lake with Unity Catalog. These are the skills that separate
production-quality RAG from toy demos.

Databricks documentation (last verified 2026-06-30) treats this as the "unstructured data
pipeline for RAG." The key current platform components are:

- **Unity Catalog Volumes** — governed storage for raw unstructured files (PDFs, HTML, text)
- **Delta Lake** — ACID-transactional table storage for parsed text, chunks, and embeddings
- **Databricks AI Search** (formerly Vector Search) — managed vector index backed by a Delta table
- **Lakeflow Spark Declarative Pipelines** (formerly Delta Live Tables) — orchestrated, incremental ETL

## Learning Outcomes

By completing this section, you will be able to:

- Choose the right parsing library for a given document type (PDF, HTML, image/OCR)
- Apply metadata extraction, deduplication, and filtering to improve corpus quality
- Select and implement a chunking strategy appropriate to the source document structure
- Use Delta Lake tables as the backbone of a GenAI data pipeline (bronze/silver/gold medallion)
- Store raw files in Unity Catalog Volumes and govern access with UC permissions
- Create a Databricks AI Search index backed by a Delta table
- Design a complete end-to-end unstructured data pipeline for a RAG application

## Topics Covered

- Document parsing: PDF (`unstructured`, `PyPDF2`), HTML (BeautifulSoup), OCR (Tesseract, cloud APIs)
- Metadata extraction: document-level, content-based, structural, contextual metadata
- Deduplication: MinHash + Spark ML locality-sensitive hashing
- Filtering: relevance, toxicity, PII detection
- Chunking strategies: fixed-size, paragraph-based, format-specific (Markdown/HTML headers), semantic
- Chunk overlap and semantic coherence trade-offs
- Embedding models for RAG: `bge-large-en-v1.5`, `text-embedding-3-small`, `GTE-Large-v1.5`
- Delta Lake fundamentals: ACID transactions, schema enforcement, time travel, Auto Loader
- Medallion architecture applied to GenAI data pipelines
- Unity Catalog Volumes: managed vs. external, path conventions, access control
- Databricks AI Search (formerly Vector Search): HNSW, hybrid keyword-similarity search, RRF scoring
- Auto Loader for incremental document ingestion

## How This Section Fits

Section 01 gave you the mental model for embeddings and context windows. This section shows you
how to build the data infrastructure those concepts depend on. Section 03 (Application Development)
assumes you have a working vector index to query — built in this section. Section 04 (Deploying)
covers putting your data pipeline into production via Model Serving and CI/CD.

## Study Tips

- **The pipeline is sequential but iterative.** Ingest → Parse → Chunk → Embed → Index is the
  forward path, but in practice you tune chunking and embedding choices based on retrieval quality
  and iterate. Design for iteration from the start.
- **Chapter 01 (Extraction and Chunking) is the higher-exam-weight chapter.** Chunking strategy
  choices and their trade-offs are a common exam topic. Know the four strategies and when each is appropriate.
- **"Databricks AI Search" is the current product name** for what was formerly called Vector Search.
  Exam questions may use either name — know both.
- **Lakeflow Spark Declarative Pipelines** (formerly Delta Live Tables / DLT) is Databricks' preferred
  orchestration layer for data pipelines. You do not need deep LFP knowledge for this section, but
  know that it exists and what it provides.
- **Delta Lake is the default table format on Databricks** — you do not need to specify it explicitly.
  Every `CREATE TABLE` or DataFrame `write` produces a Delta table unless you say otherwise.

## Chapters / Modules

- [ ] [01 Document Processing](./01-document-processing/) — 9 hrs
