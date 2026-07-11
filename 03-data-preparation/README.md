# Section 03: Data Preparation

**Estimated time:** 4 hrs | **Exam domain weight:** 14% | **Prerequisites:** Section 02 (Design Applications)

## Overview

This section covers the foundational steps of preparing data for Retrieval-Augmented Generation (RAG) and other generative AI applications. It focuses on taking raw, unstructured, or semi-structured data from various sources and processing it into a form that a vector store and LLM can efficiently utilize. 

## Learning Outcomes

By completing this section you will be able to:
- Design pipelines to extract text from unstructured document formats (like PDFs) using Databricks tools and Unstructured.
- Apply recursive, fixed, and semantic chunking strategies to preserve contextual boundaries.
- Continuously ingest data into Delta Tables using Unity Catalog Volumes and Change Data Feed for downstream vector indexing.
- Cleanse, deduplicate, and improve source document quality before they enter the vector database.
- Evaluate retrieval quality using Databricks MLflow and improve relevance with cross-encoder reranking.

## Modules

| # | Module | Est. time | Chapters |
|---|---|---|---|
| 1 | Ingestion and Chunking | 2.5 hrs | 3 |
| 2 | Retrieval Quality | 1.5 hrs | 2 |

## How This Section Fits

After Section 02 covered how to design your application logic and agents, this section ensures you have the high-quality data foundation needed to feed those agents. The concepts learned here (Chunking, Delta ingestion, filtering) directly unlock the actual RAG application development happening in Section 04.

## Study Tips

- Focus on the distinction between Unity Catalog Volumes (for raw files like PDFs) and Delta Tables (for parsed chunks).
- Understand why Change Data Feed (CDF) is critical for syncing Delta Tables with Databricks Vector Search.
- Spend time on the visual overviews in the chunking strategies chapter to internalize why character counts alone are insufficient for RAG.
