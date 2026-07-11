# Lab: Implementing and Comparing Chunking Strategies

**Lab:** LAB-03 | **Section:** 01-ingestion-and-chunking | **Module:** 03-data-preparation | **Est. time:** 1 hr

## Objective

You will build a Python script using LangChain to chunk a sample text using both naive fixed-size chunking and recursive character chunking, and visually inspect the output to understand why structural boundaries matter for RAG context.

## Prerequisites

- A Python 3.9+ environment.
- Basic understanding of Python and RAG concepts.

## Setup

Install the required LangChain text splitters package.

```bash
# Setup commands
pip install langchain-text-splitters
```

## Steps

### Step 1 — Define the Source Text

Create a Python script (`chunking_lab.py`) and define a sample text that represents a realistic, somewhat structured document.

```python
# Step 1 code/config
document_text = """
# Introduction to Databricks AI Search

Databricks AI Search is a powerful tool. It allows you to build RAG applications easily.

## Key Features

1. Vector Search: It uses the HNSW algorithm for fast retrieval.
2. Hybrid Search: Combines keyword and semantic similarity.
3. Access Control: Integrates perfectly with Unity Catalog.

Remember to always configure your chunking strategy carefully!
"""
```

### Step 2 — Implement Naive Fixed-Size Chunking

Use the basic `CharacterTextSplitter` to cut the text arbitrarily. We will set a very small chunk size to exaggerate the effect.

```python
# Step 2 code/config
from langchain_text_splitters import CharacterTextSplitter

naive_splitter = CharacterTextSplitter(
    separator="", # Force it to cut exactly at the character limit
    chunk_size=50,
    chunk_overlap=0
)

naive_chunks = naive_splitter.split_text(document_text)

print("--- Naive Fixed-Size Chunks ---")
for i, chunk in enumerate(naive_chunks):
    print(f"Chunk {i+1}: '{chunk}'")
```

### Step 3 — Implement Recursive Character Chunking

Now, use the `RecursiveCharacterTextSplitter` with an overlap, which will attempt to respect the newlines and spaces.

```python
# Step 3 code/config
from langchain_text_splitters import RecursiveCharacterTextSplitter

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10,
    separators=["\n\n", "\n", " ", ""]
)

recursive_chunks = recursive_splitter.split_text(document_text)

print("\n--- Recursive Character Chunks ---")
for i, chunk in enumerate(recursive_chunks):
    print(f"Chunk {i+1}: '{chunk}'")
```

## Validation

Run the script to verify the lab succeeded. You should observe that the naive splitter cuts words in half (e.g., "pow" and "erful"), destroying meaning. The recursive splitter will try to keep words and phrases intact.

```bash
# Validation command / expected output
python chunking_lab.py
```

*Expected observation:* The Naive chunks will end abruptly regardless of word boundaries. The Recursive chunks will overlap slightly and avoid cutting words in half, making them much more useful for an embedding model.

## Teardown

There are no persistent cloud resources to tear down for this local scripting lab. Simply delete `chunking_lab.py` if desired.

## Reflection Questions

1. Look at the output of the naive splitter. If you embedded those chunks, how would an LLM interpret a chunk that contains only half of a technical term?
2. How would you adjust the `chunk_size` and `chunk_overlap` parameters if this text was going into an embedding model with a strict 512-token limit?
3. How does this local splitting logic connect to building a Databricks Vector Search index?
