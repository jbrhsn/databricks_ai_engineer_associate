# LAB-05: Implementing a Two-Stage Retrieval Pipeline with Reranking

**Lab:** LAB-05 | **Section:** 03-Data Preparation | **Module:** 02-Retrieval Quality | **Est. time:** 1 hrs

## Objective

The learner will build a two-stage retrieval pipeline using a fast initial vector search followed by a cross-encoder reranker to improve the precision of retrieved documents.

## Prerequisites

- Completion of Lab 04 (Vector Database Setup).
- A Python environment with `langchain`, `langchain-community`, and `sentence-transformers` installed.
- (Optional but recommended) Databricks workspace with MLflow configured.

## Setup

Set up the local environment and install necessary libraries to run the open-source cross-encoder.

```bash
# Setup commands
pip install langchain langchain-community sentence-transformers faiss-cpu
```

## Steps

### Step 1 — Create a Mock Document Corpus

We will create a small corpus of documents and load them into a local FAISS vector store to simulate our base retriever. We'll include one highly specific document and several generally similar ones.

```python
# Step 1 code/config
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

docs = [
    Document(page_content="The company provides 12 weeks of parental leave for US full-time employees."),
    Document(page_content="US contractors do not receive paid parental leave."),
    Document(page_content="General leave policies depend on the local jurisdiction."),
    Document(page_content="In the UK, part-time employees are entitled to statutory parental leave pro-rata."),
    Document(page_content="UK full-time employees receive 16 weeks of parental leave.")
] * 10 # Duplicate to simulate a larger corpus (50 docs total)

# Initialize a standard bi-encoder for base retrieval
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)
```

### Step 2 — Test the Base Retriever

Query the base retriever and observe that the bi-encoder might not rank the most relevant document first due to broad semantic overlap.

```python
# Step 2 code/config
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

query = "What is the parental leave policy for part-time employees in the UK?"
print("--- Base Retrieval ---")
for i, doc in enumerate(base_retriever.invoke(query)):
    print(f"Rank {i+1}: {doc.page_content}")
```

### Step 3 — Implement the Cross-Encoder Reranker

Wrap the base retriever in a `ContextualCompressionRetriever` using a `CrossEncoderReranker`.

```python
# Step 3 code/config
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Initialize the cross-encoder model (this will download a small model locally)
reranker_model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-TinyBERT-L-2-v2")

# Configure the compressor to return only the top 2 results
compressor = CrossEncoderReranker(model=reranker_model, top_n=2)

# Combine the base retriever (which gets 5 docs) with the reranker
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)
```

### Step 4 — Test the Two-Stage Pipeline

Execute the exact same query against the new two-stage pipeline.

```python
# Step 4 code/config
print("\n--- Reranked Retrieval ---")
for i, doc in enumerate(compression_retriever.invoke(query)):
    print(f"Rank {i+1}: {doc.page_content}")
```

## Validation

Verify that the reranked retrieval correctly places the UK part-time policy at Rank 1, whereas the base retrieval may have buried it below US policies or general full-time policies.

```text
# Expected Output (approximate)
--- Base Retrieval ---
Rank 1: UK full-time employees receive 16 weeks of parental leave.
Rank 2: The company provides 12 weeks of parental leave for US full-time employees.
Rank 3: In the UK, part-time employees are entitled to statutory parental leave pro-rata.
Rank 4: ...

--- Reranked Retrieval ---
Rank 1: In the UK, part-time employees are entitled to statutory parental leave pro-rata.
Rank 2: UK full-time employees receive 16 weeks of parental leave.
```

## Teardown

If you deployed any persistent vector search endpoints or MLflow experiments during this lab, ensure they are deleted to avoid ongoing compute charges.

```python
# Cleanup (if applicable)
del vectorstore
del reranker_model
```

## Reflection Questions

1. What would happen to the query latency if you changed the base retriever's `k` parameter from 5 to 500?
2. If the base retriever entirely missed the correct document, could the reranker find it?
3. How does implementing this reranker improve the "Groundedness" metric when evaluated by MLflow Agent Evaluation?
