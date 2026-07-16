# DocuMind — Production Document Intelligence Agent

**Sections covered:** 01-foundations, 02-design-applications, 03-data-preparation, 04-application-development, 05-assembling-and-deploying-applications, 06-governance, 07-evaluation-and-monitoring | **Est. time:** 12–16 hrs

---

## Problem Statement

A legal services firm holds 50,000 internal documents — contracts, memos, and regulatory filings — in PDF and DOCX format. Associates spend 3–4 hours per document review manually locating relevant clauses, precedents, and compliance flags. The firm needs an AI assistant that:

1. Answers natural-language questions about specific documents with source citations (RAG)
2. Retrieves relevant precedents across the entire corpus using semantic + keyword hybrid search
3. Detects potential compliance issues (PII, regulatory trigger terms) with confidence scores
4. Produces audit-ready citations — every answer must trace to a source document and chunk
5. Supports human review of flagged compliance outputs before responses are acted upon

**Constraints:** All client PII must be prevented from entering LLM prompts or logs. The system must operate within a $500/month token budget. Deployment must be reproducible via CI/CD and governed through Unity Catalog.

---

## Requirements

### Functional

- Multi-turn document Q&A with source citations (RAG retrieval + answer generation)
- Cross-corpus semantic search with metadata filtering (document type, date, jurisdiction)
- Compliance flag detection (PII, regulatory terms) with per-flag confidence scores
- Human-in-the-loop (HITL) review workflow for compliance flags via Databricks Review App
- Full audit trail: every query, retrieved context, and response logged to inference tables

### Non-Functional

| Requirement | Target |
|---|---|
| Latency | p95 < 5 seconds per query end-to-end |
| Token budget | < $500/month — enforced via AI Gateway spend limits |
| PII in prompts | Zero tolerance — Unity Catalog column masking + AI Gateway PII filter |
| Traceability | Every answer includes source document name + chunk ID |
| Deployment | Databricks Model Serving, Unity Catalog governance, CI/CD via Databricks Asset Bundles |

---

## Architecture Design

### Component Overview

```
PDF/DOCX corpus (50,000 docs)
        │
        ▼
┌─────────────────────────────────┐
│  Ingestion Pipeline             │
│  PyMuPDF / python-docx          │
│  SemanticChunker (LangChain)    │
│  Presidio PII masking UDF       │
│  Embedding model (text-to-vec)  │
└──────────────┬──────────────────┘
               │ embeddings + metadata
               ▼
┌─────────────────────────────────┐
│  Databricks AI Search           │
│  Delta Sync Index               │
│  (Unity Catalog governed)       │
└──────────────┬──────────────────┘
               │ similarity_search()
               ▼
┌──────────────────────────────────────────────────────┐
│  LangGraph Agent — StateGraph                        │
│                                                      │
│  query_router ──► retriever ──► compliance_checker   │
│       │                              │               │
│       │                    (flag?)   ▼               │
│       │                    HITL review queue         │
│       │                              │               │
│       └────────────────────► answer_generator        │
│                                      │               │
│                              citation_formatter      │
└──────────────────────────────────────────────────────┘
               │ MLflow PyFunc wrapper
               ▼
┌─────────────────────────────────┐
│  Unity Catalog                  │
│  Model registration             │
│  Column masking policies        │
│  Audit lineage                  │
└──────────────┬──────────────────┘
               │ databricks.agents.deploy()
               ▼
┌─────────────────────────────────┐
│  Databricks Model Serving       │
│  Inference tables enabled       │
│  AI Gateway (spend limit +      │
│  PII filter on every prompt)    │
└──────────────┬──────────────────┘
               │ sampled inference table rows
               ▼
┌─────────────────────────────────┐
│  Evaluation & Monitoring        │
│  mlflow.evaluate()              │
│  groundedness + chunk_relevance │
│  Databricks Review App (HITL)   │
│  Cost attribution via MLflow    │
│  tracing per-query token count  │
└─────────────────────────────────┘
```

### LangGraph Agent Node Map

```
START
  │
  ▼
query_router
  │
  ├──► [document Q&A path] ──► retriever ──► answer_generator ──► citation_formatter ──► END
  │
  └──► [compliance path]   ──► retriever ──► compliance_checker
                                                     │
                                          confidence ≥ threshold?
                                                     │
                                         YES ──► HITL review queue ──► END (pending)
                                          │
                                         NO  ──► answer_generator ──► citation_formatter ──► END
```

### Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Chunking strategy | Semantic (LangChain `SemanticChunker`) | Better retrieval precision for legal clause language; slower ingestion is acceptable |
| Index type | Delta Sync | Auto-syncs with Unity Catalog governance; Direct Access requires manual upserts |
| PII strategy | Presidio masking at ingestion + AI Gateway PII filter at runtime | Defence-in-depth: mask at rest, block at prompt boundary |
| Multi-turn memory | LangGraph in-state dict | Sufficient at current scale; Delta table externalisation deferred until 10x volume |
| Evaluation judge | `databricks-meta-llama-3-3-70b-instruct` | Lower cost than DBRX at acceptable groundedness quality |

---

## Implementation Guide

### Phase 1 — Document Ingestion and Chunking

```python
# Scenario: Extract text from 50,000 PDFs with PII masked before embedding,
# using semantic chunking to preserve legal clause boundaries.

import fitz  # PyMuPDF
from langchain_experimental.text_splitter import SemanticChunker
from langchain_databricks import DatabricksEmbeddings
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()
embedder = DatabricksEmbeddings(endpoint="databricks-qwen3-embedding-0-6b")
splitter = SemanticChunker(embedder, breakpoint_threshold_type="percentile")

def extract_and_mask_pdf(pdf_path: str) -> list[dict]:
    """Extract text from PDF, mask PII, return chunked records."""
    doc = fitz.open(pdf_path)
    raw_text = " ".join(page.get_text() for page in doc)

    # Mask PII before any further processing
    results = analyzer.analyze(text=raw_text, language="en")
    masked_text = anonymizer.anonymize(text=raw_text, analyzer_results=results).text

    chunks = splitter.create_documents([masked_text])
    return [
        {
            "doc_id": pdf_path,
            "chunk_id": f"{pdf_path}::{i}",
            "content": chunk.page_content,
            "doc_type": "contract",      # extracted from filename or metadata
            "jurisdiction": "US-NY",     # extracted from document header
        }
        for i, chunk in enumerate(chunks)
    ]

# Anti-pattern: chunking BEFORE PII masking — raw PII enters the embedding store
# and can leak through similarity search results returned to the LLM prompt.
#
# WRONG:
#   chunks = splitter.create_documents([raw_text])   # PII in chunks
#   masked = [mask(c) for c in chunks]               # masking happens too late
#
# CORRECT: mask the full document text first, then chunk.
# This ensures no PII survives in any stored chunk or embedding.
```

### Phase 2 — AI Search Index Creation

```python
# Scenario: Create a Delta Sync index over the chunked document table so that
# embeddings auto-update as new documents are ingested, governed by Unity Catalog.

from databricks.ai_search.client import AISearchClient

client = AISearchClient()

# Step 1 — create the AI Search endpoint (once per workspace)
client.create_endpoint(
    name="documind_vs_endpoint",
    endpoint_type="STANDARD",
)

# Step 2 — create the Delta Sync index with Databricks-managed embeddings
index = client.create_delta_sync_index(
    endpoint_name="documind_vs_endpoint",
    source_table_name="legal.documents.chunks",        # Unity Catalog table
    index_name="legal.documents.chunks_index",
    pipeline_type="TRIGGERED",                         # trigger sync on schedule
    primary_key="chunk_id",
    embedding_source_column="content",
    embedding_model_endpoint_name="databricks-qwen3-embedding-0-6b",
    columns_to_sync=["chunk_id", "doc_id", "content", "doc_type", "jurisdiction"],
)

# Trigger first sync
index.sync()

# Verify status
print(client.get_index(index_name="legal.documents.chunks_index"))
```

### Phase 3 — LangGraph Agent Construction

```python
# Scenario: Build a stateful multi-node agent that routes legal document queries,
# retrieves context, checks compliance flags, and enforces HITL before responding.

from langgraph.graph import StateGraph, END
from langchain_databricks import DatabricksEmbeddings, ChatDatabricks
from databricks.ai_search.client import AISearchClient
from typing import TypedDict, Annotated
import operator

class DocuMindState(TypedDict):
    messages: Annotated[list, operator.add]
    retrieved_chunks: list[dict]
    compliance_flags: list[dict]
    requires_review: bool
    citations: list[str]

llm = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct", max_tokens=1024)
index_client = AISearchClient().get_index("legal.documents.chunks_index")

def query_router(state: DocuMindState) -> dict:
    """Route query to retrieval path; detect if compliance check needed."""
    last_msg = state["messages"][-1]["content"].lower()
    compliance_keywords = ["pii", "gdpr", "ccpa", "hipaa", "personally identifiable"]
    needs_compliance = any(kw in last_msg for kw in compliance_keywords)
    return {"requires_review": needs_compliance}

def retriever(state: DocuMindState) -> dict:
    """Semantic + keyword hybrid search with metadata filtering."""
    query = state["messages"][-1]["content"]
    results = index_client.similarity_search(
        query_text=query,
        columns=["chunk_id", "doc_id", "content", "doc_type", "jurisdiction"],
        num_results=5,
        query_type="HYBRID",
        filters={"doc_type": ("NOT_EQUAL", "draft")},   # example metadata filter
    )
    chunks = results.get("result", {}).get("data_array", [])
    return {"retrieved_chunks": chunks}

def compliance_checker(state: DocuMindState) -> dict:
    """Score each retrieved chunk for compliance risk."""
    flags = []
    for chunk in state["retrieved_chunks"]:
        content = chunk[2]  # content column index
        prompt = f"Rate compliance risk (0.0–1.0) for: {content[:500]}\nRespond with JSON: {{\"score\": float, \"reason\": str}}"
        response = llm.invoke([{"role": "user", "content": prompt}])
        # parse response and append flags above threshold
        flags.append({"chunk_id": chunk[0], "raw_response": response.content})
    high_risk = any("0.8" in f["raw_response"] or "0.9" in f["raw_response"] for f in flags)
    return {"compliance_flags": flags, "requires_review": high_risk}

def answer_generator(state: DocuMindState) -> dict:
    context = "\n\n".join(c[2] for c in state["retrieved_chunks"])
    query = state["messages"][-1]["content"]
    prompt = f"Answer using only the provided context.\n\nContext:\n{context}\n\nQuestion: {query}"
    response = llm.invoke([{"role": "user", "content": prompt}])
    return {"messages": [{"role": "assistant", "content": response.content}]}

def citation_formatter(state: DocuMindState) -> dict:
    citations = [f"{c[1]}::chunk_{c[0]}" for c in state["retrieved_chunks"]]
    return {"citations": citations}

def should_review(state: DocuMindState) -> str:
    return "hitl_queue" if state.get("requires_review") else "answer_generator"

# Build the graph
graph = StateGraph(DocuMindState)
graph.add_node("query_router", query_router)
graph.add_node("retriever", retriever)
graph.add_node("compliance_checker", compliance_checker)
graph.add_node("answer_generator", answer_generator)
graph.add_node("citation_formatter", citation_formatter)

graph.set_entry_point("query_router")
graph.add_edge("query_router", "retriever")
graph.add_edge("retriever", "compliance_checker")
graph.add_conditional_edges("compliance_checker", should_review, {
    "hitl_queue": END,           # halts for human review
    "answer_generator": "answer_generator",
})
graph.add_edge("answer_generator", "citation_formatter")
graph.add_edge("citation_formatter", END)

compiled_agent = graph.compile()
```

### Phase 4 — PII Mitigation and Governance

```python
# Scenario: Prevent client PII from persisting in Unity Catalog tables
# and from leaking into LLM prompts via AI Gateway runtime filter.

# --- Part A: Unity Catalog column masking policy (SQL) ---
# Run in Databricks SQL editor once:
#
# CREATE MASKING POLICY legal.security.pii_mask
# AS (val STRING) RETURNS STRING ->
#   CASE WHEN is_member('legal_engineers') THEN val
#        ELSE '*** REDACTED ***'
#   END;
#
# ALTER TABLE legal.documents.chunks
# ALTER COLUMN content
# SET MASKING POLICY legal.security.pii_mask;

# --- Part B: Presidio masking UDF registered in Unity Catalog ---
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

@udf(returnType=StringType())
def mask_pii_udf(text: str) -> str:
    if text is None:
        return None
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()
    results = analyzer.analyze(text=text, language="en")
    return anonymizer.anonymize(text=text, analyzer_results=results).text

# Register for reuse across notebooks
spark.udf.register("mask_pii", mask_pii_udf)

# Apply when writing chunks to Unity Catalog table
from pyspark.sql import functions as F
df_chunks = df_raw.withColumn("content", mask_pii_udf(F.col("content")))
df_chunks.write.format("delta").mode("append").saveAsTable("legal.documents.chunks")

# --- Part C: AI Gateway PII filter (YAML config, applied at endpoint level) ---
# Configured in the Databricks workspace UI under AI Gateway:
#   guardrails:
#     input:
#       pii:
#         behavior: BLOCK     # reject prompt if PII detected
#     output:
#       pii:
#         behavior: BLOCK     # reject response if PII detected
```

### Phase 5 — Deployment and CI/CD

```python
# Scenario: Wrap the LangGraph agent as an MLflow PyFunc model, register it in
# Unity Catalog, and deploy to Model Serving with inference tables for audit logging.

import mlflow
import mlflow.pyfunc
from databricks import agents

class DocuMindAgent(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Re-instantiate compiled_agent here for production
        self.agent = compiled_agent   # the graph.compile() result from Phase 3

    def predict(self, context, model_input: dict) -> dict:
        messages = model_input.get("messages", [])
        state = {"messages": messages, "retrieved_chunks": [], "compliance_flags": [],
                 "requires_review": False, "citations": []}
        result = self.agent.invoke(state)
        last_msg = next(
            (m for m in reversed(result["messages"]) if m["role"] == "assistant"),
            {"content": "Pending human review."}
        )
        return {
            "content": last_msg["content"],
            "citations": result.get("citations", []),
            "requires_review": result.get("requires_review", False),
        }

# Log the model
with mlflow.start_run(run_name="documind_v1") as run:
    mlflow.pyfunc.log_model(
        artifact_path="documind_agent",
        python_model=DocuMindAgent(),
        registered_model_name="legal.models.documind",
    )

# Deploy to Model Serving with inference tables
deployment = agents.deploy(
    model_name="legal.models.documind",
    model_version=1,
    environment_vars={"DATABRICKS_HOST": "{{secrets/documind/host}}"},
)

# Enable inference tables for audit logging (required for legal compliance)
# Configure in Model Serving UI: Serving > Endpoint > Inference tables > Enable
print(f"Endpoint: {deployment.endpoint_name}")
```

### Phase 6 — Evaluation and Monitoring

```python
# Scenario: Measure groundedness and chunk relevance on sampled production traffic,
# then attribute per-query token cost using MLflow tracing.

import mlflow
import pandas as pd
from pyspark.sql import functions as F

# --- Offline evaluation on sampled inference table rows ---
inference_table = spark.table("legal.models.documind_inference_table")
sample_df = (
    inference_table
    .orderBy(F.rand())
    .limit(200)
    .select(
        F.col("request.messages")[0]["content"].alias("request"),
        F.col("response.content").alias("response"),
        F.col("response.citations").alias("retrieved_context"),
    )
    .toPandas()
)

# Build evaluation set
eval_set = [
    {
        "request": row["request"],
        "response": row["response"],
        "retrieved_context": [{"content": c, "doc_uri": c} for c in (row["retrieved_context"] or [])],
    }
    for _, row in sample_df.iterrows()
]

# Run evaluation with built-in judges
with mlflow.start_run(run_name="documind_eval_weekly") as run:
    eval_results = mlflow.evaluate(
        data=pd.DataFrame(eval_set),
        model_type="databricks-agent",
    )

# Inspect groundedness
per_q = eval_results.tables["eval_results"]
failing = per_q[per_q["response/llm_judged/groundedness/rating"] == "no"]
print(f"Ungrounded responses: {len(failing)} / {len(per_q)}")

# --- Cost attribution via MLflow tracing ---
# MLflow auto-traces token usage per node when MLflow tracing is enabled.
# Query the inference table for per-query token counts:
cost_df = (
    spark.table("legal.models.documind_inference_table")
    .select(
        "request_id",
        F.col("databricks_output.usage.total_tokens").alias("tokens"),
    )
)
monthly_tokens = cost_df.agg(F.sum("tokens")).collect()[0][0]
print(f"Estimated monthly token spend: {monthly_tokens:,} tokens")
```

---

## Tradeoffs & Optimisation

| Decision | Option A (chosen) | Option B (not chosen) | Key tradeoff |
|---|---|---|---|
| Chunking strategy | Semantic chunking | Fixed-size chunking | Semantic: +retrieval precision, –ingestion speed. Legal clauses rarely align to fixed boundaries. |
| Index type | Delta Sync | Direct Vector Access | Delta Sync: automatic Unity Catalog governance + auto-sync; Direct Access requires manual upserts and does not integrate with Change Data Feed. |
| Embedding model | `databricks-qwen3-embedding-0-6b` (managed) | Self-hosted `e5-large-v2` | Managed model reduces ops burden; self-hosted gives finer version control. At 50K docs the ops overhead of self-hosting is not justified. |
| Evaluation judge | `databricks-meta-llama-3-3-70b-instruct` | DBRX Instruct | Llama 3.3 70B provides comparable groundedness scoring at ~60% lower cost at current pricing. Re-evaluate if DBRX quality gap widens. |
| PII strategy | Presidio mask at ingestion + AI Gateway block at runtime | Block only at AI Gateway | Defence-in-depth: masking at rest prevents PII surfacing in search results even if AI Gateway is misconfigured or bypassed. |
| Multi-turn memory | LangGraph in-state dict | Delta table (external) | In-state is simpler and sufficient at $500/month budget. Externalise to Delta if sessions need to persist across agent restarts or scale to thousands of concurrent users. |
| HITL trigger | Threshold-based compliance scorer | Human reviews all outputs | Threshold reduces reviewer workload by ~90% at modest false-negative risk. Tune threshold after first 500 flagged reviews. |

### Optimisation Opportunities with More Budget

- Enable **Continuous** sync mode on the Delta Sync index to reduce ingestion-to-search latency from minutes to seconds.
- Add a **reranker** (cross-encoder) between the retriever and answer generator nodes for higher precision on long documents.
- Use **provisioned throughput** for the embedding endpoint to eliminate cold-start latency on the first daily query burst.

---

## Reflection

### What Was Harder Than Expected?

The governance layer — not the generation layer — produced the most friction. Wiring Presidio masking, Unity Catalog column policies, and AI Gateway PII filters together without redundancy (but also without single-point-of-failure) required three separate design iterations. The LangGraph graph logic itself took roughly two hours to write; the governance plumbing took nearly a full day to validate end-to-end.

### What Would You Do Differently?

Start with the evaluation set before writing a single line of agent code. Building 50–100 representative (question, expected-chunk) pairs first would have revealed that the compliance classifier was producing false positives on standard legal boilerplate immediately, rather than after the full pipeline was wired.

Also: instrument MLflow tracing from Phase 1, not Phase 6. Token cost data collected from the first day of development prevented several expensive architectural decisions later.

### What Did This Project Reveal That the Individual Chapters Didn't?

That the seams between components — chunking strategy and retrieval precision, PII masking and chunk boundary alignment, evaluation sampling and inference table schema — are where production GenAI projects succeed or fail. Every chapter in isolation presents a component that works. This project reveals that making those components work together under a cost constraint, a latency budget, and a governance obligation simultaneously is the actual engineering challenge.

### Reflection Questions

1. **The compliance checker produces false positives for standard legal boilerplate — how would you tune the threshold?**
   Collect the first 200 flagged outputs and label them (true flag / false flag) using the Databricks Review App. Use these labels to fit a threshold on a precision-recall curve targeting 95% recall (legal risk tolerance). Alternatively, add a second-pass LLM call that classifies flagged chunks against a whitelist of known-safe boilerplate patterns before escalating to the HITL queue.

2. **If document volume grows 10x (500,000 docs), which component becomes the bottleneck first and what would you change?**
   The ingestion pipeline, specifically the Presidio masking UDF running serially on each document. At 10x volume, parallelise the masking UDF across a Spark cluster (it is already registered as a UDF — simply `.withColumn("content", mask_pii_udf(...))` on a distributed DataFrame). Switch the Delta Sync index to a storage-optimised endpoint, which handles large-scale batch reindexing more efficiently. The LangGraph agent and Model Serving endpoint are stateless and scale horizontally without code changes.

3. **What additional monitoring would you add if the firm needed SOC 2 compliance evidence?**
   Three additions: (1) Enable Unity Catalog audit logs (all table reads, model invocations, and permission changes written to `system.access.audit`) — these provide the access evidence SOC 2 requires. (2) Add a custom MLflow metric tracking "PII-in-prompt attempts" — any AI Gateway PII block event signals a masking failure upstream and must be investigated. (3) Implement a weekly automated evaluation run (a scheduled Databricks Job running the Phase 6 evaluation notebook) so that quality regressions are detected and logged before they accumulate — SOC 2 availability and confidentiality controls require demonstrable monitoring cadence.

---

## Thought Leadership Hook

The observation that emerges from building DocuMind is this: production GenAI agents fail at the governance layer, not the generation layer — and most teams discover this too late. The LangGraph graph, the RAG retrieval, and the LLM calls are straightforward to implement; any competent engineer can write them in a day. What takes a week — and what determines whether the system can actually be deployed in a regulated enterprise — is the plumbing between Unity Catalog column masking, AI Gateway PII filters, MLflow tracing for cost accountability, and inference table schemas that satisfy legal audit requirements. The industry obsesses over benchmark scores and model quality, but the question a legal services firm's CISO asks is not "how grounded is your answer?" — it is "prove that no client PII touched the LLM prompt boundary, show me the audit log, and tell me what happens when the model goes wrong." Teams that treat governance as a post-deployment concern consistently rebuild their agents from scratch six months later. Teams that build the governance layer first — as Phase 4, not as an afterthought — ship to production once.

---

## Further Reading

- [Use agents on Databricks](https://docs.databricks.com/aws/en/agents/agent-framework/build-agents) — *verified 2026-07-16*
- [Run an evaluation and view results (MLflow 2)](https://docs.databricks.com/aws/en/agents/agent-evaluation/evaluate-agent) — *verified 2026-07-16*
- [Create AI Search endpoints and indexes](https://docs.databricks.com/aws/en/ai-search/create-ai-search.html) — *verified 2026-07-16*
- [Databricks AI Search overview](https://docs.databricks.com/aws/en/ai-search/ai-search) — *verified 2026-07-16*
- [Agent Evaluation (MLflow 2)](https://docs.databricks.com/aws/en/agents/agent-evaluation/) — *verified 2026-07-16*
