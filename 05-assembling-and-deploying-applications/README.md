# Section 05 — Assembling and Deploying Applications

**Estimated time:** ~14 hrs | **Exam domain weight:** ~22% (Assembling & Deploying) | **Prerequisites:** Section 04 — Application Development

---

## Overview

This section covers the full deployment lifecycle for Databricks AI applications: assembling RAG chains and PyFunc wrappers, indexing and querying vector stores, registering models in Unity Catalog, serving them via Model Serving and Foundation Model APIs, and operating them through CI/CD pipelines and user-facing interfaces. It is the highest-weight section alongside Application Development, directly mapping to the "Assembling & Deploying" exam domain (22%).

---

## Learning Outcomes

By completing this section you will be able to:

- Assemble MLflow PyFunc chains with pre- and post-processing and define input/output signatures for serving compatibility
- Build end-to-end RAG pipelines using Databricks Vector Search as the retriever and MLflow model logging for deployment
- Create and configure Vector Search indexes (Delta Sync and Direct Vector Access) and execute similarity, hybrid, and filtered queries
- Register models in Unity Catalog with the three-level namespace, apply Champion/Challenger aliases, and create Model Serving endpoints with traffic splitting
- Call Foundation Model API endpoints from Python and SQL (`ai_query()`), and route traffic through AI Gateway
- Integrate MCP servers with agent tools and implement persistent memory patterns using Delta tables and Vector Search
- Deploy AI applications through Databricks Asset Bundles, manage prompt lifecycle versioning, and surface chat UIs and Genie interfaces

---

## Modules

| # | Module | Est. time | Chapters |
|---|---|---|---|
| 1 | Chains and RAG Assembly | ~4 hrs | 2 (+ LAB-11, LAB-12) |
| 2 | Vector Search and Serving | ~6 hrs | 3 (+ LAB-13, LAB-14, LAB-15) |
| 3 | CI/CD, MCP, and Interfaces | ~4 hrs | 2 (+ LAB-16, LAB-17) |

### Module 1 — Chains and RAG Assembly (`01-chains-and-rag-assembly/`)

| Chapter | Folder | Lab |
|---|---|---|
| PyFunc Chains & Pre/Post-Processing | `01-pyfunc-chains-pre-post-processing/` | LAB-11 |
| RAG Building Blocks & Signatures | `02-rag-building-blocks-and-signatures/` | LAB-12 |

### Module 2 — Vector Search and Serving (`02-vector-search-and-serving/`)

| Chapter | Folder | Lab |
|---|---|---|
| Vector Search Index & Config | `01-vector-search-index-and-config/` | LAB-13 |
| Unity Catalog Registration & Serving | `02-unity-catalog-registration-serving/` | LAB-14 |
| Foundation Model APIs & ai_query() | `03-foundation-model-apis-and-ai-query/` | LAB-15 |

### Module 3 — CI/CD, MCP, and Interfaces (`03-cicd-mcp-and-interfaces/`)

| Chapter | Folder | Lab |
|---|---|---|
| MCP Servers & Persistent Memory | `01-mcp-servers-and-persistent-memory/` | LAB-16 |
| CI/CD, Prompt Lifecycle & UIs | `02-cicd-prompt-lifecycle-and-uis/` | LAB-17 |

---

## How This Section Fits

Section 04 (Application Development) taught you to build individual LangGraph agents and LangChain chains. This section takes those artifacts and operationalises them: wrapping in PyFunc for MLflow compatibility, backing them with Vector Search retrieval, registering to Unity Catalog for governed access, serving through Model Serving endpoints, and delivering stable deployments via Asset Bundles. It feeds directly into Section 06 (Governance) — you need to understand what is being governed — and Section 07 (Evaluation & Monitoring) — you need a live endpoint to monitor.

---

## Study Tips

- **Vector Search is fast-evolving**: the product was renamed from "AI Search" in late 2024/2025; verify index type names (`DELTA_SYNC` vs `DIRECT_VECTOR_ACCESS`) and endpoint creation APIs against current docs before the exam.
- **MCP is fast-evolving**: the Databricks MCP integration (managed MCP servers, `use-mcp-in-agents`) was introduced in 2025 and is actively expanding. Check the fast-evolving callout boxes in `01-mcp-servers-and-persistent-memory/notes.md`.
- **Genie/Genie Agents**: Genie Spaces was renamed to Genie Agents as of July 2026. Exam questions may use either name; understand the underlying conversational BI concept.
- **Signature contract**: the most common deployment failure is a mismatch between the `ModelSignature` logged with the model and the payload sent to the serving endpoint. Study the `ColSpec`/`TensorSpec` section in `01-pyfunc-chains-pre-post-processing/notes.md` carefully.
- **Unity Catalog namespace**: memorise `catalog.schema.model_name` — the three-level name is required for `mlflow.register_model()` when `registry_uri = "databricks-uc"`.
- **Labs are sequential within a module**: LAB-11 output (a logged PyFunc model) feeds LAB-12 (adding a RAG retriever); LAB-13 (Vector Search index) is consumed by LAB-12's retriever. Run them in order.
