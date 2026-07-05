# Packaging and Registration

**Part of section:** Assembling and Deploying Apps
**Estimated time:** 5 hours
**Prerequisites:** Section 03 — LangGraph agents wrapped with `mlflow.pyfunc.ResponsesAgent`; Section 02 — Delta tables and embedding concepts
**Exam mapping:** Assembling and Deploying Applications domain (~20%) — model logging, MLflow, Unity Catalog registration, and AI Search index configuration

## Overview

Before an agent can be served, it must be **packaged** into an MLflow model and **registered** to
Unity Catalog, and the **AI Search index** it retrieves from must be created and configured. This
module covers both halves of "assembling" the application: turning agent code into a governed,
versioned model artifact, and standing up the vector index that grounds it.

## Learning Outcomes

By completing this module, you will be able to:

- Log a LangGraph `ResponsesAgent` using MLflow 3's models-from-code pattern and declare its resources
- Register the logged model to Unity Catalog and manage deployment status with aliases (not stages)
- Create and configure an AI Search endpoint and index with the correct type and sync mode for a workload

## Topics Covered

- MLflow 3 models-from-code logging, `code_paths`, `resources`, signature auto-inference
- Unity Catalog registration: three-level namespace, aliases vs stages, permissions, `models:/<id>` URIs
- AI Search endpoints (STANDARD vs STORAGE_OPTIMIZED) and indexes (Delta Sync managed/self-managed, Direct Access)
- TRIGGERED vs CONTINUOUS sync, Change Data Feed prerequisite, query types (ANN/full-text/hybrid), reranking

## How This Module Fits

This is the first half of Section 04. It produces the two artifacts the next module (Serving and
Integration) deploys: a registered agent model and a queryable AI Search index. You cannot serve an
agent (Module 02) until it is logged and registered here, and the agent's retrieval tool depends on
the index configured here.

## Study Tips

- Author Chapter 01 (logging/registration) before Chapter 02 (index) if learning linearly, but note
  they are independent artifacts — in practice the index is often built first so the agent can
  reference it at log time via `resources`.
- The most exam-critical facts here are the **models-from-code file-path pattern**, **UC three-level
  names**, **aliases replacing stages**, and the **TRIGGERED/CONTINUOUS + STANDARD/STORAGE_OPTIMIZED**
  matrix. Drill those.
- Do not confuse the *index* (searchable structure) with the *endpoint* (compute that serves it).

## Chapters

- [ ] [01 PyFunc, MLflow, and Unity Catalog Registration](./01-pyfunc-mlflow-and-unity-catalog-registration/notes.md) — 2.5 hrs
- [ ] [02 Vector Search Indexing and Config](./02-vector-search-indexing-and-config/notes.md) — 2.5 hrs
