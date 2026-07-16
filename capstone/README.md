# Capstone — DocuMind

DocuMind is a production-ready LangGraph-based document intelligence agent for a legal services firm. It integrates all six exam domains into a single end-to-end system: RAG retrieval over 50,000 legal documents, compliance flag detection with human-in-the-loop review, PII governance via Unity Catalog and AI Gateway, and continuous evaluation on sampled production traffic — deployed to Databricks Model Serving under a $500/month token budget.

---

## Files

| File | Purpose |
|---|---|
| `project-brief.md` | Full capstone spec: problem statement, requirements, architecture diagrams, 6-phase implementation guide with code snippets, tradeoffs table, reflection questions, and thought leadership hook |

---

## How to Use This Capstone

- **Attempt after all 7 content sections** — DocuMind draws on every domain simultaneously. Working through it before completing the sections will surface gaps and send you back to the chapters repeatedly, which is inefficient.
- **Build phase by phase** — each of the 6 phases is independently runnable. Complete Phase 1 fully (ingest 10–20 documents) before writing the LangGraph graph. Validating each component in isolation prevents compound debugging later.
- **Self-assess against the Success Criteria checklist below** — for each phase, tick off every criterion before moving to the next. Leaving criteria unchecked is a signal to revisit the corresponding chapter, not to continue.
- **Use it as a portfolio piece** — the architecture diagram, the PII mitigation design, and the Thought Leadership Hook are all interview-ready artefacts. Write a Medium post or LinkedIn article from the Thought Leadership Hook after you complete the project.
- **Treat reflection questions as exam prep** — questions 1–3 in the Reflection section mirror the analysis-level questions on the exam (domain weights: Assembling & Deploying 22%, Governance 8%, Evaluation 12%). Write out your answers before checking the suggested answers in the brief.

---

## Sections Integrated

| Section | Contribution to DocuMind |
|---|---|
| `01-foundations` | LLM fundamentals, embedding model selection, token budget reasoning, Databricks platform primitives (Unity Catalog, Model Serving, MLflow) |
| `02-design-applications` | RAG architecture design, query routing decision, LangGraph graph design, human-in-the-loop pattern, multi-turn conversation state |
| `03-data-preparation` | PDF/DOCX ingestion pipeline, semantic chunking strategy, PII masking at ingestion, Delta table schema for chunks, Change Data Feed enablement |
| `04-application-development` | LangGraph StateGraph construction, node implementation, conditional edge routing, LangChain retriever and LLM integration, MLflow tracing instrumentation |
| `05-assembling-and-deploying-applications` | MLflow PyFunc wrapping, Unity Catalog model registration, `databricks.agents.deploy()`, inference table enablement, AI Gateway spend limits and PII filter, CI/CD via Databricks Asset Bundles |
| `06-governance` | Unity Catalog column masking policy, Presidio PII masking UDF, AI Gateway PII guardrail configuration, audit log requirements for SOC 2 |
| `07-evaluation-and-monitoring` | `mlflow.evaluate()` with `databricks-agent` model type, groundedness and chunk relevance judges, Databricks Review App for HITL, per-query cost attribution via MLflow tracing, inference table sampling |

---

## Success Criteria

Use this checklist to determine whether each phase is complete before proceeding.

### Phase 1 — Document Ingestion and Chunking
- [ ] PDF/DOCX text extraction runs without error on at least 10 test documents
- [ ] Presidio PII masking UDF is registered in Unity Catalog and applied before chunking
- [ ] Semantic chunker produces variable-length chunks (verify: no chunk is exactly the same length as its neighbours)
- [ ] Chunk records written to Delta table `legal.documents.chunks` with all required metadata columns

### Phase 2 — AI Search Index Creation
- [ ] AI Search endpoint created and in ONLINE state
- [ ] Delta Sync index created with `TRIGGERED` pipeline type
- [ ] Index syncs successfully; index status shows `ONLINE` with non-zero row count
- [ ] Sample similarity search query returns results with content and metadata columns

### Phase 3 — LangGraph Agent
- [ ] `StateGraph` compiles without error; `.get_graph().draw_mermaid()` renders the expected node topology
- [ ] `query_router` correctly distinguishes compliance-sensitive queries from general Q&A in at least 5 test cases
- [ ] `retriever` returns non-empty chunks for at least 3 different natural-language queries
- [ ] `compliance_checker` returns `requires_review: True` for at least one test query with PII language
- [ ] `citation_formatter` returns source doc IDs for every non-flagged response

### Phase 4 — PII Mitigation and Governance
- [ ] Column masking policy applied to `legal.documents.chunks.content`; non-privileged user sees `*** REDACTED ***`
- [ ] AI Gateway PII filter configured at `BLOCK` level on the serving endpoint
- [ ] End-to-end test: construct a query containing a mock PII string; confirm it is blocked at AI Gateway

### Phase 5 — Deployment and CI/CD
- [ ] `DocuMindAgent.predict()` returns a valid response dict for a known test query when called via `mlflow.pyfunc.load_model()`
- [ ] Model registered in Unity Catalog at `legal.models.documind`
- [ ] Model Serving endpoint in READY state; `/invocations` returns a response
- [ ] Inference tables enabled; at least one row appears in the inference table after a test query

### Phase 6 — Evaluation and Monitoring
- [ ] `mlflow.evaluate()` completes on at least 20 sampled inference table rows
- [ ] MLflow Run shows `response/llm_judged/groundedness/percentage` metric
- [ ] At least one failing row identified and root cause traced to a specific retrieved chunk
- [ ] Monthly token count query returns a non-zero value from the inference table
