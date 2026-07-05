<!--
CAPSTONE SUBMISSION TEMPLATE
Copy this file (or fill it in place) and complete every section from YOUR own project.
See ./README.md for the project brief, requirements, phased approach, and self-assessment rubric.
The "why" fields matter more than the "what" — if you can't justify a choice in a sentence or two,
that's a gap to close before the exam. Delete these HTML comments once you start writing.
-->

# Capstone Submission — End-to-End RAG Agent on Databricks

**Author:**
**Date:**
**Corpus chosen:** <!-- e.g., "this repo's Markdown notes", "a set of 40 product-doc PDFs" -->
**Total time spent:**
**Time to complete:** <!-- honest estimate of hours -->

---

## 0. One-Paragraph Summary

<!-- In 3–5 sentences: what does your agent do, over what corpus, and what's the single most
interesting thing you learned building it? Write this LAST, after everything else is filled in. -->



---

## 1. Architecture Overview

Draw or describe the end-to-end flow. Name the Databricks product used at each stage.

```
[ corpus source ]
      │  (parse / chunk)
      ▼
[ Delta table in Unity Catalog:  ______.______.______ ]
      │  (Delta Sync)
      ▼
[ AI Search index:  ______ ]  on endpoint [ ______ ]
      │  (retrieve top-k)
      ▼
[ LangGraph agent  (ChatDatabricks: ______) ]  →  grounded answer + citations
      │  (log: models-from-code → register: UC → deploy: agents.deploy)
      ▼
[ Model Serving endpoint:  ______ ]  → (optional) [ Databricks App UI ]
      │
      ▼
[ MLflow tracing + mlflow.genai.evaluate() ]
```

**Product/name check (2026):** confirm you used current names where applicable — AI Search (not Vector
Search), Unity AI Gateway (not Mosaic AI Gateway), MLflow **aliases** (not stages), Declarative
Automation Bundles (not Asset Bundles).

---

## 2. Phase 1 — Data Preparation

**Source & size:** <!-- how many docs, what format, roughly how many tokens/chunks --> 

**Parsing approach:**

**Chunking strategy chosen:** <!-- fixed / recursive / semantic / document-structure-aware -->

**Why this strategy (1–2 sentences):**

**Metadata attached to each chunk:** <!-- source path, title, section, chunk index, ... -->

**Delta table (three-level UC name):** `______.______.______`

**Any quality/dedup handling:** <!-- e.g., MinHash dedup, empty-chunk filtering, or "none — why" -->

**Key code (ingestion → Delta):**
```python
# paste the essential ingestion/chunking/write-to-Delta code
```

---

## 3. Phase 2 — Indexing (AI Search)

**Endpoint name & type:** <!-- STANDARD or STORAGE_OPTIMIZED --> 

**Index name & type:** <!-- Delta Sync or Direct Access -->

**Embedding approach:** <!-- managed embeddings (which model?) or self-managed -->

**Embedding model & dimension:** <!-- e.g., bge-large-en (512 tokens) --> 

**Sync mode:** <!-- TRIGGERED or CONTINUOUS -->

**Why this endpoint type + sync mode (1–2 sentences):**
<!-- Remember: STORAGE_OPTIMIZED supports TRIGGERED only. Change Data Feed is required for Delta Sync. -->

**Did you enable Change Data Feed on the source table?** <!-- yes/no + how -->

**Key code (create endpoint + index + sync):**
```python
# paste the essential index-creation and sync code
```

---

## 4. Phase 3 — The Agent (LangGraph)

**Framework:** LangGraph + `ChatDatabricks` (model: `______`)

**Graph shape:** <!-- describe nodes and edges: e.g., retrieve → grounded-generate; any conditional edges? -->

**Retrieval config:** <!-- top-k, query type (ANN/hybrid), filters, reranking? -->

**Grounding prompt (paste it):**
```
# paste the grounded system/prompt template that injects retrieved context
```

**How you produce citations:**

**Key agent code:**
```python
# paste the StateGraph definition and the retrieve + generate nodes
```

---

## 5. Phase 4 & 5 — Package, Register, Deploy

**Logging pattern:** models-from-code (`python_model="______.py"`)

**`resources` declared at log time:** <!-- MUST list serving endpoint + vector index or serving fails auth -->
- `DatabricksServingEndpoint(...)`:
- `DatabricksVectorSearchIndex(...)`:

**Unity Catalog registered name:** `______.______.______`

**Alias set:** <!-- e.g., @champion / @production (NOT a stage) -->

**Deployment call:** `agents.deploy(...)` — endpoint name: `______`

**Serving config:** <!-- scale-to-zero? workload size? concurrency? -->

**Proof it's live — a real query + response:**
```python
# paste the query (OpenAI client / MLflow Deployments SDK / REST / ai_query) and the returned answer
```

**Anything that failed the first time here?** <!-- forgotten resource? signature issue? auth error? -->

---

## 6. Phase 6 — Evaluation

**Evaluation dataset:** <!-- how many Q/A pairs, how you built them --> 

**Scorers used:** <!-- correctness, groundedness/faithfulness, relevance, safety, custom judge? -->

**Is tracing enabled?** <!-- yes/no + where you viewed the traces -->

**Results (paste the numbers):**

| Scorer | Score | Notes |
|---|---|---|
|  |  |  |
|  |  |  |
|  |  |  |

**What the scores told you (1–3 sentences):** <!-- which failures were retrieval vs generation? -->

**Any before/after comparison from an improvement you made:** <!-- e.g., naive vs hybrid search -->

**Key eval code:**
```python
# paste the mlflow.genai.evaluate() call and dataset setup
```

---

## 7. Phase 7 — Governance & Safety

**Guardrails applied:** <!-- input (PII/injection), output (unsafe content), grounding check -->

**Where they run:** <!-- in-agent, endpoint guardrails, or Unity AI Gateway? --> 

**Unity Catalog grants / access controls on the corpus:** <!-- who can query the table/index/endpoint? -->

**Any row filters / column masks on sensitive data:** <!-- yes/no + why -->

**Your monitoring plan (even if not fully built):** <!-- inference tables, alerts, drift checks -->

---

## 8. Stretch Goals Attempted (optional)

- [ ] Databricks Apps chat UI
- [ ] CI/CD with Declarative Automation Bundle
- [ ] Hybrid search / reranking (with eval comparison)
- [ ] Prompt Registry versioning
- [ ] Multi-agent / supervisor pattern
- [ ] LLM-as-a-judge custom scorer with alignment

**Notes on any you attempted:**

---

## 9. Failure Modes Observed

<!-- The most instructive part. What broke, what produced wrong/ungrounded answers, and what fixed it?
List at least two. These are exactly the scenarios the exam tests. -->

1. **Symptom:** … → **Cause:** … → **Fix:** …
2. **Symptom:** … → **Cause:** … → **Fix:** …

---

## 10. Self-Assessment Rubric

Score each dimension 0–3 (see `./README.md` for the descriptors). Be honest — a low row is a
pointer to the section to revisit before the exam.

| Dimension | Score (0–3) | Justification |
|---|---|---|
| Data prep (Sec 02) |  |  |
| App development (Sec 03) |  |  |
| Assembling & deploying (Sec 04) |  |  |
| Governance/eval/monitoring (Sec 05) |  |  |
| Foundations & reasoning (Sec 01) |  |  |
| **Total** | **/ 15** |  |

**Reading:** 12–15 exam-ready · 8–11 solid, revisit ≤1 rows · ≤7 rebuild the weakest phase first.

---

## 11. Reflection

**Which lifecycle stage do you understand best now?**

**Which is still shakiest?**

**Top 1–3 things to review before sitting the exam (from your rubric gaps):**
1.
2.
3.

**If you rebuilt this from scratch, what would you do differently?**
