# Guardrails, Masking, and Data Governance

**Section:** Governance, Evaluation, and Monitoring | **Module:** Governance and Responsible AI | **Est. time:** 4 hrs
**Exam mapping:** Databricks GenAI Engineer Associate — Governance, Evaluation, and Monitoring domain (~15%); guardrails, PII handling, Unity Catalog data governance, and the AI Gateway control plane are directly tested

## Learning Objectives

By the end of this chapter, you will be able to:

- Govern GenAI data and assets with Unity Catalog securables, grants, row filters, and column masks
- Configure input and output guardrails on a serving endpoint (PII, safety, keywords, topics)
- Distinguish endpoint guardrails (PII BLOCK/MASK) from service policies (ALLOW/DENY/ASK)
- Enable and configure Unity AI Gateway features: rate limits, spend caps, usage tracking, payload logging
- Explain Databricks' responsible-AI framing and the DASF security framework at a high level
- Reason about compliance concerns: HIPAA-eligibility, data residency, and securing credentials

## Core Concepts

### Unity Catalog: one governance layer for data *and* AI

The foundational idea of governance on Databricks is that **Unity Catalog (UC) governs everything** —
not just tables, but the entire GenAI stack. Every governed asset is a **securable object** addressed
by the familiar three-level namespace `catalog.schema.object`. For a GenAI engineer, the securables
that matter are:

- **Tables, views, volumes** — the source data your agent's index and tools read
- **Functions** — Unity Catalog functions used as agent tools
- **Models** — registered agent/model versions (from Section 04)
- **Vector (AI Search) indexes** — the retrieval store
- **Prompts** — Prompt Registry entries (Section 04) are UC entities too
- **Agents, MCP services, connections** — the runtime interaction surfaces

All of these are secured with the **same `GRANT`/`REVOKE` mechanism**:

```sql
GRANT SELECT ON TABLE prod.docs.chunks TO `data-science-team`;
GRANT EXECUTE ON FUNCTION prod.tools.lookup_order TO `agent-service-principal`;
GRANT EXECUTE ON MODEL prod.agents.docs_agent TO `app-service-principal`;
```

Because it's one layer, UC also gives you **lineage** (data → model → serving endpoint → dashboard,
which is what makes an AI system *auditable*) and **audit logs** (`system.access.audit`). When an
exam question asks "how do you control which users can query the agent's underlying data," the answer
is Unity Catalog grants — not application code.

### Fine-grained access control: row filters and column masks

Grants are all-or-nothing at the object level. For **row-** and **column-level** control you use two
UC mechanisms:

- A **row filter** is a SQL UDF returning a BOOLEAN. Rows where it returns FALSE are excluded. Applied
  at the table level:

  ```sql
  CREATE FUNCTION prod.docs.region_filter(region STRING)
    RETURN is_account_group_member('global-admins') OR region = current_user_region();
  ALTER TABLE prod.docs.chunks SET ROW FILTER prod.docs.region_filter ON (region);
  ```

- A **column mask** is a SQL UDF that takes a column's value and returns it masked or unmasked:

  ```sql
  CREATE FUNCTION prod.docs.mask_email(email STRING)
    RETURN CASE WHEN is_account_group_member('pii-readers') THEN email ELSE '***MASKED***' END;
  ALTER TABLE prod.docs.chunks ALTER COLUMN author_email SET MASK prod.docs.mask_email;
  ```

One row filter and one mask per column/table. **ABAC policies** (`CREATE POLICY`, attached at
catalog/schema level and applied automatically by governed tags) are the *recommended* 2026 approach —
they're more efficient, table owners can't override them, and they scale better than per-table UDFs.

**A critical GenAI-specific limitation:** you **cannot create an AI Search / vector index from a table
that has row filters or column masks applied.** If you need both masking and a vector index, mask into
a separate curated table and index that, or handle the access control at query time in the agent.

### Guardrails, generation 1: endpoint guardrails

Guardrails filter *content* flowing through a serving endpoint. The **endpoint guardrails**
(Public Preview) are configured per serving endpoint — in the Serving UI's AI Gateway section or via
`PUT /api/2.0/serving-endpoints/{name}/ai-gateway`. There are two guardrail types, applied
independently on **input** (the request) and **output** (the response):

- **Safety filtering** — blocks unsafe/harmful content (violence, self-harm, hate speech). Built on
  **Meta Llama Guard** as the safety-filter model. It's a simple on/off toggle.
- **PII detection** — uses **Microsoft Presidio** to detect PII (credit cards, emails, phone numbers,
  bank accounts, SSNs — U.S.-scoped). Behavior is one of **BLOCK**, **MASK**, or **None**.

Plus two content controls: **invalid keywords** (reject content containing listed terms) and
**valid topics** (restrict responses to allowed subjects).

Conceptually, the config looks like:

```json
"guardrails": {
  "input":  { "pii": {"behavior": "BLOCK"}, "enable_safety_filter": true,
              "invalid_keywords": ["competitor_name"], "valid_topics": ["support", "billing"] },
  "output": { "pii": {"behavior": "MASK"}, "enable_safety_filter": true }
}
```

**PII BLOCK vs MASK — a defining exam distinction:**
- **BLOCK** rejects the whole request or response and returns a default message. Use when *any* PII is
  unacceptable.
- **MASK** redacts the detected PII entities and lets the rest through. Use when the content is still
  useful with PII removed (e.g., masking an SSN out of an otherwise valid answer).

### Guardrails, generation 2: service policies

The newer mechanism (Beta, part of **Unity AI Gateway**) is **service policies** — ABAC content
policies attached to AI *services* (Model Services and MCP Services). Built-in guardrails live in the
`system.ai` catalog:

| Built-in policy | Blocks | Phase |
|---|---|---|
| `system.ai.block_pii` | PII | request or response |
| `system.ai.block_unsafe_content` | unsafe/harmful content | request or response |
| `system.ai.block_jailbreak` | prompt injection / jailbreak | **request only (ON CALL)** |
| `system.ai.block_hallucination` | hallucinated responses | **response only (ON RESULT)** |

The crucial difference from endpoint guardrails: service policies return a **decision** —
**ALLOW / DENY / ASK** (ASK = human-in-the-loop approval) — they **do not transform content**. So
**there is no MASK in service policies; `system.ai.block_pii` only DENIES.** If a question asks how to
*mask* PII, the answer is *endpoint guardrails*, not service policies.

Service policies are **fail-closed**: any evaluation error, timeout, missing field, or misconfiguration
results in **DENY** — safety over availability. They evaluate at two points (**ON CALL** before the
model runs, **ON RESULT** after), run in **rank** order, and the chain **stops at the first DENY**.

Custom policies are authored as SQL UDFs returning a VARIANT decision object:

```sql
CREATE FUNCTION prod.policies.block_project_aurora(event VARIANT)
RETURNS VARIANT LANGUAGE SQL
RETURN CASE
  WHEN event:type::string = 'request'
   AND contains(lower(event:context.message::string), 'project aurora')
  THEN to_variant_object(named_struct('result', 'DENY', 'reason', 'Confidential project'))
  ELSE to_variant_object(named_struct('result', 'ALLOW', 'reason', ''))
END;
```

(Note the `to_variant_object(named_struct(...))` requirement — a plain `CAST(... AS VARIANT)` is not
supported — and that VARIANT paths must be cast to a scalar with `::string` before comparison.)

### Unity AI Gateway: the control plane

**Unity AI Gateway** (formerly **Mosaic AI Gateway**, **Beta** in 2026 — enabled by an account admin
from the account console Previews page) is the governance control plane for LLM traffic. Its features:

- **Access control** — register Databricks-hosted and external models, MCP services, and agents as UC
  securables and govern them with grants.
- **Rate limiting** — QPM or TPM caps at the endpoint, per-user, per-service-principal, or per-group
  level (the more restrictive limit wins).
- **Spend caps / budgets** — per-user budgets with **hard caps that stop requests** when exceeded
  (not just after-the-fact alerts). *Caveat:* budgets currently track pay-per-token and `ai_query`
  batch usage, **not** provisioned throughput or external-model inference.
- **Usage tracking** — auto-populates `system.serving.endpoint_usage` (per-request tokens, latency,
  status, requester) and `system.serving.served_entities` (metadata). **Account admins only.**
- **Payload logging (inference tables)** — logs requests/responses to UC Delta tables (payloads over
  **1 MiB** are not logged).
- **Fallbacks** — on 429/5XX, route to a backup model (external models only, max 2, ordered by
  listing not traffic %).
- **Traffic splitting** — percentage-based routing across served entities.

**Billing distinction (exam gotcha):** payload logging and usage tracking are **paid** features;
query permissions, rate limiting, fallbacks, and traffic splitting are **free**.

## Deep Dive / Advanced Topics

### Endpoint guardrails vs service policies — when each applies

The two generations aren't just old vs new; they have different support and capabilities:

| | Endpoint guardrails | Service policies (Beta) |
|---|---|---|
| Where configured | Per serving endpoint (AI Gateway section / REST) | Attached to Model/MCP services (AI Gateway UI) |
| Actions | BLOCK, **MASK**, or None (PII); safety on/off | **ALLOW / DENY / ASK** (no masking) |
| PII engine | Microsoft Presidio | LLM-judge (`system.ai.block_pii`) |
| Safety engine | Llama Guard | LLM-judge (`system.ai.block_unsafe_content`) |
| Custom logic | keywords, topics | custom SQL policy functions (CEL-transpiled) |
| Failure mode | passes or blocks | **fail-closed → DENY** |

**Endpoint guardrail limitations to remember:** batch size (or chat `n`) can't exceed **16** with
guardrails on; **output guardrails aren't supported for embeddings or streaming**; with **function
calling**, guardrails apply only to the **final output**, not intermediate tool calls; guardrails are
**not supported on custom-model endpoints**; and because moderation depends on FM-API pay-per-token,
regions requiring cross-geo don't support guardrails.

### Responsible AI and the DASF

Databricks frames responsible AI around **trust** — organizations should own and control their data
and models with monitoring, privacy controls, and governance across the lifecycle. Two artifacts show
up in exam-adjacent material:

- **Lineage + audit logs** are repeatedly cited as *the* mechanisms for **auditability** — being able
  to answer "what data produced this model's answer, and who accessed it."
- The **Databricks AI Security Framework (DASF)** is a structured risk taxonomy: an AI system is
  **data + code + models**, modeled as **12 architecture components** across 4 stages, with **62
  identified risks** (including prompt injection, model inversion, DoS, and hallucination) and **64
  prescriptive controls** (classification, lineage, versioning, model-serving isolation, auditing,
  MLOps/LLMOps, centralized LLM management). It aligns with NIST AI RMF, OWASP Top 10 for LLMs, and
  CISA guidance. You don't need to memorize all 62/64 — know DASF is Databricks' framework and the
  **12/62/64** shape.

### Compliance and residency framing

- **HIPAA-eligibility** is available under the Enhanced Security and Compliance add-on; Databricks
  also maintains SOC 2 Type II, ISO certifications, and pen-test attestations.
- **Data residency** is handled via regional deployments (Databricks Geos / Designated Services). A
  concrete gotcha: because endpoint guardrail moderation depends on pay-per-token FM APIs, regions
  requiring cross-geo enablement **do not support AI Guardrails**.
- **Secrets and credentials** are secured with UC storage credentials and connections (metastore-level
  securables), Databricks Secrets, customer-managed keys, and Private Link — never hardcode credentials
  in agent code (recall the plain-text models-from-code warning from Section 04).

## Worked Examples & Practice

### End-to-end: govern and guard the docs agent

Your `prod.agents.docs_agent` (Section 04) reads from `prod.docs.chunks` and is served on an endpoint.
Lock it down.

```sql
-- 1. Data access: only the agent's service principal and the DS team can read the source
GRANT USE CATALOG ON CATALOG prod TO `agent-sp`;
GRANT USE SCHEMA  ON SCHEMA  prod.docs TO `agent-sp`;
GRANT SELECT ON TABLE prod.docs.chunks TO `agent-sp`;

-- 2. Column mask: redact author emails except for PII-cleared readers
CREATE FUNCTION prod.docs.mask_email(e STRING)
  RETURN CASE WHEN is_account_group_member('pii-readers') THEN e ELSE '***' END;
ALTER TABLE prod.docs.chunks ALTER COLUMN author_email SET MASK prod.docs.mask_email;
-- NOTE: if this table also feeds a vector index, mask into a curated copy instead —
--       you cannot build an AI Search index from a table with masks/row filters.
```

```python
# 3. Endpoint guardrails: block PII on input, mask PII on output, safety on both
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

w.serving_endpoints.put_ai_gateway(
    name="docs-agent-endpoint",
    guardrails={
        "input":  {"pii": {"behavior": "BLOCK"}, "enable_safety_filter": True},
        "output": {"pii": {"behavior": "MASK"},  "enable_safety_filter": True},
    },
    rate_limits=[{"calls": 100, "renewal_period": "minute", "key": "user"}],
    usage_tracking_config={"enabled": True},
)
```

**Failure mode to observe:** apply the column mask to `prod.docs.chunks` *and* try to (re)create the
AI Search index directly from it. Index creation fails, because AI Search cannot index a table with a
column mask or row filter. The fix is to materialize a masked/curated table (or one that simply
excludes the sensitive column) and build the index from that — governance and retrieval have to be
reconciled at the data-modeling layer, not patched on at index time.

## Common Pitfalls & Misconceptions

- **Pitfall:** Assuming service policies can mask PII → **Why it happens:** the built-in is called `system.ai.block_pii` and "guardrail" implies redaction → **Fix:** service policies only return ALLOW/DENY/ASK — they don't transform content. **Masking is an endpoint-guardrail capability** (PII behavior = MASK). If you need masking, use endpoint guardrails.

- **Pitfall:** Trying to build a vector index on a table that has row filters or column masks → **Why it happens:** you added governance to the source table, then went to index it → **Fix:** AI Search cannot index a table with row filters/column masks. Materialize a curated/masked copy and index that instead.

- **Pitfall:** Expecting output guardrails to work on a streaming or embedding endpoint → **Why it happens:** guardrails "work" on standard chat responses → **Fix:** output guardrails aren't supported for streaming or embeddings, and guardrails aren't supported on custom-model endpoints at all. Check the feature-support matrix for your endpoint type.

- **Pitfall:** Governing agent data access in application code instead of Unity Catalog → **Why it happens:** it feels like the agent should enforce its own rules → **Fix:** access control belongs in UC (grants, row filters, masks) so it's enforced consistently and auditable via lineage/audit logs, regardless of how the data is reached. Use on-behalf-of-user auth so UC permissions follow the calling user.

- **Pitfall:** Relying on a budget spend cap to control external-model or provisioned-throughput cost → **Why it happens:** budgets look global → **Fix:** AI Gateway budgets currently track only pay-per-token and `ai_query` batch usage; provisioned-throughput and external-model inference are **not** tracked. Don't assume a hard cap covers those modes.

## Key Definitions

| Term | Definition |
|---|---|
| Unity Catalog securable | Any governed object (table, view, volume, function, model, index, prompt, agent, MCP service, connection) addressed as `catalog.schema.object` and controlled with GRANT/REVOKE |
| Row filter | A SQL UDF returning BOOLEAN that excludes rows where it's FALSE; applied with `ALTER TABLE ... SET ROW FILTER` |
| Column mask | A SQL UDF that returns a column value masked or unmasked; applied with `ALTER TABLE ... ALTER COLUMN ... SET MASK` |
| ABAC policy | The recommended 2026 approach to fine-grained access: `CREATE POLICY` attached at catalog/schema level, applied automatically via governed tags |
| Endpoint guardrails | Per-endpoint content filtering: PII (Presidio, BLOCK/MASK/None), safety (Llama Guard), invalid keywords, valid topics |
| Service policies | Unity AI Gateway ABAC content policies (Beta) that return ALLOW/DENY/ASK for AI services; fail-closed; `system.ai.block_*` built-ins; no masking |
| Unity AI Gateway | The governance control plane for LLM traffic (formerly Mosaic AI Gateway, Beta): rate limits, spend caps, usage tracking, payload logging, fallbacks, traffic splitting |
| DASF | Databricks AI Security Framework — a risk taxonomy of 12 architecture components, 62 risks, and 64 controls |

## Summary / Quick Recall

- **Unity Catalog governs data *and* AI** — tables, models, functions, indexes, prompts, agents, MCP servers are all securables governed with GRANT/REVOKE; lineage + audit logs provide auditability
- **Row filters** (BOOLEAN UDF, exclude rows) and **column masks** (transform column values); **ABAC policies** are the recommended modern approach
- **You cannot build an AI Search index from a table with row filters or column masks** — materialize a curated copy
- **Endpoint guardrails:** PII via **Presidio** (**BLOCK / MASK / None**), safety via **Llama Guard**, plus keywords/topics; on input and output independently
- **Service policies (Beta):** `system.ai.block_pii/unsafe_content/jailbreak/hallucination`; return **ALLOW/DENY/ASK**, **no masking**; **fail-closed**; jailbreak = request-only, hallucination = response-only
- **MASK is endpoint-guardrails only**; service policies only DENY
- **Unity AI Gateway** (formerly Mosaic AI Gateway, Beta): rate limits, hard spend caps, usage tracking, payload logging (>1 MiB skipped), fallbacks (external only, max 2), traffic splitting
- Guardrail limits: batch/n ≤ 16; no output guardrails on streaming/embeddings; not on custom models; function-calling → final output only
- **DASF** = 12 components / 62 risks / 64 controls; HIPAA-eligible under add-on; guardrails unavailable in cross-geo regions

## Self-Check Questions

1. A compliance team requires that any SSN appearing in a *user's question* causes the request to be rejected outright, but SSNs in the *model's answer* should merely be redacted so the rest of the answer still returns. How do you configure this?

<details>
<summary>Answer</summary>
Use endpoint guardrails with different PII behaviors on input vs output: set input `pii.behavior = BLOCK` (rejects the request when PII is detected) and output `pii.behavior = MASK` (redacts detected PII but returns the rest). This is an endpoint-guardrail capability — service policies can't mask, only deny.
</details>

2. Your team applied a column mask to `prod.docs.chunks` to hide author emails, and now the nightly job that rebuilds the AI Search index fails. Why, and what's the fix?

<details>
<summary>Answer</summary>
AI Search cannot create an index from a table that has row filters or column masks applied. The fix is to materialize a curated table without the mask (e.g., select only the columns the index needs, or write a masked/cleaned copy) and build the index from that table instead. Governance and retrieval must be reconciled at the data-modeling layer.
</details>

3. What's the fundamental difference between what a Unity Catalog grant controls and what a service policy controls?

<details>
<summary>Answer</summary>
A UC grant controls **whether** a principal can call an asset at all (access control — can this user query the model/index/function?). A service policy controls **how** an allowed interaction proceeds based on the request/response *content* and actor context (returning ALLOW/DENY/ASK). Grants gate access; service policies gate content within an already-permitted call.
</details>

4. Someone proposes using `system.ai.block_pii` to redact credit card numbers from responses. Will that work? Explain.

<details>
<summary>Answer</summary>
No. Service policies (including `system.ai.block_pii`) only return decisions — ALLOW, DENY, or ASK — they do not transform or redact content. `system.ai.block_pii` would DENY a response containing a credit card number, not redact it. To redact/mask PII you need endpoint guardrails with `pii.behavior = MASK`.
</details>

5. A team enabled AI Gateway budgets with a hard spend cap and assumes all their LLM costs are now capped. Their most expensive workload uses a provisioned-throughput endpoint. What's the problem?

<details>
<summary>Answer</summary>
AI Gateway budgets currently track only pay-per-token and `ai_query` batch inference — provisioned-throughput and external-model inference are **not** tracked. So the hard cap won't apply to their provisioned-throughput workload, and it can overspend beyond the budget. They need a different cost-control approach for that endpoint.
</details>

## Further Reading

- [Databricks — AI governance with Unity AI Gateway](https://docs.databricks.com/aws/en/ai-gateway/) — the control plane, features, and Beta status (updated 2026-07-01)
- [Databricks — Configure AI Gateway on endpoints](https://docs.databricks.com/aws/en/ai-gateway/configure-ai-gateway-endpoints) — guardrails, rate limits, logging config (updated 2026-07-01)
- [Databricks — Service policies for AI securables](https://docs.databricks.com/aws/en/data-governance/unity-catalog/service-policies/) — ALLOW/DENY/ASK policies, `system.ai.block_*` (updated 2026-07-01)
- [Databricks — Row filters and column masks](https://docs.databricks.com/aws/en/data-governance/unity-catalog/filters-and-masks/) — fine-grained access control (updated 2026-06-11)
- [Databricks — What is Unity Catalog?](https://docs.databricks.com/aws/en/data-governance/unity-catalog/) — securables, lineage, audit (updated 2026-06-30)
- [Databricks — AI Security Framework (DASF)](https://www.databricks.com/resources/whitepaper/databricks-ai-security-framework-dasf) — 12 components / 62 risks / 64 controls
