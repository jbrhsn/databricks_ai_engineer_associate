# Governance and Responsible AI

**Part of section:** Governance, Evaluation, and Monitoring
**Estimated time:** 4 hours
**Prerequisites:** Section 03 (guardrails and Unity AI Gateway intro), Section 04 (Unity Catalog registration, Model Serving, AI Search indexes)
**Exam mapping:** Governance, Evaluation, and Monitoring domain (~15%) — data governance, guardrails, PII handling, and the AI Gateway control plane

## Overview

This module covers how Databricks controls a GenAI system: **who** can access the data and assets
(Unity Catalog governance), and **what** content is allowed through the endpoints (guardrails and
service policies), all under the **Unity AI Gateway** control plane, framed by Databricks' responsible
AI and security posture.

## Learning Outcomes

By completing this module, you will be able to:

- Govern GenAI data and assets with Unity Catalog securables, grants, row filters, and column masks
- Configure and distinguish endpoint guardrails (PII BLOCK/MASK, safety) from service policies (ALLOW/DENY/ASK)
- Enable and configure Unity AI Gateway features and explain the responsible-AI/DASF framing

## Topics Covered

- Unity Catalog as the governance layer for data and AI; securables; row filters and column masks; ABAC
- Endpoint guardrails: PII (Presidio, BLOCK/MASK/None), safety (Llama Guard), keywords, topics
- Service policies (Beta): `system.ai.block_*`, ALLOW/DENY/ASK, phases (ON CALL / ON RESULT), fail-closed
- Unity AI Gateway: rate limits, spend caps/budgets, usage tracking, payload logging, fallbacks
- Responsible AI, DASF (12/62/64), lineage and audit logs; HIPAA/data-residency framing

## How This Module Fits

This is the "keep it safe and compliant" half of Section 05. It builds directly on Section 04's
deployed endpoints and UC-registered assets, applying governance and content controls to them. It
pairs with Module 02 (Evaluation and Monitoring), which measures quality — governance controls *what
is allowed*, evaluation measures *how good it is*.

## Study Tips

- The single highest-value distinction here is **endpoint guardrails (can MASK) vs service policies
  (deny-only)**. Drill it.
- Remember that **Unity Catalog is one governance layer for both data and AI** — models, functions,
  indexes, prompts, agents, and MCP servers are all securables.
- Know the **AI Gateway feature/billing matrix** at a high level: payload logging and usage tracking
  are paid; permissions, rate limiting, fallbacks, and traffic splitting are free.

## Chapters

- [ ] [01 Guardrails, Masking, and Data Governance](./01-guardrails-masking-and-data-governance/notes.md) — 4 hrs
