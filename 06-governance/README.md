# Section 06 — Governance

**Estimated time:** 9 hrs | **Exam domain weight:** ~8%  | **Prerequisites:** Section 05 — Assembling & Deploying

---

## Overview

This section covers the engineering controls required to deploy LLM applications responsibly and in compliance with regulatory requirements. It spans three tightly coupled concerns: protecting sensitive data through detection and masking pipelines, defending applications against adversarial inputs and prompt injection attacks, and ensuring that model choices and outputs meet licensing and content-safety obligations. Governance is the smallest exam domain by weight but carries outsized production risk — a single compliance failure can invalidate an entire deployment.

---

## Learning Outcomes

By completing this section you will be able to:

- Design a defence-in-depth PII mitigation pipeline that protects sensitive data at ingestion, retrieval, serving, and logging layers on Databricks
- Configure AI Gateway input validation rules and implement multi-layer guardrails that detect and block direct injection, indirect injection, and jailbreaking attempts
- Evaluate open-weight model licences for commercial use, configure output toxicity and copyright mitigation pipelines, and apply Unity Catalog and AI Gateway controls to enforce content safety at scale
- Map GDPR, HIPAA, and responsible AI requirements to concrete engineering controls in a Databricks GenAI stack
- Diagnose governance failures — leaked PII in vector indexes, bypassed guardrails, undetected licence violations — using inference tables and Unity Catalog audit logs

---

## Module

| # | Module | Est. time | Chapters |
|---|---|---|---|
| 1 | Guardrails and Compliance | 9 hrs | 3 |

---

## Chapters

### Module 01 — Guardrails and Compliance

| # | Chapter | Notes | Interview Prep | Thought Leadership | Lab |
|---|---|---|---|---|---|
| 1 | Masking and PII Mitigation | [notes.md](01-guardrails-and-compliance/01-masking-and-pii-mitigation/notes.md) | [interview-prep.md](01-guardrails-and-compliance/01-masking-and-pii-mitigation/interview-prep.md) | [thought-leadership.md](01-guardrails-and-compliance/01-masking-and-pii-mitigation/thought-leadership.md) | [LAB-18](01-guardrails-and-compliance/01-masking-and-pii-mitigation/LAB-18-masking-and-pii-mitigation.md) |
| 2 | Input Guardrails and Injection | [notes.md](01-guardrails-and-compliance/02-input-guardrails-and-injection/notes.md) | [interview-prep.md](01-guardrails-and-compliance/02-input-guardrails-and-injection/interview-prep.md) | [thought-leadership.md](01-guardrails-and-compliance/02-input-guardrails-and-injection/thought-leadership.md) | [LAB-19](01-guardrails-and-compliance/02-input-guardrails-and-injection/LAB-19-input-guardrails-and-injection.md) |
| 3 | Licensing and Text Mitigation | [notes.md](01-guardrails-and-compliance/03-licensing-and-text-mitigation/notes.md) | [interview-prep.md](01-guardrails-and-compliance/03-licensing-and-text-mitigation/interview-prep.md) | [thought-leadership.md](01-guardrails-and-compliance/03-licensing-and-text-mitigation/thought-leadership.md) | — |

---

## Chapter Summaries

### Chapter 1 — Masking and PII Mitigation
Covers PII detection methods (regex, NER, Presidio), transformation strategies (masking, pseudonymization, tokenization, anonymization), Unity Catalog dynamic masking policies, AI Gateway PII filtering, and the timing principle that determines compliance scope. Key exam focus: masking is not anonymization, and defence-in-depth requires protection at every pipeline stage — not just at model output.

### Chapter 2 — Input Guardrails and Injection
Covers the taxonomy of prompt injection attacks (direct, indirect via RAG, jailbreaking), AI Gateway input validation configuration, LangGraph validation node patterns, multi-tenant isolation strategies, and inference table analysis for detecting attack patterns in production traffic. Key exam focus: guardrails must validate both syntactic structure and semantic intent, and indirect injection via retrieved content is the hardest attack vector to eliminate.

### Chapter 3 — Licensing and Text Mitigation
Covers the open-weight model licensing ecosystem (Apache 2.0, MIT, Llama Community, RAIL variants), permissive vs. restrictive commercial use constraints, toxicity detection (Detoxify, Llama Guard), copyright and IP risk mitigation, AI Gateway safety layer configuration, and responsible AI engineering obligations. Key exam focus: licence restrictions must be evaluated before deployment, not after; output safety requires a pipeline — not just a system prompt.

---

## How This Section Fits

**Before this section:** Section 05 (Assembling & Deploying) established how to build, register, deploy, and scale LLM applications on Databricks. Governance is the enforcement layer that those deployed applications must satisfy before they are production-ready.

**After this section:** Section 07 (Evaluation & Monitoring) — 12% exam weight — closes the loop: once governed applications are running, evaluation and monitoring verify that governance controls are working as intended, detect drift, and surface new failure modes as traffic patterns change.

---

## Study Tips

- **Governance is 8% but high-stakes.** Expect 3–4 exam questions. They tend to test the *decision boundary* — "when do you block vs. mask?", "which layer of the stack should enforce X?" — not just vocabulary.
- **Layer confusion is the most common mistake.** Work through the PII pipeline and guardrails architecture diagrams in the notes files until you can draw them from memory and label which Databricks feature owns each layer.
- **Licence names are testable.** Know Apache 2.0, MIT, Llama Community Licence, and RAIL/OpenRAIL cold. The difference between permissive with attribution and restrictive with use-case bans has appeared in practice exams.
- **AI Gateway vs. application-layer controls.** AI Gateway is a centralised enforcement point; application-layer controls (Presidio UDFs, LangGraph validation nodes) are defence-in-depth backstops. The exam may give a scenario and ask which control is the right place to enforce a policy — understand the tradeoff (coverage vs. flexibility).
- **No lab for Chapter 3.** The licensing and text mitigation chapter has no accompanying lab. Compensate by working through its worked example and the implementation snippets in the notes until you can reconstruct the licence decision logic and toxicity pipeline from scratch.
- **Read the thought-leadership articles last.** They synthesise the chapters into production framings and are useful for multi-step scenario questions that require architectural judgment, not just recall.
