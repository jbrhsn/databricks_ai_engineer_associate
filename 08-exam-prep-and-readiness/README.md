# Section 08 — Exam Prep and Readiness

**Estimated time:** 5 hrs | **Exam domain weight:** Supporting content (draws on all 6 scored domains) | **Prerequisites:** Sections 01–07

---

## Overview

This section consolidates the entire certification arc into exam-ready form. It does not introduce new technical concepts — instead it synthesises all six scored domains into per-domain cheatsheets, surfaces the cross-domain confusables that most often produce wrong answers, and equips you with a systematic question-elimination strategy for the 90-minute, 45-question proctored exam. Work through this section in the final 2 weeks before your exam date.

---

## Learning Outcomes

By completing this section you will be able to:

- Produce a time-weighted study plan from a domain score breakdown
- Recall the highest-yield API, config, and concept for each of the six exam domains in under 60 seconds
- Apply a 4-step answer-elimination strategy to reduce distractor interference on unfamiliar questions
- Distinguish the 10 most commonly confused concept pairs across the exam (e.g. Delta Sync vs Direct Access index, `make_genai_metric` vs built-in judges, column masking vs AI Gateway PII filter)
- Identify the cognitive level of any exam question (recall / application / analysis) and adjust your time budget accordingly

---

## Modules

| # | Module | Est. time | Chapters |
|---|---|---|---|
| 1 | Mock Exams and Strategy | 5 hrs | 2 chapters — Domain Review & Cheatsheets, Mock Exams & Pitfalls |

---

## Chapter Map

| Chapter | Key topics |
|---|---|
| [01 — Domain Review and Cheatsheets](01-mock-exams-and-strategy/01-domain-review-and-cheatsheets/notes.md) | Per-domain cheatsheets for all 6 domains; master API parameter reference table; weight-adjusted study-plan triage; cross-domain glossary of ~20 exam-distractor terms |
| [02 — Mock Exams and Exam Pitfalls](01-mock-exams-and-strategy/02-mock-exams-and-pitfalls/notes.md) | Question anatomy and stem patterns; 5 distractor types; time management (2 min/question); multi-select mechanics; 10-pair confusable-pairs table; 90-minute time-budget protocol |

---

## How This Section Fits

Sections 01–07 built the technical knowledge base domain by domain. Section 08 is the integration layer: it assumes all prior content is understood and focuses entirely on translating that knowledge into exam performance. The capstone project (`capstone/project-brief.md`) runs in parallel — it is the applied proof that the knowledge holds together under realistic end-to-end constraints.

---

## Exam Quick Reference

| Dimension | Value |
|---|---|
| Questions | 45 |
| Duration | 90 minutes |
| Time per question | ~2 minutes |
| Passing score | ~70% (Databricks does not publish exact threshold) |
| Fee | $200 USD |
| Format | Proctored online (Examity); single-answer and multi-select |
| Retake policy | Pay full fee to retake; no cooling-off period stated |
| Source of truth | [Exam Guide Mar 2026](https://www.databricks.com/sites/default/files/2026-03/Databricks-Certified-Generative-AI-Engineer-Associate-Exam-Guide-Mar26.pdf) |

**Domain weights at a glance:**

```
Application Development   ████████████████████████████████  30%
Assembling & Deploying    ████████████████████████          22%
Design Applications       ████████████████                  14%
Data Preparation          ████████████████                  14%
Evaluation & Monitoring   ██████████████                    12%
Governance                █████████                          8%
```

---

## Final-Week Study Tips

1. **Weight your revision time by domain**: spend ~30% of study time on Application Development, ~22% on Assembling & Deploying. Governance (8%) should be last.
2. **Use the confusable-pairs table** in chapter 02 as a daily 5-minute drill — these pairs account for a disproportionate share of wrong answers.
3. **Simulate exam conditions**: 45 questions in 90 minutes with no notes. Flag uncertain questions and return — do not stall.
4. **Fast-evolving features**: AI Gateway rate limits, MLflow GenAI Scorers, and Databricks Vector Search (now "AI Search") change frequently. Re-verify against current docs before exam day.
5. **The capstone is your best revision tool**: if you can explain every architectural decision in DocuMind and map it to a domain objective, you are exam-ready.
