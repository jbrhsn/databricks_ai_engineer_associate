# Learning Roadmap — Databricks Generative AI Engineer Associate

**Exam:** Databricks Certified Generative AI Engineer Associate
**Blueprint version this repo targets:** 2026-07-05 (verify official guide if authoring >6 months after this date)
**Exam format:** 60 questions · 90 minutes · passing score ~70% · Pearson VUE (remote or test center)
**Official exam guide:** https://www.databricks.com/learn/certification/generative-ai-engineer-associate

> **Blueprint drift warning:** Databricks exams evolve quickly. The domain weights and objectives below are accurate as of the repo creation date. Before sitting the exam, cross-check the current official exam guide and update affected READMEs if anything has shifted.

---

## Exam Domain Weights

| Domain | Exam weight | Corresponding repo section |
|--------|-------------|---------------------------|
| Design and build LLM-powered applications | ~30% | `03-application-development` |
| Data preparation and management | ~20% | `02-data-preparation` |
| Assembling and deploying applications | ~20% | `04-assembling-and-deploying-apps` |
| Governance, evaluation, and monitoring | ~15% | `05-governance-evaluation-and-monitoring` |
| Foundations (GenAI concepts, prompting, tools) | ~15% | `01-foundations` |

> The foundations section is foundational knowledge that feeds all other domains; it is tested both directly and implicitly throughout the exam.

---

## Section Overview

| # | Section | Est. hours | Exam weight | Key topics |
|---|---------|-----------|-------------|-----------|
| 01 | Foundations | 8 hrs | ~15% | LLM mechanics, prompt engineering, model selection, Databricks workspace setup |
| 02 | Data Preparation | 9 hrs | ~20% | Document processing, chunking strategies, embeddings, Delta Lake, Unity Catalog |
| 03 | Application Development | 12 hrs | ~30% | Chains, agents, LangChain, RAG pipelines, guardrails, Databricks-native agents |
| 04 | Assembling & Deploying Apps | 10 hrs | ~20% | MLflow, PyFunc, Model Serving, Vector Search, CI/CD, Databricks Apps |
| 05 | Governance, Evaluation & Monitoring | 8 hrs | ~15% | MLflow tracing, AI Gateway, responsible AI, evaluation frameworks |
| 06 | Exam Prep | 5 hrs | — | Blueprint review, mock exams, gap analysis, timing strategy |
| Capstone | Capstone Project | — | — | End-to-end RAG application built on Databricks |
| **Total** | | **52 hrs** | | |

---

## Recommended Study Path

Follow the sections in order. Each section builds on the one before it — skipping ahead
will leave gaps in foundational vocabulary and Databricks-specific context that make later
material harder to absorb.

```
01-foundations
    └── 01-prerequisites-and-setup        (environment, workspace orientation)
    └── 02-genai-and-prompting-fundamentals (LLM concepts, prompting, model selection)

02-data-preparation
    └── 01-document-processing            (extraction, chunking, Delta Lake, Unity Catalog)

03-application-development
    └── 01-chains-agents-and-frameworks   (chains, agents, LangChain, Databricks-native)
    └── 02-embeddings-guardrails-and-model-selection (embeddings, guardrails, safety)

04-assembling-and-deploying-apps
    └── 01-packaging-and-registration     (PyFunc, MLflow, UC registration, Vector Search)
    └── 02-serving-and-integration        (Model Serving, APIs, CI/CD, Databricks Apps)

05-governance-evaluation-and-monitoring
    └── 01-governance-and-responsible-ai  (guardrails, masking, data governance)
    └── 02-evaluation-and-monitoring      (MLflow tracing, scoring, AI Gateway)

06-exam-prep
    └── 01-review-and-practice            (blueprint review, mock exams, final checklist)
```

---

## Time Budget by Study Pattern

| Pattern | Daily commitment | Estimated completion |
|---------|-----------------|---------------------|
| Intensive | 4 hrs/day | ~13 days (~2 weeks) |
| Moderate | 2 hrs/day | ~26 days (~4 weeks) |
| Part-time | 1 hr/day | ~52 days (~7–8 weeks) |

> These are working hours, not calendar time. Factor in review cycles and the capstone project
> if you plan to complete it.

---

## Exam Logistics

- **Format:** Multiple choice and multi-select questions (single best answer or select-all-that-apply)
- **Duration:** 90 minutes
- **Questions:** 60
- **Passing score:** ~70% (42/60); Databricks does not publish the exact cut score
- **Delivery:** Pearson VUE — remote proctored or at a test center
- **Validity:** Certification is valid for 2 years
- **Retake policy:** 14-day waiting period after a failed attempt; third attempt requires a 30-day wait
- **Cost:** ~$200 USD (check Databricks website for current pricing)
- **Prerequisites:** None formal; practical experience with Python and Databricks recommended

---

## What the Exam Tests vs. What Makes You Good at the Job

This repo covers both, but they are not identical. Throughout the chapters you will see notes
distinguishing:

- **Cert-required:** knowledge you must have to pass the exam
- **Depth/mastery:** knowledge that makes you effective as a practitioner, sometimes beyond exam scope

Both are worth learning. If you are time-constrained and purely cramming for the exam, focus on
chapters' **Summary / Quick Recall** and **Self-Check Questions** sections, and use the exam
objective mapping in each chapter header to prioritize.

---

## Prerequisites for This Repo

Before starting Section 01, you should be comfortable with:

- Python 3.x — functions, classes, list comprehensions, f-strings, virtual environments
- Basic command line usage (creating files, navigating directories)
- Git basics (clone, add, commit, push)
- High-level awareness of what machine learning is (no ML practitioner experience needed)

You do **not** need prior experience with:

- LLMs, GenAI, or NLP
- Databricks or Spark
- MLflow
- Cloud platforms (AWS, Azure, GCP)

---

## Keeping This Roadmap Current

This roadmap was built against the exam blueprint as of **2026-07-05**. Databricks typically
updates exam objectives when major platform features ship. Signs the roadmap may be stale:

- You find official exam guide objectives that don't appear anywhere in this repo's section/module READMEs
- New Databricks features (e.g., a new agent framework or Vector Search version) are widely referenced in official docs but not covered here
- The exam guide shows a domain weight shift of ≥5 percentage points from the table above

If any of these apply: update this roadmap first, then update affected section/module READMEs,
then flag any chapter stubs that need new or revised content before authoring them.
