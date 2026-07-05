# Databricks Certified Generative AI Engineer Associate — Learning Repo

A structured, self-paced study repository for the **Databricks Certified Generative AI
Engineer Associate** exam. It takes you from GenAI fundamentals through building,
deploying, governing, and evaluating LLM applications on Databricks, ending in an
end-to-end capstone.

> **Blueprint version targeted:** 2026-07-05. Databricks exams evolve quickly — verify the
> [official exam guide](https://www.databricks.com/learn/certification/generative-ai-engineer-associate)
> before you sit the exam. See `00-roadmap/learning-roadmap.md` for drift-detection guidance.

## Exam at a Glance

| | |
|---|---|
| **Format** | 60 questions · multiple choice / multi-select |
| **Duration** | 90 minutes |
| **Passing score** | ~70% (Databricks does not publish the exact cut score) |
| **Delivery** | Pearson VUE — remote proctored or test center |
| **Cost** | ~$200 USD (check current pricing) |
| **Validity** | 2 years |
| **Prerequisites** | None formal; Python + Databricks experience recommended |

## How to Use This Repo

1. Start with **`00-roadmap/learning-roadmap.md`** — the full study plan, domain weights, and
   time budget by study pattern.
2. Work through the sections **in order** (`01` → `06`). Each builds on the previous one.
3. Each chapter lives in its own folder with a `notes.md` file. Read the section and module
   `README.md` first for context, then work the chapter.
4. Track your progress in **`progress-tracker.md`**.
5. Finish with the **`capstone/`** — an end-to-end RAG application that exercises the whole blueprint.

## Sections

| # | Section | Est. hrs | Exam weight | Focus |
|---|---------|----------|-------------|-------|
| 01 | [Foundations](./01-foundations/) | 8 | ~15% | LLM mechanics, prompt engineering, model selection, workspace setup |
| 02 | [Data Preparation](./02-data-preparation/) | 9 | ~20% | Document processing, chunking, embeddings, Delta Lake, Unity Catalog |
| 03 | [Application Development](./03-application-development/) | 12 | ~30% | Chains, LangGraph agents, RAG pipelines, guardrails, Databricks-native agents |
| 04 | [Assembling & Deploying Apps](./04-assembling-and-deploying-apps/) | 10 | ~20% | MLflow, PyFunc, Model Serving, Vector Search, CI/CD, Databricks Apps |
| 05 | [Governance, Evaluation & Monitoring](./05-governance-evaluation-and-monitoring/) | 8 | ~15% | MLflow tracing, AI Gateway, responsible AI, evaluation frameworks |
| 06 | [Exam Prep](./06-exam-prep/) | 5 | — | Blueprint review, mock exams, gap analysis, timing strategy |
| — | [Capstone](./capstone/) | — | — | End-to-end RAG application on Databricks |
| | **Total** | **52** | | |

## Repo Structure

```
00-roadmap/              Full learning roadmap, domain weights, logistics
01-foundations/ … 06-exam-prep/   Sections → modules → chapters (each with notes.md)
capstone/                Capstone project brief and submission template
templates/               Authoring templates + quality rubric
progress-tracker.md      Per-chapter checklist and quick stats
```

## Authoring / Contributing

This repo is built to a consistent standard. Before adding or revising content, read
**`templates/authoring-guidelines.md`** and use the chapter/section/module templates in
`templates/`. Content should teach, not just list — see the depth proportions and quality
checklist in the guidelines.
