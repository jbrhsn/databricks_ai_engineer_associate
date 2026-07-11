# Databricks Certified Generative AI Engineer Associate — Learning Repository

A structured, Markdown-based study repository for the **Databricks Certified Generative AI Engineer Associate** certification (March 2026 exam guide) — built to take you from GenAI beginner to production-grade practitioner and public thought leader.

## Goal

This repo is for an intermediate-Python, regular-Databricks user who is new to GenAI/LLM concepts and wants **real depth in MLOps, not just a passing score**. After working through it you will be able to:

- Design, build, deploy, govern, evaluate, and monitor a production RAG/agent application on Databricks
- Pass the exam (45 questions / 90 minutes / $200 / 6 domains)
- Speak and write credibly about agent frameworks (LangGraph primary, LangChain secondary, CrewAI for breadth)
- Ship a portfolio-grade capstone agent ("DocuMind") that touches every exam domain

**Framework stack:** LangGraph (primary orchestration) → LangChain (component library) → CrewAI (breadth/comparison). The Databricks Agent Framework is framework-agnostic (LangGraph, LangChain, LlamaIndex, OpenAI).

**Source of truth:** [Databricks Certified Generative AI Engineer Associate Exam Guide (Mar 2026)](https://www.databricks.com/sites/default/files/2026-03/Databricks-Certified-Generative-AI-Engineer-Associate-Exam-Guide-Mar26.pdf) — *verified 2026-07-11*

## Learning Path

| Phase | Section | Est. hours | Focus area |
|---|---|---|---|
| Foundations | `01-foundations` | 10.5–21 | LLM/NLP fundamentals, RAG, agentic AI, framework landscape (prerequisite, not scored) |
| Domain 1 (14%) | `02-design-applications` | 6–12 | Prompt/task design, chain components, tool ordering, Agent Bricks |
| Domain 2 (14%) | `03-data-preparation` | 7.5–15 | Extraction, chunking, Delta Lake/Unity Catalog, retrieval eval, re-ranking |
| Domain 3 (30%) | `04-application-development` | 12–24 | LangGraph/LangChain, model & embedding selection, MLflow, agents, Genie |
| Domain 4 (22%) | `05-assembling-and-deploying-applications` | 10.5–21 | pyfunc chains, Vector Search, UC registration, serving, MCP, CI/CD, UIs |
| Domain 5 (8%) | `06-governance` | 4.5–9 | Masking, input guardrails, licensing, text mitigation |
| Domain 6 (12%) | `07-evaluation-and-monitoring` | 7.5–15 | Metrics, MLflow scoring/tracing, SME feedback, inference tables, AI Gateway, cost control |
| Exam prep | `08-exam-prep-and-readiness` | 3–6 | Domain review, cheat sheets, mock exams, pitfalls, timing |
| Capstone | `capstone` | — | DocuMind: a deployed, governed, monitored LangGraph agent |
| **Total** | **41 chapters** | **~61–123 hrs** | Budget target: 15–20 hrs/week over 4 weeks |

The 4-week daily cadence (which chapters to study on which days) lives in [`00-roadmap/learning-roadmap.md`](00-roadmap/learning-roadmap.md).

## Repository Structure

```
databricks_ai_engineer_associate/
├── AGENTS.md                 # Authoring rules + content depth standards
├── templates/                # The only pre-populated content (authoring templates)
├── 00-roadmap/               # 4-week daily study plan
├── 01-foundations/           # Prerequisite groundwork (LLM, RAG, agentic AI)
├── 02-design-applications/           # Domain 1 (14%)
├── 03-data-preparation/              # Domain 2 (14%)
├── 04-application-development/        # Domain 3 (30%)
├── 05-assembling-and-deploying-applications/  # Domain 4 (22%)
├── 06-governance/                    # Domain 5 (8%)
├── 07-evaluation-and-monitoring/     # Domain 6 (12%)
├── 08-exam-prep-and-readiness/       # Mock exams + strategy
├── capstone/                 # DocuMind capstone project
└── progress-tracker.md       # Self-rated objective confidence tracker
```

Each section contains objective-cluster **modules**, and each module contains **chapters**. Every chapter folder holds `notes.md`, `thought-leadership.md`, and `interview-prep.md`; hands-on chapters also hold a `LAB-XX-*.md` file (global lab sequence LAB-01 → LAB-22).

## Section Summaries

- [`01-foundations`](01-foundations/) — LLM/NLP fundamentals, prompt engineering, RAG concepts, agentic AI, and the LangGraph/LangChain/CrewAI landscape. Prerequisite groundwork, not a scored domain.
- [`02-design-applications`](02-design-applications/) — Domain 1: prompt design for formatted output, model-task selection, chain components, tool ordering, and when to use Agent Bricks.
- [`03-data-preparation`](03-data-preparation/) — Domain 2: document extraction and filtering, chunking strategies, writing chunks to Delta Lake/Unity Catalog, retrieval evaluation, and re-ranking.
- [`04-application-development`](04-application-development/) — Domain 3 (largest): LangGraph orchestration, LangChain components, prompt augmentation/guardrails, LLM and embedding selection, MLflow + Agent Framework, multi-agent systems and Genie Spaces.
- [`05-assembling-and-deploying-applications`](05-assembling-and-deploying-applications/) — Domain 4: pyfunc chains, RAG building blocks, Vector Search config, Unity Catalog registration and serving, Foundation Model APIs, MCP servers, CI/CD, and user-facing interfaces.
- [`06-governance`](06-governance/) — Domain 5: PII masking, input guardrails against malicious inputs, data licensing, and problematic-text mitigation.
- [`07-evaluation-and-monitoring`](07-evaluation-and-monitoring/) — Domain 6: evaluation metrics, MLflow scoring/judges/custom Scorers, SME feedback loops, inference tables, AI Gateway, and cost control.
- [`08-exam-prep-and-readiness`](08-exam-prep-and-readiness/) — Full domain review, cheat sheets, timed mock exams, and common pitfalls.
- [`capstone`](capstone/) — DocuMind: an internal knowledge agent (RAG + structured data via Genie) built in LangGraph, deployed as a pyfunc model behind a Databricks App, with governance, evaluation, and monitoring.

## File Type Guide

| File type | Pattern | Purpose | Created at scaffold? |
|---|---|---|---|
| Chapter notes | `notes.md` | Core learning content per chapter | Yes (stub) |
| Thought leadership | `thought-leadership.md` | Blog/LinkedIn/talk angle per chapter | Yes (stub) |
| Interview prep | `interview-prep.md` | Interview Q&A and answer frames | Yes (stub) |
| Lab | `LAB-XX-name.md` | Hands-on exercise (global numbering) | Yes (stub, where applicable) |
| Section index | `README.md` in section folder | Section overview and module map | Yes (stub) |
| Module index | `README.md` in module folder | Module overview | No — created on request |
| Templates | `templates/*.md` | Authoring templates | Yes (populated) |

## Certification Target

- **Exam:** Databricks Certified Generative AI Engineer Associate
- **Format:** 45 scored multiple-choice questions · 90 minutes · $200 USD
- **Domains & weights:** Design Applications 14% · Data Preparation 14% · Application Development 30% · Assembling & Deploying 22% · Governance 8% · Evaluation & Monitoring 12%
- **Prerequisites:** None required; 6+ months hands-on experience recommended
- **Validity:** 2 years; recertify on the then-current exam version
- **Official page:** [Databricks Certified Generative AI Engineer Associate](https://www.databricks.com/learn/certification/genai-engineer-associate) — *verified 2026-07-11*

## How to Use This Repo

All content files are blank stubs. Ask the agent to populate any file, chapter, or section and it will follow the Standard Chapter Template in [`templates/chapter-notes-template.md`](templates/chapter-notes-template.md) and the Content Depth Rules in [`AGENTS.md`](AGENTS.md). Start with [`templates/authoring-guidelines.md`](templates/authoring-guidelines.md), then populate section by section.

> ⚠️ Several topics (Agent Bricks, MCP, AI Gateway, MLflow GenAI Scorers, Vector Search, Genie Spaces) evolve quickly. Verify against current official docs before authoring — see the Blueprint Drift Warning in the authoring guidelines.
