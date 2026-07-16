# Project Handoff

## Project Summary

This is a **Markdown-only learning repository** for the **Databricks Certified Generative AI Engineer Associate** certification (March 2026 exam guide). There is no build system, no tests, and no application code — every file is a `.md` document. The agent's job is always one of: populate a stub, write an index, or update structured Markdown.

**Goal:** pass the exam (45 questions / 90 min / $200 / 6 domains), reach production-grade MLOps mastery, and build a public thought-leadership + interview-ready track record. Framework priority for all authored content: **LangGraph (primary)** → **LangChain (secondary)** → **CrewAI (breadth/comparison)**. The Databricks Agent Framework is framework-agnostic.

**Key artifacts:**
- `AGENTS.md` — authoritative authoring rules and Content Depth Rules (Rules 1–9). Read before authoring anything.
- `templates/` — the ONLY directory populated at scaffold with reusable authoring templates (8 templates + `README.md`). Never edit template originals; copy from them.
- `01-foundations/` … `08-exam-prep-and-readiness/` — 8 sections mapping to the 6 exam domains + foundations + exam prep. **All 8 sections are now fully authored.**
- `09-questions-dump/` — overhauled practice question bank: **177 questions across 8 files** (Q1-19, Q20-39, Q40-59, Q60-89, Q90-119, Q120-149, Q150-162, Q163-179). 23 multi-select questions (13%). All factual errors corrected.
- `capstone/` — DocuMind LangGraph agent: `project-brief.md` (539 lines, fully authored) + `README.md` (navigation index).
- `progress-tracker.md` — progress log (not updated this session; still reflects state through Section 04).

**Architecture:** Sections → objective-cluster modules → chapters. Each chapter folder has `notes.md`, `thought-leadership.md`, `interview-prep.md`; hands-on chapters also have `LAB-XX-*.md`. **40 chapters authored across Sections 01–08. Labs LAB-01–LAB-22 all authored. Capstone authored. Repo is content-complete.**

**Critical constants (do NOT change):** Filenames/folders are LOCKED after scaffold — renaming breaks forward-links. Lab numbering is global and continuous. Section numbering `01`–`08`. Source of truth: Databricks Exam Guide Mar 2026 PDF; domain weights 14/14/30/22/8/12%.

---

## Session Log

### Session: 2026-07-16 (current)
**Files touched:**
- `07-evaluation-and-monitoring/` — 19 files: 5 chapters × (notes.md + thought-leadership.md + interview-prep.md) + LAB-20, LAB-21, LAB-22 + section README.md. Chapters: `01-metrics-and-llm-choice`, `02-mlflow-scoring-judges-scorers`, `03-sme-feedback-loops`, `01-inference-tables-and-ai-gateway`, `02-cost-control-and-agent-monitoring`.
- `08-exam-prep-and-readiness/` — 7 files: 2 chapters × (notes.md + thought-leadership.md + interview-prep.md) + section README.md. Chapters: `01-domain-review-and-cheatsheets`, `02-mock-exams-and-pitfalls`.
- `capstone/` — `project-brief.md` (539 lines) + `README.md` (77 lines).
- `09-questions-dump/` — all 8 files overhauled: fixed Q3 (wrong answer), Q7 (wrong rationale), Q11 (factually wrong "cost guarantees"), Q39 (rate limiting wrongly attributed to Model Serving instead of AI Gateway); reframed 8 duplicate question pairs; replaced 10 trivially-easy distractor sets; added Databricks-specificity to 3 questions; added fast-evolving flags to Q46 and Q108; wrote Q150-162 (13 new D3/D4 questions: LangGraph, PyFunc, AISearchClient SDK, `agents.deploy()`, `mlflow.register_model()`, `infer_signature`, AI Gateway route config, provisioned throughput sizing); wrote Q163-179 (17 new multi-select questions across all 6 domains).

**Summary:** Session authored all remaining stubs — Section 07 (Evaluation & Monitoring, 5 chapters + 3 labs), Section 08 (Exam Prep, 2 chapters), and the DocuMind capstone project brief. All 5 Section 07 notes.md files passed the mechanical QA gate (Key Parameters = 1, Worked Example = 1, `<details>` = 5, multi-select ≥ 1, STUB/TODO = 0). The 149-question bank was then evaluated against the exam guide: 4 factually wrong answers identified, 8 duplicate pairs, 10 trivial distractor sets, 11% non-Databricks-specific questions, and a 13-question D3 coverage shortfall. All issues were fixed via 6 parallel file-fix subagents plus 2 new question files. Final bank: 177 questions, 23 multi-select (13%), 0 known wrong answers.

**Outcome:** Repository is content-complete. All 40 chapters, 22 labs, capstone, and 177-question bank are authored and committed across 3 commits (`009d3e0`, `6ed9f30`, `f34efb1`). The only remaining task is updating `progress-tracker.md`.

### Session: 2026-07-16 (previous)
**Files touched:**
- `06-governance/` — 12 files total: `README.md` (section index, new); all 3 chapters fully authored — `01-masking-and-pii-mitigation/` (notes.md 524 lines, interview-prep.md 474 lines, thought-leadership.md 206 lines, LAB-18 421 lines); `02-input-guardrails-and-injection/` (notes.md 613 lines, interview-prep.md 433 lines, thought-leadership.md 271 lines, LAB-19 490 lines); `03-licensing-and-text-mitigation/` (notes.md 616 lines, interview-prep.md 705 lines, thought-leadership.md 153 lines).
- `09-questions-dump/` — 6 files touched (Q1-19, Q20-39, Q40-59, Q60-89, Q90-119, Q120-149); 1,661 lines total. These are the practice question bank files.
**Summary:** This session evaluated `06-governance/`, identified 3 stubs (README.md, LAB-18, LAB-19), and populated all three. The section README was authored as a full navigation index with chapter summaries, exam-weight framing, and governance-specific study tips. LAB-18 covers a 4-step PII mitigation pipeline (Presidio detection → masking UDF → Unity Catalog column masking policy → AI Gateway PII blocking), and LAB-19 covers AI Gateway guardrail configuration, injection/jailbreak blocking, a LangGraph validation node for indirect injection in RAG chunks, and inference table attack-pattern analysis. All labs are simulation/illustrative style. The `09-questions-dump/` question bank (149 questions) was also present and populated, apparently from an earlier activity within the same calendar day.
**Outcome:** Section 06 (Governance, 8% exam domain) fully populated — all 12 files authored, committed as part of `009d3e0`.

---

## Open Items / Next Steps

- [ ] `progress-tracker.md` — update to reflect completion of Sections 05–08, capstone, and the 177-question bank overhaul.

---

## Quick Reference

- **No build/test/lint** — this is a Markdown-only repo. No `make`, `pytest`, `npm`, etc.
- **Authoring rules:** read `AGENTS.md` (root) first — Content Depth Rules 1–9 govern every notes file.
- **Templates location:** `templates/` — `chapter-notes-template.md`, `section-index-template.md`, `module-index-template.md`, `authoring-guidelines.md`, `thought-leadership-template.md`, `interview-prep-template.md`, `lab-template.md`, `capstone-template.md`, `README.md`.
- **Never edit template originals** — copy their content into the target file.
- **Chapter notes structure (in order):** TL;DR → ELI5 → Learning Objectives → Visual Overview → Key Concepts → **`## Key Parameters / Configuration Knobs`** → **`## Worked Example: Requirement → Decision`** → Implementation (≥2 snippets, ≥1 anti-pattern) → Common Pitfalls → Key Definitions → Summary → Self-Check (5 Qs, ≥1 multi-select, 5 `<details>`) → Further Reading.
- **CRITICAL structural rule:** `## Key Parameters` and `## Worked Example` must be standalone `##` sections — never `###` nested inside `## Key Concepts`.
- **Post-authoring mechanical check (run per notes file):** `grep -c '^## Key Parameters'` = 1; `grep -c '^## Worked Example'` = 1; `grep -c '<details>'` = 5; `grep -c 'Which TWO\|Which THREE'` ≥ 1; `grep -c 'STUB\|TODO'` = 0.
- **Authoring workflow that worked:** dispatch one `general` subagent per chapter (all 4 files including lab) running in parallel; manually write section READMEs after subagents complete; run mechanical QA batch; commit.
- **QA workflow (if needed):** dispatch one `general` subagent to read all notes files and produce a gap analysis table, then dispatch one subagent per file to apply targeted fixes in parallel.
- **Lab numbering:** global and continuous. All labs LAB-01–LAB-22 authored. No more labs to write.
- **Filenames are LOCKED** — do not rename/renumber folders or files (breaks forward-links).
- **External links:** official docs only, format `[Title](url) — *verified YYYY-MM-DD*`, verify with `webfetch` before writing.
- **Framework priority in content:** LangGraph primary, LangChain component library, CrewAI breadth only.
- **Flag fast-evolving features** (Agent Bricks, MCP, AI Gateway, MLflow GenAI Scorers, Vector Search/"AI Search", Genie Spaces → "Genie Agents" as of July 2026, FMAPI model list, `ResponsesAgent`) with a `> ⚠️ Fast-evolving:` note.
- **Do not populate stubs without explicit user instruction.**
- **Git:** branch `main`, remote `origin` = `git@github.com:jbrhsn/databricks_ai_engineer_associate.git`. Latest commit: `f34efb1`. Repo is fully committed — no pending changes.
- **Source of truth:** [Exam Guide Mar 2026](https://www.databricks.com/sites/default/files/2026-03/Databricks-Certified-Generative-AI-Engineer-Associate-Exam-Guide-Mar26.pdf); domains 14/14/30/22/8/12%.
- **Content metrics:** ALL sections complete (01–08). 40 chapters, 22 labs, capstone authored. Question bank: 177 questions in 8 files in `09-questions-dump/` — 23 multi-select (13%), 0 known wrong answers. `progress-tracker.md` still reflects state through Section 04 only (needs update).
