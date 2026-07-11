# Project Handoff

## Project Summary

This is a **Markdown-only learning repository** for the **Databricks Certified Generative AI Engineer Associate** certification (March 2026 exam guide). There is no build system, no tests, and no application code — every file is a `.md` document. The agent's job is always one of: populate a stub, write an index, or update structured Markdown.

**Goal:** pass the exam (45 questions / 90 min / $200 / 6 domains), reach production-grade MLOps mastery, and build a public thought-leadership + interview-ready track record. Framework priority for all authored content: **LangGraph (primary)** → **LangChain (secondary)** → **CrewAI (breadth/comparison)**. The Databricks Agent Framework is framework-agnostic.

**Key artifacts:**
- `AGENTS.md` — authoritative authoring rules and Content Depth Rules (Rules 1–9). Read before authoring anything.
- `templates/` — the ONLY directory populated at scaffold with reusable authoring templates (8 templates + `README.md`). Never edit template originals; copy from them.
- `01-foundations/` … `08-exam-prep-and-readiness/` — 8 sections mapping to the 6 exam domains + foundations + exam prep.
- **Authored so far:** `01-foundations/` (7 chapters, LAB-01), `02-design-applications/` (4 chapters), `03-data-preparation/` (5 chapters, LAB-02–LAB-05), `04-application-development/` (9 chapters, LAB-06–LAB-10), `05-assembling-and-deploying-applications/` (7 chapters, LAB-11–LAB-17). All fully committed.
- `capstone/` — DocuMind LangGraph agent project brief (still a stub).
- `progress-tracker.md` — progress log (not updated this session; reflects state through Section 04).

**Architecture:** Sections → objective-cluster modules → chapters. Each chapter folder has `notes.md`, `thought-leadership.md`, `interview-prep.md`; hands-on chapters also have `LAB-XX-*.md`. 42 chapters total, 22 labs (LAB-01→LAB-22, global sequence). **32 chapters authored; 10 remain (Sections 06–08 + capstone).**

**Critical constants (do NOT change):** Filenames/folders are LOCKED after scaffold — renaming breaks forward-links. Lab numbering is global and continuous. Section numbering `01`–`08`. Source of truth: Databricks Exam Guide Mar 2026 PDF; domain weights 14/14/30/22/8/12%.

---

## Session Log

### Session: 2026-07-11 (current)
**Files touched:** `05-assembling-and-deploying-applications/` — all 29 files: 7 chapters × 4 files (notes.md, thought-leadership.md, interview-prep.md, LAB-XX) + section README.md. Chapters: `01-pyfunc-chains-pre-post-processing` (LAB-11), `02-rag-building-blocks-and-signatures` (LAB-12), `01-vector-search-index-and-config` (LAB-13), `02-unity-catalog-registration-serving` (LAB-14), `03-foundation-model-apis-and-ai-query` (LAB-15), `01-mcp-servers-and-persistent-memory` (LAB-16), `02-cicd-prompt-lifecycle-and-uis` (LAB-17).
**Summary:** Authored all of Section 05 (Assembling & Deploying Applications, 22% exam domain) using 7 parallel subagents — one per chapter, each writing all 4 files. Each subagent passed its own quality gate before writing. Post-authoring mechanical QA verified all 7 notes.md files: `## Key Parameters` standalone = 1/1, `## Worked Example` standalone = 1/1, `<details>` = 5/5, multi-select ≥ 1, STUB/TODO = 0. Total: 7,398 lines inserted across 29 files. Committed as `c6c55d3`.
**Outcome:** Sections 01–05 fully authored, QA-verified, and committed to `main`. Sections 06–08 and capstone remain single-line stubs. Lab sequence now at LAB-17; next lab is LAB-18.

### Session: 2026-07-11 (previous)
**Files touched:** `03-data-preparation/` — all 19 content files (5 chapters × notes/TL/ip + LAB-02–LAB-05 + section README); then all 5 notes.md files expanded after QA audit (fixed ELI5 misconceptions, section structure, pitfall labels, diagrams, missing Key Concepts for PII/toxicity/OCR/BM25, converted open-answer Self-Check to MCQ). `04-application-development/` — all 34 files (9 chapters × notes/TL/ip + LAB-06–LAB-10 + section README); structural fix (### → ## promotion for Key Parameters and Worked Example) applied to 6 notes files post-authoring. `progress-tracker.md` updated.
**Summary:** Authored Section 03 (Data Preparation, 5 chapters) using 5 parallel subagents, then ran a full QA audit and expanded all 5 notes files with targeted fixes. Authored Section 04 (Application Development, 9 chapters — 30% exam domain) using 9 parallel subagents. Applied structural fix (Key Parameters and Worked Example as standalone `##`) to 6 Section 04 files. All committed in 3 git commits.
**Outcome:** Sections 01–04 fully authored, QA-verified, and committed to `main`. Sections 05–08 and capstone remained single-line stubs at end of session.

---

## Open Items / Next Steps

- [ ] Author Section 06 (`06-governance/`) — all stubs. Exam domain weight: 8%.
- [ ] Author Section 07 (`07-evaluation-and-monitoring/`) — all stubs. Exam domain weight: 12%.
- [ ] Author Section 08 (`08-exam-prep-and-readiness/`) — all stubs. Includes mock exams and readiness checklists.
- [ ] Populate `capstone/project-brief.md` from `templates/capstone-template.md` — DocuMind LangGraph agent.

---

## Quick Reference

- **No build/test/lint** — this is a Markdown-only repo. No `make`, `pytest`, `npm`, etc.
- **Authoring rules:** read `AGENTS.md` (root) first — Content Depth Rules 1–9 govern every notes file.
- **Templates location:** `templates/` — `chapter-notes-template.md`, `section-index-template.md`, `module-index-template.md`, `authoring-guidelines.md`, `thought-leadership-template.md`, `interview-prep-template.md`, `lab-template.md`, `capstone-template.md`, `README.md`.
- **Never edit template originals** — copy their content into the target file.
- **Chapter notes structure (in order):** TL;DR → ELI5 → Learning Objectives → Visual Overview → Key Concepts → **`## Key Parameters / Configuration Knobs`** → **`## Worked Example: Requirement → Decision`** → Implementation (≥2 snippets, ≥1 anti-pattern) → Common Pitfalls → Key Definitions → Summary → Self-Check (5 Qs, ≥1 multi-select, 5 `<details>`) → Further Reading.
- **CRITICAL structural rule:** `## Key Parameters` and `## Worked Example` must be standalone `##` sections — never `###` nested inside `## Key Concepts`. Apply the post-authoring check immediately after each batch.
- **Post-authoring mechanical check (run per notes file):** `grep -c '^## Key Parameters'` = 1; `grep -c '^## Worked Example'` = 1; `grep -c '<details>'` = 5; `grep -c 'Which TWO\|Which THREE'` ≥ 1; `grep -c 'STUB\|TODO'` = 0.
- **Authoring workflow that worked:** dispatch one `general` subagent per chapter (all 4 files including lab) running in parallel; manually write section READMEs after subagents complete; run mechanical QA batch; commit.
- **QA workflow (if needed):** dispatch one `general` subagent to read all notes files and produce a gap analysis table, then dispatch one subagent per file to apply targeted fixes in parallel.
- **Lab numbering:** global and continuous (LAB-01 → LAB-22). Last lab authored: LAB-17. **Next lab to author: LAB-18.**
- **Filenames are LOCKED** — do not rename/renumber folders or files (breaks forward-links).
- **External links:** official docs only, format `[Title](url) — *verified YYYY-MM-DD*`, verify with `webfetch` before writing.
- **Framework priority in content:** LangGraph primary, LangChain component library, CrewAI breadth only.
- **Flag fast-evolving features** (Agent Bricks, MCP, AI Gateway, MLflow GenAI Scorers, Vector Search/"AI Search", Genie Spaces → "Genie Agents" as of July 2026, FMAPI model list, `ResponsesAgent`) with a `> ⚠️ Fast-evolving:` note.
- **Do not populate stubs without explicit user instruction.**
- **Git:** branch `main`, remote `origin` = `git@github.com:jbrhsn/databricks_ai_engineer_associate.git`. Latest commit: `c6c55d3 feat: author Section 05`.
- **Source of truth:** [Exam Guide Mar 2026](https://www.databricks.com/sites/default/files/2026-03/Databricks-Certified-Generative-AI-Engineer-Associate-Exam-Guide-Mar26.pdf); domains 14/14/30/22/8/12%.
- **Content metrics:** Sections 01–05 complete. Section 05 = 7 chapters + LAB-11–17. Total: 32 chapters, ~22,000+ lines of notes content, 10 chapters remaining (Sections 06–08 + capstone).
