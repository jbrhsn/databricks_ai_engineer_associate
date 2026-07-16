# Project Handoff

## Project Summary

This is a **Markdown-only learning repository** for the **Databricks Certified Generative AI Engineer Associate** certification (March 2026 exam guide). There is no build system, no tests, and no application code — every file is a `.md` document. The agent's job is always one of: populate a stub, write an index, or update structured Markdown.

**Goal:** pass the exam (45 questions / 90 min / $200 / 6 domains), reach production-grade MLOps mastery, and build a public thought-leadership + interview-ready track record. Framework priority for all authored content: **LangGraph (primary)** → **LangChain (secondary)** → **CrewAI (breadth/comparison)**. The Databricks Agent Framework is framework-agnostic.

**Key artifacts:**
- `AGENTS.md` — authoritative authoring rules and Content Depth Rules (Rules 1–9). Read before authoring anything.
- `templates/` — the ONLY directory populated at scaffold with reusable authoring templates (8 templates + `README.md`). Never edit template originals; copy from them.
- `01-foundations/` … `08-exam-prep-and-readiness/` — 8 sections mapping to the 6 exam domains + foundations + exam prep.
- **Authored so far:** `01-foundations/` (7 chapters, LAB-01), `02-design-applications/` (4 chapters), `03-data-preparation/` (5 chapters, LAB-02–LAB-05), `04-application-development/` (9 chapters, LAB-06–LAB-10), `05-assembling-and-deploying-applications/` (7 chapters, LAB-11–LAB-17), `06-governance/` (3 chapters, LAB-18–LAB-19, section README). **All 6 content sections through 06 are fully authored.**
- `09-questions-dump/` — practice question bank with 149 questions across 6 files (Q1-19, Q20-39, Q40-59, Q60-89, Q90-119, Q120-149).
- `capstone/` — DocuMind LangGraph agent project brief (still a stub).
- `progress-tracker.md` — progress log (not updated this session; reflects state through Section 04).

**Architecture:** Sections → objective-cluster modules → chapters. Each chapter folder has `notes.md`, `thought-leadership.md`, `interview-prep.md`; hands-on chapters also have `LAB-XX-*.md`. **35 chapters authored; 7 remain (Sections 07–08 + capstone).** Lab sequence: LAB-01–LAB-19 authored; next is LAB-20.

**Critical constants (do NOT change):** Filenames/folders are LOCKED after scaffold — renaming breaks forward-links. Lab numbering is global and continuous. Section numbering `01`–`08`. Source of truth: Databricks Exam Guide Mar 2026 PDF; domain weights 14/14/30/22/8/12%.

---

## Session Log

### Session: 2026-07-16 (current)
**Files touched:**
- `06-governance/` — 12 files total: `README.md` (section index, new); all 3 chapters fully authored — `01-masking-and-pii-mitigation/` (notes.md 524 lines, interview-prep.md 474 lines, thought-leadership.md 206 lines, LAB-18 421 lines); `02-input-guardrails-and-injection/` (notes.md 613 lines, interview-prep.md 433 lines, thought-leadership.md 271 lines, LAB-19 490 lines); `03-licensing-and-text-mitigation/` (notes.md 616 lines, interview-prep.md 705 lines, thought-leadership.md 153 lines).
- `09-questions-dump/` — 6 files touched (Q1-19, Q20-39, Q40-59, Q60-89, Q90-119, Q120-149); 1,661 lines total. These are the practice question bank files.

**Summary:** This session evaluated `06-governance/`, identified 3 stubs (README.md, LAB-18, LAB-19), and populated all three. The section README was authored as a full navigation index with chapter summaries, exam-weight framing, and governance-specific study tips. LAB-18 covers a 4-step PII mitigation pipeline (Presidio detection → masking UDF → Unity Catalog column masking policy → AI Gateway PII blocking), and LAB-19 covers AI Gateway guardrail configuration, injection/jailbreak blocking, a LangGraph validation node for indirect injection in RAG chunks, and inference table attack-pattern analysis. All labs are simulation/illustrative style. The `09-questions-dump/` question bank (149 questions) was also present and populated, apparently from an earlier activity within the same calendar day.

**Outcome:** Section 06 (Governance, 8% exam domain) is fully populated — all 12 files authored, no stubs remain. Uncommitted; 4,977 insertions pending as of session end.

### Session: 2026-07-11 (previous)
**Files touched:** `05-assembling-and-deploying-applications/` — all 29 files: 7 chapters × 4 files (notes.md, thought-leadership.md, interview-prep.md, LAB-XX) + section README.md. Chapters: `01-pyfunc-chains-pre-post-processing` (LAB-11), `02-rag-building-blocks-and-signatures` (LAB-12), `01-vector-search-index-and-config` (LAB-13), `02-unity-catalog-registration-serving` (LAB-14), `03-foundation-model-apis-and-ai-query` (LAB-15), `01-mcp-servers-and-persistent-memory` (LAB-16), `02-cicd-prompt-lifecycle-and-uis` (LAB-17).
**Summary:** Authored all of Section 05 (Assembling & Deploying Applications, 22% exam domain) using 7 parallel subagents — one per chapter, each writing all 4 files. Each subagent passed its own quality gate before writing. Post-authoring mechanical QA verified all 7 notes.md files: `## Key Parameters` standalone = 1/1, `## Worked Example` standalone = 1/1, `<details>` = 5/5, multi-select ≥ 1, STUB/TODO = 0. Total: 7,398 lines inserted across 29 files. Committed as `c6c55d3`.
**Outcome:** Sections 01–05 fully authored, QA-verified, and committed to `main`. Sections 06–08 and capstone remain single-line stubs. Lab sequence now at LAB-17; next lab is LAB-18.

---

## Open Items / Next Steps

- [ ] Commit Section 06 changes — 4,977 insertions across 12 files are staged but not committed. Suggested message: `feat: author Section 06 — Governance (3 chapters, LAB-18–LAB-19, section README)`.
- [ ] Author Section 07 (`07-evaluation-and-monitoring/`) — all stubs. Exam domain weight: 12%.
- [ ] Author Section 08 (`08-exam-prep-and-readiness/`) — all stubs. Includes mock exams and readiness checklists.
- [ ] Populate `capstone/project-brief.md` from `templates/capstone-template.md` — DocuMind LangGraph agent.
- [ ] Update `progress-tracker.md` to reflect Sections 05–06 completion and the 149-question dump.

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
- **Lab numbering:** global and continuous (LAB-01 → LAB-22). Last lab authored: LAB-19. **Next lab to author: LAB-20.**
- **Filenames are LOCKED** — do not rename/renumber folders or files (breaks forward-links).
- **External links:** official docs only, format `[Title](url) — *verified YYYY-MM-DD*`, verify with `webfetch` before writing.
- **Framework priority in content:** LangGraph primary, LangChain component library, CrewAI breadth only.
- **Flag fast-evolving features** (Agent Bricks, MCP, AI Gateway, MLflow GenAI Scorers, Vector Search/"AI Search", Genie Spaces → "Genie Agents" as of July 2026, FMAPI model list, `ResponsesAgent`) with a `> ⚠️ Fast-evolving:` note.
- **Do not populate stubs without explicit user instruction.**
- **Git:** branch `main`, remote `origin` = `git@github.com:jbrhsn/databricks_ai_engineer_associate.git`. Latest commit: `6cc4798 progress update`. Section 06 changes are **uncommitted**.
- **Source of truth:** [Exam Guide Mar 2026](https://www.databricks.com/sites/default/files/2026-03/Databricks-Certified-Generative-AI-Engineer-Associate-Exam-Guide-Mar26.pdf); domains 14/14/30/22/8/12%.
- **Content metrics:** Sections 01–06 complete. Section 06 = 3 chapters + LAB-18–19. Total: 35 chapters authored, ~27,000+ lines of notes content, 7 chapters remaining (Sections 07–08 + capstone). Practice question bank: 149 questions in `09-questions-dump/`.
