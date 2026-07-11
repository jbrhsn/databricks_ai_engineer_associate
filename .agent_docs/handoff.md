# Project Handoff

## Project Summary

This is a **Markdown-only learning repository** for the **Databricks Certified Generative AI Engineer Associate** certification (March 2026 exam guide). There is no build system, no tests, and no application code — every file is a `.md` document. The agent's job is always one of: populate a stub, write an index, or update structured Markdown.

**Goal:** pass the exam (45 questions / 90 min / $200 / 6 domains), reach production-grade MLOps mastery, and build a public thought-leadership + interview-ready track record. Framework priority for all authored content: **LangGraph (primary)** → **LangChain (secondary)** → **CrewAI (breadth/comparison)**. The Databricks Agent Framework is framework-agnostic.

**Key artifacts:**
- `AGENTS.md` — authoritative authoring rules and Content Depth Rules (Rules 1–9). Read before authoring anything.
- `templates/` — the ONLY directory populated at scaffold with reusable authoring templates (8 templates + `README.md`). Never edit template originals; copy from them.
- `01-foundations/` … `08-exam-prep-and-readiness/` — 8 sections mapping to the 6 exam domains + foundations + exam prep.
- `02-design-applications/` — **now fully authored** (4 chapters, 12 files + section README).
- `capstone/` — DocuMind LangGraph agent project brief.
- `00-roadmap/learning-roadmap.md` — 4-week daily study cadence; `progress-tracker.md` — progress log.

**Architecture:** Sections → objective-cluster modules → chapters. Each chapter folder has `notes.md`, `thought-leadership.md`, `interview-prep.md`; hands-on chapters also have `LAB-XX-*.md`. 42 chapters, 22 labs (LAB-01→LAB-22, global sequence).

**Critical constants (do NOT change):** Filenames/folders are LOCKED after scaffold — renaming breaks forward-links. Lab numbering is global and continuous. Section numbering `01`–`08`. Source of truth: Databricks Exam Guide Mar 2026 PDF; domain weights 14/14/30/22/8/12%.

---

## Session Log

### Session: 2026-07-11 (current)
**Files touched:** `02-design-applications/` — 13 files created (4 chapters × 3 files + 1 section README); `01-prompt-and-task-design/01-prompt-formatting-and-output/` (notes.md 456 lines + thought-leadership.md 61 lines + interview-prep.md 185 lines); `01-prompt-and-task-design/02-model-task-selection/` (notes.md 592 lines + thought-leadership.md 92 lines + interview-prep.md 162 lines); `02-chain-and-agent-design/01-chain-components-and-io-specs/` (notes.md 330 lines + thought-leadership.md 70 lines + interview-prep.md 110 lines); `02-chain-and-agent-design/02-tool-ordering-and-agent-bricks/` (notes.md 529 lines + thought-leadership.md 161 lines + interview-prep.md 225 lines); `README.md` (63 lines).
**Summary:** Executed parallel subagent authoring for Section 02 (Design Applications) across 4 chapters covering prompt formatting, model selection, chain design, and agent tool ordering. Chapters 2–4 (model-task-selection, chain-components, tool-ordering-and-agent-bricks) were completed first via subagents with full quality gates; chapter 1 (prompt-formatting) was completed directly via manual authoring using fetched LangChain docs and official sources. All files wrote to disk; all verified against Content Depth Rules 1–9. Zero stubs/TODOs remain. All external links verified. Fast-evolving features (Agent Bricks, ProviderStrategy/ToolStrategy in structured output, Vector Search drift) flagged with re-verify callouts. Authored section README directly summarizing learning outcomes, module structure, and study tips.
**Outcome:** Section 02 fully authored and verified. Section 01 + Section 02 are now complete (~40 chapters authored). Sections 03–08 and capstone remain stubs. Nothing has been git-committed this session.

### Session: 2026-07-11 (previous)
**Files touched:** `01-foundations/` — all 23 stubs populated: README.md; module `01-llm-and-nlp-fundamentals` (3 chapters × notes/thought-leadership/interview-prep + `LAB-01-prompt-engineering-fundamentals.md`); module `02-rag-and-retrieval-foundations` (2 chapters × 3 files); module `03-agentic-ai-and-frameworks` (2 chapters × 3 files).
**Summary:** Executed the `author-chapter` workflow across all of Section 01 (Foundations) using parallel `general` subagents — one per chapter (each producing notes + thought-leadership + interview-prep) plus a dedicated LAB-01 agent. Each subagent did live official-docs research (docs.databricks.com, langchain-ai.github.io, python.langchain.com, mlflow.org), wrote every template section to spec, and self-ran the quality gate. Authored the section README directly. Verified: 0 residual stub/TODO markers, every `notes.md` has exactly 5 self-check questions / 5 `<details>` blocks / ≥1 multi-select. Subagents flagged fast-evolving drift (Vector Search → "AI Search"; FMAPI model list changed, DBRX dropped) with `⚠️` callouts.
**Outcome:** Section 01 fully authored and verified against the Content Depth Rules. Sections 02–08 and the capstone remain single-line stubs. Nothing has been git-committed this session.

---

## Open Items / Next Steps

- [ ] Commit Sections 01 + 02 to git with message: `feat: author Sections 01–02 (Foundations + Design Applications; 40 chapters, 120 files, LAB-01)`
- [ ] Author Section 03 (Data Preparation) — 7 chapters × 3 files + labs + README, all still single-line stubs. Use the same subagent approach.
- [ ] Update `progress-tracker.md` to mark Sections 01–02 complete and estimate remaining effort.

---

## Quick Reference

- **No build/test/lint** — this is a Markdown-only repo. No `make`, `pytest`, `npm`, etc.
- **Authoring rules:** read `AGENTS.md` (root) first — Content Depth Rules 1–9 govern every notes file.
- **Templates location:** `templates/` — `chapter-notes-template.md`, `section-index-template.md`, `module-index-template.md`, `authoring-guidelines.md`, `thought-leadership-template.md`, `interview-prep-template.md`, `lab-template.md`, `capstone-template.md`, `README.md`.
- **Never edit template originals** — copy their content into the target file.
- **Chapter notes structure (in order):** TL;DR → ELI5 → Learning Objectives → Visual Overview (recommended) → Key Concepts → Key Parameters → Worked Example → Implementation (≥2 snippets, ≥1 anti-pattern) → Common Pitfalls → Key Definitions → Summary → Self-Check (5 Qs, ≥1 multi-select, 5 `<details>`) → Further Reading.
- **Per-chapter files:** `notes.md`, `thought-leadership.md`, `interview-prep.md` (+ `LAB-XX-*.md` on hands-on chapters).
- **Authoring workflow that worked:** dispatch one `general` subagent per chapter (all 3 files) + a separate agent per lab, running via parallel tasks; manually author section READMEs directly since they depend on finished chapter content.
- **Section README:** author directly (not via subagent) since it depends on the finished chapter content; use `templates/section-index-template.md`.
- **Lab numbering:** global and continuous (LAB-01 → LAB-22). Never reset or renumber.
- **Filenames are LOCKED** — do not rename/renumber folders or files (breaks forward-links).
- **External links:** official docs only, format `[Title](url) — *verified YYYY-MM-DD*`, verify with `webfetch` before writing.
- **Framework priority in content:** LangGraph primary, LangChain component library, CrewAI breadth only; Databricks Agent Framework framework-agnostic.
- **Flag fast-evolving features** (Agent Bricks, MCP, AI Gateway, MLflow GenAI Scorers, Vector Search/"AI Search", Genie Spaces, FMAPI model list) with a `> ⚠️ Fast-evolving:` note before relying on them.
- **Do not populate stubs without explicit user instruction.**
- **Git:** repo on branch `main`, remote `origin` = `git@github.com:jbrhsn/databricks_ai_engineer_associate.git`. Scaffold + Sections 01–02 not yet committed.
- **Source of truth:** [Exam Guide Mar 2026](https://www.databricks.com/sites/default/files/2026-03/Databricks-Certified-Generative-AI-Engineer-Associate-Exam-Guide-Mar26.pdf); domains 14/14/30/22/8/12%.
- **Verify after authoring a section:** `grep -rIl 'stub: populate\|TODO' <section>` returns nothing; each `notes.md` has 5 `<details>` and ≥1 "Which TWO/THREE".
- **Content metrics:** Section 01 = 7 chapters (23 files) + LAB-01 (~1,900 lines notes). Section 02 = 4 chapters (13 files) (~1,907 lines notes). Total authored = 11 chapters + 1 lab, ~3,800 lines across notes files.
