# Project Handoff

## Project Summary

This is a **Markdown-only learning repository** for the **Databricks Certified Generative AI Engineer Associate** certification (March 2026 exam guide). There is no build system, no tests, and no application code — every file is a `.md` document. The agent's job is always one of: populate a stub, write an index, or update structured Markdown.

**Goal:** pass the exam (45 questions / 90 min / $200 / 6 domains), reach production-grade MLOps mastery, and build a public thought-leadership + interview-ready track record. Framework priority for all authored content: **LangGraph (primary)** → **LangChain (secondary)** → **CrewAI (breadth/comparison)**.

**Key artifacts:**
- `AGENTS.md` — authoritative authoring rules and Content Depth Rules (Rules 1–9). Read before authoring anything.
- `templates/` — the ONLY directory with real content; 8 authoring templates + a `README.md`. Never edit template originals; copy from them.
- `01-foundations/` … `08-exam-prep-and-readiness/` — 8 sections mapping to the 6 exam domains + foundations + exam prep.
- `capstone/` — DocuMind LangGraph agent project brief.
- `00-roadmap/learning-roadmap.md` — holds the 4-week daily study cadence.

**Architecture:** Sections → objective-cluster modules → chapters. Each chapter folder has `notes.md`, `thought-leadership.md`, `interview-prep.md`; hands-on chapters also have `LAB-XX-*.md`. 42 chapters, 22 labs (LAB-01→LAB-22, global sequence).

**Critical constants (do NOT change):** Filenames/folders are LOCKED after scaffold — renaming breaks forward-links. Lab numbering is global and continuous. Section numbering `01`–`08`. Source of truth: Databricks Exam Guide Mar 2026 PDF.

---

## Session Log

### Session: 2026-07-11 (current)
**Files touched:** entire repo scaffold — `AGENTS.md` (populated), `README.md` (populated), `templates/` (9 files populated), all 8 section `README.md` stubs, 42 × `notes.md`/`thought-leadership.md`/`interview-prep.md` stubs, 22 × `LAB-XX-*.md` stubs, `00-roadmap/learning-roadmap.md` stub, `progress-tracker.md` stub, `capstone/README.md` + `capstone/project-brief.md` stubs.
**Summary:** Executed the full `create-learning-repo` workflow (Phases 0–6). Verified the exam blueprint against live Databricks sources, designed a 6-domain + foundations + exam-prep structure with objective-cluster modules, wrote `AGENTS.md` and all templates verbatim, then scaffolded ~213 files via parallel subagents. Ran a planned-vs-actual verification pass (all counts matched, all README links resolve, all stubs single-line).
**Outcome:** Repo fully scaffolded and verified. All content files are blank single-line stubs; only `AGENTS.md`, `README.md`, and `templates/*` contain real content. Nothing has been git-committed yet.

*No prior session recorded.*

---

## Open Items / Next Steps
- [ ] Repo root — an earlier scaffold structure exists in git history (e.g. old `01-foundations/01-prerequisites-and-setup/`); git shows those as deleted. Stage and commit the reorganized tree with: `git add . && git commit -m "chore: initial skeleton, templates, and stubs for Databricks GenAI Engineer Associate"` (only when the user confirms no old content needs recovering).
- [ ] `01-foundations/01-llm-and-nlp-fundamentals/01-how-llms-work-tokens-embeddings/notes.md` — natural first chapter to author using `templates/chapter-notes-template.md` (currently a stub).

---

## Quick Reference

- **No build/test/lint** — this is a Markdown-only repo. No `make`, `pytest`, `npm`, etc.
- **Authoring rules:** read `AGENTS.md` (root) first — Content Depth Rules 1–9 govern every notes file.
- **Templates location:** `templates/` — `chapter-notes-template.md`, `section-index-template.md`, `module-index-template.md`, `authoring-guidelines.md`, `thought-leadership-template.md`, `interview-prep-template.md`, `lab-template.md`, `capstone-template.md`, `README.md`.
- **Never edit template originals** — copy their content into the target file.
- **Chapter notes structure (in order):** TL;DR → ELI5 → Learning Objectives → Visual Overview (recommended) → Key Concepts → Key Parameters → Worked Example → Implementation (≥2 snippets, ≥1 anti-pattern) → Common Pitfalls → Key Definitions → Summary → Self-Check (5 Qs, ≥1 multi-select) → Further Reading.
- **Per-chapter files:** `notes.md`, `thought-leadership.md`, `interview-prep.md` (+ `LAB-XX-*.md` on hands-on chapters).
- **Lab numbering:** global and continuous (LAB-01 → LAB-22). Never reset or renumber.
- **Filenames are LOCKED** — do not rename/renumber folders or files (breaks forward-links).
- **Stub content:** plain stubs contain `<!-- stub: populate using templates/ -->`; section READMEs use `<!-- stub: populate using templates/section-index-template.md -->`; `capstone/project-brief.md` uses the capstone-template variant.
- **Module-level index files:** NOT created at scaffold — create on request only.
- **External links:** official docs only, format `[Title](url) — *verified YYYY-MM-DD*`, verify with `webfetch` before writing.
- **Framework priority in content:** LangGraph primary, LangChain component library, CrewAI breadth only.
- **Flag fast-evolving features** (Agent Bricks, MCP, AI Gateway, MLflow GenAI Scorers, Vector Search, Genie Spaces) with a drift warning before relying on them.
- **Do not populate stubs without explicit user instruction.**
- **Git:** repo on branch `main`, remote `origin` = `git@github.com:jbrhsn/databricks_ai_engineer_associate.git`. Scaffold not yet committed.
- **Source of truth:** [Exam Guide Mar 2026](https://www.databricks.com/sites/default/files/2026-03/Databricks-Certified-Generative-AI-Engineer-Associate-Exam-Guide-Mar26.pdf); domains 14/14/30/22/8/12%.
