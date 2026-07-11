# AGENTS.md

This file provides guidance to agents working in this repository.

---

## What This Repo Is

A Markdown-only learning repository for **Databricks Certified Generative AI Engineer Associate**. No build system, no tests. All content is `.md` files. The agent's job is always one of: populate a stub, write an index, or update structured Markdown.

**Goal:** Pass the Databricks Certified Generative AI Engineer Associate exam (45 questions / 90 min / $200), reach deep production-grade MLOps mastery, and build a public thought-leadership + interview-ready track record. Framework stack: LangGraph (primary) → LangChain (secondary) → CrewAI (breadth/comparison).

**Source of truth:** Databricks Certified Generative AI Engineer Associate Exam Guide, March 2026 revision — https://www.databricks.com/sites/default/files/2026-03/Databricks-Certified-Generative-AI-Engineer-Associate-Exam-Guide-Mar26.pdf (verified 2026-07-11). Six domains with weights: Design Applications 14%, Data Preparation 14%, Application Development 30%, Assembling & Deploying 22%, Governance 8%, Evaluation & Monitoring 12%.

---

## Critical Conventions

### Stub files are intentionally empty
All `.md` files (except those in `templates/`) were created as single-line stubs. An empty file is NOT missing content — populate only what the user explicitly requests.

### Never edit template originals
Templates in `templates/` are reference-only. Copy content from a template into the target file; never modify the template itself.

### Standard Chapter Template
The authoritative template is `templates/chapter-notes-template.md`. Every notes file must follow this structure in this order:

1. **TL;DR** — 2–4 sentences ending with a bolded "one thing to remember"
2. **ELI5** — Mandatory plain-language analogy section, no jargon
3. **Learning Objectives** — Specific, testable, action-verb outcomes
4. **Visual Overview** — Recommended when the topic has a visualisable process; 2–4 ASCII diagrams in plain fenced blocks under `###` sub-headers; placed after Learning Objectives, before Key Concepts
5. **Key Concepts** — Each sub-section: definition + mechanism + Databricks/framework manifestation
6. **Implementation** — ≥2 snippets (different angles) including one anti-pattern
7. **Common Pitfalls** — Each: bolded label + why beginners make it + correct mental model
8. **Key Definitions** — Precise, scoped definitions only
9. **Summary / Quick Recall** — 3–7 scannable takeaways
10. **Self-Check Questions** — 5 questions spanning recall → application → analysis; ≥1 multi-select
11. **Further Reading** — Official docs only, all links verified

### Template → destination mapping

| Template | Destination |
|---|---|
| `chapter-notes-template.md` | `[chapter]/notes.md` |
| `thought-leadership-template.md` | `[chapter]/thought-leadership.md` |
| `interview-prep-template.md` | `[chapter]/interview-prep.md` |
| `lab-template.md` | `[chapter]/LAB-XX-name.md` |
| `module-index-template.md` | `[module]/README.md` or `[module]/INDEX.md` |
| `section-index-template.md` | `[section]/README.md` or `[section]/INDEX.md` |
| `capstone-template.md` | `capstone/project-brief.md` |

### Lab numbering is global
Lab numbers run continuously across the entire repo (LAB-01 … LAB-22). Never reset per section or module.

### Index placement
- Section-level index exists at scaffold time (`README.md` in each section folder) — do not recreate.
- Module-level index does not exist at scaffold time — create on request.

### External links
Official documentation only. No third-party blogs, Medium, or YouTube.
Format: `[Title](url) — *verified YYYY-MM-DD*`
Verify every URL with `webfetch` before writing.

### Framework priority
When authoring orchestration content, LangGraph is primary (graph-based, stateful agents), LangChain is the component library used inside LangGraph nodes, CrewAI is breadth/comparison only. The Databricks Agent Framework is framework-agnostic (supports LangGraph, LangChain, LlamaIndex, OpenAI) — reflect this in content.

---

## Content Depth Rules

These rules are topic-agnostic and govern authoring quality regardless of subject matter.

### Rule 1 — ELI5 is mandatory and must use a structural analogy
Every notes file must open with an ELI5 section after TL;DR:
- Plain English, zero jargon
- A concrete everyday analogy that maps structurally onto the technical concept
- Specific enough that a complete beginner could build the correct mental model from it
- 3–6 sentences, prose only, no bullet lists
- Non-compliant: "Think of X as a way to represent Y." (too vague — no structure)
- Compliant: names a familiar object, maps its mechanism to the technical process, explicitly corrects the most common misconception

### Rule 2 — Every concept sub-section must explain the mechanism
Each Key Concepts sub-header must answer three questions:
1. What is it? (1–2 sentence definition)
2. How does it work under the hood? (2–4 sentences on the mechanism — the process or system behaviour that produces the result)
3. Where does it appear in Databricks/the framework ecosystem? (specific command, API call, UI location, config field, or observable output)
Answering only question 1 is non-compliant.

### Rule 3 — Key Parameters sub-section is required for configurable topics
Any chapter covering a component with tunable settings must include a **Key Parameters / Configuration Knobs** table: `Parameter | What it controls | Decision rule`. The Decision rule must be a concrete actionable rule, not a restatement of the parameter's purpose. If no configurable parameters exist, write "No configurable parameters for this topic." and continue.

### Rule 4 — Worked Example is required in every chapter
Every notes file must include a **Worked Example: Requirement → Decision** sub-section following this structure:
- Given: a realistic scenario in plain English
- Step 1 — Identify the goal
- Step 2 — Define inputs
- Step 3 — Define outputs
- Step 4 — Apply constraints (constraints relevant to this domain and topic)
- Step 5 — Select the approach with a one-sentence rationale vs alternatives
If no selection decision exists, substitute a realistic failure diagnosis walkthrough.

### Rule 5 — Snippets must be scenario-first, not topic-first
Every code or config snippet must begin with a comment naming the real-world problem being solved.
Non-compliant: a comment that only names the feature or command being demonstrated.
Compliant: a comment that states the concrete operational goal the snippet achieves and the constraint that makes it the right choice.
At least one snippet per file must be an anti-pattern (`# Anti-pattern:`) immediately followed by the corrected version with an explanation of what breaks.

### Rule 6 — Pitfalls must have three parts
Each pitfall bullet: (1) **bolded label**, (2) one sentence on why beginners make this mistake, (3) one sentence on the correct mental model. Bare bullets are non-compliant.

### Rule 7 — Answer rationales must cover all options
Every Self-Check answer must explain why the correct answer is right AND why the main distractor(s) are wrong. One-word rationales are non-compliant. For multi-select, explain why both correct answers qualify AND why the most tempting wrong answer fails.

### Rule 8 — Self-Check questions must span cognitive levels
Required distribution: Q1 recall, Q2–Q3 application, Q4–Q5 analysis/trade-off. Five recall questions is non-compliant even if one is multi-select.

### Rule 9 — Visual Overview is recommended for visualisable topics
When a topic involves a pipeline, decision path, architecture, or before/after contrast, include a `## Visual Overview` section placed **after `## Learning Objectives` and before `## Key Concepts`**. Format: each diagram under its own `### [Diagram Title]` sub-header inside a plain fenced code block (no language tag). Use `──►` for flow arrows and `│ ├ └ ─ ┌ ┐` for tree/box structure. Aim for 2–4 diagrams. Omit this section only for purely conceptual topics where no process or structure exists to diagram.

---

## File Naming Rules

- Notes: `notes.md`
- Thought leadership: `thought-leadership.md`
- Interview prep: `interview-prep.md`
- Labs: `LAB-XX-kebab-name.md` (global sequence, zero-padded)
- Section folders: `NN-section-name/`
- Module folders: `NN-module-name/`
- Chapter folders: `NN-chapter-name/`
- Auxiliary files (thought leadership, interview prep) sit alongside `notes.md` in each chapter folder.

---

## Markdown Style
- H1 for file title, H2 for major sections, H3 for sub-sections
- All code blocks carry a language tag
- Horizontal rules (`---`) separate every major section
- Self-Check answers use `<details><summary>Answer</summary>` collapsible blocks
- HTML comments (`<!-- -->`) carry authoring guidance in templates — preserve them

## What Not to Do
- Do not populate stubs without explicit user instruction
- Do not paraphrase syllabus or exam objectives — quote verbatim from the source of truth
- Do not add content not traceable to the authoritative source
- Do not renumber or rename files or folders after scaffold — filenames are locked from Phase 5. Authored chapters may forward-link to not-yet-written stubs; those links resolve when the target is populated. Renaming to "fix" a link breaks every other reference to that file.
- Do not link to third-party blogs, Medium, or YouTube
- Do not skip or merge sections without user approval
- Do not assume a fast-evolving feature (Agent Bricks, MCP, AI Gateway, MLflow GenAI Scorers, Vector Search, Genie Spaces) is unchanged — verify against current official docs before authoring, per the Blueprint Drift Warning in `templates/authoring-guidelines.md`.
