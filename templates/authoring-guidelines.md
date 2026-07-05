# Authoring Guidelines & Quality Rubric

This document defines how to populate the chapter, section, and module templates in this
repo consistently. Read it once before authoring your first chapter, then use the checklist
below as a pre-submission gate for every chapter after that.

## Expectations for Authors

### Tone & Voice

- Write for **learning**, not documentation. Explain *why* a concept matters before or
  alongside explaining *what* it is.
- Use active voice. Favor concrete examples over abstract definitions.
- Assume the reader has completed prior chapters in sequence but has zero prior context on
  this specific chapter's topic.
- Conversational but authoritative — like a knowledgeable colleague explaining something
  clearly, not a textbook and not a casual chat.

### Depth

- **Core Concepts** should be roughly 80% of the chapter's effort — this is where the actual
  teaching happens.
- **Deep Dive** is the next ~15%; **Worked Examples & Practice** is the remaining ~5% of
  authoring effort (though examples may take real estate on the page — effort here means
  "don't over-invest in polish here at the expense of Core Concepts").
- Do not invert these proportions. A chapter that is 80% deep-dive nuance and a thin Core
  Concepts section will lose beginner and intermediate readers.
- Cut anything that doesn't serve a stated Learning Objective. No filler, no padding.

### Worked Examples

- Every chapter needs **at least one end-to-end worked example** — not a fragment.
- Examples should be realistic (drawn from actual Databricks workflows / real GenAI
  engineering scenarios), not contrived toy problems.
- Where relevant, show a failure mode or edge case, not just the happy path — this is
  often where real understanding gets tested on the exam.

### Self-Check Questions

- Questions must be **answerable only after working through the chapter** — if a question
  can be answered from general knowledge or a quick search, revise it.
- Prefer questions that test application/understanding over recall of a definition.
- Provide a brief (1–2 sentence) answer for each, collapsed under a `<details>` toggle so
  learners can self-test honestly before checking.

### Exam Alignment (specific to this repo)

- Every chapter should note which exam objective(s) it maps to (or explicitly mark it as
  "supporting content" if it's prerequisite material not directly scored).
- This repo tracks a specific, versioned exam blueprint (see `00-roadmap/learning-roadmap.md`
  for the version and date this repo was built against). If you're authoring content more
  than ~6 months after that date, **check the current official exam guide first** — this
  is a fast-evolving exam and objectives have already shifted once.
- Distinguish "certification-required knowledge" (what's needed to pass) from "depth/mastery
  knowledge" (what's needed to be good at the job) explicitly where they diverge. Both have
  a place in this repo, but a learner cramming for the exam should be able to tell which is
  which at a glance.

## Quality Checklist Before Submission

- [ ] Learning objectives are specific and testable (not vague like "understand X")
- [ ] Core Concepts section explains foundational ideas clearly, assuming no prior context
      on this specific topic
- [ ] Deep Dive adds meaningful depth beyond intro-level material — not a restatement
- [ ] At least one worked example is realistic and fully solved/walked through
- [ ] A failure mode or edge case is shown somewhere in the examples, if applicable
- [ ] Common Pitfalls section addresses real learner confusion points, not hypothetical ones
- [ ] Key Definitions are accurate, complete, and scoped to this chapter
- [ ] Summary is scannable — a learner could review it alone the night before the exam
- [ ] Self-Check Questions are answerable *only* after reading, with concise answers provided
- [ ] Further Reading links are current, working, and high-quality (official docs preferred)
- [ ] Exam objective mapping is filled in (or explicitly marked "supporting content")
- [ ] No filler or padding; every section earns its space
- [ ] Tone is conversational but authoritative throughout

## Templates as Living Documents

- These templates evolve. If authoring a chapter reveals a gap (a section that never fits,
  a section that's always missing), flag it rather than silently working around it.
- If the official exam blueprint changes (check `00-roadmap/learning-roadmap.md` for the
  version this repo targets), the roadmap and affected section/module READMEs should be
  updated before continuing to author new chapters against stale objectives.
