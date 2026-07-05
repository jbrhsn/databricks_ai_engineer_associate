# Mock Exam and Timing Strategy

**Section:** Exam Prep | **Module:** Review and Practice | **Est. time:** 2 hrs
**Exam mapping:** Not directly scored — timed rehearsal and question technique across all five domains

## Learning Objectives

By the end of this chapter, you will be able to:

- Execute a pacing plan that finishes 60 questions in 90 minutes with time to review
- Apply a repeatable triage-and-elimination process to individual questions
- Handle multi-select questions without over- or under-selecting
- Take a realistic, domain-weighted mock exam and interpret your score
- Review mock results to convert wrong answers into durable learning

## Core Concepts

### The pacing math

60 questions in 90 minutes is **1.5 minutes (90 seconds) per question** on average. But you should not
spend the full time linearly — the winning strategy is:

- **Target ~75 seconds per question on the first pass.** That banks time.
- **Flag anything that takes longer than ~90 seconds** and move on. Don't let one hard question eat
  the time of three easy ones.
- **Reserve the final ~10–15 minutes** for flagged questions and a sanity re-read of anything you
  second-guessed.

A simple checkpoint discipline: at **question 20 you should be near the 30-minute mark**, at
**question 40 near 60 minutes**. If you're behind at a checkpoint, speed up your first-pass triage
(answer-and-flag more aggressively) rather than trying to be perfect on every item.

### Per-question triage: read → classify → eliminate → answer

For each question, run the same loop:

1. **Read the full stem and every option before answering.** GenAI exam questions frequently hinge on
   a single qualifier ("in production," "with no code," "storage-optimized endpoint") that changes the
   right answer. Skimming is the top cause of avoidable misses.
2. **Classify the question:** Is it a *recall* question (name/definition), a *decision* question
   (which tool/approach for this scenario), or a *distractor* question (two options are technically
   real but one is stale/wrong for the context)? Most of this exam is decision and distractor questions.
3. **Eliminate** obviously wrong options first. On a 4-option question, removing two wrong answers
   turns a guess from 25% to 50%. Watch for **renamed-product distractors** (an option using an old
   name like "Vector Search" or "Mosaic AI Gateway" may be a deliberate trap — or may be correct,
   since the exam sometimes uses old names; judge by the rest of the option's accuracy).
4. **Answer**, and **flag** it if you weren't confident. Move on.

### Multi-select discipline

Multi-select questions ("select all that apply" / "choose two") are where candidates lose points by
sloppiness:

- **Read how many to select.** If it says "choose two," select exactly two.
- **Evaluate each option independently** as true/false against the stem — don't rank them relative to
  each other.
- **Beware the plausible-but-wrong extra.** Test-writers pad multi-selects with an option that's true
  in general but false *for this scenario*. If an option is only true under a different condition than
  the stem states, exclude it.
- Multi-selects are often **all-or-nothing scored** — partial credit is not guaranteed — so a single
  wrong inclusion can cost the whole question. When genuinely unsure between including a borderline
  option, weigh whether the stem's qualifiers support it.

### Guessing policy

There is **no penalty for wrong answers** — the score is number correct. Therefore **never leave a
question blank.** For anything you can't resolve, eliminate what you can and make your best guess
before time runs out. Flagged-but-answered is always better than flagged-and-blank.

## Deep Dive / Advanced Topics

### Reviewing a mock exam so it actually teaches

Taking a mock is only half the value; the review is where learning happens. For every question you got
wrong (and every one you guessed and happened to get right):

1. **Categorize the miss:** knowledge gap (didn't know the fact), misread (knew it, misread the stem),
   or trap (fell for a distractor). Each has a different fix.
2. **Knowledge gaps** → go to the coverage map in `06/01/01`, hit the relevant chapter, redo its
   Self-Check Questions.
3. **Misreads** → a process problem; slow your first pass slightly and underline qualifiers.
4. **Traps** → study the Pitfalls chapter (`06/01/03`); these are the cheapest points to recover
   because the knowledge is there, you just need to recognize the trick.

Track your miss categories across mocks. If misreads dominate, your problem is pacing/attention, not
knowledge — and cramming more facts won't help.

### When to take mocks

Take at least one **full timed mock early** (to expose gaps and pacing issues while there's time to
fix them) and one **in the final 48 hours** (to rehearse under conditions and build confidence). Don't
take a mock the night before if it rattles you — end on your terms.

## Worked Examples & Practice

### Mini mock exam (12 questions, domain-weighted)

Simulate the real ratio: give yourself **18 minutes** (90 seconds each). Answers follow in a collapsed
toggle — don't peek until you've committed to all 12.

**Domain 1 — Application Development**

1. You need a Q&A chatbot over 300 internal PDFs already in a UC Volume, with no custom logic, as fast
   as possible. What's the best choice?
   A. Custom LangGraph ReAct agent  B. Knowledge Assistant  C. `ai_query()` batch job  D. External model endpoint

2. In LangGraph, which primitive pauses a graph for human approval before continuing?
   A. `checkpointer`  B. `add_conditional_edges`  C. `interrupt()`  D. `create_react_agent`

3. Why must a custom LangGraph agent be wrapped in `mlflow.pyfunc.ResponsesAgent` before it can be
   tested in the AI Playground? (choose the best answer)
   A. It compresses the model  B. Playground/tracing/evaluation require the ResponsesAgent interface
   C. It trains the model  D. It encrypts the prompts

**Domain 2 — Data Preparation**

4. Which is required on a source Delta table before a Delta Sync index on a standard endpoint will
   sync? A. Z-ordering  B. Change Data Feed enabled  C. Liquid clustering  D. A primary key of type UUID

5. AI Search ranks with which distance metric by default? A. Cosine  B. Dot product  C. L2 (Euclidean)  D. Manhattan

**Domain 3 — Assembling & Deploying**

6. When logging a GenAI agent with MLflow 3, what do you pass to `python_model`?
   A. The agent object instance  B. A pickled file  C. The path to the agent's `.py` file  D. A model URI

7. In Unity Catalog, how do you express that a model version is the live production version?
   A. Move it to the Production *stage*  B. Set an *alias* (e.g., `@production`)  C. Rename the model
   D. Set an environment variable

8. Which sync mode is the *only* option supported on a STORAGE_OPTIMIZED AI Search endpoint?
   A. CONTINUOUS  B. TRIGGERED  C. STREAMING  D. BATCH

9. Which SQL function runs LLM inference over a Delta table for batch jobs (DBR 18.2+, serverless)?
   A. `predict()`  B. `ai_query()`  C. `vector_search()`  D. `ai_generate()`

**Domain 4 — Governance, Evaluation, Monitoring**

10. A response should have its PII **redacted** (not the whole response blocked). Which mechanism?
    A. `system.ai.block_pii` service policy  B. Endpoint guardrail with PII behavior = MASK
    C. A row filter  D. A column mask

11. Which built-in MLflow scorer checks whether a response is grounded in the retrieved documents
    (anti-hallucination) and requires a trace? A. `Safety`  B. `RelevanceToQuery`  C. `RetrievalGroundedness`  D. `Correctness`

**Domain 5 — Foundations**

12. You need deterministic, structured JSON output from an LLM for a downstream parser. Which pairing
    best supports this? A. High temperature + free text  B. Low temperature + a response format / structured-output schema
    C. Top-p = 1.0 + no schema  D. Increasing max_tokens only

<details>
<summary>Answers &amp; rationale</summary>

1. **B** — Knowledge Assistant is the no-code RAG chatbot for document Q&A; fastest path when no custom logic is needed (`03/01/03`).
2. **C** — `interrupt()` pauses a LangGraph graph for human-in-the-loop approval (`03/01/01`).
3. **B** — Playground, tracing, and Agent Evaluation all require the `ResponsesAgent` interface (`03/01/03`, `04/01/01`).
4. **B** — Change Data Feed must be enabled for Delta Sync on standard endpoints (`04/01/02`).
5. **C** — AI Search uses L2 distance with HNSW; normalize vectors for cosine-equivalent ranking (`04/01/02`).
6. **C** — models-from-code passes the **file path** string to `python_model`, not the object (`04/01/01`).
7. **B** — UC has no stages; deployment status is an **alias** like `@production` (`04/01/01`).
8. **B** — STORAGE_OPTIMIZED endpoints support **TRIGGERED only** (`04/01/02`).
9. **B** — `ai_query()` runs batch/real-time inference over tables (DBR 18.2+, serverless) (`04/02/01`).
10. **B** — Masking is an **endpoint guardrail** capability (PII behavior = MASK); service policies only DENY (`05/01/01`).
11. **C** — `RetrievalGroundedness` measures grounding in retrieved docs and needs a trace with a RETRIEVER span (`05/02/01`).
12. **B** — Low temperature reduces randomness; a structured-output/response-format schema constrains to valid JSON (`01/02/02`).

**Scoring:** 10–12 correct = on track (≥70% pace). 7–9 = passing but review your misses. ≤6 = do a
focused gap-analysis pass before your next mock.
</details>

**Failure mode to observe:** a candidate who reads question 10 as "block PII" instead of the actual
"redact (not block)" picks A (`system.ai.block_pii`) and loses a point they knew the answer to. The
stem's qualifier — *redact, not block* — is the entire question. This is a **misread**, not a
knowledge gap, and the fix is process (slow the first pass, underline qualifiers), not more studying.

## Common Pitfalls & Misconceptions

- **Pitfall:** Spending 5 minutes on one hard question → **Why it happens:** stubbornness / sunk cost → **Fix:** cap first-pass time at ~90 seconds, flag, and move on. Banked time on easy questions is worth more than grinding one hard one.

- **Pitfall:** Leaving questions blank → **Why it happens:** hoping to return and running out of time → **Fix:** there's no wrong-answer penalty; always submit a best guess after elimination. Blank is a guaranteed zero.

- **Pitfall:** Over-selecting on multi-select questions → **Why it happens:** an option is "true in general" → **Fix:** evaluate each option against *this stem's* conditions; exclude options that are only true under different conditions. Respect the "choose N" count.

- **Pitfall:** Taking mocks but not reviewing the misses → **Why it happens:** the score feels like the point → **Fix:** categorize every miss (gap / misread / trap) and remediate accordingly. The review teaches more than the mock itself.

## Key Definitions

| Term | Definition |
|---|---|
| First pass | The initial sweep answering easy questions quickly and flagging hard ones |
| Triage | Deciding fast whether to answer now or flag and return |
| Checkpoint | A time marker (e.g., Q20 ≈ 30 min) used to detect if you're behind pace |
| Multi-select | A question requiring selection of multiple correct options, often all-or-nothing scored |
| Miss category | Classifying a wrong answer as a knowledge gap, a misread, or a trap, to guide remediation |

## Summary / Quick Recall

- **90 seconds/question average**; target ~75s on the first pass to bank time; reserve ~10–15 min at the end
- **Checkpoints:** ~Q20 by 30 min, ~Q40 by 60 min; if behind, triage faster
- Per question: **read fully → classify (recall/decision/distractor) → eliminate → answer → flag if unsure**
- **Never leave a question blank** — no penalty for wrong answers
- Multi-select: evaluate each option independently, respect the count, watch the plausible-but-wrong extra
- **Review misses by category:** gap → restudy the chapter; misread → slow down; trap → study Pitfalls
- Take one **timed mock early** and one in the **final 48 hours**

## Self-Check Questions

1. At question 30 you notice you've used 55 minutes. Are you on pace, and what should you do?

<details>
<summary>Answer</summary>
You're behind. At 90 seconds each, question 30 should land near the 45-minute mark; 55 minutes means you're ~10 minutes over. Speed up your first pass — answer-and-flag more aggressively, cap hard questions at ~90 seconds, and stop re-reading questions you've already reasoned through. Protect enough time to at least attempt every remaining question.
</details>

2. A "choose two" question has one clearly correct option, one clearly wrong, and two that both seem plausible. How do you proceed?

<details>
<summary>Answer</summary>
Lock in the clearly correct option and exclude the clearly wrong one. Then evaluate the two plausible options independently against the stem's specific conditions — pick the one that is unambiguously true *for this scenario*. If both seem equally valid, favor the one whose truth doesn't depend on an assumption the stem never states. You must select exactly two.
</details>

3. You get a question right by guessing after eliminating two options. Should you review it?

<details>
<summary>Answer</summary>
Yes. A lucky guess masks a real knowledge gap — next time the eliminated options might differ and the guess could go the other way. Treat guessed-correct like a miss: identify the underlying gap and remediate it from the relevant chapter.
</details>

4. Why is a misread a fundamentally different problem from a knowledge gap, and how does the fix differ?

<details>
<summary>Answer</summary>
A misread means you *had* the knowledge but answered the wrong question (missed a qualifier like "in production" or "redact, not block"). The fix is process — read stems fully, underline qualifiers, slow the first pass slightly — not more studying. A knowledge gap means you lacked the fact, and the fix is targeted review of the relevant chapter. Cramming won't fix misreads, and slowing down won't fix gaps.
</details>

## Further Reading

- [Databricks — Generative AI Engineer Associate certification](https://www.databricks.com/learn/certification/generative-ai-engineer-associate) — official exam guide, sample questions, and any current practice-exam offering
- `06/01/01-full-blueprint-review-and-gap-analysis/notes.md` — the coverage map to remediate misses by chapter
- `06/01/03-common-pitfalls-and-final-checklist/notes.md` — the trap list to convert trap-misses into points
