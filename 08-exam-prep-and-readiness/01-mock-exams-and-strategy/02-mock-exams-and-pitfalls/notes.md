# Mock Exams and Exam Pitfalls

**Section:** 08 Exam Prep | **Module:** 01 Mock Exams and Strategy | **Est. time:** 2 hrs | **Exam mapping:** Supporting content — exam strategy

---

## TL;DR

The Databricks Certified Generative AI Engineer Associate exam is 45 questions in 90 minutes (~2 min/question), passing at roughly 70%, proctored online via Examity. Most candidates who fail do not fail because they lack knowledge — they fail because they misread the question stem, fall for a plausible-but-wrong distractor, or run out of time on easy questions they second-guessed. Developing a systematic elimination protocol is as important as content mastery. **The one thing to remember: read the constraint in the stem first — the correct answer almost always depends on a single constraining word or phrase that most candidates skip.**

---

## ELI5 — Explain It Like I'm 5

Imagine you are playing a board game where every card has a trick hidden in its fine print. If you just glance at the card and grab the first answer that sounds right, you will keep landing on the wrong space. The trick is to read the fine print first — specifically the one word that changes everything — and only then look at the answers. Most people do the opposite: they read the answers first and keep jumping to the first one that sounds familiar. On this exam, the wrong answers are designed to sound familiar — that is the whole point of a distractor. The candidate who wins is not the one who knows the most facts; it is the one who spots the hidden constraint in the question before even looking at the options.

---

## Learning Objectives

By the end of this chapter you will be able to:
- [ ] Identify the five recurring distractor patterns in Databricks GenAI Associate exam questions and apply an elimination rule to each
- [ ] Execute a time-budget protocol that allocates 2 min/question with a flag-and-return strategy for hard questions
- [ ] Apply the stem-first reading technique to extract the constraining phrase before evaluating answer options
- [ ] Distinguish at least 10 confusable API / concept pairs that appear as wrong-answer traps on the exam
- [ ] Explain why multi-select questions require both correct answers simultaneously and describe the common over-selection trap

---

## Visual Overview

### 90-Minute Exam Time Budget

```
START ──► Read exam instructions (2 min)
         │
         ▼
         Questions 1–45: TARGET 2 min/question
         │
         ├── Confident? ──► Answer immediately ──► Continue
         │
         └── Uncertain? ──► Flag + best guess ──► Continue
                                    │
                                    ▼
                           End of first pass (~80 min)
                                    │
                                    ▼
                           Review flagged questions
                           (remaining ~8–10 min)
                                    │
                                    ▼
                           Submit — never leave blank
```

### Question-Difficulty Triage Flow

```
Read question stem
         │
         ▼
Extract the constraint phrase
("wants to avoid...", "must preserve...", "without retraining...")
         │
         ▼
Does a single answer satisfy the constraint clearly?
         ├── Yes ──► Mark it. Move on. (< 90 sec)
         │
         └── No ──► Eliminate obviously wrong options (wrong layer / wrong product)
                    │
                    ▼
                    Two options remain?
                    ├── Yes ──► Apply the distinguishing rule for this pair ──► Mark
                    └── No  ──► Flag + mark best guess ──► Return later
```

### Distractor Taxonomy Tree

```
Wrong-answer patterns
│
├── (1) Correct concept, wrong layer
│        e.g. Unity Catalog masking ≠ AI Gateway guardrails
│
├── (2) Correct tool, wrong parameter
│        e.g. model_type="llm/v1/completions" ≠ model_type="databricks-agent"
│
├── (3) Outdated API
│        e.g. MLflow flavors before pyfunc standardisation
│
├── (4) Correct for a different Databricks product
│        e.g. Delta Live Tables answer ≠ Vector Search answer
│
└── (5) Partially correct but missing a constraint
         e.g. deploys model but skips Unity Catalog registration step
```

### Domain Weight vs. Expected Question Count

```
Domain                          Weight   ~Questions
──────────────────────────────────────────────────
Application Development          30%       ~13–14
Assembling & Deploying           22%        ~9–10
Design Applications              14%         ~6
Data Preparation                 14%         ~6
Evaluation & Monitoring          12%         ~5
Governance                        8%         ~3–4
──────────────────────────────────────────────────
TOTAL                           100%          45
```

---

## Key Concepts

### Question Anatomy

**What it is:** Each exam question follows a fixed structure — a scenario stem (the "Given that..." or "A data engineer wants to...") followed by exactly four options, one of which is the best answer. Multi-select questions ("Which TWO...") specify the count explicitly and require all specified correct answers to be selected.

**How it works:** The stem encodes the constraint that distinguishes the correct answer from the distractors. Common stem patterns include:
- "A data engineer wants to [goal] while [constraint]" — the constraint is the key
- "Which of the following is the MOST appropriate way to..." — asks for best, not just valid
- "Which TWO of the following..." — both must be correct; selecting three is wrong even if all three are true in isolation
- "A team is building a RAG application and needs to..." — context narrows to a specific Databricks feature area

**In exam questions:** The stem will almost always contain one of: a resource constraint (cost, latency, memory), an operational constraint (no cluster restart, production endpoint already exists), or a governance constraint (PII masking required, Unity Catalog enforced). The correct answer satisfies all stated constraints; distractors satisfy some but not all.

---

### Distractor Taxonomy

**What it is:** The five recurring wrong-answer patterns in Databricks GenAI Associate exams. Recognising the pattern type is faster than reasoning from first principles under time pressure.

**How it works — the five patterns:**

1. **Correct concept, wrong layer** — The answer names a real Databricks feature but applies it at the wrong architectural level. Example: a question about blocking PII in LLM responses uses AI Gateway guardrails, not Unity Catalog column masking. Unity Catalog masking operates on structured data at query time; AI Gateway operates on unstructured text at inference time.

2. **Correct tool, wrong parameter** — The tool name is right but the configuration value is wrong. Example: `mlflow.evaluate()` with `model_type="databricks-agent"` is correct for agent evaluation; `model_type="llm/v1/chat"` is the Foundation Model endpoint flavor, not the evaluation parameter.

3. **Outdated API** — References an older MLflow or Databricks API that has been superseded. Example: logging a LangChain model with `mlflow.langchain.log_model()` and setting the serving endpoint manually, when the current pattern is `databricks.agents.deploy()` after Unity Catalog registration.

4. **Correct for a different Databricks product** — The answer is correct in a different product context. Example: `OPTIMIZE` and `ZORDER` are correct for Delta Lake query acceleration but are not the answer for Vector Search index freshness questions.

5. **Partially correct but missing a constraint** — The answer describes a valid approach but omits a required step that the stem specifies. Example: an answer that deploys a model via `mlflow.register_model()` but does not set `databricks_resource_name` when the stem specifies Unity Catalog is required.

**In exam questions:** Distractors are almost never obviously wrong — they are wrong *for this scenario*. Always anchor to the constraint in the stem, not to whether the option describes something true in general.

---

### Time Management

**What it is:** A pacing discipline that prevents time starvation on easy questions caused by over-investing in hard ones.

**How it works:** 90 minutes ÷ 45 questions = exactly 2 minutes per question. The protocol is: (1) attempt every question on the first pass; (2) flag uncertain questions and mark your best guess before moving on; (3) never leave a question blank — the exam has no penalty for guessing; (4) return to flagged questions in the final 8–10 minutes. The cognitive trap is that hard questions feel important and pull more time; in reality, a medium question skipped costs you the same point as a hard question abandoned.

**In exam questions:** Time management does not appear as a content question but as a meta-skill. Candidates who finish with 20 minutes remaining consistently outperform candidates who run out of time, because the time pressure causes premature commitment to the first plausible option.

---

### Multi-Select Mechanics

**What it is:** Questions labelled "Which TWO of the following..." (occasionally "Which THREE...") that require selecting exactly the specified number of correct options. The exam interface allows selecting more, but partial credit is not awarded — all specified answers must be correct and no extras selected.

**How it works:** Multi-select questions test that the candidate knows the complete set of correct answers, not just one. The common trap is selecting three options when only two are correct, because all three appear individually true. The elimination approach: first identify any option that is clearly wrong and remove it, then evaluate the remaining options against the constraint, then select exactly the count specified.

**In exam questions:** Multi-select questions appear on topics where two components are jointly required — for example, the two required parameters for `mlflow.evaluate()` when using Agent Evaluation, or the two index types available in Databricks Vector Search. The exam will not always place both correct options adjacently in the list.

---

### High-Yield Topic Clusters

**What it is:** The sub-topics within Application Development (30% weight, ~13–14 questions) and Assembling & Deploying (22%, ~9–10 questions) that generate the most questions and the most confusion.

**How it works — Application Development high-yield sub-topics:**

- **LangGraph state management** — how `StateGraph`, `TypedDict` state schemas, `add_messages` reducer, and conditional edges work; common trap: treating LangGraph state as a simple dict instead of a typed schema with reducers
- **RAG chain assembly** — the order of steps (embed query → Vector Search retrieval → context injection → LLM call) and which MLflow flavor logs each component (`mlflow.langchain.log_model` for LangChain chains, `mlflow.pyfunc.log_model` for custom chains)
- **PyFunc wrapping** — when to use `mlflow.pyfunc.PythonModel` subclass vs. `mlflow.langchain.log_model`; PyFunc is required when the inference logic is not a standard LangChain chain
- **Tool calling / function calling** — how agents bind tools to an LLM, the difference between tool definitions (JSON schema) and tool execution (Python function), and how LangGraph nodes route based on tool call presence in the message

**Assembling & Deploying high-yield sub-topics:**

- **`databricks.agents.deploy()` vs. `mlflow.register_model()`** — `register_model` moves a model from a run to the Unity Catalog registry; `agents.deploy()` creates an Agents endpoint with review app and inference tables; they are sequential steps, not alternatives
- **Model Serving endpoint configuration** — `workload_size`, `scale_to_zero_enabled`, `min_provisioned_concurrency`, `max_provisioned_concurrency`; concurrency values must be multiples of 4

**In exam questions:** Questions on these sub-topics are most likely to use distractor pattern (5) — partially correct but missing a step.

---

### Known Exam Pitfalls

**What it is:** Specific API and concept confusions that recur in practice exams and are known exam traps.

**How it works — the four primary traps:**

1. **`mlflow.evaluate()` model_type values** — `"databricks-agent"` triggers Agent Evaluation with AI judges; `"llm/v1/chat"` and `"llm/v1/completions"` are Foundation Model endpoint flavors used in `mlflow.deployments`. Using `"llm/v1/chat"` in `mlflow.evaluate()` will not run AI judges.

2. **Vector Search index types** — **Delta Sync Index** syncs automatically from a Delta table (managed, eventual consistency, suitable for batch-updated document stores); **Direct Access Index** is updated programmatically via the SDK (real-time, suitable for streaming or event-driven pipelines). Common trap: using Delta Sync index for a use case requiring immediate updates after every event.

3. **AI Gateway guardrails vs. Unity Catalog masking** — AI Gateway guardrails filter or transform unstructured LLM inputs/outputs (PII detection in free text, topic blocking); Unity Catalog column masking applies to structured data queries. They operate at different layers and are not interchangeable.

4. **`databricks.agents.deploy()` vs. `mlflow.register_model()`** — `mlflow.register_model()` registers a model in Unity Catalog (makes it versioned and auditable); `databricks.agents.deploy()` deploys the registered model as an Agents serving endpoint with a review app. Calling `deploy()` without prior registration will fail if the model is not yet in Unity Catalog.

**In exam questions:** These four traps appear as distractors (pattern 2 and pattern 4) in scenario questions. Memorise the distinguishing rule for each pair, not just the names.

---

## Key Parameters / Configuration Knobs

This section is a **Confusable Pairs** reference. Each row is a pair of concepts that commonly appear as wrong-answer distractors for each other.

| Concept A | Concept B | How to distinguish in exam context |
|---|---|---|
| `mlflow.evaluate()` `model_type="databricks-agent"` | `mlflow.evaluate()` `model_type="llm/v1/chat"` | `"databricks-agent"` activates AI judges (groundedness, safety, etc.); `"llm/v1/chat"` is a serving endpoint flavor, not an evaluation mode |
| `databricks.agents.deploy()` | `mlflow.register_model()` | `register_model` → Unity Catalog entry (versioning); `agents.deploy()` → live endpoint (serving); they are sequential, not alternative steps |
| Vector Search **Delta Sync Index** | Vector Search **Direct Access Index** | Delta Sync: automatic, managed, eventual consistency, for Delta-backed stores; Direct Access: programmatic SDK updates, real-time, for streaming ingestion |
| **AI Gateway guardrails** | **Unity Catalog column masking** | AI Gateway: operates on unstructured LLM text at inference time; UC masking: operates on structured SQL query results at read time |
| `mlflow.langchain.log_model()` | `mlflow.pyfunc.log_model()` | Use `langchain` flavor for standard LangChain chains/agents; use `pyfunc` when wrapping arbitrary Python logic or non-LangChain frameworks |
| LangGraph **StateGraph** `add_messages` reducer | LangGraph **StateGraph** overwrite (default) | `add_messages` appends new messages to the list; default assignment overwrites the entire field; wrong reducer = lost conversation history |
| **Foundation Model API** (pay-per-token) | **External Models** in AI Gateway | Foundation Model API: Databricks-hosted models billed per token; External Models: proxied access to third-party providers (OpenAI, Anthropic) through AI Gateway |
| `mlflow.pyfunc.PythonModel` subclass | `mlflow.langchain.log_model` with `loader_fn` | `PythonModel` for fully custom inference logic; `loader_fn` for LangChain models needing a custom environment setup at load time |
| **Review App** (Databricks Agents) | **Inference Tables** (Model Serving) | Review app: human feedback UI for labelling agent traces; Inference tables: automatic logging of request/response pairs to Delta for offline analysis |
| `mlflow.set_registry_uri("databricks-uc")` | `mlflow.set_tracking_uri("databricks")` | `registry_uri` controls where models are registered (Unity Catalog vs Workspace Registry); `tracking_uri` controls where runs/experiments are logged |

---

## Worked Example: Requirement → Decision

**Given:** An exam question reads: *"A machine learning engineer has built a RAG application using LangChain. They want to evaluate the application's groundedness and relevance using Databricks AI judges, log the results to an MLflow experiment, and pass previously generated responses rather than calling the application at evaluation time. Which of the following `mlflow.evaluate()` calls is correct?"*

Options:
- A. `mlflow.evaluate(data=df, model=chain, model_type="llm/v1/chat")`
- B. `mlflow.evaluate(data=df, model_type="databricks-agent")`
- C. `mlflow.evaluate(data=df, model_type="databricks-agent")` where `df` includes a `response` column
- D. `mlflow.evaluate(data=df, model=chain, model_type="databricks-agent")`

**Step 1 — Identify what the question is testing:**
Two things: (1) which `model_type` activates AI judges, and (2) the difference between passing pre-generated responses vs. passing the live model.

**Step 2 — Eliminate clearly wrong options:**
Option A uses `model_type="llm/v1/chat"` — that is a serving flavor, not an evaluation mode. Eliminate A (distractor pattern 2: correct tool, wrong parameter).

**Step 3 — Identify the distractor trap:**
Options B and C both use `model_type="databricks-agent"`. The stem says "pass previously generated responses rather than calling the application." That means no `model=` argument — pass a DataFrame that already contains the `response` column. Option D includes `model=chain`, which means the model will be called at evaluation time — violating the stem constraint.

**Step 4 — Apply the distinguishing rule:**
When `model` is omitted from `mlflow.evaluate()`, the call uses pre-generated outputs from the `data` DataFrame. The `response` column must be present. Option B omits both `model=` and the `response` column — that will raise a missing-column error. Option C omits `model=` and the description states `df` includes a `response` column.

**Step 5 — Select the answer:**
**C is correct.** It uses `model_type="databricks-agent"` to activate AI judges, omits `model=` so pre-generated responses are used, and the DataFrame contains the `response` column as required. The key distinguishing rule: omit `model=` argument when passing pre-generated outputs, include `response` column in the data.

---

## Implementation

```text
# Scenario: Executing a 90-minute exam with no questions left blank
# Constraint: Must balance deep thinking on hard questions with coverage of all 45

EXAM DAY TIME-BUDGET PROTOCOL
==============================

Before clicking Start:
  - Confirm Examity proctoring connection is stable (test audio + video)
  - Close all browser tabs except the exam
  - Have water. No phone.

First pass (target: 80 minutes for 45 questions = 1.78 min avg):
  For each question:
    1. Read the STEM only. Extract the constraint phrase.
    2. Predict the answer category before reading options.
    3. Read all four options.
    4. If confident → select + move on (target: < 90 sec)
    5. If uncertain → select best guess + FLAG + move on (target: < 2 min)
    6. Never spend > 3 min on a single question in first pass.

Second pass (remaining time, flagged questions only):
  1. Re-read stem. Look for constraint word you may have missed.
  2. Apply elimination: cross out the two most obvious wrong-pattern distractors.
  3. Choose between the two remaining options using the distinguishing rule.
  4. Change your answer only if you have a specific reason — not because of doubt.

Final 2 minutes:
  - Confirm every question has a selected answer (no blanks).
  - Submit.

Budget checkpoints:
  Q15 done → should be at ~30 min elapsed
  Q30 done → should be at ~60 min elapsed
  Q45 done → should be at ~80 min elapsed (10 min buffer for review)
```

```text
# Anti-pattern: Reading all answer options sequentially and selecting the first one
# that sounds plausible — the "recognition trap"
#
# What breaks: Distractors are designed to trigger recognition before analysis.
# Option A often contains a familiar term (e.g. "mlflow.evaluate") that causes
# premature selection. Candidates stop reading at the first match.
#
# Why this is dangerous on Databricks exams:
# The exam frequently places the correct answer at C or D.
# The wrong answer often uses the right tool with the wrong parameter (pattern 2),
# which sounds correct on first read.

# Correct approach: Elimination-first protocol
# ─────────────────────────────────────────────
# Step 1: Read STEM. Identify constraint phrase. Write it mentally.
# Step 2: Without reading options, predict: what type of answer satisfies
#         the constraint? (a function? a config parameter? a product name?)
# Step 3: Read ALL options before committing.
# Step 4: Eliminate options using distractor patterns:
#           - Wrong layer? → Eliminate (pattern 1)
#           - Wrong parameter for correct tool? → Eliminate (pattern 2)
#           - Outdated API? → Eliminate (pattern 3)
#           - Different product? → Eliminate (pattern 4)
#           - Missing a required step? → Eliminate (pattern 5)
# Step 5: Select the option that satisfies ALL constraints.
# Step 6: If two options survive elimination, apply the confusable-pairs table.
```

---

## Common Pitfalls & Misconceptions

- **Treating `model_type` as a generic string** — Candidates memorise that `mlflow.evaluate()` accepts `model_type` but do not learn the exact valid values. The exam exploits this by offering `"llm/v1/chat"` (a serving endpoint flavor) next to `"databricks-agent"` (the evaluation mode). The correct mental model: `"databricks-agent"` is the only value that activates Databricks AI judges in MLflow 2 Agent Evaluation.

- **Confusing Vector Search index creation with index freshness** — Many candidates learn that Delta Sync Index exists but do not know when to use Direct Access Index. The mistake is always defaulting to Delta Sync. The correct mental model: if the ingestion pipeline pushes updates in real time (streaming, event-driven), Direct Access Index is required because Delta Sync index refresh is scheduled/triggered, not instantaneous.

- **Thinking Unity Catalog and AI Gateway are interchangeable for PII** — Both involve access control, so candidates mix them up. The correct mental model: Unity Catalog masking applies at the SQL/structured-data layer (column masking policies on Delta tables); AI Gateway guardrails apply at the LLM inference layer (input/output filtering). A RAG system needs both at different layers; they do not substitute for each other.

- **Misreading "Which TWO" as "Which of the following"** — Under time pressure, candidates miss the word "TWO" and select only one option, then move on. The correct mental model: if the question says "Which TWO," the interface will let you submit with one selection but it will be wrong. Always check the question type before selecting.

- **Equating `mlflow.register_model()` with deployment** — Registration and deployment are separate steps. `register_model()` creates a versioned entry in Unity Catalog. `databricks.agents.deploy()` creates the serving endpoint. Selecting `register_model()` as the answer to "how do you deploy an agent" is a pattern-5 distractor.

- **Over-relying on keyword matching instead of constraint satisfaction** — Candidates see "LangGraph" in the stem and immediately select any answer containing "LangGraph." The correct mental model: the correct answer satisfies the specific constraint stated in the stem. A LangGraph answer that doesn't address the stated constraint (e.g., "preserving conversation history") is still wrong even if it's about LangGraph.

- **Second-guessing correct first answers** — During review of flagged questions, candidates change their answer from the one they marked first to a different one, often incorrectly. Research consistently shows first-instinct answers are more often correct. The correct mental model: change an answer during review only when you identify a specific reason (you misread the stem, or you now recognise a distractor pattern you missed), not because the other option "also sounds good."

---

## Key Definitions

| Term | Definition |
|---|---|
| `model_type="databricks-agent"` | The `mlflow.evaluate()` parameter value that activates Databricks AI judges (groundedness, relevance, safety) in MLflow 2 Agent Evaluation |
| `model_type="llm/v1/chat"` | A Foundation Model serving endpoint flavor specifying the chat completions API; used in endpoint config, not in `mlflow.evaluate()` |
| Delta Sync Index | A Vector Search index type that automatically syncs embeddings from a Delta table on a schedule or trigger; managed by Databricks |
| Direct Access Index | A Vector Search index type updated programmatically via the SDK; suited for real-time or streaming ingestion pipelines |
| AI Gateway guardrails | Input/output filters applied at the LLM serving layer to detect or block PII, off-topic content, or unsafe responses in unstructured text |
| Unity Catalog column masking | A row/column-level security policy applied to structured Delta table data at SQL query time; not applicable to LLM text streams |
| `databricks.agents.deploy()` | SDK function that creates an Agents serving endpoint from a Unity Catalog–registered model, with review app and inference tables |
| `mlflow.register_model()` | MLflow function that copies a logged model from a run to the Unity Catalog model registry, creating a versioned, auditable entry |
| Distractor | A wrong answer option designed to be plausible; on Databricks exams, distractors are almost always correct in a different context |
| Elimination protocol | A systematic approach to exam questions that crosses off wrong-pattern answers before committing to a selection |
| Stem constraint | The specific word, phrase, or condition in a question stem that narrows the correct answer; identifying it is the first step of sound exam technique |
| Multi-select question | An exam question requiring exactly N correct options (e.g., "Which TWO"); no partial credit — all N must be selected and no extras |
| `add_messages` reducer | A LangGraph state annotation that appends new messages to the state list rather than overwriting; required for conversational agents |
| Review App | A Databricks Agents feature providing a UI for human labellers to review and annotate agent traces after `databricks.agents.deploy()` |
| Inference Tables | Automatic logging of serving endpoint request/response pairs to a Delta table, enabled during endpoint creation or update |

---

## Summary / Quick Recall

- **2 min/question** — 90 min ÷ 45 questions. Flag uncertain questions and move on; return at the end.
- **Read the stem constraint first** — the correct answer satisfies every constraint; distractors satisfy some but not all.
- **Five distractor patterns**: wrong layer, wrong parameter, outdated API, wrong product, missing constraint — learn to spot them by name.
- **`"databricks-agent"` ≠ `"llm/v1/chat"`** — these are different `model_type` values for different purposes; the first is for evaluation, the second is a serving flavor.
- **Delta Sync Index ≠ Direct Access Index** — Sync: managed, eventual; Direct: SDK-updated, real-time.
- **`register_model()` ≠ `deploy()`** — registration and deployment are sequential, not alternative.
- **Multi-select: select exactly N** — if the question says "Which TWO," selecting three is wrong even if all three facts are true.

---

## Self-Check Questions

1. What `model_type` value must be passed to `mlflow.evaluate()` to activate Databricks AI judges such as groundedness and safety?

   <details><summary>Answer</summary>

   **`"databricks-agent"`** is the correct value. It instructs MLflow to use Databricks Agent Evaluation AI judges. The most common distractor is `"llm/v1/chat"`, which is the Foundation Model endpoint flavor used in serving endpoint configuration — it is not a valid evaluation mode and will not activate AI judges. Using it in `mlflow.evaluate()` would either raise an error or produce no judge output.

   </details>

2. A candidate is 70 minutes into the exam and has 12 questions remaining. They have 3 flagged questions and 9 new questions to answer. What is the correct action?

   <details><summary>Answer</summary>

   **Continue answering new questions first, then return to flagged ones.** With 20 minutes and 12 questions remaining, the pace is still manageable (~1.67 min/question). Returning to flagged questions before finishing new ones risks leaving new questions unanswered entirely. The protocol: complete the first pass of all 45 questions (marking a best guess on every flagged item), then use remaining time to revisit flags. No question should ever be left blank — the exam does not penalise wrong guesses.

   </details>

3. **Which TWO** of the following are valid approaches to passing a model to `mlflow.evaluate()` when using `model_type="databricks-agent"` according to current Databricks documentation?

   - A. Pass a pandas DataFrame with a `response` column and omit the `model=` argument (pre-generated outputs)
   - B. Pass `model="endpoints:/your-agent-endpoint"` to call a deployed Agents endpoint at evaluation time
   - C. Pass `model_type="llm/v1/chat"` to enable streaming responses during evaluation
   - D. Pass a Delta table path as the `data` argument and set `model=None`
   - E. Pass a local Python function as the `model=` argument that accepts and returns the expected schema

   <details><summary>Answer</summary>

   **A and B are correct.**

   - **A** is correct: omitting `model=` and including `response` in the DataFrame passes pre-generated outputs — this is the recommended approach when evaluating an already-deployed or previously-run application.
   - **B** is correct: `"endpoints:/endpoint-name"` is a valid model URI format that calls a deployed Agents endpoint at evaluation time (requires `databricks-agents` SDK ≥ 0.8.0).
   - **C** is wrong: `"llm/v1/chat"` is a serving endpoint flavor, not a valid `model_type` for Agent Evaluation — this is distractor pattern 2 (correct tool, wrong parameter).
   - **D** is wrong: `data` must be a pandas DataFrame, not a Delta table path; `model=None` is not a documented option.
   - **E** is wrong as stated: a local Python function is a valid `model=` option (Option 4 in the docs), but option E as written is the most tempting distractor because it sounds plausible — the issue is that E describes the general pyfunc pattern, not the Agents-specific interface. The test of whether to select E is whether the function signature matches the expected schema; if the question does not confirm this, B is the safer choice.

   </details>

4. A team uses a Delta Sync Vector Search index for a customer support chatbot. They observe that newly added support articles do not appear in retrieval results for 15–20 minutes after ingestion. An engineer proposes switching to a Direct Access Index. Under what condition is this the correct decision?

   <details><summary>Answer</summary>

   **Switching to Direct Access Index is correct when the ingestion pipeline can be modified to call the Vector Search SDK to push updates immediately after each article is ingested.** Delta Sync Index syncs on a schedule or trigger from a Delta table — it is not designed for sub-minute freshness. Direct Access Index accepts programmatic upserts/deletes via the SDK with no scheduling lag, but it requires the application team to manage the update calls explicitly. If the team cannot modify the ingestion pipeline, the correct fix is to shorten the Delta Sync refresh interval, not to switch index types. The distractor here is assuming Direct Access is always better — it is better only when the team owns the write path and needs real-time freshness.

   </details>

5. On a high-stakes exam, a candidate finishes their first pass with 12 minutes remaining and has 6 flagged questions. They feel uncertain about 2 of the flagged questions and want to change one answer they originally marked confidently. Which approach has the highest expected score?

   <details><summary>Answer</summary>

   **Revisit only the flagged questions using the elimination protocol; do not change the confident answer unless you can identify a specific distractor pattern you missed the first time.** The research-backed principle is that first-instinct answers are more often correct than changes made under time pressure. The correct process for flagged questions: re-read the stem, extract the constraint phrase, apply the five distractor pattern rules, and then decide. For the confidently-marked answer: change it only if you can now articulate *which distractor pattern* made the original wrong option attractive — vague doubt is not a sufficient reason to switch. Changing answers due to anxiety rather than analysis is the single most common source of score degradation in review phases.

   </details>

---

## Further Reading

- [Run an evaluation and view the results (MLflow 2) — Databricks on AWS](https://docs.databricks.com/aws/en/generative-ai/agent-evaluation/evaluate-agent.html) — *verified 2026-07-16* — Official documentation for `mlflow.evaluate()` with `model_type="databricks-agent"`, AI judge selection, and pre-generated output evaluation patterns
- [Create custom model serving endpoints — Databricks on AWS](https://docs.databricks.com/aws/en/machine-learning/model-serving/create-manage-serving-endpoints.html) — *verified 2026-07-16* — Endpoint configuration parameters including `workload_size`, concurrency settings, and `scale_to_zero_enabled`
- [Databricks Certified Generative AI Engineer Associate Exam Guide, March 2026](https://www.databricks.com/sites/default/files/2026-03/Databricks-Certified-Generative-AI-Engineer-Associate-Exam-Guide-Mar26.pdf) — *verified 2026-07-11* — Authoritative source for domain weights, topic list, and exam format (45 questions, 90 min, Examity proctoring)
