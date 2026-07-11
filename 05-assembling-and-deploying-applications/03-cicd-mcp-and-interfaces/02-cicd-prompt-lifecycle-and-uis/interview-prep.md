# CI/CD, Prompt Lifecycle, and User Interfaces — Interview Prep

**Section:** 05-assembling-and-deploying-applications | **Role target:** Senior ML Engineer, MLOps Engineer, Generative AI Engineer, Solutions Architect

---

## Core Conceptual Questions

| Question | Key points to cover | Common weak-answer trap |
|---|---|---|
| What is a Declarative Automation Bundle and how does it differ from running a notebook manually? | IaC approach; `databricks.yml` as single source of truth; `targets:` block for env separation; atomic resource management via CLI; audit trail in Git | "It's just a YAML wrapper for a job" — misses the IaC model, multi-environment promotion, and the fact that bundles manage endpoints, experiments, and registered models, not only jobs |
| Explain the difference between `databricks bundle deploy` and `databricks bundle run`. | `deploy` creates/updates all declared resources in the target workspace (infra provisioning); `run` executes a specific workflow resource (job or pipeline) that was previously deployed. Deploy ≠ execute. | Conflating the two; saying "deploy runs the job" — in reality, deploy only provisions; run is a separate step |
| What are the two primary strategies for prompt versioning, and when would you choose one over the other? | Git-tracked YAML (SHA = version, good for tight code+prompt coupling); MLflow Prompt Registry (alias-based promotion, good for decoupled prompt/code teams). Choice axis: release cadence — same team/PR → Git YAML; independent cadences → registry | "We just store prompts in the notebook" — no version identity, no audit trail, no programmatic retrieval |
| What does `agents.deploy()` provision, and what permissions do domain experts need to use the Review App? | Creates Model Serving endpoint, provisions Review App, enables real-time MLflow tracing, creates inference tables. Domain experts need: Databricks account access + `CAN_EDIT` on MLflow experiment. No workspace access required. | Saying domain experts need workspace access; not knowing the Review App exists; confusing Review App with the serving endpoint |
| Distinguish the use cases for AI Playground, Genie Agents, and Databricks Apps. | AI Playground: engineer prototyping, model comparison, no external access. Genie Agents: conversational BI, SQL/Unity Catalog only, business users. Databricks Apps: custom external-facing UIs with Streamlit/Gradio, serverless, OAuth/UC inheritance. | "They all do the same thing" or "Genie can replace a RAG chatbot" |

---

## Applied / Scenario Questions

**Q:** Your team's production RAG agent started returning lower-quality answers three weeks ago. Your MLflow traces show the requests and responses but you cannot identify the cause. What should you have instrumented differently, and how would you prevent this in future deployments?

**Strong answer framework:**
- Identify the missing artifact: the system prompt (and few-shot examples) was not tagged on the MLflow run. Add `mlflow.set_tag("prompt_sha", get_git_sha())` at agent init.
- Correlate the answer quality drop to a Git commit timeline: with the SHA tag, search MLflow traces filtered by date range and compare `prompt_sha` values across the regression window.
- Add a `"prompt_path"` tag as well so the exact file is traceable, not just the commit.
- For future deployments: move prompts to a YAML file, log SHA + path on every run, integrate a nightly offline eval (`mlflow.genai.evaluate()`) against a held-out dataset so regressions surface before users report them.
- Show tradeoff awareness: Git-tracked YAML is sufficient if prompt and code are reviewed together; MLflow Prompt Registry adds governance but increases operational surface — choose based on team structure.

**Q:** A colleague proposes deploying the Streamlit chatbot UI on an EC2 instance rather than Databricks Apps to avoid "platform lock-in." How do you evaluate this proposal?

**Strong answer framework:**
- Concrete costs of EC2 approach: must manage credentials/secrets for Unity Catalog and Model Serving access; no automatic OAuth; manual TLS cert management; no Unity Catalog audit log for app data access; separate infrastructure cost and ops burden.
- Concrete benefits of EC2: more control over the OS and network config; not tied to Databricks compute pricing for the app server.
- When EC2 is genuinely justified: if the app has non-Databricks dependencies (e.g., calling a third-party payment API that requires an inbound webhook), or if the organization has a strict "no vendor lock-in for web tier" policy.
- The "lock-in" framing is often overstated: Databricks Apps runs standard Python (Streamlit/Gradio) code; porting to another host is a configuration change, not a rewrite.
- Show tradeoff awareness: for regulated data (HIPAA, FINRA), the governance benefits of Databricks Apps typically outweigh the lock-in concern.

---

## System Design / Architecture Questions

**Q:** Design a CI/CD pipeline for a LangGraph agent that must pass automated evaluation and human Review App approval before reaching production. Sketch the pipeline stages and identify the artifacts that gate each stage.

**Approach:**

1. **Clarify requirements:** What is the evaluation threshold for automated tests? Who are the domain experts for the Review App? What is the rollback strategy? Is zero-downtime required?

2. **Propose structure:**
   - **Stage 1 — PR validation (CI):** `databricks bundle validate`; unit tests for graph nodes; offline eval (`mlflow.genai.evaluate()`) against a frozen test dataset; gate: eval score above threshold AND no YAML validation errors.
   - **Stage 2 — Staging deploy (CI/CD):** `databricks bundle deploy -t staging`; `agents.deploy()` against staging UC model; Review App URL posted to PR comment; gate: ≥N domain-expert approvals in Review App (checked via MLflow Assessment count via SDK).
   - **Stage 3 — Production deploy (CD):** `databricks bundle deploy -t prod`; `agents.deploy(scale_to_zero_enabled=False)`; post-deploy smoke test hits endpoint and asserts response schema; gate: smoke test passes.
   - **Artifacts at each gate:** Git SHA (identifies code + prompt), UC model version (identifies model), MLflow run ID (links traces to eval), Assessment count (Review App approvals).

3. **Justify choices and name tradeoffs:**
   - Automated eval gates catch prompt regressions fast but can miss nuanced quality issues — that is why the human Review App gate is non-negotiable for production.
   - Running `agents.deploy()` separately from the bundle (vs. declaring the endpoint only in the bundle) allows the Review App to be provisioned with feedback infrastructure; a bundle-only deploy does not provision the Review App automatically.
   - The staging Review App gives domain experts a realistic test surface with live inference, not just static example outputs.

---

## Vocabulary That Signals Expertise

Use these terms naturally — don't force them:

- **"targets block"** — when discussing environment promotion; signals you understand DAB's layered override model vs. env-var hacks
- **"prompt SHA tag"** — signals you treat prompts as versioned artifacts, not anonymous strings
- **"Assessment objects"** — when discussing Review App feedback; signals you know the MLflow data model, not just that feedback "gets saved somewhere"
- **"scale_to_zero_enabled: false"** — when discussing production endpoint config; signals you know the cold-start cost tradeoff
- **"labeling session"** — when describing the Review App workflow; signals you know the structured review mechanism, not just the UI
- **"mode: development"** — when discussing DAB dev targets; signals you know resource-name prefixing and its purpose
- **"Candidate / Production alias"** — when discussing MLflow Prompt Registry promotion; signals registry fluency
- **"atomic deploy"** — when describing what `bundle deploy` does; signals you understand it is not a sequential script

---

## Vocabulary That Signals Weakness

Avoid these — they signal outdated or shallow understanding:

- **"I use dbx"** — dbx is deprecated; the current tool is Declarative Automation Bundles (DAB) / Databricks CLI `bundle` commands
- **"bundle.yaml"** — the correct filename is `databricks.yml`; using the wrong name signals you have not actually used the tool
- **"Genie Spaces"** — the current name is Genie Agents (renamed July 2026); using the old name signals you have not checked current docs
- **"the Review App shows users the chatbot"** — conflates the stakeholder feedback tool with a production UI; the Review App is for internal domain experts, not end users
- **"we just put the prompt in an environment variable"** — signals no audit trail strategy; cannot answer "which prompt was active on date X?"
- **"deploy runs the pipeline"** — confuses `bundle deploy` (provisioning) with `bundle run` (execution)

---

## STAR Answer Frame

**Situation:** I was on a team that had deployed a Databricks-hosted RAG chatbot. Three months in, the legal team raised a compliance concern: they could not verify which system prompt was active when a specific set of user conversations occurred, because the prompt was a hardcoded string in a notebook.

**Task:** I was responsible for designing and implementing an audit-ready prompt versioning system that could answer "what prompt produced this response?" for any historical MLflow trace, without disrupting the existing deployment cadence.

**Action:** I extracted the system prompt to a `prompts/rag_system.yaml` file, committed it to the repository, and added three lines to the agent initialization code: load the YAML, capture the Git SHA via subprocess, and call `mlflow.set_tag("prompt_sha", sha)` inside every MLflow run. I updated the CI/CD pipeline to fail the bundle validate step if `prompts/rag_system.yaml` was absent or the SHA tag was not present in a canary trace from staging. For future prompt changes owned by the legal team independently, I proposed migrating to the MLflow Prompt Registry with `Candidate` and `Production` aliases — but deferred that to a later phase to avoid disrupting an active deployment.

**Result:** Within two weeks, every production trace had a `prompt_sha` tag. The compliance team could answer any audit question about historical prompt state in under five minutes by filtering MLflow traces by date and running `git show <sha>`. The change required no model retraining, no endpoint redeployment, and no downtime.

---

## Red Flags Interviewers Watch For

- **Describing a prompt as a "config value" stored in an environment variable** — signals no understanding of auditability requirements; environment variables are not versioned, not logged, and not queryable
- **Saying `databricks bundle deploy` executes the job** — fundamental CLI command confusion; deploy provisions infrastructure, run executes workflows
- **Proposing separate `databricks.yml` files for dev, staging, and prod** — the entire value of DAB is one file with a `targets:` block; separate files negate the approach and create drift
- **Recommending Genie Agents for a use case that involves unstructured document retrieval** — shows failure to understand the SQL-only constraint; would result in a product that cannot answer the user's actual questions
- **Not knowing what the Review App URL is or who can access it** — signals the candidate has not actually deployed an agent end-to-end; a common surface-level DAB gap
- **Treating `mode: development` as optional on all targets** — if set on prod, it prefixes resource names and breaks external references; shows incomplete understanding of bundle modes
