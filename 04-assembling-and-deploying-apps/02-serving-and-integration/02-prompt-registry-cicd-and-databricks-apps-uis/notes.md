# Prompt Registry, CI/CD, and Databricks Apps UIs

**Section:** Assembling and Deploying Apps | **Module:** Serving and Integration | **Est. time:** 2.5 hrs
**Exam mapping:** Databricks GenAI Engineer Associate — Assembling and Deploying Applications domain (~20%); prompt versioning, CI/CD with bundles, and Databricks Apps deployment are directly tested

## Learning Objectives

By the end of this chapter, you will be able to:

- Register and load versioned prompts in the MLflow Prompt Registry using aliases
- Explain why prompts are version-controlled separately from application code
- Describe Declarative Automation Bundles (formerly Asset Bundles) and the bundle CLI lifecycle
- Set up a dev → staging → prod CI/CD flow using UC catalogs, aliases, and deployment jobs
- Deploy a chat UI for an agent using Databricks Apps and bind it to a serving endpoint
- Explain the operational lifecycle that ties prompts, CI/CD, and Apps together

## Core Concepts

### The MLflow Prompt Registry

Prompts are as much a part of your application's behavior as code — but hardcoding them in source
files makes them hard to iterate on, review, roll back, or let non-engineers edit. The **MLflow
Prompt Registry** (part of MLflow 3 / `mlflow.genai`) is a **Git-inspired, versioned repository for
prompt templates**. On Databricks it is backed by **Unity Catalog** (prompts are UC entities with
three-level names), so it inherits UC governance and audit.

Its data model mirrors Git:

- **Prompt** — a named entity (UC name `catalog.schema.name`).
- **Version** — an **immutable** snapshot with an auto-incrementing number. You cannot edit a version;
  to change a prompt you register a new version.
- **Alias** — a **mutable** named pointer to a version (`production`, `staging`, `beta`). `@latest` is
  reserved.

```python
import mlflow

# Register a prompt (double-brace {{variable}} template syntax)
prompt = mlflow.genai.register_prompt(
    name="prod.agents.support_system_prompt",
    template="You are a Databricks support assistant. Answer using only the provided context.\n"
             "Context: {{context}}\nQuestion: {{question}}",
    commit_message="Initial support prompt",
    tags={"author": "eng@co.com", "task": "rag_qa"},
)
print(prompt.version)   # 1

# Point the "production" alias at version 1
mlflow.genai.set_prompt_alias(
    name="prod.agents.support_system_prompt", alias="production", version=1
)

# Load by alias or by version in your agent code
p = mlflow.genai.load_prompt("prompts:/prod.agents.support_system_prompt@production")
p_v2 = mlflow.genai.load_prompt("prompts:/prod.agents.support_system_prompt/2")

# Fill in the variables at runtime
text = p.format(context=retrieved_docs, question=user_question)
```

Two syntax notes the exam likes:
- MLflow uses **double braces** `{{variable}}`. **LangChain/LlamaIndex use single braces** `{variable}`.
  Convert with `prompt.to_single_brace_format()`.
- Templates can be text (`str`) or chat (`List[dict]` with `role`/`content`), and Jinja2 is
  auto-detected when the template contains `{% %}` control flow.

**Why version prompts at all?** Immutable, reproducible history; alias-based deployment/rollback/A-B
testing **without touching app code** (flip the alias, not a deploy); non-engineers can edit prompts
via the UI; lineage integrated with tracing and evaluation; and UC governance + audit. It's the same
alias-driven promotion idea you saw for models — applied to prompts.

### Declarative Automation Bundles (DABs)

**Declarative Automation Bundles** — *"formerly known as Databricks Asset Bundles"* — are
infrastructure-as-code for Databricks. A single `databricks.yml` file declares your resources (jobs,
pipelines, **model serving endpoints**, **registered models**, **apps**, experiments, catalogs,
vector search endpoints/indexes) plus your source code and config, as one deployable, version-controlled
unit. The **rename is a known exam gotcha:** the name changed, but the config file is still
`databricks.yml` and the CLI is still `databricks bundle`.

The CLI lifecycle:

```bash
databricks bundle init mlops-stacks      # scaffold from a template
databricks bundle validate               # check config against the schema
databricks bundle deploy -t dev          # deploy resources to the "dev" target
databricks bundle run my_training_job    # run a resource by its YAML key
databricks bundle destroy                # tear down
```

**Targets** are named environments (`dev`, `staging`, `prod`), each with its own workspace `host`,
`root_path`, and `mode` (`development` vs `production`). This is how one bundle deploys to multiple
environments. A minimal bundle that deploys an agent's serving endpoint and its chat app:

```yaml
# databricks.yml
bundle:
  name: docs-agent

targets:
  dev:
    mode: development
    workspace:
      host: https://dev.cloud.databricks.com
  prod:
    mode: production
    workspace:
      host: https://prod.cloud.databricks.com

resources:
  model_serving_endpoints:
    docs_agent_endpoint:
      name: docs-agent-endpoint
      config:
        served_entities:
          - entity_name: prod.agents.docs_agent
            entity_version: "3"
            scale_to_zero_enabled: false
  apps:
    docs_agent_app:
      name: docs-agent-app
      source_code_path: ./app
      resources:
        - name: agent-endpoint
          serving_endpoint:
            name: docs-agent-endpoint
            permission: CAN_QUERY
```

### CI/CD patterns for GenAI

The recommended MLOps pattern on Databricks is **deploy code, not models**: the agent/training *code*
moves dev → staging → prod, and the model is (re)built in each environment. Environments map to
**Unity Catalog catalogs** (`dev.*`, `staging.*`, `prod.*`), and promotion flips **model aliases** (and
prompt aliases) rather than moving artifacts around.

A GitHub Actions / Azure DevOps pipeline typically runs `databricks bundle validate` then
`databricks bundle deploy -t <env>`, authenticating via **workload identity federation (OIDC)** so no
long-lived secret is stored (`DATABRICKS_AUTH_TYPE=github-oidc`, `DATABRICKS_HOST`, `DATABRICKS_CLIENT_ID`).

**MLflow 3 Deployment Jobs** (Public Preview) provide the automated promotion gate. A Lakeflow Job is
triggered on every new UC model version and runs three standard tasks: **evaluation → approval →
deployment**:

- The **evaluation** task runs `mlflow.genai.evaluate` (with scorers like relevance and safety);
  results surface on the model-version page.
- The **approval** task is human-in-the-loop via **UC governed tags** — a task whose name starts with
  `approval` passes only when a UC tag is applied. **The first run always fails on the approval task**
  (no tag yet); clicking **Approve** in the UI applies the tag and resumes the run. Governed tags can
  enforce a two-person rule (the model owner can't self-approve).
- Required job parameters `model_name` and `model_version` must be set at the **job level**, not the
  task level — a common misconfiguration. The job runs as the **model owner's** identity, so use a
  minimal-permission service principal.

### Databricks Apps — the user-facing chat UI

Section 03 introduced Databricks Apps as the hosting layer; here's how you actually deploy one. An app
is defined by three files:

- **`app.py`** — the application code and UI entry point
- **`app.yaml`** — the run command, environment variables, and config
- **`requirements.txt`** — Python dependencies

```yaml
# app.yaml
command:
  - streamlit
  - run
  - app.py
env:
  - name: MODEL_ENDPOINT
    value: docs-agent-endpoint
```

Supported frameworks include **Streamlit, Dash, Gradio, Flask, FastAPI** (Python) and React/Angular/
Svelte/Express (Node). Databricks ships **agent app templates** (github.com/databricks/app-templates)
with a **built-in chat UI, no setup required** — streaming output, tool-call rendering, Databricks
auth, and optional persistent chat history. The recommended template is `e2e-chatbot-app-next`. These
templates work with any current-schema agent: agents on Apps, agents on Model Serving, Knowledge
Assistant, Supervisor Agent, and foundation-model chat endpoints.

Deploy via the CLI or as a bundle resource (shown above). Authentication: by default the app runs as
its **service principal**; **user authorization** (on-behalf-of-user, Public Preview) forwards the
end-user's token and is required by Supervisor Agent endpoints and endpoints with API scopes.

## Deep Dive / Advanced Topics

### The prompt caching subtlety

When you load a prompt by URI, MLflow caches it. **Version-based URIs** (`prompts:/name/2`) are cached
with **infinite TTL** — they're immutable, so caching forever is safe. **Alias-based URIs**
(`prompts:/name@production`) are cached with a **60-second TTL** by default, because an alias can be
repointed at any time and the app must eventually pick up the change. This is why flipping a prompt
alias in production takes effect within ~a minute without a redeploy — and why, if you need instant
propagation in a test, you might tune `MLFLOW_ALIAS_PROMPT_CACHE_TTL_SECONDS`.

Also note: the prompt **template is immutable**, but the associated **`model_config`** (temperature,
max_tokens, etc.) is **mutable** and version-specific. So you can tweak generation parameters on an
existing prompt version without creating a new one — a distinction the exam may probe.

### The Databricks Apps CI/CD gotcha

A deploy that "succeeds" in CI can still leave the app serving stale or broken code. The reason:
`databricks bundle deploy` **uploads code but does not restart the app**. You must run
`databricks bundle run <app_key>` afterward, then **poll `databricks apps get` until
`app_status.state == RUNNING`**. If you skip the poll, CI reports green while the app might be failing
on startup. Also, app names are **workspace-unique** and the app URL is **immutable** — if an app of
that name already exists, use `databricks bundle deployment bind` to adopt it rather than colliding.

One more auth subtlety: after changing user-authorization **scopes** on an app, you must **clear
browser cookies** for the app URL, because the browser session reuses the old token and won't pick up
the new scopes until the session is refreshed.

### Deploy-code vs deploy-model, and why code wins

- **Deploy code (recommended):** the training/agent code is promoted through environments and the model
  is rebuilt in each. Safer automated retraining (the *code* is reviewed and tested), and it works even
  when production data is access-restricted from lower environments.
- **Deploy model:** the built model artifact is promoted. Use only when training is prohibitively
  expensive to reproduce or you're single-workspace with no CI/CD. Downsides: automated retraining is
  awkward and supporting pipelines must be deployed separately.

For GenAI agents (which have no expensive training step), deploy-code is almost always right — you're
promoting agent code, prompt versions, and config, then rebuilding/re-registering the model per env.

## Worked Examples & Practice

### End-to-end: version a prompt, promote via bundle, surface in an app

Tie the whole section together. The `prod.agents.docs_agent` is registered and deployed; now version
its system prompt, promote a change through CI/CD, and expose it in a chat app.

```python
import mlflow

# 1. Register v2 of the system prompt after improving it
prompt = mlflow.genai.register_prompt(
    name="prod.agents.support_system_prompt",
    template="You are a Databricks documentation expert. Answer ONLY from the provided context; "
             "if the context is insufficient, say so.\nContext: {{context}}\nQuestion: {{question}}",
    commit_message="Add explicit refusal instruction to reduce hallucination",
)

# 2. Test v2 by pointing the 'staging' alias at it (production still on v1)
mlflow.genai.set_prompt_alias("prod.agents.support_system_prompt", "staging", prompt.version)

# 3. After evaluation passes in staging, promote: flip 'production' to v2 (no redeploy!)
mlflow.genai.set_prompt_alias("prod.agents.support_system_prompt", "production", prompt.version)
```

The agent loads `prompts:/prod.agents.support_system_prompt@production`, so within ~60s of flipping the
alias the live agent uses the new prompt — no code change, no redeploy. Rolling back is just pointing
`production` back at v1.

The bundle and app that serve it:

```bash
# CI pipeline (GitHub Actions with OIDC auth)
databricks bundle validate -t prod
databricks bundle deploy   -t prod        # uploads code + resources; does NOT restart the app
databricks bundle run docs_agent_app       # actually (re)starts the app
databricks apps get docs-agent-app --output json   # poll until state == RUNNING
```

**Failure mode to observe:** run `bundle deploy` in CI, see it pass, and mark the release done —
skipping `bundle run` and the status poll. The app keeps serving the *previous* code (deploy uploaded
new files but didn't restart it), or it silently failed on startup while CI showed green. Always
`bundle run` and poll for `RUNNING` before declaring an app release successful.

## Common Pitfalls & Misconceptions

- **Pitfall:** Editing a prompt version in place to change wording → **Why it happens:** it feels like editing a file → **Fix:** prompt versions are immutable. Register a **new version** and move the alias. (The `model_config` on a version *is* mutable, but the template text is not.)

- **Pitfall:** Using single-brace `{variable}` placeholders in a Prompt Registry template → **Why it happens:** LangChain/f-strings use single braces → **Fix:** MLflow Prompt Registry uses **double braces** `{{variable}}`. Use `to_single_brace_format()` when handing the prompt to LangChain.

- **Pitfall:** Answering that "Databricks Asset Bundles" and "Declarative Automation Bundles" are different tools → **Why it happens:** the rename → **Fix:** they're the same thing; the CLI (`databricks bundle`) and config file (`databricks.yml`) are unchanged.

- **Pitfall:** Treating `databricks bundle deploy` as sufficient to update a running app → **Why it happens:** "deploy" sounds complete → **Fix:** deploy uploads code but does not restart the app. Follow with `databricks bundle run <app_key>` and poll `databricks apps get` for `state == RUNNING`.

- **Pitfall:** Expecting the deployment-job approval task to pass on the first run → **Why it happens:** it looks like a normal task → **Fix:** the approval task **always fails the first run** because no approval tag exists yet. Click **Approve** in the UI to apply the UC governed tag, which resumes the run. Also set `model_name`/`model_version` at the **job** level, not the task level.

## Key Definitions

| Term | Definition |
|---|---|
| MLflow Prompt Registry | A Git-inspired, UC-backed versioned repository for prompt templates (`mlflow.genai.register_prompt` / `load_prompt`) |
| Prompt version | An immutable, auto-incrementing snapshot of a prompt template |
| Prompt alias | A mutable named pointer (`@production`, `@staging`) to a prompt version, enabling promotion/rollback without code changes |
| Declarative Automation Bundle (DAB) | Infrastructure-as-code for Databricks (formerly Databricks Asset Bundles); a `databricks.yml` describing resources deployed via the `databricks bundle` CLI |
| Target | A named environment (dev/staging/prod) in a bundle, each with its own workspace host and mode |
| Deploy-code | The recommended MLOps pattern: promote code through environments and rebuild the model per environment (vs promoting the model artifact) |
| Deployment job | An MLflow 3 Lakeflow Job triggered on new UC model versions, running evaluation → approval (via UC governed tags) → deployment |
| Databricks Apps | The Databricks serverless hosting platform for user-facing apps (chat UIs), defined by `app.py`, `app.yaml`, `requirements.txt` |
| User authorization (OBO) | An Apps auth mode that forwards the end-user's token so the app acts with the user's permissions |

## Summary / Quick Recall

- **Prompt Registry:** UC-backed, versioned prompts; **versions immutable**, **aliases mutable**; use **double-brace** `{{var}}` (convert with `to_single_brace_format()` for LangChain)
- Flipping a prompt **alias** promotes/rolls back **without a redeploy** (alias URIs cache 60s; version URIs cache forever)
- **DABs = Declarative Automation Bundles** (formerly Asset Bundles); same `databricks.yml` + `databricks bundle` CLI; **targets** = environments
- CLI: `bundle init` → `validate` → `deploy -t <env>` → `run`
- Recommended MLOps = **deploy code**; environments = **UC catalogs**; promotion flips **model/prompt aliases**; CI auth via **OIDC** (no stored secret)
- **Deployment jobs**: evaluation → approval → deployment; **first approval run always fails** (apply UC tag to resume); `model_name`/`model_version` set at **job** level; run as a minimal-permission SP
- **Databricks Apps:** `app.py` + `app.yaml` + `requirements.txt`; built-in chat UI templates; bind a `serving_endpoint` resource
- **Apps CI/CD gotcha:** `bundle deploy` uploads but does **not restart** — you must `bundle run` and poll for `RUNNING`; app URLs are immutable

## Self-Check Questions

1. You improved a production prompt's wording. A teammate suggests editing the existing version directly. Why is that impossible, and what's the correct workflow to get the new wording live and reversible?

<details>
<summary>Answer</summary>
Prompt versions are immutable, so you cannot edit one in place. Register a **new version** with `mlflow.genai.register_prompt(...)`, optionally validate it via the `staging` alias, then flip the `production` alias to the new version with `set_prompt_alias(...)`. Because the agent loads `prompts:/name@production`, the change goes live within ~60 seconds with no redeploy, and rollback is just pointing the alias back at the old version.
</details>

2. An exam question mentions "Declarative Automation Bundles." A colleague insists your team uses "Databricks Asset Bundles," not that. Are these different, and what commands/files are involved either way?

<details>
<summary>Answer</summary>
They are the same thing — Databricks Asset Bundles were renamed to Declarative Automation Bundles. The tooling is unchanged: the config file is `databricks.yml` and the CLI is `databricks bundle` (`init`, `validate`, `deploy`, `run`, `destroy`). Don't be thrown by the name in an exam question.
</details>

3. Describe the recommended dev → staging → prod promotion approach for a GenAI agent on Databricks. What plays the role of "environments," and how is a version promoted?

<details>
<summary>Answer</summary>
Use the **deploy-code** pattern: the agent code, prompt versions, and bundle config are promoted through environments, and the model is rebuilt/re-registered per environment. Environments correspond to **Unity Catalog catalogs** (`dev.*`, `staging.*`, `prod.*`). Promotion flips **model aliases** (and prompt aliases) — e.g., set the `production` alias to the validated version — rather than moving artifacts. An MLflow 3 deployment job can gate promotion with evaluation + human approval.
</details>

4. Your CI pipeline runs `databricks bundle deploy -t prod`, reports success, and you close the release ticket. Users report the chat app still shows old behavior. What did you forget?

<details>
<summary>Answer</summary>
`bundle deploy` uploads code and resource definitions but does **not restart** the Databricks App. You must run `databricks bundle run <app_key>` to (re)start it, then poll `databricks apps get` until `app_status.state == RUNNING`. Without the run + poll, the app keeps serving the previously running code (or may have failed on startup while CI still showed green).
</details>

5. In an MLflow 3 deployment job, the pipeline fails on the "approval" task the very first time it runs, even though nothing is misconfigured. Is this a bug, and how do you proceed?

<details>
<summary>Answer</summary>
It's expected behavior, not a bug. The approval task passes only when a Unity Catalog governed tag (keyed by the task name, value `Approved`) is present, and on the first run no tag exists yet — so it fails. A reviewer clicks **Approve** in the model-version UI, which applies the tag and repairs/resumes the run so it proceeds to deployment. (Also ensure `model_name` and `model_version` are set at the job level, and consider disabling retries on the approval task.)
</details>

## Further Reading

- [MLflow — Prompt Registry (Databricks)](https://docs.databricks.com/aws/en/mlflow3/genai/prompt-version-mgmt/prompt-registry/) — versions, aliases, `{{variable}}` templates, UC backing (updated 2026-06-23)
- [MLflow — Prompt Registry (OSS)](https://mlflow.org/docs/latest/genai/prompt-registry/) — the `mlflow.genai` prompt APIs
- [Databricks — Declarative Automation Bundles](https://docs.databricks.com/aws/en/dev-tools/bundles/) — `databricks.yml`, targets, CLI lifecycle (updated 2026-03-16)
- [Databricks — MLflow deployment jobs](https://docs.databricks.com/aws/en/mlflow/deployment-job) — evaluation/approval/deployment gates (updated 2026-06-25)
- [Databricks — Databricks Apps](https://docs.databricks.com/aws/en/dev-tools/databricks-apps/) — app structure, frameworks, deployment (updated 2026-06-30)
- [Databricks — Chat UI for agents](https://docs.databricks.com/aws/en/agents/agent-framework/chat-app) — built-in chat UI templates (updated 2026-07-02)
