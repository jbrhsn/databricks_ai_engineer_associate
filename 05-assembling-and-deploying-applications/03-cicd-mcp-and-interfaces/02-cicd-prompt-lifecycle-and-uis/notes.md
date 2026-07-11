# CI/CD, Prompt Lifecycle, and User Interfaces on Databricks

**Section:** 05-assembling-and-deploying-applications | **Module:** 03-cicd-mcp-and-interfaces | **Est. time:** 3 hrs | **Exam mapping:** Assembling & Deploying Applications (22%)

---

## TL;DR

Declarative Automation Bundles (DAB, formerly Databricks Asset Bundles) are the infrastructure-as-code mechanism for deploying AI/ML pipelines on Databricks: a single `databricks.yml` file declares every resource, and `databricks bundle deploy -t prod` promotes them atomically across environments. Prompt versioning must be treated like code — stored in Git-tracked YAML or in MLflow's Prompt Registry — so that every model change traces to a specific prompt version. User-facing surfaces range from zero-code (AI Playground for prototyping, Genie Agents for conversational BI) to fully custom chatbot UIs built with Gradio or Streamlit on Databricks Apps, with the `databricks.agents` Review App bridging production traffic and human feedback.

**The one thing to remember: a bundle's `targets:` block is what separates dev from staging from prod — every environment difference lives there, not scattered across notebooks or environment variables.**

---

## ELI5 — Explain It Like I'm 5

Imagine you are writing a recipe book and your kitchen has three rooms: a test kitchen (dev), a tasting room (staging), and a restaurant (prod). You write the recipe once in one master notebook. The notebook says "in the test kitchen, use cheap cheese; in the restaurant, use the expensive kind." When you hand the notebook to a sous chef and say "cook this in the restaurant," they follow the restaurant instructions automatically. Databricks Asset Bundles work exactly the same way: `databricks.yml` is the master recipe, the `targets:` section is the room-by-room instruction set, and `databricks bundle deploy -t prod` is the order to the sous chef. The most common misconception is that you need separate files for dev and prod — you do not. One file, one deploy command, different targets.

---

## Learning Objectives

By the end of this chapter you will be able to:

- [ ] Construct a `databricks.yml` bundle file with dev, staging, and prod targets and deploy it using the Databricks CLI
- [ ] Compare Git-tracked YAML prompt versioning against MLflow Prompt Registry and select the right approach given a governance constraint
- [ ] Trace a prompt through its lifecycle stages (draft → review → production) and identify the artifact produced at each stage
- [ ] Distinguish the use cases for AI Playground, Genie Agents, and a custom Gradio/Streamlit app on Databricks Apps
- [ ] Explain how `agents.deploy()` provisions the Review App and how human feedback flows back to MLflow traces

---

## Visual Overview

### Bundle Deployment Pipeline

```
 Developer Workstation
 ┌─────────────────────────────────────────────────┐
 │  databricks.yml                                 │
 │  ├── bundle: name: my-llm-app                   │
 │  ├── resources: (jobs, endpoints, experiments)  │
 │  └── targets:                                   │
 │       ├── dev   (default: true)                 │
 │       ├── staging                               │
 │       └── prod                                  │
 └───────────────────┬─────────────────────────────┘
                     │
         databricks bundle validate
                     │
         databricks bundle deploy -t prod
                     │
                     ▼
 Databricks Workspace (prod)
 ┌─────────────────────────────────────────────────┐
 │  Lakeflow Job  ──► Model Serving Endpoint       │
 │  MLflow Experiment  ──► Unity Catalog Model     │
 └─────────────────────────────────────────────────┘
```

### Prompt Lifecycle: Draft → Production

```
 Git Repo / MLflow Prompt Registry
 ┌───────────────────────────────────────────────────────┐
 │                                                       │
 │  draft (local YAML / unregistered version)            │
 │      │                                                │
 │      ▼                                                │
 │  review (PR / MLflow version with "Candidate" alias)  │
 │      │                                                │
 │      ▼                                                │
 │  production (merged to main / "Production" alias)     │
 │      │                                                │
 │      └──► referenced by model serving endpoint        │
 └───────────────────────────────────────────────────────┘
```

### User-Facing Surfaces Comparison

```
 Use case
 │
 ├── Prototype / explore models ──► AI Playground (UI, no code)
 │
 ├── Business users ask data questions ──► Genie Agents (conversational BI)
 │
 ├── Internal stakeholders test agent ──► Review App (databricks.agents)
 │
 └── Custom chatbot for end users ──► Gradio / Streamlit on Databricks Apps
```

### CI/CD Flow with Bundle + Review App

```
 Code Push (Git)
      │
      ▼
 CI: databricks bundle validate
      │
      ▼
 CI: databricks bundle deploy -t staging
      │
      ▼
 Human feedback via Review App ──► MLflow traces
      │  (labeling sessions, Assessment objects)
      ▼
 CD: databricks bundle deploy -t prod
      │
      ▼
 Model Serving Endpoint (production traffic)
      │
      └──► Inference Tables + MLflow real-time tracing
```

---

## Key Concepts

### Declarative Automation Bundles (DAB)

**What is it?** Declarative Automation Bundles (formerly Databricks Asset Bundles) are an infrastructure-as-code approach to managing Databricks projects. A bundle is an end-to-end project definition: source files, resource declarations, and environment targets all expressed in YAML alongside the application code.

**How does it work under the hood?** The Databricks CLI reads the `databricks.yml` at the project root and merges top-level resource definitions with target-specific overrides. When you run `databricks bundle deploy`, the CLI serializes the merged configuration and creates or updates every declared resource (jobs, pipelines, model serving endpoints, MLflow experiments, registered models) in the target workspace via the Databricks REST API. State is tracked so subsequent deploys produce minimal diffs rather than recreating resources from scratch. The `databricks bundle run` command then triggers a specific workflow resource by its key in the YAML.

**Where does it appear in Databricks?** The bundle manifest must be named `databricks.yml` (not `bundle.yaml`) at the project root. Resources created by a bundle are labelled with the bundle name and target in the workspace UI. CI/CD pipelines (e.g., GitHub Actions) invoke `databricks bundle validate && databricks bundle deploy -t prod` as pipeline steps. The CLI version must be ≥ 0.218.0.

### `databricks.yml` Structure

**What is it?** The single required configuration file for a bundle. It contains four primary top-level blocks: `bundle` (name and metadata), `resources` (every Databricks object to manage), `targets` (per-environment overrides), and optionally `workspace`, `variables`, `artifacts`, and `include`.

**How does it work under the hood?** YAML merging follows a layered override model: values in a named `targets:` block override the corresponding values in the top-level `resources:` or `workspace:` blocks for that deployment. This means you declare a resource once and override only what differs per environment (cluster ID, workspace host, permissions). The `default: true` flag on a target makes it the implicit target for commands that omit `-t`.

**Where does it appear in Databricks?** The file lives at the repository root alongside application code. The `include:` mapping lets you split large bundles into multiple YAML files. The `databricks bundle schema` command outputs a JSON schema usable for IDE autocompletion validation.

### Prompt Versioning Strategies

**What is it?** Prompt versioning is the practice of tracking the exact text and parameters of every system prompt (and few-shot examples) as versioned, auditable artifacts — analogous to code commits. Two primary strategies exist: Git-tracked YAML files and the MLflow Prompt Registry.

**How does it work under the hood?** With **Git-tracked YAML**, prompt templates are stored as `.yaml` files in the repository. The model code reads the prompt at load time from the file path; the Git commit SHA doubles as the version identifier. With the **MLflow Prompt Registry**, prompts are stored as versioned MLflow objects using `mlflow.register_prompt()` and retrieved by name and alias (e.g., `"Production"`). The registry stores each version as an immutable artifact with metadata, enabling A/B comparison and staged promotion. The lifecycle aliases `"Candidate"`, `"Champion"`, and `"Production"` mirror the model registry promotion pattern.

**Where does it appear in Databricks?** Git-tracked YAML is visible in the repository and referenced in notebooks or Python modules. The MLflow Prompt Registry appears under the MLflow Experiments or Models section of the Databricks workspace UI. Both strategies integrate with bundle deployments: YAML files are synced with `databricks bundle deploy`; registry entries are created programmatically and can be referenced by alias in serving endpoint configuration.

### Prompt Lifecycle Stages

**What is it?** A formalized three-stage promotion path — draft, review, production — that mirrors software release management and ensures no unapproved prompt reaches a production endpoint.

**How does it work under the hood?** In the **draft** stage, the prompt exists locally or as an unaliased MLflow version; it is iterable by the engineer and not yet tied to any serving endpoint. In the **review** stage, the prompt is submitted for human evaluation — either as a pull request (Git strategy) or by assigning the `"Candidate"` alias (registry strategy) and routing test traffic through the Review App or an offline evaluation run. In the **production** stage, the prompt is merged to main (Git) or assigned the `"Production"` alias (registry) and the serving endpoint is redeployed to reference the new version via bundle redeploy.

**Where does it appear in Databricks?** The Review App (provisioned automatically by `agents.deploy()`) is the primary review surface. MLflow experiment traces record which prompt version produced each response, enabling direct comparison across versions. The bundle YAML references the prompt version to deploy (by file path or registry alias) ensuring the environment and prompt are promoted atomically.

### Databricks Genie Agents

> ⚠️ **Fast-evolving:** Genie Spaces was renamed to Genie Agents as of July 2026 and is part of the broader "Genie" family (also including Genie One and Genie Code). Verify current feature names and capabilities against the [official docs](https://docs.databricks.com/en/genie/index.html) before authoring or deploying.

**What is it?** A Genie Agent is a domain-specific natural-language chat interface that lets business users ask questions of structured data and receive SQL queries, result tables, and visualizations in return. Authors curate each agent with Unity Catalog datasets, example SQL queries, business-semantics SQL expressions, and text instructions.

**How does it work under the hood?** When a user submits a question, the Genie Agent translates it to SQL using an LLM grounded by the curated instructions and example queries. It executes the SQL against the Unity Catalog tables through a SQL warehouse, formats the result, and optionally renders a chart. The Conversation API allows external applications to integrate Genie Agent conversations programmatically.

**Where does it appear in Databricks?** Genie Agents are created and managed in the Databricks workspace under the Genie section. They can be embedded in external apps or accessed via the Conversation API. The Genie One surface provides a unified chat interface across data, dashboards, and apps for business users.

### AI Playground

**What is it?** A zero-code chat environment in the Databricks workspace for interactively testing, prompting, and comparing LLMs, and for prototyping tool-calling agents and question-answering bots.

**How does it work under the hood?** AI Playground routes requests to Foundation Model API endpoints (PAYG or provisioned throughput). You can add up to several endpoints side-by-side using the `+` button for direct response comparison. System prompts and tool definitions can be configured in the UI without writing code. Prototype agents built in Playground can be exported to a notebook to begin productionization.

**Where does it appear in Databricks?** Accessible under **AI/ML → Playground** in the left navigation pane. Requires a workspace region that supports Foundation Models. No installation required; the workspace account team can enable it if absent.

### Chatbot UIs on Databricks Apps

**What is it?** Databricks Apps is a serverless hosting platform for Python and Node.js applications deployed directly within the Databricks platform, inheriting Unity Catalog governance and OAuth authentication. Gradio and Streamlit are the primary Python frameworks for building chatbot UIs.

**How does it work under the hood?** The developer writes a Python app (e.g., a Streamlit or Gradio app) locally, defines the runtime in `app.yaml`, and deploys via `databricks apps deploy` or as part of a bundle. The app runs on Databricks' serverless compute and is billed per hour of compute time. The app inherits the deploying user's OAuth credentials for authenticating calls to Model Serving, Unity Catalog, and SQL warehouses. The framework process is the entry point defined in `app.yaml`'s `command` field.

**Where does it appear in Databricks?** Apps appear under the **Apps** section of the workspace. The app URL is workspace-scoped. Databricks recommends deploying agents on Apps (rather than only Model Serving) for full control over agent code, server configuration, and deployment workflow.

### Review App and Human Feedback

**What is it?** The Review App is a web interface automatically provisioned by `agents.deploy()` that allows stakeholders to interact with a deployed agent and provide structured feedback — thumbs up/down, free-text comments, or expected-answer labels — directly against production or staging traces.

**How does it work under the hood?** `agents.deploy()` creates a Model Serving endpoint and simultaneously provisions the Review App as part of the same deployment. User interactions and feedback are logged as MLflow `Assessment` objects attached to traces in the MLflow experiment. Feedback of type `"expectation"` (correct answers) can be directly converted into evaluation datasets for `mlflow.genai.evaluate()`. The Review App can be customized by deploying an open-source template as a Databricks App.

**Where does it appear in Databricks?** The Review App URL is returned by `agents.deploy()` and printed to the notebook output. It is visible on the serving endpoint's detail page in the workspace. Domain experts need only account-level provisioning and `CAN_EDIT` on the MLflow experiment — no workspace access required.

---

## Key Parameters / Configuration Knobs

| Parameter | What it controls | Decision rule |
|---|---|---|
| `targets.<name>.default: true` | Which target is used when no `-t` flag is passed | Set to `true` on `dev` only; force explicit `-t staging` and `-t prod` to prevent accidental production deploys |
| `targets.<name>.workspace.host` | The Databricks workspace URL for that target | Always set explicitly for staging and prod; omit only for dev where the CLI's `DEFAULT` profile applies |
| `targets.<name>.mode` | Sets `development` mode (adds dev prefix to resource names, relaxes permissions) | Set `mode: development` on dev targets to avoid name collisions with production resources |
| `bundle.databricks_cli_version` | Minimum CLI version required to deploy this bundle | Pin to the CLI version used in CI to catch version drift early |
| `variables.<name>.default` | Default value for a bundle variable used across resources | Use variables for cluster IDs, model names, and warehouse IDs that differ by target rather than hardcoding them in each target block |
| `resources.model_serving_endpoints.<name>.config.served_entities[].scale_to_zero_enabled` | Whether the endpoint scales to zero when idle | Set `true` for staging to save cost; set `false` for prod to eliminate cold-start latency |
| `agents.deploy()` `scale_to_zero_enabled` | Python API counterpart for endpoint scale-to-zero | Same rule as above; pass `scale_to_zero_enabled=False` for production workloads requiring low latency |

---

## Worked Example: Requirement → Decision

**Given:** A team has built a LangGraph-based RAG agent. The system prompt and retrieval instruction are stored in `prompts/rag_system.yaml` in the repo. They need to deploy the agent to dev for testing, staging for stakeholder review via the Review App, and prod for external users — with zero-downtime upgrades and full audit trail of which prompt version served each response.

**Step 1 — Identify the goal:** Promote a LangGraph agent with a versioned prompt through three environments atomically, capturing human feedback before prod promotion.

**Step 2 — Define inputs:** `prompts/rag_system.yaml` (Git-tracked prompt), `agent.py` (LangGraph graph definition), `databricks.yml` (bundle), Unity Catalog model registry entry.

**Step 3 — Define outputs:** Three Databricks Model Serving endpoints (one per environment), MLflow experiment traces per environment, Review App URL for staging, prod endpoint URL for external traffic.

**Step 4 — Apply constraints:** Prod must have no cold-start latency (`scale_to_zero_enabled: false`). Staging stakeholders need Review App access without workspace permissions. The prompt YAML must be pinned to a Git SHA so every MLflow trace references an auditable version. Rollback must be achievable within one `bundle deploy` command.

**Step 5 — Select the approach:** Use DAB with three targets (`dev`, `staging`, `prod`). The `bundle deploy -t staging` step runs in CI after unit tests pass; a human-approval gate checks Review App feedback before the CD pipeline triggers `bundle deploy -t prod`. The prompt is loaded from `prompts/rag_system.yaml` at agent initialization; the Git SHA is logged as an MLflow tag (`mlflow.set_tag("prompt_sha", git_sha)`) so every trace is traceable. Git-tracked YAML is preferred over the MLflow Prompt Registry here because the prompt and agent code change together in the same PR, making atomic review natural. The MLflow Prompt Registry would be the better choice if the prompt team and the engineering team operate on independent release cadences.

---

## Implementation

```yaml
# Scenario: Deploy a LangGraph RAG agent across dev/staging/prod with
# per-environment endpoint configuration and no accidental prod deploys.

bundle:
  name: langgraph-rag-agent
  databricks_cli_version: ">=0.218.0"

variables:
  model_name:
    description: "UC model name for the registered agent"
    default: "main.ml.rag_agent"

resources:
  registered_models:
    rag_agent_model:
      name: ${var.model_name}
      catalog_name: main
      schema_name: ml
      comment: "LangGraph RAG agent — managed by DAB"

  model_serving_endpoints:
    rag_agent_endpoint:
      name: "rag-agent-${bundle.target}"
      config:
        served_entities:
          - entity_name: ${var.model_name}
            entity_version: "1"
            workload_size: "Small"
            scale_to_zero_enabled: true  # overridden per target below

targets:
  dev:
    default: true
    mode: development
    workspace:
      host: https://dev-workspace.azuredatabricks.net

  staging:
    workspace:
      host: https://staging-workspace.azuredatabricks.net

  prod:
    workspace:
      host: https://prod-workspace.azuredatabricks.net
    resources:
      model_serving_endpoints:
        rag_agent_endpoint:
          config:
            served_entities:
              - entity_name: ${var.model_name}
                entity_version: "1"
                workload_size: "Medium"
                scale_to_zero_enabled: false  # no cold-start in prod
```

```python
# Scenario: Register a LangGraph RAG agent to Unity Catalog, deploy it with
# the Review App enabled for staging stakeholder feedback, and log the
# Git-tracked prompt SHA as an MLflow tag on every trace.

import mlflow
import subprocess
from databricks import agents
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

# ── Prompt loading (Git-tracked YAML strategy) ──────────────────────────────
import yaml, pathlib

PROMPT_PATH = pathlib.Path("prompts/rag_system.yaml")

def load_prompt() -> str:
    with open(PROMPT_PATH) as f:
        return yaml.safe_load(f)["system_prompt"]

def get_git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()

SYSTEM_PROMPT = load_prompt()
PROMPT_SHA = get_git_sha()

# ── LangGraph agent definition ───────────────────────────────────────────────
class AgentState(TypedDict):
    messages: List[dict]
    retrieved_docs: List[str]

def retrieve(state: AgentState) -> AgentState:
    # Scenario: call Vector Search index for top-k documents
    query = state["messages"][-1]["content"]
    # ... (Vector Search call omitted for brevity)
    state["retrieved_docs"] = ["doc1 content", "doc2 content"]
    return state

def generate(state: AgentState) -> AgentState:
    context = "\n".join(state["retrieved_docs"])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\n{state['messages'][-1]['content']}"},
    ]
    # ... (LLM call omitted for brevity)
    state["messages"].append({"role": "assistant", "content": "answer"})
    return state

graph = StateGraph(AgentState)
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)
graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)
agent = graph.compile()

# ── Log and register to Unity Catalog ───────────────────────────────────────
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Shared/rag-agent-staging")

with mlflow.start_run() as run:
    mlflow.set_tag("prompt_sha", PROMPT_SHA)
    mlflow.set_tag("prompt_path", str(PROMPT_PATH))
    # Log the agent (using langchain flavor as proxy for LangGraph)
    mlflow.langchain.log_model(
        agent,
        artifact_path="agent",
        registered_model_name="main.ml.rag_agent",
    )
    model_version = mlflow.register_model(
        f"runs:/{run.info.run_id}/agent",
        "main.ml.rag_agent"
    )

# ── Deploy with Review App ────────────────────────────────────────────────────
deployment = agents.deploy(
    model_name="main.ml.rag_agent",
    model_version=model_version.version,
    scale_to_zero_enabled=True,  # staging: ok to scale to zero
)
print(f"Review App URL: {deployment.review_app_url}")
print(f"Query endpoint: {deployment.query_endpoint}")
```

```python
# Anti-pattern: Hardcoding the system prompt string directly in the agent
# module — no version tracking, no audit trail, prompt and code diverge silently.

def generate(state):
    messages = [
        # BAD: prompt is an anonymous string literal with no version identifier
        {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
        {"role": "user", "content": state["messages"][-1]["content"]},
    ]
    ...

# What breaks: when the prompt is updated, there is no record of which agent
# responses were generated by which prompt version. MLflow traces cannot be
# correlated to a specific prompt, making evaluation, rollback, and compliance
# audits impossible. A bug introduced by a prompt change is undetectable.

# Correct approach: load from a versioned file or registry and tag every run.
SYSTEM_PROMPT = load_prompt()  # from prompts/rag_system.yaml at known Git SHA

def generate(state):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": state["messages"][-1]["content"]},
    ]
    ...
# Tag the MLflow run: mlflow.set_tag("prompt_sha", PROMPT_SHA)
```

---

## Common Pitfalls & Misconceptions

- **Treating `databricks bundle deploy` as idempotent with no side effects** — Beginners assume re-running deploy on prod is always safe. In reality, if resource keys are renamed in the YAML, the old resource is destroyed and a new one created, causing downtime. The correct mental model: treat YAML resource keys as permanent identifiers — rename them only with a planned migration, just as you would rename a database table.

- **Storing prompts only in code comments or notebook markdown** — Engineers new to LLM ops treat prompts as documentation rather than artifacts. This makes it impossible to reconstruct which prompt produced a given response weeks later. The correct mental model: prompts are first-class versioned artifacts; every production trace must be traceable to a specific prompt version via Git SHA tag or MLflow registry alias.

- **Conflating AI Playground with a production UI** — The Playground is a workspace-internal prototyping tool that requires Databricks credentials. It is not accessible to external users or embeddable in a customer-facing app. The correct mental model: Playground is for engineers to prototype and compare; Databricks Apps (Gradio/Streamlit) is for deploying chatbot UIs to end users.

- **Assuming Genie Agents replace custom LLM agents** — Because both involve natural language interaction, beginners conflate them. Genie Agents are specialized for structured data (SQL/BI) and curated datasets in Unity Catalog; they do not support arbitrary tool use or retrieval-augmented generation over unstructured documents. The correct mental model: Genie Agents = conversational BI for business users; custom LangGraph agents = flexible AI applications for any use case.

- **Using `mode: development` in the prod target** — Development mode adds a user-name prefix to resource names, which prevents name collisions in dev but breaks production resource references and permissions. The correct mental model: `mode: development` belongs only in the `dev` target; staging and prod targets should never set it.

- **Expecting the Review App to work for external users without Databricks accounts** — Domain experts using the Review App need to be provisioned at the Databricks account level (even without workspace access), but external end users without any Databricks account cannot use it. The correct mental model: Review App is for internal domain experts and stakeholders; it is not a customer-facing feedback widget.

---

## Key Definitions

| Term | Definition |
|---|---|
| Declarative Automation Bundle (DAB) | An IaC project for Databricks that declares all resources and environments in `databricks.yml` and is deployed atomically via the Databricks CLI |
| `databricks.yml` | The mandatory root configuration file for a DAB bundle; defines bundle name, resources, targets, variables, and workspace settings |
| Target | A named deployment environment within a bundle (e.g., `dev`, `staging`, `prod`) whose workspace host and resource overrides are declared in the `targets:` block |
| Prompt versioning | The practice of tracking prompt text as a versioned artifact — either as a Git-tracked YAML file (version = Git SHA) or an MLflow Prompt Registry entry (version = integer with aliases) |
| MLflow Prompt Registry | An MLflow service for storing, versioning, and promoting prompt templates with lifecycle aliases (`Candidate`, `Production`) analogous to the model registry |
| Prompt lifecycle | The three-stage promotion path for a prompt: draft (local/unaliased) → review (PR or `Candidate` alias) → production (merged or `Production` alias) |
| Genie Agent | A Databricks domain-specific conversational BI interface (formerly Genie Spaces) that translates natural-language questions to SQL against curated Unity Catalog datasets |
| AI Playground | A workspace UI for zero-code interactive testing, comparison, and prototyping of LLMs and tool-calling agents; not accessible to external users |
| Databricks Apps | A serverless hosting platform for Python and Node.js apps deployed within Databricks, inheriting Unity Catalog and OAuth; supports Gradio, Streamlit, Dash, React |
| Review App | A web interface automatically provisioned by `agents.deploy()` for collecting structured human feedback (assessments) on MLflow traces from a deployed agent |
| `agents.deploy()` | The Python API function in `databricks.agents` that creates a Model Serving endpoint, provisions the Review App, enables real-time tracing, and configures inference tables in one call |
| Assessment | An MLflow entity that stores a domain expert's feedback label (feedback or expectation type) attached to a specific trace within a labeling session |

---

## Summary / Quick Recall

- **One `databricks.yml`, multiple targets** — all environment differences live in the `targets:` block; never create separate YAML files per environment.
- **`databricks bundle deploy -t prod`** promotes all declared resources atomically; resource key renames destroy and recreate, causing downtime — treat keys as permanent.
- **Prompt = versioned artifact** — Git SHA (YAML strategy) or registry alias (`Production`) must be logged as an MLflow tag on every production trace.
- **AI Playground is prototype-only** — requires Databricks credentials; not embeddable for external users; use Databricks Apps for customer-facing chatbots.
- **Genie Agents = conversational BI over structured data** — SQL/Unity Catalog only; not a substitute for LangGraph RAG agents over unstructured documents.
- **`agents.deploy()` bundles endpoint + Review App + tracing** — one call provisions the full feedback loop; domain experts need only account access and `CAN_EDIT` on the experiment, not full workspace access.
- **`mode: development`** adds user-name prefixes to resource names — never use it in staging or prod targets.

---

## Self-Check Questions

1. What is the name of the required root configuration file for a Declarative Automation Bundle, and what top-level key defines per-environment differences?

   <details><summary>Answer</summary>

   The file must be named `databricks.yml` (not `bundle.yaml` or any other name — one and only one per project). Per-environment differences are defined in the `targets:` top-level key, where each named target can override `workspace.host`, `resources`, `variables`, and `permissions`. The most common distractor is `bundle.yaml`, which is not a valid DAB filename and will cause the CLI to fail with a missing configuration error.

   </details>

2. A team discovers that a production agent started giving degraded answers last Tuesday. They need to identify which system prompt was active on that date. They currently load prompts as hardcoded strings in a Python module committed to Git. What must they add to their workflow to make this investigation possible, and what is the minimal change?

   <details><summary>Answer</summary>

   They must log the Git commit SHA as an MLflow tag on every run that produced production traces. The minimal change is: (1) move the prompt string to a YAML file tracked in Git, (2) call `subprocess.check_output(["git", "rev-parse", "HEAD"])` at agent initialization, and (3) call `mlflow.set_tag("prompt_sha", sha)` inside the MLflow run context before inference. With this in place, every MLflow trace has a `prompt_sha` tag, and the team can `git show <sha>` to reconstruct the exact prompt active on any date. Without this change, they can look at the Git log for the Python module, but if the prompt string is buried in code and the module had other changes on the same commit, attribution is ambiguous.

   </details>

3. **Which TWO** of the following statements correctly describe the Review App provisioned by `agents.deploy()`?

   - A. It is accessible to anyone with the URL, without a Databricks account.
   - B. Domain experts need only Databricks account access and `CAN_EDIT` on the MLflow experiment — they do not need workspace access.
   - C. Feedback collected is stored as `Assessment` objects attached to MLflow traces.
   - D. The Review App replaces the Model Serving endpoint for production traffic.
   - E. It requires MLflow 1.x compatibility mode to function.

   <details><summary>Answer</summary>

   **B and C** are correct.

   **B** is correct: the Review App is designed so that domain experts (e.g., legal reviewers, subject-matter experts) can give feedback without needing a full Databricks workspace seat. They need account-level provisioning and experiment `CAN_EDIT` permission.

   **C** is correct: feedback is stored as `Assessment` MLflow entities (type `"feedback"` or `"expectation"`) attached to traces in a labeling session, and can be retrieved programmatically via `mlflow.search_traces()`.

   **A** is wrong: the Review App requires Databricks account authentication — it is not a publicly accessible URL.

   **D** is wrong: the Review App is a separate web interface for human feedback; it does not replace the serving endpoint, which continues to handle API traffic.

   **E** is wrong: the Review App requires MLflow 3 (specifically MLflow ≥ 3.1.3 for `agents.deploy()` with full functionality); MLflow 1.x compatibility mode is irrelevant.

   </details>

4. A team uses a DAB bundle with `mode: development` on their dev target. They want to promote the same bundle to a prod target. A colleague suggests simply changing `default: true` from dev to prod in the YAML and running `databricks bundle deploy`. What is the primary risk of this approach?

   <details><summary>Answer</summary>

   The primary risk is that `mode: development` (if accidentally left on the prod target, or if the resource key names are the same as the dev deployment's prefixed names) creates resources with user-name-prefixed names that differ from what production consumers expect. More critically, if the colleague's suggestion means removing `default: true` from dev without adding a proper `workspace.host` to the prod target, the CLI falls back to the DEFAULT profile, which likely points at dev — silently deploying "prod" to the dev workspace. The correct approach is to define prod as a named target with an explicit `workspace.host`, never set `mode: development` on prod, and always pass `-t prod` explicitly to prevent accidental deployments.

   </details>

5. Your organization requires that every prompt change go through a formal approval process before reaching production users. You are choosing between Git-tracked YAML and the MLflow Prompt Registry. Under which specific constraint is the MLflow Prompt Registry the strictly superior choice, and why?

   <details><summary>Answer</summary>

   The MLflow Prompt Registry is strictly superior when the **prompt team and the engineering/serving team operate on independent release cadences** — that is, when non-engineers (e.g., product managers, legal reviewers, or domain experts) need to update prompt text without touching application code, or when prompt changes must go through a separate approval workflow that does not align with the Git PR cycle. In the registry, a domain expert can update a prompt version and a reviewer can promote it from `"Candidate"` to `"Production"` alias via the MLflow UI without a code deploy. The serving endpoint references the alias, so the change takes effect at next load without a bundle redeploy.

   By contrast, Git-tracked YAML is superior when prompt and code changes are tightly coupled (same PR, same review), because the atomic commit guarantees that the code and prompt are always in sync. Choosing the registry when the teams are co-located and work in the same repo adds unnecessary complexity without governance benefit.

   </details>

---

## Further Reading

- [What are Declarative Automation Bundles?](https://docs.databricks.com/en/dev-tools/bundles/index.html) — *verified 2026-07-11* — Overview, when to use, and CLI requirements
- [Declarative Automation Bundles configuration](https://docs.databricks.com/en/dev-tools/bundles/settings.html) — *verified 2026-07-11* — Full `databricks.yml` schema with all top-level mappings and examples
- [Develop Declarative Automation Bundles](https://docs.databricks.com/en/dev-tools/bundles/work-tasks.html) — *verified 2026-07-11* — Bundle lifecycle: create, validate, deploy, run, destroy
- [Deploy an agent for generative AI applications](https://docs.databricks.com/en/generative-ai/deploy-agent.html) — *verified 2026-07-11* — `agents.deploy()` API, Review App provisioning, and deployment actions table
- [Collect feedback and expectations by labeling existing traces](https://docs.databricks.com/en/mlflow3/genai/human-feedback/expert-feedback/label-existing-traces.html) — *verified 2026-07-11* — Review App, labeling sessions, Assessment objects, and feedback-to-evaluation-dataset pipeline
- [Genie Agents](https://docs.databricks.com/en/genie/index.html) — *verified 2026-07-11* — Genie Agents overview (formerly Genie Spaces), Conversation API, and agent-mode features
- [Chat with LLMs and prototype generative AI apps using AI Playground](https://docs.databricks.com/en/large-language-models/ai-playground.html) — *verified 2026-07-11* — AI Playground requirements, side-by-side model comparison, and agent prototyping
- [Databricks Apps](https://docs.databricks.com/en/dev-tools/databricks-apps/index.html) — *verified 2026-07-11* — Serverless app hosting, supported frameworks (Streamlit, Gradio, Dash, React), CI/CD integration
