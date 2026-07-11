# LAB-17: CI/CD Pipeline, Prompt Lifecycle, and Chatbot UI for a LangGraph Agent

**Lab:** LAB-17 | **Section:** 05-assembling-and-deploying-applications | **Module:** 03-cicd-mcp-and-interfaces | **Est. time:** 2.5 hrs

---

## Objective

Build a complete CI/CD-ready project for a LangGraph RAG agent: author a `databricks.yml` bundle with dev and staging targets, implement Git-SHA-tagged prompt versioning, deploy the agent with `agents.deploy()` to enable the Review App, and deploy a Gradio chatbot UI on Databricks Apps.

---

## Prerequisites

- Completed LAB-16 (or equivalent: a registered LangGraph agent in Unity Catalog)
- Databricks CLI ≥ 0.218.0 installed locally (`databricks --version`)
- A Databricks workspace with Unity Catalog enabled and at least two targets (dev and staging can point to the same workspace with different resource prefixes)
- Python 3.10+, `mlflow>=3.1.3`, `databricks-agents>=1.1.0`, `gradio>=4.0`, `pyyaml`
- A Unity Catalog catalog and schema where you have `CREATE MODEL` privilege (e.g., `main.ml`)
- Git initialized in the project directory (`git init` if needed)

---

## Setup

```bash
# Install required packages
pip install mlflow>=3.1.3 databricks-agents>=1.1.0 gradio>=4.0 pyyaml

# Verify Databricks CLI version
databricks --version

# Authenticate CLI (OAuth U2M — recommended)
databricks auth login --host https://<your-workspace>.azuredatabricks.net

# Create and enter the project directory
mkdir langgraph-rag-cicd && cd langgraph-rag-cicd
git init

# Create the directory structure
mkdir -p prompts src
```

```bash
# Expected project structure after setup:
# langgraph-rag-cicd/
# ├── databricks.yml
# ├── prompts/
# │   └── rag_system.yaml
# ├── src/
# │   ├── agent.py
# │   └── app.py          (Gradio UI)
# └── app.yaml            (Databricks Apps runtime config)
```

---

## Steps

### Step 1 — Author the prompt YAML and capture the Git SHA

Create `prompts/rag_system.yaml` with the system prompt:

```yaml
# prompts/rag_system.yaml
# Version is tracked by Git SHA — do not hardcode version numbers here.
system_prompt: |
  You are a helpful assistant that answers questions using only the provided
  context documents. If the answer is not in the context, say "I don't know."
  Always cite the document ID when referencing information.

retrieval_instruction: "Retrieve the top 3 most relevant passages for the user query."
```

Now create `src/agent.py` that loads the prompt and logs the Git SHA as an MLflow tag:

```python
# src/agent.py
# Scenario: LangGraph RAG agent with Git-SHA prompt versioning and MLflow trace tagging.

import mlflow
import subprocess
import yaml
import pathlib
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict

PROMPT_PATH = pathlib.Path("prompts/rag_system.yaml")

def load_prompt() -> Dict:
    with open(PROMPT_PATH) as f:
        return yaml.safe_load(f)

def get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except subprocess.CalledProcessError:
        return "unversioned"

PROMPT_CONFIG = load_prompt()
SYSTEM_PROMPT = PROMPT_CONFIG["system_prompt"]
GIT_SHA = get_git_sha()

class AgentState(TypedDict):
    messages: List[Dict]
    retrieved_docs: List[str]

def retrieve(state: AgentState) -> AgentState:
    # Scenario: mock retrieval — replace with actual Vector Search call
    query = state["messages"][-1]["content"]
    state["retrieved_docs"] = [
        f"[doc-1] The return policy allows returns within 30 days.",
        f"[doc-2] Exchanges require a receipt dated within 60 days.",
    ]
    return state

def generate(state: AgentState) -> AgentState:
    from databricks_openai import DatabricksOpenAI
    client = DatabricksOpenAI()
    context = "\n".join(state["retrieved_docs"])
    response = client.chat.completions.create(
        model="databricks-claude-sonnet-4",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {state['messages'][-1]['content']}"},
        ],
    )
    state["messages"].append({
        "role": "assistant",
        "content": response.choices[0].message.content,
    })
    return state

graph = StateGraph(AgentState)
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)
graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)
agent = graph.compile()

def tag_mlflow_run():
    """Call inside an active MLflow run to tag with prompt provenance."""
    mlflow.set_tag("prompt_sha", GIT_SHA)
    mlflow.set_tag("prompt_path", str(PROMPT_PATH))
```

Commit the initial state so the SHA is meaningful:

```bash
git add prompts/ src/agent.py
git commit -m "feat: initial LangGraph RAG agent with versioned prompt"
```

---

### Step 2 — Register the agent to Unity Catalog with prompt tags

```python
# src/register_agent.py
# Scenario: Log LangGraph agent to MLflow, tag with prompt SHA, and register
# to Unity Catalog so it can be referenced in the bundle.

import mlflow
from src.agent import agent, tag_mlflow_run

UC_MODEL_NAME = "main.ml.rag_agent_lab17"

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Shared/lab17-rag-agent")

with mlflow.start_run() as run:
    tag_mlflow_run()

    mlflow.langchain.log_model(
        agent,
        artifact_path="agent",
        registered_model_name=UC_MODEL_NAME,
    )
    model_version = mlflow.register_model(
        f"runs:/{run.info.run_id}/agent",
        UC_MODEL_NAME
    )
    print(f"Registered: {UC_MODEL_NAME} version {model_version.version}")
    print(f"Run ID: {run.info.run_id}")
```

```bash
python src/register_agent.py
# Expected output:
# Registered: main.ml.rag_agent_lab17 version 1
# Run ID: <mlflow-run-id>
```

---

### Step 3 — Author the `databricks.yml` bundle with dev and staging targets

```yaml
# databricks.yml
bundle:
  name: lab17-rag-cicd
  databricks_cli_version: ">=0.218.0"

variables:
  uc_model_name:
    description: "Unity Catalog model name for the registered agent"
    default: "main.ml.rag_agent_lab17"
  model_version:
    description: "Model version to deploy"
    default: "1"

resources:
  registered_models:
    rag_agent:
      name: ${var.uc_model_name}
      catalog_name: main
      schema_name: ml
      comment: "Lab 17 RAG agent — managed by DAB"

  model_serving_endpoints:
    rag_endpoint:
      name: "lab17-rag-${bundle.target}"
      config:
        served_entities:
          - entity_name: ${var.uc_model_name}
            entity_version: ${var.model_version}
            workload_size: "Small"
            scale_to_zero_enabled: true

targets:
  dev:
    default: true
    mode: development
    workspace:
      host: https://<YOUR-DEV-WORKSPACE>.azuredatabricks.net

  staging:
    workspace:
      host: https://<YOUR-STAGING-WORKSPACE>.azuredatabricks.net
    resources:
      model_serving_endpoints:
        rag_endpoint:
          config:
            served_entities:
              - entity_name: ${var.uc_model_name}
                entity_version: ${var.model_version}
                workload_size: "Small"
                scale_to_zero_enabled: true
```

Validate and deploy to dev:

```bash
# Validate YAML syntax and resource schema
databricks bundle validate

# Deploy to dev (default target)
databricks bundle deploy

# Deploy to staging explicitly
databricks bundle deploy -t staging
```

---

### Step 4 — Enable the Review App with `agents.deploy()`

```python
# src/deploy_with_review.py
# Scenario: Deploy the registered agent to staging with Review App enabled
# for domain expert feedback before prod promotion.

from databricks import agents

UC_MODEL_NAME = "main.ml.rag_agent_lab17"
MODEL_VERSION = 1

deployment = agents.deploy(
    model_name=UC_MODEL_NAME,
    model_version=MODEL_VERSION,
    scale_to_zero_enabled=True,
)

print(f"Endpoint: {deployment.query_endpoint}")
print(f"Review App: {deployment.review_app_url}")
# Share the review_app_url with domain experts (they need only account access
# and CAN_EDIT on the MLflow experiment — no workspace login required).
```

```bash
python src/deploy_with_review.py
# Expected output:
# Endpoint: https://<workspace>/serving-endpoints/agents-main-ml-rag_agent_lab17-1/invocations
# Review App: https://<workspace>/ml/review-app/main.ml.rag_agent_lab17/1
```

---

### Step 5 — Build and deploy the Gradio chatbot UI on Databricks Apps

Create the Gradio app and its runtime config:

```python
# src/app.py
# Scenario: Gradio chatbot UI that calls the deployed Model Serving endpoint
# using the workspace OAuth identity inherited from Databricks Apps.

import gradio as gr
import os
import requests

ENDPOINT_URL = os.environ.get(
    "ENDPOINT_URL",
    "https://<workspace>/serving-endpoints/lab17-rag-staging/invocations"
)
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")

def chat(message: str, history: list) -> str:
    messages = [{"role": m["role"], "content": m["content"]} for m in history]
    messages.append({"role": "user", "content": message})
    payload = {"messages": messages}
    headers = {"Authorization": f"Bearer {DATABRICKS_TOKEN}"}
    response = requests.post(ENDPOINT_URL, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

demo = gr.ChatInterface(
    fn=chat,
    title="Lab 17 — RAG Agent Chatbot",
    description="Ask questions about our return and exchange policy.",
    type="messages",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 8080)))
```

```yaml
# app.yaml — Databricks Apps runtime configuration
command: ["python", "src/app.py"]
env:
  - name: ENDPOINT_URL
    value: "https://<workspace>/serving-endpoints/lab17-rag-staging/invocations"
```

Deploy the app:

```bash
# Deploy the Gradio app to Databricks Apps
databricks apps deploy lab17-chatbot --source-code-path .

# Expected output:
# App lab17-chatbot deployed successfully.
# App URL: https://<workspace>/apps/lab17-chatbot
```

---

## Validation

After completing all steps, run the following checks:

```bash
# 1. Verify bundle deploy created the staging endpoint
databricks serving-endpoints list | grep lab17-rag-staging

# Expected: lab17-rag-staging   READY
```

```python
# 2. Verify prompt SHA tag is present on the MLflow run
import mlflow
mlflow.set_tracking_uri("databricks")

runs = mlflow.search_runs(
    experiment_names=["/Shared/lab17-rag-agent"],
    filter_string="tags.prompt_sha != ''",
)
assert len(runs) > 0, "No runs with prompt_sha tag found"
print(f"Prompt SHA on most recent run: {runs.iloc[0]['tags.prompt_sha']}")
# Expected: Prompt SHA on most recent run: <40-char Git SHA>
```

```python
# 3. Verify the staging endpoint responds correctly
import requests, os

endpoint_url = "https://<workspace>/serving-endpoints/lab17-rag-staging/invocations"
token = os.environ["DATABRICKS_TOKEN"]
payload = {"messages": [{"role": "user", "content": "What is the return policy?"}]}

resp = requests.post(
    endpoint_url,
    json=payload,
    headers={"Authorization": f"Bearer {token}"}
)
assert resp.status_code == 200
assert "choices" in resp.json()
print("Endpoint validation passed:", resp.json()["choices"][0]["message"]["content"][:100])
```

```bash
# 4. Verify the Gradio app is accessible
databricks apps list | grep lab17-chatbot
# Expected: lab17-chatbot   RUNNING   https://<workspace>/apps/lab17-chatbot
```

---

## Teardown

```bash
# Destroy all bundle-managed resources (endpoints, registered model entry)
# WARNING: this permanently deletes deployed resources — do not run on shared workspaces
databricks bundle destroy -t staging --auto-approve
databricks bundle destroy --auto-approve  # dev target

# Delete the Databricks App
databricks apps delete lab17-chatbot

# Optionally delete the MLflow experiment
databricks experiments delete /Shared/lab17-rag-agent
```

---

## Reflection Questions

1. If a colleague updates `prompts/rag_system.yaml` but forgets to commit before running `src/register_agent.py`, the logged `prompt_sha` will point to the previous commit — not the actual prompt used at inference. What validation step would you add to the CI pipeline to detect and block this condition?

2. In Step 3, `mode: development` is set on the dev target. If you accidentally set it on the staging target instead, what observable difference would you see in the Databricks workspace, and why would this break the `agents.deploy()` call in Step 4?

3. The current Gradio app calls the Model Serving endpoint with a static bearer token. In a production Databricks Apps deployment, what is the preferred authentication mechanism and how does it change the code in `src/app.py`?
