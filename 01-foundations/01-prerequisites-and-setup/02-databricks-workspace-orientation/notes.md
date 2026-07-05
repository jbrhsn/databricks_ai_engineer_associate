# Databricks Workspace Orientation

**Section:** Foundations | **Module:** Prerequisites and Setup | **Est. time:** 1.5 hrs
**Exam mapping:** Supporting content — not directly scored; workspace navigation knowledge is assumed for all practical exam scenarios

## Learning Objectives

By the end of this chapter, you will be able to:

- Navigate the Databricks workspace UI and locate Compute, Workspace, Data, and Workflows sections
- Create and configure a cluster appropriate for GenAI development
- Create notebooks, write and run cells, and use magic commands
- Understand the difference between DBFS and Unity Catalog Volumes for file storage
- Connect a workspace repo to a Git remote using Databricks Repos
- Describe the relationship between notebooks, jobs, and workflows at a conceptual level

## Core Concepts

### The Workspace UI — Mental Model First

Before clicking through menus, build the mental model. The Databricks workspace is organized around
four core resources:

| Resource | What it is | Where in the UI |
|----------|-----------|-----------------|
| **Clusters** | The compute that runs your code | Compute → All-purpose clusters |
| **Notebooks** | Interactive documents with code + output | Workspace (left sidebar) |
| **Data** | Tables, files, and Unity Catalog objects | Data (left sidebar) |
| **Workflows / Jobs** | Scheduled or triggered notebook/task runs | Workflows (left sidebar) |

Everything else in the UI is a variation or orchestration layer on top of these four.

### Clusters

A cluster is a set of virtual machines that run your Spark and Python workloads. For this repo,
you need an **all-purpose cluster** (as opposed to a job cluster, which auto-terminates after a
job run).

**Key configuration choices when creating a cluster:**
- **Databricks Runtime version:** Use `14.3 LTS ML` or `15.x LTS ML` (see the previous chapter)
- **Node type:** For GenAI exercises that don't require GPU, a single-node CPU cluster (e.g., `m5.xlarge` on AWS, `Standard_DS3_v2` on Azure) is sufficient and cheapest
- **Auto-termination:** Set to 30–60 minutes of inactivity to avoid charges while you are reading

**Cluster states you will encounter:**
- `Running` — ready, attached to your notebook
- `Terminated` — shut down; will auto-start when you attach a notebook
- `Starting` — initializing (takes 3–5 minutes for a cold start)
- `Error` — something went wrong during startup; check the event log

### Notebooks

Databricks notebooks look like Jupyter notebooks but run against a cluster. Key differences:

- **Language default:** Set per notebook (Python, SQL, Scala, R), but you can override per cell
- **Magic commands:** `%python`, `%sql`, `%scala`, `%r`, `%sh`, `%md`, `%fs`
- **`display()`:** Databricks-specific function that renders DataFrames, images, and charts inline
- **Auto-attach:** The first time you run a cell, the notebook attaches to the default cluster

**Magic commands you will use regularly:**
```python
%python   # switch cell to Python (useful in a SQL-default notebook)
%sql      # run a SQL cell
%sh       # run a shell command (useful for quick file checks)
%md       # write Markdown documentation in a cell
%fs ls /  # list files in DBFS root
```

### File Storage: DBFS vs. Unity Catalog Volumes

This distinction trips up many learners. They look similar but serve different purposes:

**DBFS (Databricks File System):**
- Legacy file system abstraction that maps to cloud object storage (S3, ADLS, GCS)
- Paths start with `/dbfs/` (from the terminal) or `dbfs:/` (from notebooks/`%fs`)
- Still works but is being de-emphasized in favor of Unity Catalog
- **Use it for:** Temporary scratch space, reading legacy data during transition periods
- **Avoid it for:** New projects — Unity Catalog Volumes are the preferred path

**Unity Catalog Volumes:**
- First-class storage objects governed by Unity Catalog's access control
- Paths: `/Volumes/<catalog>/<schema>/<volume>/path/to/file`
- Full governance: audit logs, access control, lineage
- **Use it for:** All new data storage in Databricks workspaces with UC enabled

For the exam: Unity Catalog Volumes is the current Databricks-preferred approach for managing
unstructured files (PDFs, text files, images) in GenAI pipelines.

### Unity Catalog — Brief Overview

Unity Catalog (UC) is Databricks' data governance layer. Its object hierarchy:

```
Metastore (one per region, shared across workspaces)
  └── Catalog
        └── Schema (= database)
              ├── Table
              ├── View
              ├── Function
              └── Volume (for files)
```

You will encounter UC throughout this repo. For now, the key points:
- All table and volume references in Databricks should use three-level namespace: `catalog.schema.object`
- UC controls who can read, write, and modify each object via grants
- The default catalog for most workspaces is `main` or a workspace-specific catalog

### Databricks Repos (Git Integration)

Databricks Repos syncs your workspace notebooks with a Git remote (GitHub, GitLab, Bitbucket, etc.).

**To connect a repo:**
1. Left sidebar → Workspace → Repos → Add Repo
2. Paste your Git remote URL
3. Clone → the repo appears as a folder in your workspace
4. You can now `git pull`, `git commit`, and `git push` directly from the Databricks UI

**When to use Repos vs. plain Workspace folders:**
- Use **Repos** when you want version control, code review, and CI/CD integration
- Use plain **Workspace folders** for throwaway experiments and scratch notebooks

### Workflows (Jobs) — Conceptual Overview

A **Job** is a scheduled or API-triggered run of one or more tasks (notebooks, Python scripts,
dbt models, etc.). This is how you productionize code that runs in notebooks.

For this section, you only need to know that Jobs exist and that they run notebooks against
**job clusters** (auto-provisioned, auto-terminated) rather than your always-on all-purpose
cluster. The full treatment is in Section 04.

## Deep Dive / Advanced Topics

### Cluster Access Modes and What They Mean for GenAI Work

Databricks clusters have three access modes that affect what you can do:

| Access mode | Multi-user? | Spark + Python | UC enforced |
|------------|------------|----------------|-------------|
| Single user | No | Full access | Yes |
| Shared | Yes | Python only (no Spark shell) | Yes |
| No isolation (legacy) | Yes | Full access | No |

For GenAI work that involves calling LLM APIs (not Spark-heavy workloads), **Shared** mode is
fine and cheaper. If you need Spark DataFrame operations on large datasets in the same session,
use **Single user** mode.

### Secrets Management — Don't Hardcode API Keys

You will need API keys (e.g., for OpenAI, Anthropic, or Databricks Foundation Model APIs)
throughout this repo. Never paste them directly into a notebook cell.

**Databricks Secrets:**
```python
# Store a secret (done once, from CLI or Secrets UI):
# databricks secrets create-scope genai-keys
# databricks secrets put-secret genai-keys openai-api-key

# Retrieve in a notebook:
import os
os.environ["OPENAI_API_KEY"] = dbutils.secrets.get(scope="genai-keys", key="openai-api-key")
```

Secrets are redacted in notebook output — you will see `[REDACTED]` if you accidentally print one.
This is intentional and correct behavior.

### `dbutils` — The Databricks Utility Library

`dbutils` is pre-available in all Databricks notebooks (no import needed). It provides:

| Submodule | Key use |
|-----------|---------|
| `dbutils.fs` | File system operations (list, copy, move, rm) |
| `dbutils.secrets` | Retrieve secrets from Databricks Secrets |
| `dbutils.notebook` | Run a notebook from another notebook, pass parameters |
| `dbutils.widgets` | Create interactive UI widgets in notebooks |
| `dbutils.library` | `restartPython()` — restart the Python kernel after `%pip install` |

## Worked Examples & Practice

### End-to-End Setup Verification: Cluster → Notebook → Data → Secret

Work through this sequence in a fresh workspace to confirm everything is connected.

**Step 1: Create a cluster**
- Compute → Create compute
- Runtime: `14.3 LTS ML`
- Node type: smallest available single-node
- Auto-terminate: 30 minutes
- Create

**Step 2: Create a notebook and attach it**
```python
# Cell 1 — confirm cluster is attached and Spark is live
print(f"Spark version: {spark.version}")
print(f"Python version: {spark.conf.get('spark.databricks.clusterUsageTags.sparkVersion')}")
```

**Step 3: Write and read a file via Unity Catalog Volume**
```python
# Cell 2 — write a test file to a UC Volume
# Replace <catalog>, <schema>, <volume> with your actual values
volume_path = "/Volumes/main/default/my_test_volume"

dbutils.fs.put(f"{volume_path}/hello.txt", "Hello, Unity Catalog!", overwrite=True)

# Read it back
content = dbutils.fs.head(f"{volume_path}/hello.txt")
print(content)  # Hello, Unity Catalog!
```

**Step 4: Store and retrieve a secret**
```python
# Cell 3 — retrieve a pre-stored secret (requires you to have set one up)
# If you haven't set one up yet, skip this and return to it when you need an API key
try:
    secret_val = dbutils.secrets.get(scope="genai-keys", key="test-key")
    print("Secret retrieved successfully — value is redacted in output")
except Exception as e:
    print(f"Secret not yet configured: {e}")
```

**Step 5: List your DBFS root (for reference — prefer UC Volumes for new work)**
```python
# Cell 4
display(dbutils.fs.ls("dbfs:/"))
```

If all cells run without errors, your workspace is correctly configured for this repo.

## Common Pitfalls & Misconceptions

- **Pitfall:** Attaching a notebook to a terminated cluster and running a cell immediately → **Why it happens:** The cluster must auto-start before the cell can execute — this takes 3–5 minutes, and the UI can be silent about it → **Fix:** Click "Start" on the cluster explicitly and wait for the `Running` state before running cells, or simply be patient when the cell takes longer than expected

- **Pitfall:** Using `dbfs:/` paths in production code when Unity Catalog Volumes are available → **Why it happens:** Most tutorials and older examples use DBFS; it still works, so people keep using it → **Fix:** Use `/Volumes/<catalog>/<schema>/<volume>/...` paths for all new file storage; reserve DBFS for legacy compatibility only

- **Pitfall:** Hardcoding an API key as a string in a notebook cell → **Why it happens:** It is the fastest way to get something working → **Fix:** Use `dbutils.secrets.get()` — even in a personal dev workspace; building the habit now prevents accidental key exposure in shared workspaces or version control

- **Pitfall:** Confusing the three-level Unity Catalog namespace (`catalog.schema.table`) with a two-level namespace (just `schema.table`) → **Why it happens:** Traditional SQL databases use two-level namespaces; Databricks UC adds a third level → **Fix:** Always use the full three-level reference: `SELECT * FROM main.default.my_table`

- **Pitfall:** Creating a Shared-mode cluster and then being confused why `sc` (SparkContext) is unavailable → **Why it happens:** Shared-mode clusters restrict direct SparkContext access for isolation reasons → **Fix:** Use `spark.sparkContext` or switch to Single-user mode if you need direct SparkContext access

## Key Definitions

| Term | Definition |
|---|---|
| All-purpose cluster | A long-running Databricks cluster you attach notebooks to interactively; as opposed to a job cluster, which is auto-created and auto-terminated for a single job run |
| DBFS | Databricks File System — a legacy virtual file system abstraction over cloud object storage, accessible via `dbfs:/` paths; being superseded by Unity Catalog Volumes |
| Unity Catalog (UC) | Databricks' governance layer that provides centralized access control, lineage, and auditing for tables, views, functions, and volumes |
| Volume | A Unity Catalog object for storing unstructured files (analogous to a folder in cloud object storage, but with UC governance) |
| Magic command | A notebook-level directive prefixed with `%` that changes the language of a cell (`%sql`, `%python`) or invokes utilities (`%fs`, `%sh`) |
| `dbutils` | A pre-available Databricks utility library in notebooks providing file system operations, secrets access, notebook orchestration, and more |
| Databricks Repos | A workspace feature that syncs a folder of notebooks with a Git remote, enabling version control and CI/CD integration |
| Secrets scope | A named container in Databricks Secrets that holds key-value pairs; accessed via `dbutils.secrets.get(scope, key)` |

## Summary / Quick Recall

- The four core workspace resources: Clusters, Notebooks, Data (Unity Catalog), Workflows
- Use **ML-enabled** runtime (`14.3 LTS ML` or newer) for all GenAI work
- Prefer **Unity Catalog Volumes** over DBFS for file storage in new projects; paths: `/Volumes/<catalog>/<schema>/<volume>/`
- Unity Catalog uses a three-level namespace: `catalog.schema.object`
- Magic commands: `%sql`, `%python`, `%sh`, `%md`, `%fs` — switch cell language or invoke utilities
- Never hardcode API keys — use `dbutils.secrets.get(scope, key)`
- `dbutils.library.restartPython()` restarts the Python process after `%pip install`
- Databricks Repos = Git integration for version-controlled notebooks

## Self-Check Questions

1. You need to store a 500 MB PDF collection for a RAG pipeline. Should you use DBFS or Unity Catalog Volumes, and why?

<details>
<summary>Answer</summary>
Use Unity Catalog Volumes. UC Volumes are the current Databricks-preferred approach for unstructured file storage — they are governed by UC access controls, support audit logging, and participate in data lineage. DBFS is a legacy path that still works but is being de-emphasized for new projects.
</details>

2. You have a Unity Catalog table called `documents` in the `rag_data` schema of the `dev` catalog. Write the correct three-level reference to query it.

<details>
<summary>Answer</summary>
`SELECT * FROM dev.rag_data.documents` — Unity Catalog always uses the three-level format `catalog.schema.object`. Using a two-level reference (`rag_data.documents`) may work if `dev` is the active catalog, but explicit three-level references are the correct and unambiguous form.
</details>

3. What is the difference between an all-purpose cluster and a job cluster?

<details>
<summary>Answer</summary>
An all-purpose cluster is long-running and attached to notebooks interactively — you start it, it stays up, and you run cells against it. A job cluster is auto-provisioned when a job starts and auto-terminated when it finishes; it is ephemeral and used for production scheduled workloads. All-purpose clusters cost more per hour but are appropriate for development work.
</details>

4. A colleague pastes their API key directly into a shared notebook. What is the risk, and what is the correct approach?

<details>
<summary>Answer</summary>
The API key is now visible to anyone with access to that notebook and will appear in version history if the notebook is committed to Git. The correct approach: store the key in a Databricks Secrets scope and retrieve it with `dbutils.secrets.get(scope="<scope>", key="<key>")`. Secrets are redacted in notebook output and never exposed in plain text.
</details>

5. You are on a Shared-mode cluster and your code fails with an error about `sc` not being available. What is the cause?

<details>
<summary>Answer</summary>
Shared-mode clusters restrict direct `SparkContext` (`sc`) access to enforce isolation between users. Use `spark.sparkContext` as an alternative, or switch to a Single-user cluster if your code requires direct SparkContext access (e.g., for RDD operations).
</details>

## Further Reading

- [Databricks — Compute configuration reference](https://docs.databricks.com/en/compute/configure.html) — cluster access modes, runtime versions, and node types
- [Databricks — Unity Catalog Volumes](https://docs.databricks.com/en/connect/unity-catalog/volumes.html) — creating and using volumes for unstructured data
- [Databricks — Secrets management](https://docs.databricks.com/en/security/secrets/index.html) — creating scopes and storing secrets securely
- [Databricks — Repos (Git integration)](https://docs.databricks.com/en/repos/index.html) — connecting workspaces to Git remotes
