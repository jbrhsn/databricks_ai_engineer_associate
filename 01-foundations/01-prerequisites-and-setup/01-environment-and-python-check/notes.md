# Environment and Python Check

**Section:** Foundations | **Module:** Prerequisites and Setup | **Est. time:** 1 hr
**Exam mapping:** Supporting content — not directly scored; ensures you can run practical exercises throughout the repo

## Learning Objectives

By the end of this chapter, you will be able to:

- Verify your local Python version is compatible with Databricks Runtime requirements
- Confirm key GenAI libraries are installed and importable at the expected versions
- Understand what Databricks Runtime (DBR) bundles and why the local/cluster version gap matters
- Set up or confirm a virtual environment that isolates project dependencies
- Identify and fix the three most common environment mismatches before they block later work

## Core Concepts

### Python Version Requirements

Databricks Runtime (DBR) ships with a specific Python version. As of DBR 14.x and 15.x (the
versions this repo targets), the bundled Python is **3.11**. Your local Python version matters
for two reasons:

1. **Local development and testing** — if you test code locally before running it on a cluster,
   version mismatches cause subtle bugs (f-string behavior changes, type hint syntax, match
   statements, etc.)
2. **Library compatibility** — some GenAI libraries drop support for older Python versions quickly.
   `langchain`, `mlflow`, and `databricks-sdk` all require Python ≥ 3.10 in their current releases.

**Minimum:** Python 3.10. **Recommended:** Python 3.11 (matches DBR 14.x/15.x exactly).

To check your version:
```bash
python --version
# or, if you have multiple pythons:
python3 --version
```

### Databricks Runtime (DBR) Versioning

DBR is the software layer that runs on your Databricks cluster. It bundles:
- A specific Python version
- Apache Spark
- A curated set of pre-installed Python libraries (NumPy, pandas, scikit-learn, MLflow, etc.)
- For **ML-enabled** runtimes (e.g., `14.3 LTS ML`): additional libraries like PyTorch, Hugging Face Transformers, and LangChain

The important pattern: **DBR version number + LTS + ML flag**
- `14.3 LTS` — Long Term Support release of DBR 14.3; stable, supported for 2+ years
- `14.3 LTS ML` — Same, plus ML libraries pre-installed; **use this for all GenAI work**
- `15.x` — Newer, may not yet have LTS designation; fine for experimentation, less stable for production

For this repo, use **DBR 14.3 LTS ML or newer** on your cluster. The ML-enabled runtime saves you
from manually installing PyTorch and Hugging Face libraries on every cluster start.

### Virtual Environments for Local Work

When working locally (writing code, testing before pushing to Databricks), use a virtual
environment to isolate this project's dependencies from your system Python.

**Option 1 — venv (built-in, recommended for simplicity):**
```bash
python3.11 -m venv .venv
source .venv/bin/activate       # macOS/Linux
.venv\Scripts\activate          # Windows
pip install -r requirements.txt  # if the repo has one
```

**Option 2 — conda (preferred if you manage multiple Python versions):**
```bash
conda create -n databricks-genai python=3.11
conda activate databricks-genai
```

**Key libraries to have locally for this repo:**
```bash
pip install databricks-sdk mlflow langchain openai pydantic
```

### Key Library Versions to Verify

After installing, confirm these import cleanly and are at expected versions:

```python
import mlflow
import langchain
import databricks.sdk
import pydantic

print(mlflow.__version__)       # expect 2.x or 3.x
print(langchain.__version__)    # expect 0.2.x or newer
```

If an import fails with `ModuleNotFoundError`, the library is not installed in the active
environment — a common mistake when multiple Python installs or environments exist.

## Deep Dive / Advanced Topics

### Why DBR Version Mismatches Cause Hard-to-Debug Failures

When you `%pip install` a library inside a Databricks notebook cell, it installs into the
cluster's Python environment — but the cluster already has many pre-installed packages. If your
`%pip install langchain==0.1.x` conflicts with a pre-installed dependency, pip may silently
downgrade another package, breaking unrelated code.

The safest pattern: **do not pin exact library versions unless you have a specific reason to**.
Let pip resolve against the DBR-bundled set, and only pin if you see a conflict.

### Cluster-Local vs. DBFS-Installed Libraries

Databricks offers three ways to make libraries available on a cluster:

| Method | Scope | Persistence |
|--------|-------|-------------|
| `%pip install` in a notebook cell | Session only | Restarts on cluster restart |
| Cluster Libraries UI (PyPI) | All notebooks on that cluster | Persists while cluster config exists |
| Unity Catalog Volumes or DBFS (`/FileStore`) | File storage only — not Python installs | Permanent |

For this repo's exercises, `%pip install` at the top of each notebook is sufficient. For
production code, add libraries to the cluster configuration.

### Checking the Databricks Connect Version (Optional)

Databricks Connect lets you run cluster-bound code from a local IDE. If you plan to use it:

```bash
pip install databricks-connect==14.3.*  # must match your cluster's DBR major.minor
databricks-connect configure
```

This is optional for this repo — all exercises can run directly in Databricks notebooks.

## Worked Examples & Practice

### Verifying Your Full Stack in One Notebook Cell

Create a new notebook on a DBR 14.3 LTS ML cluster and run this cell. All assertions should pass:

```python
import sys
import importlib

# 1. Python version check
major, minor = sys.version_info.major, sys.version_info.minor
assert (major, minor) >= (3, 10), f"Python 3.10+ required, got {major}.{minor}"
print(f"Python {major}.{minor} — OK")

# 2. Key library imports
required = ["mlflow", "langchain", "pydantic", "databricks.sdk"]
for lib in required:
    try:
        mod = importlib.import_module(lib)
        version = getattr(mod, "__version__", "version unknown")
        print(f"{lib} {version} — OK")
    except ImportError:
        print(f"MISSING: {lib} — install with %pip install {lib.replace('.', '-')}")

# 3. Spark availability (only meaningful on a cluster)
try:
    spark  # noqa: F821 — pre-defined in Databricks notebooks
    print(f"Spark {spark.version} — OK")
except NameError:
    print("Spark not available (running locally — OK for local dev)")
```

**Expected output on DBR 14.3 LTS ML:**
```
Python 3.11 — OK
mlflow 2.x.x — OK
langchain 0.x.x — OK
pydantic 2.x.x — OK
databricks.sdk 0.x.x — OK
Spark 3.5.x — OK
```

**What to do if a library is missing:**
Add `%pip install <library>` as the first cell in the notebook, run it, then restart the Python
kernel (the notebook will prompt you) before proceeding.

## Common Pitfalls & Misconceptions

- **Pitfall:** Running `python` and getting Python 2.7 or 3.8 on a machine with multiple installs → **Why it happens:** macOS ships Python 2.7 as system Python; older conda envs may default to 3.8 → **Fix:** Always use `python3` or activate your virtual environment first; check `which python`

- **Pitfall:** Installing libraries locally with `pip install` but they are not available on the Databricks cluster → **Why it happens:** Local and cluster environments are entirely separate → **Fix:** Use `%pip install` at the top of your notebook or add the library to cluster Libraries config

- **Pitfall:** Selecting a non-ML DBR (e.g., `14.3 LTS` without the ML flag) and wondering why PyTorch and Transformers are missing → **Why it happens:** ML libraries are only bundled in the `...ML` variant → **Fix:** When creating a cluster, always select the `ML` runtime variant for GenAI work

- **Pitfall:** `%pip install` works but code breaks after it runs → **Why it happens:** pip may have downgraded a pre-installed library to satisfy a version constraint → **Fix:** After a `%pip install`, always restart the Python kernel when prompted; if problems persist, remove explicit version pins

## Key Definitions

| Term | Definition |
|---|---|
| Databricks Runtime (DBR) | The versioned software layer running on a Databricks cluster — bundles Python, Spark, and pre-installed libraries |
| LTS (Long Term Support) | A DBR designation indicating Databricks will maintain and patch this version for 2+ years; preferred for stable workloads |
| ML Runtime | A DBR variant (`X.Y LTS ML`) that includes additional pre-installed machine learning libraries: PyTorch, Hugging Face, LangChain, etc. |
| Virtual environment | An isolated Python installation directory that has its own packages, separate from the system Python — created with `venv` or `conda` |
| `%pip install` | A Databricks notebook magic command that installs a Python package into the current cluster's notebook-scoped environment |

## Summary / Quick Recall

- Python 3.10+ required; 3.11 is ideal (matches DBR 14.x/15.x)
- Always use an **ML-enabled DBR** (e.g., `14.3 LTS ML`) for GenAI work — it pre-installs PyTorch, Hugging Face, LangChain
- Local and cluster environments are independent; `%pip install` in a notebook only affects the cluster session
- Use `venv` or `conda` locally to avoid polluting your system Python
- After `%pip install` in a notebook, restart the Python kernel before importing the new library
- Verify your stack with a quick import check cell before starting any chapter's exercises

## Self-Check Questions

1. What Python version does DBR 14.3 LTS bundle, and why does your local version need to match?

<details>
<summary>Answer</summary>
DBR 14.3 LTS bundles Python 3.11. Your local version should match (or be close) because code you test locally may use syntax or library behavior specific to that Python version. A mismatch can cause code that passes locally to fail on the cluster, or vice versa.
</details>

2. You create a new cluster using `DBR 14.3 LTS` (without the ML flag) and find that `import torch` fails. What is the cause and fix?

<details>
<summary>Answer</summary>
The non-ML DBR variant does not pre-install PyTorch. Fix: recreate or edit the cluster to use `DBR 14.3 LTS ML`, which bundles PyTorch and other ML libraries. Alternatively, `%pip install torch` at the top of the notebook, but this is slower and less reliable.
</details>

3. You `%pip install langchain` in a notebook cell, but the next cell still raises `ModuleNotFoundError: No module named 'langchain'`. What is the most likely cause?

<details>
<summary>Answer</summary>
You need to restart the Python kernel after `%pip install`. Databricks prompts you to do this automatically. The install succeeded, but the current Python process has not yet loaded the new package. Click "Restart Python" when prompted, or run `dbutils.library.restartPython()`.
</details>

4. What is the difference between adding a library via `%pip install` in a notebook vs. adding it in the cluster's Libraries configuration?

<details>
<summary>Answer</summary>
`%pip install` installs the library only for the current notebook session; it is lost if the cluster restarts. Adding a library to the cluster's Libraries configuration installs it every time the cluster starts, making it available to all notebooks on that cluster persistently.
</details>

5. Why should you avoid pinning exact library versions (e.g., `%pip install langchain==0.1.0`) in Databricks notebooks unless necessary?

<details>
<summary>Answer</summary>
Exact version pins can conflict with other pre-installed packages in the DBR environment, causing pip to silently downgrade dependencies and break unrelated code. Let pip resolve against the DBR-bundled set; only pin when you have diagnosed a specific version conflict that requires it.
</details>

## Further Reading

- [Databricks Runtime release notes](https://docs.databricks.com/en/release-notes/runtime/index.html) — official DBR version history, Python versions, and bundled library versions
- [Databricks — Manage cluster libraries](https://docs.databricks.com/en/libraries/cluster-libraries.html) — official guide to the three library installation methods
- [Python virtual environments (venv)](https://docs.python.org/3/library/venv.html) — official Python docs for the built-in venv module
