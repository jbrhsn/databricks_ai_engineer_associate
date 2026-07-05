# Module 01 — Prerequisites and Setup

**Part of section:** 01 Foundations
**Estimated time:** 2.5 hours
**Prerequisites:** None — this is the entry point for the repo
**Exam mapping:** Supporting content — not directly scored; required to do any practical work in Databricks

## Overview

Before you can build or experiment with anything in this repo, you need a working environment.
This module covers two things: verifying your local Python setup meets Databricks compatibility
requirements, and orienting yourself to the Databricks workspace UI so nothing in later modules
comes as a surprise.

## Learning Outcomes

By completing this module, you will be able to:

- Confirm your Python version and key library versions are compatible with Databricks Runtime
- Create, configure, and connect to a Databricks cluster
- Navigate the workspace UI confidently: notebooks, repos, DBFS, Unity Catalog, and jobs
- Run a notebook end-to-end and interpret the output
- Set up Git integration in Databricks Repos

## Topics Covered

- Python version requirements and virtual environments
- Databricks Runtime (DBR) versioning and what it bundles
- Cluster creation and configuration basics
- Notebook interface: cells, magic commands (`%python`, `%sql`, `%sh`), display()
- Databricks File System (DBFS) vs. Unity Catalog Volumes — when to use which
- Git integration via Databricks Repos
- Jobs and Workflows: what they are (overview only — details in Section 04)

## How This Module Fits

This module is pure setup. Its output is a working Databricks environment you can use for every
subsequent chapter. The GenAI and Prompting Fundamentals module (Module 02 in this section)
assumes you can open a notebook and call the Foundation Model API — both of which require the
workspace setup covered here.

## Study Tips

- **Spin up a cluster before reading Module 02.** Databricks cluster startup can take 3–5 minutes;
  start it in the background while you read.
- **Use the Community Edition or a trial workspace** if you don't have an employer-provided
  Databricks account. Most of what this repo covers works on a single-node cluster.
- **Don't memorize UI locations** — the Databricks UI updates frequently. Learn the conceptual
  model (what clusters, notebooks, repos, and volumes *are*) and you will find your way around
  regardless of which UI version you encounter on exam day.
- **DBR version matters.** The exam tests against specific Databricks Runtime features. Prefer
  DBR 14.x or 15.x (ML-enabled) for this repo's practical exercises.

## Chapters

- [ ] [01 Environment and Python Check](./01-environment-and-python-check/notes.md) — 1 hr
- [ ] [02 Databricks Workspace Orientation](./02-databricks-workspace-orientation/notes.md) — 1.5 hrs
