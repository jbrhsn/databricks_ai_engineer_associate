# Section 04: Application Development

**Estimated time:** 8 hrs | **Exam domain weight:** ~30%** | **Prerequisites:** Section 02 (Design Applications), Section 03 (Data Preparation)

> **This is the heaviest exam domain (30% of questions).** Prioritise all three modules and their labs.

---

## Overview

This section covers the end-to-end development of GenAI applications on Databricks — from writing your first LangGraph state machine through deploying a monitored, production-grade multi-agent system. It bridges the data pipeline work of Section 03 with the deployment and governance topics of Sections 05–06. You will build working agents using LangGraph (primary framework), LangChain components (LCEL chains, retrievers, output parsers), and the Databricks MLflow Agent Framework, then learn how to select the right models, evaluate quality offline, and monitor it online.

---

## Learning Outcomes

By completing this section you will be able to:
- Build stateful agents using LangGraph `StateGraph`, conditional edges, and `MemorySaver` / `SqliteSaver` checkpointing.
- Compose production RAG chains with LangChain's LCEL pipe operator, `ChatDatabricks`, `DatabricksVectorSearch.as_retriever()`, and typed output parsers.
- Harden applications with input guardrails (PII masking, prompt injection detection), system prompt constraints, output schema validation, and fallback chains.
- Select LLMs and embedding models by evaluating context window, inference speed, cost, task-type benchmark alignment, and model card limitations.
- Log, trace, evaluate, register, and deploy agents through the full MLflow lifecycle using `mlflow.langchain.log_model()`, `@mlflow.trace`, and `mlflow.evaluate(model_type="databricks-agent")`.
- Design multi-agent systems using the LangGraph supervisor pattern, agent-as-tool registration, and shared `TypedDict` state.
- Detect post-deployment quality drift using inference tables, Lakehouse Monitoring, and A/B traffic splitting.

---

## Modules

| # | Module | Est. time | Chapters | Labs |
|---|---|---|---|---|
| 1 | Frameworks and Prompting | 3 hrs | 3 | LAB-06, LAB-07, LAB-08 |
| 2 | Model and Embedding Selection | 2 hrs | 3 | — |
| 3 | Agents, MLflow, and Genie | 3 hrs | 3 | LAB-09, LAB-10 |

---

## Chapter Map

### Module 1 — Frameworks and Prompting
| Chapter | Key topics | Lab |
|---|---|---|
| `01-langgraph-fundamentals` | StateGraph, nodes, conditional edges, checkpointing, streaming | LAB-06 |
| `02-langchain-components` | LCEL, ChatDatabricks, DatabricksVectorSearch.as_retriever(), output parsers | LAB-07 |
| `03-prompt-augmentation-and-guardrails` | Context injection, input/output guardrails, AI Gateway filters, fallback chains | LAB-08 |

### Module 2 — Model and Embedding Selection
| Chapter | Key topics | Lab |
|---|---|---|
| `01-llm-selection-by-attributes` | Context window, latency, cost model, FMAPI, External Models | — |
| `02-embedding-models-and-context-length` | Embedding dimensions, max sequence length, distance metrics, DatabricksEmbeddings | — |
| `03-model-hub-cards-and-metrics` | Model cards, MMLU, HumanEval, MT-Bench, HELM, bias/limitations | — |

### Module 3 — Agents, MLflow, and Genie
| Chapter | Key topics | Lab |
|---|---|---|
| `01-mlflow-and-agent-framework` | `log_model`, Model Registry, MLflow Tracing, `mlflow.evaluate()`, Agent Framework | LAB-09 |
| `02-multi-agent-and-genie-spaces` | Supervisor pattern, agent-as-tool, shared state, Genie Spaces, CrewAI comparison | LAB-10 |
| `03-eval-vs-monitoring-lifecycle` | Offline eval, online monitoring, inference tables, drift detection, A/B testing | — |

---

## How This Section Fits

Section 03 produced clean, chunked data in Delta Tables ready for retrieval. **This section consumes that data** — building the agents and chains that sit on top of it. The deployment and governance skills from Section 05 (Assembling & Deploying) and Section 06 (Governance) build directly on the MLflow lifecycle, guardrail patterns, and monitoring instrumentation introduced here.

---

## Study Tips

- **Do the labs in order.** LAB-06 (LangGraph) → LAB-07 (LCEL chain) → LAB-08 (guardrails) → LAB-09 (MLflow) → LAB-10 (multi-agent) form a progressive build — each lab extends the previous one.
- **LangGraph is primary.** When the exam presents a scenario requiring stateful, multi-turn, or multi-agent orchestration, LangGraph is the expected answer. LangChain LCEL is the component library *inside* LangGraph nodes, not a replacement.
- **Know the MLflow lifecycle cold.** The exam heavily tests `log_model` → register → `mlflow.evaluate()` → deploy. Understand what each step does and what happens if you skip one.
- **Model selection is maths.** Practice the token-budget calculation: `context_window - system_prompt_tokens - query_tokens = max_retrievable_context`. Know that MMLU ≠ task accuracy — benchmark alignment to your domain is what matters.
- **Inference tables are your monitoring foundation.** `auto_capture_config` must be enabled at endpoint creation — you cannot add it retroactively. Know this for the exam.
- **Fast-evolving features to re-verify:** Genie Spaces (now "Genie Agents" as of July 2026), FMAPI model catalogue, `ResponsesAgent` interface, MLflow GenAI Scorers.
