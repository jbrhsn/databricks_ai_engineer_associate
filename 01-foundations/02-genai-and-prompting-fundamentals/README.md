# Module 02 — GenAI and Prompting Fundamentals

**Part of section:** 01 Foundations
**Estimated time:** 5.5 hours
**Prerequisites:** Module 01 (Prerequisites and Setup) — you should have a working Databricks cluster before starting this module
**Exam mapping:** Foundations domain (~15% of exam); concepts here also appear as assumed knowledge in App Development (30%) and Governance/Evaluation (15%) domains

## Overview

This is the conceptual core of the entire repo. Before you can build RAG pipelines, agents, or
evaluation frameworks, you need a precise understanding of how LLMs actually work — not a vague
sense that "AI predicts the next word," but a working model of tokens, context windows, sampling
parameters, and the mechanics of prompt parsing. This module covers that machinery, then builds
on it to develop practical prompting skills and model selection judgment.

## Learning Outcomes

By completing this module, you will be able to:

- Explain how transformer-based LLMs process input and generate output at the token level
- Describe the role of embeddings, context windows, and sampling parameters (temperature, top-p)
- Use the Foundation Model API on Databricks to call hosted LLMs
- Write effective prompts using zero-shot, few-shot, and chain-of-thought techniques
- Control output format with system prompts, JSON mode, and Pydantic schema enforcement
- Select an appropriate model for a given task based on latency, cost, capability, and openness
- Decompose a complex user problem into a sequence of focused LLM sub-tasks

## Topics Covered

- Transformer architecture (high level): attention, tokenization, next-token prediction
- Embeddings: what they are, why they enable semantic search, the difference between token and document embeddings
- Context windows: tokens in, tokens out, why length matters for cost and quality
- Temperature and sampling: greedy decoding, temperature, top-p, top-k
- Foundation Model APIs on Databricks: pay-per-token hosted endpoints for models like DBRX, Llama, Mistral
- Prompt anatomy: system message, user message, assistant message
- Zero-shot, few-shot, and chain-of-thought prompting
- Structured output: JSON mode, schema enforcement, Pydantic integration
- Task taxonomy and model selection criteria
- Problem decomposition patterns for multi-step LLM workflows

## How This Module Fits

Module 01 (Prerequisites and Setup) gave you an environment. This module gives you the mental
model you need to use it. Every chapter in Sections 02–05 assumes the vocabulary and intuitions
built here. In particular:

- Section 02 (Data Preparation) assumes you understand embeddings and context windows
- Section 03 (Application Development) assumes you can write effective prompts and know what structured output means
- Section 05 (Governance) assumes you understand how sampling parameters affect output determinism

## Study Tips

- **Chapter 01 (LLM Core Concepts) is the highest-leverage chapter in this module.** If exam
  time is short, spend most of your time here — its concepts appear across all other exam domains
  as assumed knowledge.
- **Try every prompting technique in a notebook as you read about it.** The difference between
  zero-shot and few-shot performance is something you need to see, not just read about. Use
  Databricks Foundation Model API (or any OpenAI-compatible endpoint) for this.
- **Temperature = 0 for deterministic outputs, higher for creative tasks.** This sounds simple
  but exam questions often use temperature as a distractor. Know exactly what it does and when
  to change it.
- **Chapter 03 (model selection) is lighter reading** but tests practical judgment. The exam
  does not ask you to memorize benchmark scores — it tests whether you can reason about
  trade-offs given a scenario.

## Chapters

- [ ] [01 LLM and GenAI Core Concepts](./01-llm-and-genai-core-concepts/notes.md) — 2 hrs
- [ ] [02 Prompt Engineering and Structured I/O](./02-prompt-engineering-and-structured-io/notes.md) — 2 hrs
- [ ] [03 Matching Models, Tasks, and Problem Decomposition](./03-matching-models-tasks-and-problem-decomposition/notes.md) — 1.5 hrs
