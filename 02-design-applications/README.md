# Design Applications

**Estimated time:** 8.5 hrs | **Exam domain weight:** 14% | **Prerequisites:** Section 01 (Foundations)

## Overview

Design Applications is the practical bridge between foundational LLM knowledge and building production systems. This section covers how to architect generative AI applications: from crafting effective prompts and selecting appropriate models to designing multi-step chains and orchestrating agentic workflows. You'll learn the design patterns that separate toy prototypes from scalable, maintainable systems.

## Learning Outcomes

By completing this section you will be able to:
- Design and validate prompts for reliability and consistency
- Select the right model for a given task based on capabilities and constraints
- Compose chain components with explicit I/O specifications
- Implement tool selection and ordering strategies for agents
- Apply error handling and retry logic in production LLM applications

## Modules

| # | Module | Est. time | Chapters |
|---|---|---|---|
| 1 | Prompt and Task Design | 4.5 hrs | 2 |
| 2 | Chain and Agent Design | 4 hrs | 2 |

### Module 1: Prompt and Task Design (4.5 hrs)

**Chapters:**
1. **Prompt Formatting and Output Specification** (2.5 hrs) — Designing effective prompts with few-shot examples, system messages, and output schemas. Implementing structured output validation (Pydantic, ProviderStrategy, ToolStrategy).
2. **Model and Task Selection** (2 hrs) — Selecting the right model for your task based on capability (general-purpose, embeddings, vision, reasoning), deployment patterns (pay-per-token, provisioned throughput), and cost-quality-latency tradeoffs.

**Why it matters:** Poor prompt design or mismatched models cascade into failures downstream. This module teaches you to start from the contract (what output do I need?) and work backward to the right prompt and model.

### Module 2: Chain and Agent Design (4 hrs)

**Chapters:**
1. **Chain Components and I/O Specifications** (2 hrs) — Designing composable chain components with explicit input/output schemas. Using LangChain's Runnable interface and LCEL for deterministic workflows.
2. **Tool Ordering and Agent Bricks** (2 hrs) — Designing tool selection and sequencing strategies for agents. Understanding Agent Bricks (framework-agnostic reusable agent modules) and tool routing patterns.

**Why it matters:** Once you've designed individual prompts and models, you need to compose them into multi-step workflows. This module teaches orchestration patterns that scale from simple chains to complex agent systems.

## How This Section Fits

**From Section 01 (Foundations):** Section 01 covered LLM and RAG fundamentals. This section assumes you understand what LLMs are and how retrieval augmentation works. Now you design applications that use these foundations.

**To Section 03+ (Data & Development):** Section 03 will cover data preparation (cleaning, tokenization, embeddings). Sections 04–08 cover deployment, governance, and evaluation. Design (this section) is the first opportunity to apply your foundations in practice; later sections will teach you how to build around these designs.

## Study Tips

- **Set up a local LangChain environment early.** You don't need to run code for every concept, but being able to test a few snippets (PromptTemplate, ProviderStrategy, FewShotChatMessagePromptTemplate) will anchor abstract concepts.
- **Focus on "why" over "how."** The mechanics of prompt formatting are less important than understanding *why* explicit prompts work better than vague ones. Internalize the principles; the syntax varies by framework.
- **Practice the Worked Examples.** Each chapter includes a realistic scenario (e.g., customer review classification, agent design). Walk through the example and try tweaking it—add an extra few-shot example, change a constraint, see how output changes.
- **Review the Common Pitfalls.** This section's pitfalls are the most common mistakes teams make in practice. They're worth memorizing verbatim—you'll recognize them in code reviews and production incidents.
- **Use LangSmith for visibility.** Tracing tools like LangSmith let you see exactly what prompts are being sent and what the model returns. This is invaluable for debugging output validation failures or unexpected model behavior. Set up tracing early.

## Key Themes Across This Section

**Explicitness scales quality:** Vague prompts produce vague outputs. Explicit prompts with examples and schemas produce predictable, high-quality outputs.

**Schemas are contracts:** Define your output schema (Pydantic, JSON Schema, dataclass) before writing your prompt. Your contract with the model comes from the schema, not hope.

**Validation + retries = reliability:** Validate all LLM output against a schema. Build automatic retries into your chain so occasional model failures recover gracefully.

**Measure and iterate:** Use LangSmith or similar tools to trace requests, measure extraction rates, retry rates, and quality. Iterate prompts based on data, not guesses.
