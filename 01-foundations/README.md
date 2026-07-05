# Section 01 — Foundations

**Estimated time:** 8 hours
**Prerequisites:** Python 3.x basics, command line familiarity, git basics (see `00-roadmap/learning-roadmap.md` for the full prerequisite list)
**Exam mapping:** Databricks GenAI Engineer Associate — Foundations domain (~15% of exam); also provides vocabulary and conceptual grounding for all other exam domains

## Overview

This section builds the conceptual and environmental foundation for everything that follows. You will
set up your Databricks workspace, confirm your Python environment, and develop a solid mental model
of how large language models work — including prompting mechanics, tokenization, and model selection
criteria. Without this foundation, the application-development and deployment sections will feel
like pattern-matching without understanding.

## Learning Outcomes

By completing this section, you will be able to:

- Configure a Databricks workspace and notebook environment suitable for GenAI development
- Explain how transformer-based LLMs process input, generate output, and why sampling parameters matter
- Write effective prompts using zero-shot, few-shot, and chain-of-thought techniques
- Control output format using system prompts and structured output constraints
- Select an appropriate model for a given task based on size, speed, cost, and accuracy trade-offs
- Decompose a complex user request into a sequence of focused LLM sub-tasks

## Topics Covered

- Python environment verification and Databricks Runtime compatibility
- Databricks workspace UI, clusters, notebooks, Git integration, and Unity Catalog basics
- LLM mechanics: transformers, tokenization, embeddings, context windows, sampling parameters
- Foundation Model APIs on Databricks (pay-per-token endpoints for hosted models)
- Prompt anatomy: system prompt, user message, assistant response
- Zero-shot, few-shot, and chain-of-thought prompting
- Structured output: JSON mode, schema enforcement, Pydantic integration
- Task taxonomy: classification, extraction, generation, summarization
- Model selection: open vs. closed models, size vs. speed vs. cost trade-offs
- Problem decomposition for multi-step LLM workflows

## How This Section Fits

This is the entry point for the entire repo. Section 02 (Data Preparation) assumes you can navigate
the Databricks workspace and understand embeddings at a conceptual level. Section 03 (Application
Development) assumes you can write effective prompts and understand context windows. Skipping this
section means working with borrowed vocabulary — possible, but error-prone.

## Study Tips

- **Do the environment setup before anything else.** It is tempting to read ahead, but you will
  retain the LLM concepts better if you can run small experiments in a notebook as you go.
- **The LLM core concepts chapter is the most important in this section** for exam purposes.
  Tokenization, context windows, and temperature/sampling appear as distractors in many exam
  questions — knowing them precisely prevents careless mistakes.
- **Prompting is a skill, not a topic.** Read the chapter, then immediately try the techniques on
  real tasks in a notebook. Passive reading of prompting patterns does not build the judgment you
  need for exam scenarios that test whether a given prompt would actually work.
- **Don't over-invest in the setup chapters.** They are supporting content, not exam-scored. Get
  your environment working and move on — you will learn the Databricks UI organically as you build
  things in later sections.

## Chapters / Modules

- [ ] [01 Prerequisites and Setup](./01-prerequisites-and-setup/) — 2.5 hrs
- [ ] [02 GenAI and Prompting Fundamentals](./02-genai-and-prompting-fundamentals/) — 5.5 hrs
