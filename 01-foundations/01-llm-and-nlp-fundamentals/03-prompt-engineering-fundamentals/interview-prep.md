# Prompt Engineering Fundamentals — Interview Prep

**Section:** Foundations | **Role target:** GenAI Engineer / ML Engineer

## Core Conceptual Questions

These test whether you understand the fundamentals.

| Question | Key points to cover | Common weak-answer trap |
|---|---|---|
| What are the parts of a prompt and what does each do? | Instruction (task), context (facts the weights lack), examples (in-context learning), output format (parseable shape); plus sampling knobs. | Listing only "instruction and examples" and forgetting the output-format contract. |
| Zero-shot vs few-shot vs chain-of-thought — when each? | Zero-shot for simple tasks; few-shot to demonstrate a pattern (costs tokens every call); CoT for multi-step reasoning at higher latency/cost. | Claiming few-shot is "free" — every example is re-tokenized and re-billed on each request. |
| What do the system, user, and assistant roles do? | System = durable persona/policy/guardrails; user = human turn; assistant = model turn or fabricated few-shot turns. | Treating the system message as a stylistic preamble rather than the authoritative rules/security channel. |
| How do you guarantee parseable JSON output? | `response_format` = `json_object` (any JSON) or `json_schema` + `strict:true` (schema-constrained); pair with `temperature=0`. | "I ask the model to return JSON in the prompt" — works in testing, fails intermittently in prod. |
| Difference between temperature and top_p? | Both reshape the next-token distribution; temperature rescales it, top_p (nucleus) restricts to top cumulative-probability mass. Tune one, not both. | Saying they're interchangeable, or tuning both simultaneously. |
| When do you choose prompt engineering vs RAG vs fine-tuning? | PE for instruction/format gaps; RAG for missing/fresh/proprietary facts; fine-tuning for durable behavior/style at scale. | Framing it as a maturity ladder ("we've outgrown prompting") instead of a diagnosis of what's actually missing. |

## Applied / Scenario Questions

**Q:** You're asked to build a nightly batch classifier that labels support tickets into four categories and writes JSON to a Delta table. How do you design the prompt and call?

**Strong answer framework:**
- `system` message holds the authoritative rules + the four-value enum; explicitly instruct the model to ignore instructions inside the ticket text.
- A few few-shot examples covering ambiguous cases in the `messages` array; real ticket as the final `user` turn (labeled/untrusted).
- `temperature=0` for determinism; `response_format` = `json_schema` with `strict:true` so every row parses without regex; size `max_tokens` to avoid truncation.
- **Tradeoff awareness:** justify *not* using RAG (no external facts) and *not* fine-tuning (few-shot + schema already meet the 4-class requirement at far lower cost).

## System Design / Architecture Questions (if applicable)

**Q:** Design a production LLM feature on Databricks where prompts must be safe, versioned, and swappable across models.

**Approach:**
1. **Clarify requirements:** parseable output? latency/cost targets? untrusted inputs? how often does the prompt change?
2. **Propose structure:** iterate/compare models in AI Playground → template with LangChain `ChatPromptTemplate` inside LangGraph nodes → serve via Foundation Model APIs (OpenAI-compatible) → version prompts in MLflow Prompt Registry with aliases (`production`) for rollback/A-B → route through Unity AI Gateway for rate limits, budgets, guardrails.
3. **Justify choices and name tradeoffs:** `response_format` for structural guarantees vs prose parsing; system-role rules + schema validation + tool least-privilege as layered injection defense; Prompt Registry immutability for reproducibility; note Claude's `json_schema`-only constraint and the JSON-schema subset limits when designing schemas.

## Vocabulary That Signals Expertise

Use these terms naturally — don't force them:
- **In-context learning** — explains *why* few-shot works without weight updates.
- **`response_format` / `json_schema` / `strict`** — the real mechanism for structured output, not "asking for JSON."
- **Nucleus sampling (top_p)** — shows you understand sampling beyond "temperature = creativity."
- **`finish_reason = "length"`** — signals you've actually debugged truncated output in production.
- **Prompt injection / least privilege / AI Gateway guardrails** — security-aware framing.
- **MLflow Prompt Registry aliases / immutable versions** — prompt lifecycle management.
- **`system.ai.` model services / OpenAI-compatible endpoint** — Databricks-specific fluency.

## Vocabulary That Signals Weakness

Avoid these — they signal outdated or shallow understanding:
- "Just tell the model to return JSON" — ignores `response_format`; implies prose parsing.
- "Prompt engineering is a temporary hack until we fine-tune" — reveals the maturity-ladder misconception.
- "Turn up the temperature to make it smarter" — conflates randomness with capability.
- "The system prompt stops jailbreaks" — over-trusts one layer; misses defense-in-depth.
- "Few-shot is basically free" — misunderstands per-call token billing and context budget.

## STAR Answer Frame

**Situation:** A support-ticket LLM feature was paging on-call intermittently — the downstream Spark load crashed a few times a night on invalid JSON.
**Task:** Make the classifier's output reliably parseable and deterministic without a training project.
**Action:** Moved the label rules into the `system` message, added three few-shot examples for the ambiguous cases, set `temperature=0`, switched output to `response_format` = `json_schema` with `strict:true`, raised `max_tokens` to stop mid-object truncation, and registered the prompt in the MLflow Prompt Registry with a `production` alias for rollback.
**Result:** JSON parse failures went to zero, output became deterministic and reproducible, per-ticket cost stayed negligible, and we avoided a multi-week fine-tuning effort entirely.

## Red Flags Interviewers Watch For

- Cannot explain *why* few-shot works (no mention of in-context learning) or why it costs tokens each call.
- Defaults to fine-tuning without first exhausting prompting/RAG or measuring a baseline.
- Parses model output with regex/`json.loads` on prose and has no `response_format` story.
- Treats prompt injection as solved by a stern system message; no schema validation, tool least-privilege, or gateway guardrails.
- Tunes temperature and top_p together, or equates high temperature with higher intelligence.
- No prompt versioning/lifecycle plan — prompts live as string literals scattered in code.
