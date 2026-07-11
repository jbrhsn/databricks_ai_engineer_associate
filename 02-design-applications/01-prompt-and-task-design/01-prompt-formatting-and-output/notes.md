# Prompt Formatting and Output Specification

**Section:** Design Applications | **Module:** Prompt and Task Design | **Est. time:** 2.5 hrs | **Exam mapping:** Domain 1 – Design Applications (14%)

---

## TL;DR

Prompt formatting and output specification are the foundation of reliable LLM applications. A well-structured prompt with clear instructions, examples, and format constraints dramatically improves model accuracy and enables structured output that your application can parse reliably. **The one thing to remember: explicit prompts with schema-validated outputs beat vague requests relying on natural language parsing.**

---

## ELI5 — Explain It Like I'm 5

Imagine you're teaching a new friend how to bake cookies. If you just say "make cookies," they might guess at the ingredients, skip steps, or use the wrong oven temperature—and the results will be unpredictable. But if you give them a recipe card with exact measurements, step-by-step instructions, and a photo of what good cookies should look like, they'll follow it precisely and produce consistent results. Prompts work the same way: vague instructions like "summarize this" lead to inconsistent summaries, but a detailed prompt that names the format ("JSON with fields 'title', 'bullets', 'takeaway'"), shows an example, and specifies constraints (e.g., "max 3 bullets") produces predictable, machine-readable output. The misconception is that LLMs are mind readers—they're not. The more explicit your instructions and examples, the more reliable the output becomes.

---

## Learning Objectives

By the end of this chapter you will be able to:
- [ ] Design prompt templates with variables, role definitions, and context
- [ ] Apply prompt engineering techniques (few-shot, chain-of-thought, role-based prompting) to improve output quality
- [ ] Define and validate output schemas using Pydantic, dataclasses, or JSON Schema
- [ ] Implement structured output strategies (ProviderStrategy vs. ToolStrategy) in LangChain agents
- [ ] Diagnose and recover from prompt failures using error handling and retry logic

---

## Visual Overview

### Prompt Execution Pipeline

```
User Input ──► Prompt Template ──► LLM ──► Raw Output
                 (role, context,         (text, tokens)
                  examples, format)
                       │
               ┌───────┴────────┐
               │                │
            Parsed Output   Validation
            (structured)    Schema Check
               │                │
               └───────┬────────┘
                       │
                    Result
                  (validated or error)
```

### Output Strategy Decision Tree

```
Need structured output?
├── Yes, model supports native output? (OpenAI, Claude, Gemini, Grok)
│   ├── Yes ──► Use ProviderStrategy (native API support)
│   └── No  ──► Use ToolStrategy (tool calling fallback)
└── No ──► Plain text / natural language parsing
```

### Few-Shot Prompt Structure

```
System Role
    │
Instruction (what to do)
    │
Context (background data)
    │
Examples (1–3 demonstrations)
    │  ├── Input example 1 ──► Output example 1
    │  ├── Input example 2 ──► Output example 2
    │  └── [optional] Input example 3 ──► Output example 3
    │
Output Format (schema, constraints)
    │
User Query
```

---

## Key Concepts

### Prompt Templates and Parameterization

A prompt template is a reusable, parameterized instruction with placeholders for dynamic context. Instead of hard-coding text for every request, templates separate static instructions from variable data. Mechanisms: templates use named placeholders (e.g., `{variable_name}`) or positional slots; the application fills them at runtime with actual user data, context, or retrieved knowledge. In LangChain, `PromptTemplate` (for text) and `ChatPromptTemplate` (for multi-turn conversations) handle this; they accept a list of `Message` objects (system, human, assistant) and merge in variable values at invocation time. Use templates to ensure consistency across API calls and to reduce prompt engineering overhead.

### Few-Shot Prompting

Few-shot prompting includes 1–3 labelled input–output examples in the prompt to guide the model's behavior without fine-tuning. Mechanism: models are trained on vast corpora of example-label pairs; providing in-context examples activates the model's pattern-matching capability more directly than abstract instructions. Few-shot outperforms zero-shot (no examples) for complex tasks like classification, extraction, or reasoning, but risks overfitting if examples are unrepresentative. In LangChain, use `FewShotChatMessagePromptTemplate` to inject examples into a `ChatPromptTemplate`; pair it with an `ExampleSelector` to choose relevant examples dynamically. Few-shot works best when examples are diverse, representative, and labeled clearly.

### Role and System Prompting

A system prompt defines the LLM's role (e.g., "You are a data analyst") and high-level behavior before the user's query. Mechanism: transformers process the entire message sequence; a system message at the start of the sequence establishes framing and constraints that influence all subsequent reasoning. Role prompts improve coherence by anchoring the model's response style and scope. In LangChain agents, the `system_prompt` parameter in `create_agent` sets the system message; in `ChatPromptTemplate`, the first message is typically a `SystemMessage`. Effective system prompts are brief, specific, and role-focused (not a list of dos and don'ts).

### Structured Output Specification

Structured output specifies the exact format (JSON schema, fields, types) that the model must return. Mechanism: the model's final output is constrained to match a predefined schema before being returned to the application; if the output is malformed, validation fails or a retry is triggered. LangChain supports two strategies: `ProviderStrategy` (uses the model's native structured output API if supported, e.g., OpenAI's GPT-4) and `ToolStrategy` (uses tool calling as a fallback for models without native support). Schemas can be defined as Pydantic `BaseModel` subclasses, Python dataclasses, `TypedDict`, or raw JSON Schema. Structured output eliminates post-processing and ensures type safety.

### Output Parsing and Validation

An output parser extracts structured data from the model's raw response and validates it against a schema. Mechanism: parsers decode JSON, invoke validation logic (e.g., Pydantic field checks), and raise informative errors if validation fails. LangChain's `OutputParser` base class defines the interface; subclasses include `JsonOutputParser`, `PydanticOutputParser`, and `OutputFixingParser`. `OutputFixingParser` wraps another parser and retries with error feedback if parsing fails. Use output parsing when natural language must be converted to application-consumable types (dictionaries, lists, objects).

### Error Handling and Retry Strategies

When structured output fails validation or the model produces malformed output, retry mechanisms attempt recovery. Mechanism: validation errors trigger a feedback loop where the error message and original output are sent back to the model with instructions to fix the mistake; the model regenerates output, often successfully. LangChain's `ToolStrategy` has a `handle_errors` parameter that controls retry behavior: `True` (default) retries all errors, a string retries with custom feedback, or a callable applies custom logic. Use retries to improve robustness; combine with exponential backoff to avoid rate limits.

### Constraint-Aware Prompting

Constraints are limits on output format, length, or content that guide the model and prevent wasteful token usage. Mechanism: constraints like "respond in fewer than 100 tokens," "only use JSON," or "cite your sources" prime the model to optimize for the stated goal during generation. Token-aware constraints reduce inference cost; format constraints enable reliable parsing. In prompts, constraints are best stated early and reinforced in system messages or examples. Constraints should be specific ("JSON with fields 'summary', 'confidence', 'sources'") rather than vague ("be concise").

### Key Parameters / Configuration Knobs

| Parameter | What it controls | Decision rule |
|---|---|---|
| **Temperature** | Randomness of output (0 = deterministic, 1 = high variance) | Use 0–0.3 for fact-based tasks (extraction, classification); 0.7–1.0 for creative tasks. Keep ≤0.5 for structured output (reliability over variety). |
| **Max Tokens** | Upper bound on output length | Set to ~150–200% of expected output; too low truncates useful data, too high wastes cost. Pair with prompt constraints ("keep answer under 50 tokens"). |
| **Top-K / Top-P** | Sampling strategy (nucleus sampling) | Leave at default unless you need to suppress rare/nonsensical outputs. Top-P 0.9–0.95 is safe for most tasks. |
| **Stop Sequences** | Tokens that terminate generation early | Use `["\n\n", "End"]` to prevent overly long outputs; use `["}"]` for JSON to stop after closing brace. |
| **Prompt Format (System + Few-shot)** | Structure and content of the prompt | Always use system message + role. Include 1–3 representative examples for complex tasks. Keep examples concise (< 200 tokens each). |
| **Schema Type (Pydantic / JSON / Dataclass)** | How output format is specified | Use Pydantic if type hints + validation are useful (application will access fields); use JSON Schema if minimal dependencies; use dataclass for lightweight Python objects. |

### Worked Example: Requirement → Decision

**Given:** A content moderation team needs to classify customer reviews as "positive," "negative," or "neutral" and extract key phrases. Currently, manual review takes hours per week. They want an LLM-powered classification system with reliable, parseable output.

**Step 1 — Identify the goal:** Classify reviews into three sentiment categories and extract 2–4 key phrases per review without human rework.

**Step 2 — Define inputs:** Customer review text (50–500 tokens), review category context (e.g., "product review" vs. "customer service"), and optional reference examples.

**Step 3 — Define outputs:** Structured JSON with fields `sentiment: str` (one of "positive", "negative", "neutral"), `confidence: float` (0–1), and `key_phrases: list[str]` (2–4 short phrases).

**Step 4 — Apply constraints:** Must be parseable by Python JSON parser; must not require human post-processing; must handle edge cases (sarcasm, mixed sentiment). Inference should complete in <2 seconds per review.

**Step 5 — Select the approach:** Define a Pydantic model with the output schema, use LangChain's `ProviderStrategy` with an OpenAI model (has native structured output), and wrap the response with `OutputFixingParser` to retry if validation fails. Rationale: Pydantic + ProviderStrategy ensures type safety and reliability; `OutputFixingParser` handles edge cases. Alternative (if using a model without native structured output support): use `ToolStrategy` with tool calling instead—slightly higher latency but same reliability.

---

## Implementation

### Scenario 1: Structured Output with LangChain ProviderStrategy

```python
# Scenario: Build a product review classifier that always returns valid JSON
# with sentiment, confidence, and key phrases—no manual parsing needed.

from pydantic import BaseModel, Field
from typing import Literal
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

class ReviewAnalysis(BaseModel):
    """Structured output for review classification."""
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Overall sentiment of the review"
    )
    confidence: float = Field(
        ge=0, le=1, 
        description="Confidence score from 0 to 1"
    )
    key_phrases: list[str] = Field(
        description="2–4 key phrases that justify the sentiment"
    )

# Initialize model with native structured output support
model = init_chat_model("openai:gpt-5.5")

# Create agent with structured output schema
agent = create_agent(
    model=model,
    response_format=ReviewAnalysis,  # Auto-selects ProviderStrategy
    system_prompt=(
        "You are a product review analyst. Classify the review's sentiment "
        "and extract key phrases that support your classification. "
        "Be precise and conservative with confidence scores."
    )
)

# Invoke with a review
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "The product arrived on time and works great, but the packaging was damaged."
    }]
})

# Structured response is directly accessible and validated
response = result["structured_response"]
print(f"Sentiment: {response.sentiment}")
print(f"Confidence: {response.confidence}")
print(f"Key phrases: {response.key_phrases}")
# Output:
# Sentiment: positive
# Confidence: 0.85
# Key phrases: ['arrived on time', 'works great', 'damaged packaging']
```

### Anti-pattern: Unstructured Output with Manual Parsing

```python
# Anti-pattern: Asking the model for JSON but not validating the schema,
# leading to silent parsing failures and runtime crashes.

from langchain.chat_models import init_chat_model

model = init_chat_model("openai:gpt-5.5")

# Vague prompt without explicit schema
response = model.invoke([{
    "role": "user",
    "content": "Analyze this review and return JSON with sentiment, confidence, and key phrases: 'Great product!'"
}])

# Raw response is unstructured text
raw_text = response.content
print(raw_text)
# Output: '{"sentiment": "positive", "confidence": 0.9, "phrases": ["great", "product"]}'

# Manual parsing is fragile and error-prone
import json
try:
    data = json.loads(raw_text)
    sentiment = data["sentiment"]  # Works by luck here
    confidence = data.get("confidence")  # Missing error handling
    phrases = data.get("key_phrases", data.get("phrases", []))  # Key name differs
except (json.JSONDecodeError, KeyError) as e:
    print(f"Parsing failed: {e}")  # Silent failures accumulate

# What breaks: If the model returns invalid JSON, uses different field names,
# or includes extra fields, the parser breaks. No type validation occurs.
# Schema evolves over time, and code doesn't adapt.
```

**Explanation of what breaks:** The anti-pattern relies on hope—hope that the model returns valid JSON, hope that field names stay consistent, hope that there's no extra text. In production, models occasionally deviate: they add explanatory text before the JSON, use `"phrases"` instead of `"key_phrases"`, or return an incomplete structure. Each deviation causes a silent parsing failure or a crash. The corrected version uses LangChain's structured output support, which guarantees schema compliance and raises clear validation errors with retry opportunities.

### Scenario 2: Few-Shot Prompting with Examples

```python
# Scenario: Classify support tickets into categories (bug, feature_request, question)
# using examples to improve accuracy on edge cases.

from langchain.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate
from langchain.chat_models import init_chat_model

# Define representative examples
examples = [
    {
        "ticket": "The app crashes when I upload a file over 100MB.",
        "category": "bug"
    },
    {
        "ticket": "Can we add dark mode support?",
        "category": "feature_request"
    },
    {
        "ticket": "How do I reset my password?",
        "category": "question"
    },
]

# Build a few-shot prompt template
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "Ticket: {ticket}"),
    ("ai", "Category: {category}")
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="You are a support ticket classifier. Classify the ticket into one of: bug, feature_request, question.",
    suffix="Ticket: {ticket}\nCategory:",
    input_variables=["ticket"]
)

# Create full prompt with examples
full_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful support ticket classifier."),
    ("placeholder", "{examples}"),
    ("human", "{ticket}")
])

# Invoke with a new ticket
model = init_chat_model("openai:gpt-4")
response = model.invoke(full_prompt.format_messages(
    examples=few_shot_prompt.format(ticket="I can't log in with my email."),
    ticket="I can't log in with my email."
))

print(response.content)
# Output: "Category: question"
```

---

## Common Pitfalls & Misconceptions

**Vague Prompts Expect Miracles:** Beginners assume that LLMs will magically infer what format you want if you just ask for "JSON" or "a list." This leads to inconsistent output where the model sometimes returns valid JSON and sometimes returns prose. The correct mental model is: the model is a pattern matcher that responds to explicit instructions. Specificity in the prompt directly translates to specificity in the output. Always include examples and format specs.

**Ignoring Token Constraints:** Many prompts don't mention output length, so the model generates as much text as the token limit allows, wasting cost and causing truncation on downstream systems. Token limits are not just about saving money—they're about predictability. The correct mental model is: `max_tokens` and prompt-level constraints ("keep your answer under 50 tokens") are **output design** decisions, not optimization afterthoughts. Set them first, test, then adjust.

**Trusting Raw LLM Output Without Validation:** Beginners write code that assumes the model will always return valid JSON, matching field names, and correct types. When the model fails (which it will occasionally), the parser crashes or silently corrupts data. The correct mental model is: **the model is not a function**—it's probabilistic and occasionally erratic. Always validate output against a schema; use retry logic and error handling to gracefully recover from failures.

**Over-Constraining Prompts:** Some teams add so many rules and edge cases to prompts that they become impossible to follow. A 500-token prompt about what NOT to do overwhelms the model. The correct mental model is: constraints are powerful, but quality comes from clear positive direction (what TO do) plus 1–3 representative examples, not a laundry list of negatives. Prioritize clarity and examples over exhaustive rules.

**Ignoring Provider Capabilities:** LangChain supports many model providers, each with different structured output support. Beginners assume all models work the same, then deploy code using ProviderStrategy on a model that doesn't support native structured output, causing silent fallbacks to ToolStrategy (slower, higher token usage). The correct mental model is: check your model's profile (supports native structured output? tool calling?) before deploying. LangChain's `init_chat_model` does this automatically, but be explicit if you're optimizing for latency or cost.

---

## Key Definitions

| Term | Definition |
|---|---|
| **Prompt Template** | A reusable instruction with named placeholders for dynamic variables; rendered at runtime by substituting actual values. |
| **Few-Shot Prompting** | Including 1–3 labelled input–output examples in a prompt to guide model behavior without fine-tuning. |
| **System Message** | The initial message in a conversation that defines the model's role, tone, and constraints. |
| **Structured Output** | Output constrained to a predefined schema (JSON, Pydantic model, dataclass) that is validated before being returned. |
| **ProviderStrategy** | LangChain's strategy for leveraging native structured output APIs (e.g., OpenAI, Claude, Grok). |
| **ToolStrategy** | LangChain's fallback strategy for structured output using tool calling when native APIs are unavailable. |
| **Output Parser** | A component that decodes the model's raw response and validates it against a schema. |
| **Token Budget** | The maximum number of tokens allocated for a model's output; controls cost and prevents runaway generation. |

---

## Summary / Quick Recall

- **Explicit beats implicit:** LLMs respond to specific, structured prompts better than vague requests. Invest in prompt engineering to improve accuracy and output reliability.
- **Few-shot examples anchor behavior:** 1–3 representative examples in a prompt outperform zero-shot for complex tasks like classification and extraction.
- **Schema validation is non-negotiable:** Always define and validate output schemas (Pydantic, JSON Schema, dataclass). Use retry logic to handle occasional failures gracefully.
- **ProviderStrategy > ToolStrategy when available:** If your model supports native structured output (OpenAI, Claude, Gemini), use it—it's faster and more reliable than tool-calling fallbacks.
- **Constraints are output design:** Token limits, format specs, and length targets should be set upfront as part of prompt design, not as afterthoughts.
- **System prompts frame understanding:** A clear, role-based system message significantly improves coherence and reduces hallucination.
- **Error handling is mandatory:** Models occasionally fail to parse, return malformed output, or violate schema. Build retries and error feedback into your application.

---

## Self-Check Questions

**Q1 — Recall:** What is the primary purpose of a prompt template?

A) To make prompts shorter  
B) To reuse prompts with different variable values at runtime  
C) To ensure all prompts are identical  
D) To remove the need for system messages  

<details><summary>Answer</summary>

**Correct: B**

Prompt templates separate static instructions from dynamic data, allowing the same prompt structure to be reused with different variable values (e.g., different user reviews, different product categories). This reduces redundancy and ensures consistency.

**Why A is wrong:** Brevity is a side effect, not the purpose. Templates enable reusability, which is the core benefit.

**Why C and D are wrong:** Templates don't enforce identical prompts—they enable variation through variables. Templates work *alongside* system messages, not as replacements.

</details>

---

**Q2 — Application:** You're building a customer feedback classifier. Your initial prompt says "Classify the feedback as positive or negative." The model is inconsistent: sometimes it returns "pos" / "neg," sometimes "good" / "bad." What should you do?

A) Increase the temperature to add randomness  
B) Use few-shot examples with exact labels in output and a structured output schema (ProviderStrategy or ToolStrategy)  
C) Add more constraints to the prompt ("Do not abbreviate")  
D) Use a different model  

<details><summary>Answer</summary>

**Correct: B**

Few-shot examples demonstrate the exact output format (e.g., "positive" / "negative"), and structured output schemas enforce this at the API level. This removes ambiguity and ensures consistency.

**Why A is wrong:** Increasing temperature adds randomness, which worsens inconsistency—the opposite of what you want.

**Why C is wrong:** Prose constraints ("Do not abbreviate") can be ignored. Schema validation cannot be.

**Why D is wrong:** A different model won't solve the core issue: lack of structure. The current model would be fine with proper output specification.

</details>

---

**Q3 — Application (Multi-Select): Which TWO of the following best improve structured output reliability? (Select all that apply.)

A) Using ToolStrategy for all models  
B) Defining output as a Pydantic model and using ProviderStrategy when native support is available  
C) Including `handle_errors=True` in ToolStrategy to retry on validation failures  
D) Removing system prompts to reduce token overhead  

<details><summary>Answer</summary>

**Correct: B and C**

**B** is correct: Pydantic models enforce schema, and ProviderStrategy uses native APIs (faster, more reliable) when available. LangChain automatically falls back to ToolStrategy for models without native support.

**C** is correct: `handle_errors=True` (the default) automatically retries when validation fails, catching occasional model mistakes and recovering gracefully.

**Why A is wrong:** ToolStrategy is a fallback for models without native structured output support. Using it universally is slower and more token-expensive than ProviderStrategy when native support is available.

**Why D is wrong:** Removing system prompts reduces clarity and increases hallucination. Token savings are minimal; clarity gains are large. Keep system prompts.

</details>

---

**Q4 — Analysis:** You have two strategies for structured output: ProviderStrategy (using a model's native API) and ToolStrategy (using tool calling). You're deploying for a cost-sensitive batch job that processes 10,000 reviews overnight. Which strategy should you choose, and why?

A) ProviderStrategy, because it's faster  
B) ToolStrategy, because it works with all models  
C) ProviderStrategy, because it uses fewer tokens in most cases  
D) Neither—use plain text parsing instead  

<details><summary>Answer</summary>

**Correct: C**

In a cost-sensitive batch scenario, ProviderStrategy is the right choice. While both strategies are reliable, ProviderStrategy typically uses fewer tokens because it doesn't require tool calling syntax overhead. Over 10,000 requests, this adds up. ProviderStrategy is also faster (lower latency), so your batch completes quicker.

**Why A is incomplete:** Speed is a secondary benefit in a batch job; cost is primary. Token efficiency (C) is the stronger argument.

**Why B is wrong:** True, ToolStrategy works with more models, but if you're choosing a model, pick one with native structured output support so you can use the more efficient ProviderStrategy.

**Why D is wrong:** Plain text parsing is fragile and defeats the purpose of structured output. Stick with a structured approach.

</details>

---

**Q5 — Analysis:** A production system processes customer reviews with a structured output schema. After 3 months, errors increase 30% because the model occasionally returns null for `key_phrases` even though the schema requires a non-empty list. Your options are: (A) increase max_tokens, (B) add a retry mechanism with custom error feedback, (C) switch to a different model, (D) remove the `key_phrases` field to simplify. Which is the best long-term solution?

A) Increase max_tokens  
B) Add a retry mechanism with custom error feedback  
C) Switch to a different model  
D) Remove the `key_phrases` field  

<details><summary>Answer</summary>

**Correct: B**

Adding a retry mechanism with custom error feedback (e.g., using `OutputFixingParser` or `handle_errors` with a callable) addresses the root cause: occasional model failures. The retry loop sends validation errors back to the model with instructions to fix the mistake, and it succeeds most of the time. This is the standard pattern in production LLM systems.

**Why A is wrong:** Increasing tokens won't reliably fix schema violations. The issue isn't truncation; it's that the model occasionally generates null.

**Why C is wrong:** Switching models is expensive and doesn't guarantee the problem disappears. The new model will also occasionally fail—it's the nature of LLMs.

**Why D is wrong:** Removing required fields compromises your data quality. The field is useful; the model just needs a retry pathway.

</details>

---

## Further Reading

- [LangChain Structured Output](https://python.langchain.com/docs/how_to/structured_output/) — *verified 2026-07-11* — Complete guide to ProviderStrategy, ToolStrategy, and error handling in LangChain agents.
- [LangChain Prompt Templates](https://python.langchain.com/docs/concepts/prompt_templates/) — *verified 2026-07-11* — Designing and parameterizing prompts with `PromptTemplate` and `ChatPromptTemplate`.
- [LangChain Output Parsers](https://python.langchain.com/docs/concepts/output_parsers/) — *verified 2026-07-11* — Converting and validating model output with built-in and custom parsers.
- [LangChain Few-Shot Prompting](https://python.langchain.com/docs/how_to/few_shot/) — *verified 2026-07-11* — Implementing few-shot examples dynamically with `FewShotChatMessagePromptTemplate` and `ExampleSelector`.
- [LangChain Models Overview](https://python.langchain.com/docs/concepts/models/) — *verified 2026-07-11* — Understanding model capabilities, profiles, and provider-specific features.
