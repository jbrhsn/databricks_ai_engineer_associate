# Chain Components and I/O Specifications — Interview Prep

**Section:** Design Applications | **Role target:** Senior Engineer, Solutions Architect, LLM Application Engineer

## Core Conceptual Questions

These test whether you understand the fundamentals.

| Question | Key points to cover | Common weak-answer trap |
|---|---|---|
| **What is a chain in the context of LLM applications, and why do I/O schemas matter?** | A chain is a composition of components (prompt, model, parser, tool) where each step's output feeds into the next step's input. I/O schemas enforce type contracts at each step, preventing silent failures when a component receives unexpected data. Without schemas, type mismatches propagate downstream and cause errors far from their source. | Saying "a chain is just calling the LLM multiple times" without mentioning the schema / contract aspect. Or conflating chains with agents—agents have decision logic, chains are deterministic sequences. |
| **What is the Runnable interface, and why is it important?** | Runnable is LangChain's universal abstraction for any component that takes input and produces output. All components (models, prompts, parsers, tools, chains) implement it, providing a consistent API: invoke, batch, stream, astream. This enables composition with the pipe operator `\|` and ensures all Runnable methods work on composed chains. | Saying "Runnable is just a function that calls the LLM" or treating it as a detail rather than a foundational abstraction. Missing the point that Runnable enables composability and schema-aware execution. |
| **What is the difference between a parser and a prompt, and why do you need both?** | A prompt is a template that formats input data into natural language instructions for the LLM. A parser transforms the LLM's output (raw text) into a structured, validated object. Both are necessary: the prompt shapes what the LLM sees; the parser ensures downstream components get valid, typed data. | Conflating the two or saying "the LLM will just output the right format, so a parser is optional." Parsers are load-bearing; they enforce data contracts. |

## Applied / Scenario Questions

**Q: You are building a customer ticket routing system. Tickets come in as unstructured text. The LLM must classify the ticket (urgent/normal), extract the intent, and route to a tool. The LLM occasionally outputs malformed JSON. How would you design this chain?**

**Strong answer framework:**
- Start by defining the I/O contracts: input is `{"ticket_text": str}`, output is a Pydantic model with `category`, `intent`, `requires_escalation`.
- Use a Pydantic model to represent the output schema; this is non-negotiable for type safety.
- Chain: `prompt | model | parser`. The parser should be `PydanticOutputParser` or `OutputFixingParser` with retry logic to handle occasional LLM format drift.
- Test the chain's `input_schema` and `output_schema` to verify compatibility with upstream and downstream systems.
- Use `batch()` to route multiple tickets in parallel, not `invoke()` in a loop.
- Tradeoff awareness: Retry logic adds latency; tune `max_retries` based on your SLA (0 for fail-fast debugging, 1–2 for production).

**Q: Your team is debating whether to use LangChain chains or LangGraph for a multi-step workflow. What factors would you consider?**

**Strong answer framework:**
- Use chains if: the flow is deterministic (same steps always), no branching, no long-running state, no human interrupts needed. Chains are lighter-weight and easier to compose.
- Use LangGraph if: the flow is dynamic (conditional branching), needs persistence across failures, needs human-in-the-loop interrupts, or needs to maintain state across multiple turns.
- Example: A classification + extraction chain → use LangChain. A multi-turn agent that plans, acts, and checks → use LangGraph.
- Tradeoff: LangGraph has more setup but buys you durable execution and observability; LangChain is quick to prototype but less powerful for complex workflows.

**Q: A chain in production is occasionally failing with a TypeError that says a downstream tool received `name` instead of `user_name`. The LLM output is correct, so the problem is not there. Where would you look, and how would you fix it?**

**Strong answer framework:**
- The error is a schema mismatch between the parser output and the tool input. The parser outputs `name`, but the tool expects `user_name`.
- Debug by printing `parser.output_schema` and `tool.input_schema` to see the mismatch.
- Fix: Add a `RunnableLambda` to transform the parser output: `chain = prompt | model | parser | RunnableLambda(lambda x: {"user_name": x.name, ...}) | tool`.
- Alternatively, modify the parser to output the correct field names, or modify the tool to accept both names (less preferred—adds confusion).
- Prevention: Always check schema compatibility at compose time before deploying to production.

## System Design / Architecture Questions (if applicable)

**Q: Design a production-grade system for processing customer feedback at scale. Customers submit free-text feedback; you must classify sentiment, extract actionable insights, and store results. Feedback volume is 10k/day. What chain design and component choices would you make?**

**Approach:**
1. **Clarify requirements:**
   - Input: unstructured text feedback
   - Output: structured records with sentiment (positive/negative/neutral), insight summary, and suggested action
   - Constraints: 10k/day volume, latency < 5s per feedback, reliability > 99%, cost-conscious
   - Must handle occasional LLM output format drift
   
2. **Propose structure:**
   - Define I/O schemas: input Pydantic model with feedback text, output model with sentiment, insights, action.
   - Chain: `prompt | model_with_structured_output | parser | validator`.
   - Use LangChain chains (not LangGraph) because the flow is deterministic.
   - Add a custom validator Runnable that checks outputs against business rules (e.g., sentiment must be non-empty).
   - Use `batch()` mode with batch size 100 to process 10k feedback in ~100 API calls instead of 10k.
   - Store I/O schemas in a central schema registry (e.g., Databricks Unity Catalog) for audit and evolution tracking.
   - Use LangSmith to log all invocations, including schema version and parse success/failure rate.

3. **Justify choices and name tradeoffs explicitly:**
   - Tradeoff 1: Batch processing vs streaming. Chosen batch because volume is high and latency SLA allows it. Streaming would be more complex and not needed.
   - Tradeoff 2: Retry logic on parser vs fail-fast. Chosen 1 retry because feedback quality is variable and we want to recover from minor format issues.
   - Tradeoff 3: LangChain vs LangGraph. Chosen LangChain because the flow is deterministic; LangGraph would add overhead without benefit.
   - Tradeoff 4: Structured output model vs JSON parser. Chosen structured output (if available with your model) to reduce parsing errors at source.

## Vocabulary That Signals Expertise

Use these terms naturally — don't force them:

- **"I/O contract"** — When discussing how components should communicate, reference the explicit contract between output and input schemas. E.g., "We need to ensure the I/O contract between the parser and the router is validated before composition."
- **"Schema enforcement"** — When talking about reliability, mention that explicit schema enforcement (not optional validation) is what prevents silent failures.
- **"Component compatibility"** — When designing chains, discuss checking component compatibility at compose time: "I'd verify that the model's output schema matches the parser's input schema."
- **"Runnable abstraction"** — When discussing composability, reference the Runnable interface: "All components implement the Runnable interface, so we can compose them with the pipe operator."
- **"Output fixing"** — When discussing parser robustness, mention OutputFixingParser or equivalent retry logic: "We use output fixing so occasional LLM format drift doesn't break the chain."
- **"Batch mode"** — When discussing performance at scale, distinguish batch from invoke: "We use batch mode to process 10k items in a single API call instead of 10k sequential invokes."

## Vocabulary That Signals Weakness

Avoid these — they signal outdated or shallow understanding:

- **"The LLM will always format it correctly."** — This ignores empirical evidence of format drift and signals over-reliance on the model without defenses.
- **"I'll just catch the exception in the application code."** — This is reactive and scatters error handling; it signals ignorance of schema-first design.
- **"Chains are just wrappers around function calls."** — This misses the core value: schema validation, composability, streaming, and batch support.
- **"We don't need types; we'll validate with unit tests."** — This conflates unit testing with type safety; they're orthogonal. Types catch issues at design/runtime; tests catch logic errors.
- **"LangGraph is better than LangChain chains."** — This is tool-agnostic comparison; both have tradeoffs. Saying one is universally better signals unfamiliarity with the tradeoffs.
- **"I parse the LLM output however I want in my app."** — This signals a lack of standardization and explains why many LLM apps are brittle.

## STAR Answer Frame

**Situation:** At my previous role, we were building a classification system where the LLM had to categorize support tickets into 10+ categories. We did have a chain with a prompt and model, but we weren't explicitly enforcing the output format.

**Task:** I was responsible for improving the reliability of the classification step, which was occasionally failing with cryptic errors downstream when the LLM output format drifted.

**Action:** I led the effort to redesign the chain with explicit I/O contracts: (1) Created a Pydantic model representing the classification output (category, confidence, fallback_category). (2) Added a `PydanticOutputParser` with `OutputFixingParser` wrapping it, to handle LLM format drift gracefully. (3) Tested the chain's `input_schema` and `output_schema` to verify compatibility with the router that consumed the output. (4) Updated the chain to use `batch()` mode for 1000s of tickets, reducing API calls by 95%. (5) Integrated logging into LangSmith so we could monitor parse success rate and identify patterns in LLM format errors.

**Result:** Error rate dropped from ~5% (silent failures downstream) to <0.2% (caught immediately at the parser with clear diagnostics). Processing time for 10k tickets dropped from 2 hours (sequential) to 15 minutes (batch). New team members could now understand the data contract just by reading the Pydantic model, reducing onboarding time by a week. The same pattern was adopted across the team for all downstream chains.

## Red Flags Interviewers Watch For

- **Designing a chain without mentioning I/O schemas or contracts.** This signals they haven't thought deeply about data flow or type safety.
- **Treating parsing as optional or an afterthought.** Production-ready systems always have parsers; parsing is not nice-to-have.
- **Conflating chains with agents.** Agents have decision logic and branching; chains are deterministic. Confusion here suggests shallow understanding of orchestration patterns.
- **Recommending error handling in application code rather than at the component level.** This is reactive and indicates a lack of schema-first thinking.
- **Saying "the LLM will always format correctly" or "we'll handle edge cases later."** These are red flags for under-architected systems.
- **Not distinguishing between LangChain chains, LangGraph, and plain function composition.** Strong candidates understand the tradeoffs and make intentional choices.
- **Inability to explain what the Runnable interface is or why it matters.** This is core to LangChain's design; knowing it signals familiarity with the ecosystem.
