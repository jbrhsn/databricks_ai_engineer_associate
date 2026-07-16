# Evaluation Metrics and LLM Choice

**Section:** 07 — Evaluation and Monitoring | **Module:** 01 — Evaluation | **Est. time:** 2.5 hrs | **Exam mapping:** Domain 6 — Evaluate the performance of a generative AI application (12%)

---

## TL;DR

Generative AI applications require a layered evaluation strategy: reference-based string metrics (BLEU, ROUGE) for offline regression testing, task-specific LLM-as-judge metrics for RAG and agents, and cost/latency profiling for LLM selection decisions. Databricks surfaces these through `mlflow.evaluate()` with `model_type="databricks-agent"` and a set of built-in judges (groundedness, chunk relevance, context sufficiency, correctness, relevance to query, safety). Choosing the wrong metric class for a task produces misleading pass/fail signals that mask real quality regressions. **The one thing to remember: match the metric class to the failure mode you actually care about — no single metric tells the whole story, and using the wrong one is worse than using none.**

---

## ELI5 — Explain It Like I'm 5

Imagine you hired five different judges to score a cooking competition. One judge only cares about whether you followed the recipe exactly (reference-based). Another only cares whether the dish is safe to eat (safety). A third checks whether the food you served actually came from the ingredients in front of you and not from a secret stash (groundedness). If you only use the recipe judge, a chef who creatively adapts a dish but makes it delicious will fail — even though your customers love it. The biggest misconception is that there is one "accuracy score" for AI: in reality, you need different judges for different failure modes, and the question you ask when picking a judge is always "what kind of mistake am I most afraid of right now?"

---

## Learning Objectives

By the end of this chapter you will be able to:
- [ ] Classify evaluation metrics by category (reference-based, reference-free/LLM-as-judge, task-specific) and explain when each category applies
- [ ] Map RAG-specific quality dimensions — groundedness, chunk relevance, context sufficiency, relevance to query, correctness — to the Databricks built-in judges and their required inputs
- [ ] Identify agent-specific metrics (task completion rate, tool call accuracy, trajectory correctness) and explain how they differ from single-turn RAG metrics
- [ ] Select an appropriate LLM from Databricks Foundation Model APIs given capability, cost, latency, and context-window constraints
- [ ] Implement `mlflow.evaluate()` with `model_type="databricks-agent"` and configure per-row and global judges, then interpret aggregate metric output

---

## Visual Overview

### Evaluation Metric Taxonomy

```
                    EVALUATION METRIC TAXONOMY
                    ──────────────────────────
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
     Reference-Based    Reference-Free    Task-Specific
     ─────────────────  ───────────────   ─────────────
     BLEU, ROUGE,       LLM-as-Judge      RAG metrics:
     METEOR, BERTScore  (GPT-4, Claude)     groundedness
                                            chunk_relevance
     Best for:          Best for:           context_sufficiency
     • regression       • open-ended        relevance_to_query
     • summarization      generation        correctness
       smoke tests      • no golden
                          answer exists   Agent metrics:
                                            task completion
                                            tool call accuracy
                                            trajectory score
```

### RAG Evaluation Pipeline Flow

```
User Query ──► Retriever ──► Retrieved Chunks ──► Generator ──► Response
     │               │               │                  │             │
     │         chunk_relevance  context_sufficiency  groundedness  relevance_
     │         (are chunks      (enough to answer    (answer from   to_query
     │          relevant?)       ground truth?)       context?)    (on-topic?)
     │                                                              │
     └──────────────────────────────────── correctness ────────────┘
                                          (matches expected facts?)
```

### LLM Selection Decision Tree

```
New task requirement
        │
        ▼
Is latency < 500ms required?
├── Yes ──► Is task simple (classification, extraction)?
│           ├── Yes ──► Llama-3.1-8B / GPT-OSS-20B (low cost, fast)
│           └── No  ──► Llama-3.3-70B / Claude Haiku 4.5 (balanced)
└── No  ──► Does task require >100K token context?
            ├── Yes ──► Gemini 2.5 Pro (1M ctx) / GPT-5.x (400K ctx)
            └── No  ──► Does task require fine-tuning?
                        ├── Yes ──► Choose model family with PT support
                        └── No  ──► Match capability tier to task complexity
```

### Offline vs Online Evaluation Workflow

```
Development Phase                    Production Phase
─────────────────                    ────────────────
Curated eval set                     Live traffic traces
        │                                    │
        ▼                                    ▼
mlflow.evaluate()               Production monitoring scorers
(offline judges)                (same judges, scheduled runs)
        │                                    │
        ▼                                    ▼
Compare runs in                    Alert on metric drift
MLflow Experiment UI               Dashboard in Delta table
```

---

## Key Concepts

### Reference-Based Metrics (BLEU, ROUGE, METEOR)

**What is it?** Reference-based metrics measure the string-level or n-gram overlap between a model's generated output and one or more human-written reference answers. BLEU (Bilingual Evaluation Understudy) counts matching n-grams and applies a brevity penalty; ROUGE (Recall-Oriented Understudy for Gisting Evaluation) focuses on recall of n-gram overlap and is the standard for summarization; METEOR adds synonym matching and stemming to reduce sensitivity to paraphrase.

**How does it work mechanistically?** For ROUGE-L, the algorithm computes the longest common subsequence (LCS) between the candidate and reference strings, then calculates precision and recall against the reference length. The F1 score of LCS overlap becomes the ROUGE-L score. Because these are deterministic string comparisons, they can be computed without a second model call and at very low cost. However, they fail completely when a correct answer is expressed differently from the reference (high paraphrase sensitivity).

**Where does it appear in Databricks/framework?** MLflow's `mlflow.evaluate()` supports ROUGE and BLEU via built-in `extra_metrics` when `model_type="text-summarization"` or when passed explicitly. These metrics are most useful as smoke-test regression gates in CI/CD pipelines, not as primary quality signals in production RAG evaluation.

---

### LLM-as-Judge (Reference-Free Evaluation)

**What is it?** LLM-as-judge uses a powerful, separate language model (the "judge") to score the quality of another model's output without requiring a human-written ground-truth answer. The judge is prompted with the original request, the generated response, and optionally context, then returns a binary (yes/no) or numeric score plus a written rationale.

**How does it work mechanistically?** The judge model receives a structured prompt that defines a scoring rubric (e.g., "Is this response grounded in the provided context? Answer yes or no and explain.") and the inputs. It generates a text response that the evaluation framework parses for the score token. Because the judge is itself a stochastic model, there are known biases: verbosity bias (longer answers score higher), position bias (first option in a pairwise comparison scores higher), and self-preference bias (a model tends to prefer its own outputs). Mitigations include using a different family model as judge, calibrating on human-labeled data (Cohen's Kappa), and running multiple judge passes with prompt variations.

**Where does it appear in Databricks/framework?** Databricks built-in judges (groundedness, relevance_to_query, safety, etc.) are all LLM-as-judge implementations backed by partner-powered models (including Azure OpenAI). They are invoked automatically when `model_type="databricks-agent"` is set in `mlflow.evaluate()`. The callable `databricks.agents.evals.judges` module also exposes individual judges for ad-hoc use.

---

### RAG-Specific Metrics

**What is it?** RAG evaluation decomposes quality into distinct pipeline stages: retrieval quality (did we fetch the right chunks?) and generation quality (did we produce a grounded, correct answer from those chunks?). The RAGAS framework popularized a four-metric taxonomy: faithfulness (groundedness), answer relevance, context precision (chunk relevance), and context recall (context sufficiency). Databricks implements closely analogous metrics as built-in judges.

**How does it work mechanistically?** Each judge targets a specific component:
- **Groundedness** (`groundedness`): asks the judge whether every claim in the response can be directly traced to a retrieved chunk. Inputs: request, response, retrieved\_context. No ground truth needed.
- **Chunk relevance** (`chunk_relevance`): scores each retrieved chunk individually for relevance to the query; aggregated into a precision score (% of chunks that are relevant). Inputs: request, retrieved\_context.
- **Context sufficiency** (`context_sufficiency`): asks whether the retrieved context contains enough information to produce the expected answer. Inputs: request, retrieved\_context, expected\_response or expected\_facts. Requires ground truth.
- **Relevance to query** (`relevance_to_query`): checks whether the response addresses the user's actual question (independent of how it was answered). Inputs: request, response. No ground truth needed.
- **Correctness** (`correctness`): compares the response to expected\_facts or expected\_response. Inputs: request, response, expected\_facts[] or expected\_response. Requires ground truth.

**Where does it appear in Databricks/framework?** These are first-class judges in Databricks Agent Evaluation, accessible via `mlflow.evaluate(..., model_type="databricks-agent")` and via the callable SDK (`from databricks.agents.evals import judges`). Aggregate run-level metrics are logged as `response/llm_judged/groundedness/rating/percentage` etc. and visible in the MLflow Experiment UI under the Overview and Model Metrics tabs.

---

### Agent-Specific Metrics

**What is it?** When evaluating multi-step agents (rather than single-turn RAG), quality cannot be measured solely at the response level. Agent-specific metrics assess the entire execution trajectory: whether the agent completed the assigned goal, whether it called the right tools with the right arguments, and whether the path it took was efficient.

**How does it work mechanistically?** Key agent metrics include:
- **Task completion rate**: binary or graded score for whether the agent accomplished the user's stated goal. Evaluated by an LLM judge or by checking a verifiable postcondition (e.g., a database row was created).
- **Tool call accuracy**: fraction of tool calls where the agent selected the correct tool AND passed correct arguments. Can be evaluated by comparing against a reference trajectory.
- **Trajectory correctness**: LLM-judged score for whether the sequence of steps (thought → action → observation) was logically coherent and necessary.
- **Step efficiency**: ratio of minimum steps needed to actual steps taken; penalizes unnecessary loops or redundant calls.

For multi-turn conversations, Databricks judges evaluate only the last entry in the conversation by default.

**Where does it appear in Databricks/framework?** Databricks Agent Evaluation supports trajectory evaluation through the `trace` column in the evaluation dataset. When a `trace` is provided, the evaluation framework can compute token counts, latency, and run judges over intermediate spans. In MLflow 3, `mlflow.genai.evaluate()` (replacing `mlflow.evaluate()` for agent workloads) extends this with conversation-level scorers.

---

### LLM Selection Criteria and Foundation Model APIs

**What is it?** LLM selection is the decision process for choosing which model to use as the backbone of a GenAI application, balancing capability (benchmark performance, reasoning depth), cost (per-token pricing), latency (time to first token, throughput), context window size, and fine-tuning availability.

**How does it work mechanistically?** Selection proceeds by constraining on non-negotiable requirements first (e.g., context window must exceed document length, latency SLA must be met), then optimizing for capability within the remaining options. Databricks Foundation Model APIs provide pay-per-token access to multiple model families: Meta Llama (open weights, fine-tune-able in Provisioned Throughput mode), Anthropic Claude (strong reasoning, instruction following), Google Gemini (long context, multimodal), and OpenAI GPT families. Each model tier serves different needs: small models (Llama-3.1-8B, GPT-OSS-20B) for high-throughput, cost-sensitive tasks; mid-tier (Llama-3.3-70B, Claude Haiku 4.5) for production RAG and agent routing; large frontier models (Claude Sonnet 5, GPT-5.x) for complex reasoning and agentic coding.

**Where does it appear in Databricks/framework?** Model endpoints are referenced by name (e.g., `databricks-meta-llama-3-3-70b-instruct`, `databricks-claude-sonnet-5`) in `mlflow.deployments.get_deploy_client("databricks").predict(endpoint=...)` calls, in `ai_query()` SQL function calls, and in LangChain/LangGraph integrations via the `ChatDatabricks` class. The AI Playground in the Databricks workspace allows interactive comparison of models on the same prompt before committing to a selection.

---

### Offline vs Online Evaluation

**What is it?** Offline evaluation runs judges against a curated, static evaluation set during development before deployment. Online (production) evaluation applies the same judges to live traffic traces after deployment, enabling continuous quality monitoring.

**How does it work mechanistically?** In offline mode, `mlflow.evaluate()` is called in a notebook or CI pipeline against a pandas DataFrame containing requests (and optionally expected responses). In online mode, MLflow 3 production monitoring schedules scorers to run asynchronously against logged production traces in the Databricks workspace. Because the same scorer definitions are reused for both modes, drift between development and production quality is directly measurable: if groundedness was 94% offline and is 80% online, something about production traffic differs from the eval set.

**Where does it appear in Databricks/framework?** Offline: `mlflow.evaluate(..., model_type="databricks-agent")` in notebooks. Online: Production monitoring under MLflow 3 (`/aws/en/mlflow3/genai/eval-monitor/production-monitoring`). Results in both modes are visible in the MLflow Experiment UI and can be persisted to Delta tables for dashboarding.

---

## Key Parameters / Configuration Knobs

| Parameter | What it controls | Decision rule |
|---|---|---|
| `model_type` in `mlflow.evaluate()` | Selects the evaluator backend and default judge set | Set to `"databricks-agent"` for any RAG or agent workload on Databricks; use `"text-summarization"` only if you need ROUGE/BLEU as primary metrics |
| `evaluator_config["databricks-agent"]["metrics"]` | Restricts which built-in judges run | Specify explicitly (e.g., `["groundedness", "chunk_relevance"]`) when you want to reduce judge cost or focus on a single failure mode; omit to run the full default suite |
| `evaluator_config["databricks-agent"]["global_guidelines"]` | Applies a guideline adherence judge to every row | Use for cross-cutting constraints (language, tone, rejection policy) that apply to all responses; do not use for per-question correctness — use `expected_facts` per row instead |
| `expected_facts[]` vs `expected_response` | Controls how the correctness judge compares outputs | Prefer `expected_facts` (minimal required facts) over `expected_response` (full reference answer); `expected_response` makes correctness overly strict and penalizes valid paraphrases |
| `data` column `retrieved_context[].content` | Enables groundedness and chunk relevance judges | Required when passing pre-generated outputs (no `model` argument); omit if using `model` argument and tracing is enabled — context is extracted from the trace automatically |
| `model` argument in `mlflow.evaluate()` | Passes a live callable, logged model URI, or endpoint | Use a live callable or endpoint URI during development iteration; use a registered Unity Catalog model URI for reproducible evaluation runs in CI |

---

## Worked Example: Requirement → Decision

**Given:** An enterprise team has built a RAG application that answers questions from internal HR policy documents. During user testing, some employees report receiving answers that sound confident but contradict the actual policy. The team needs to set up an evaluation pipeline before the next release.

**Step 1 — Identify the goal:** Detect responses that make claims not supported by the retrieved policy document chunks (hallucination / groundedness failures).

**Step 2 — Define inputs:** A curated evaluation set of 50 HR policy questions with known expected answers (`expected_facts`), plus the application's retrieved context and responses.

**Step 3 — Define outputs:** Per-question groundedness scores (`response/llm_judged/groundedness/rating`), context sufficiency scores, and aggregate run metrics logged to an MLflow experiment for comparison across releases.

**Step 4 — Apply constraints:**
- The team has no dedicated annotation budget, so reference answers must be minimal (`expected_facts` not full `expected_response`).
- The evaluation must run in under 10 minutes on a Databricks notebook.
- Privacy policy prohibits sending HR document text to external APIs beyond what Databricks already operates.

**Step 5 — Select the approach:** Use `mlflow.evaluate()` with `model_type="databricks-agent"`, passing the evaluation set as a DataFrame with `request`, `response`, `retrieved_context`, and `expected_facts` columns. Enable the `groundedness`, `context_sufficiency`, and `correctness` judges. The `groundedness` judge catches hallucinations that exist even when context was retrieved correctly; `context_sufficiency` catches retrieval failures before blaming the generator; `correctness` with `expected_facts` catches factually wrong answers against the minimal ground truth. This approach is preferred over ROUGE-based metrics (which would fail on legitimate paraphrases of policy language) and over manual review (which is not scalable for regression testing across releases).

---

## Implementation

```python
# Scenario: Run RAG evaluation with groundedness and correctness judges
# to detect hallucination regressions before shipping a new retrieval configuration.

import mlflow
import pandas as pd

eval_set = pd.DataFrame([
    {
        "request": "How many days of PTO do I accrue per month?",
        "response": "You accrue 1.5 days of PTO per month after 90 days of employment.",
        "retrieved_context": [
            {"content": "Employees accrue 1.5 days of PTO per month after completing the 90-day probationary period."}
        ],
        "expected_facts": [
            "1.5 days of PTO per month",
            "after 90 days of employment"
        ]
    },
    {
        "request": "Can I carry over unused PTO to the next year?",
        "response": "Yes, you can carry over up to 10 days of unused PTO.",
        "retrieved_context": [
            {"content": "Unused PTO may be carried over up to a maximum of 10 days at year-end."}
        ],
        "expected_facts": [
            "carry over up to 10 days of unused PTO"
        ]
    }
])

with mlflow.start_run(run_name="rag_eval_v2_groundedness"):
    results = mlflow.evaluate(
        data=eval_set,
        model_type="databricks-agent",
        evaluator_config={
            "databricks-agent": {
                "metrics": ["groundedness", "context_sufficiency", "correctness"]
            }
        }
    )

# Inspect aggregate metrics
print(results.metrics)
# e.g., {'response/llm_judged/groundedness/rating/percentage': 1.0,
#         'response/llm_judged/correctness/rating/percentage': 1.0}

# Inspect per-question results
per_q = results.tables["eval_results"]
failed = per_q[per_q["response/llm_judged/groundedness/rating"] == "no"]
print(failed[["request", "response", "response/llm_judged/groundedness/rationale"]])
```

```python
# Scenario: Call judges directly via SDK to debug a single suspicious response
# without running a full evaluation run — useful for rapid root cause analysis.

from databricks.agents.evals import judges

request = "What is the parental leave policy?"
response = "You receive 16 weeks of fully paid parental leave."
retrieved_context = [
    {"content": "Eligible employees receive 12 weeks of paid parental leave."}
]

# Check groundedness: response claims 16 weeks but context says 12 weeks
g = judges.groundedness(
    request=request,
    response=response,
    retrieved_context=retrieved_context
)
print(g)
# Expected: rating='no', rationale='The response states 16 weeks but the
# retrieved context only supports 12 weeks...'
```

```python
# Anti-pattern: Using ROUGE to evaluate a RAG Q&A application's factual accuracy.
# This fails silently because ROUGE measures word overlap, not factual correctness.

import mlflow

# WRONG: ROUGE penalizes valid paraphrases and rewards verbatim copying.
# A response that rephrases the correct fact ("fourteen days" vs "2 weeks")
# will score 0 on ROUGE even though it is semantically correct.
# A response that copies the retrieved passage verbatim will score high
# even if it includes hallucinated sentences.
results_wrong = mlflow.evaluate(
    data=eval_set,
    targets="expected_response",           # requires single reference string
    model_type="text-summarization",       # activates ROUGE suite
    evaluator_config={"default": {"metric_prefix": "rag_"}}
)
# This will show rag_rouge1 = 0.3 for a correct but paraphrased answer,
# giving a false signal that the model is underperforming.

# Correct approach: use LLM-as-judge correctness with expected_facts,
# which evaluates semantic accuracy, not string overlap.
results_correct = mlflow.evaluate(
    data=eval_set,          # DataFrame with request, response, expected_facts, retrieved_context
    model_type="databricks-agent",
    evaluator_config={
        "databricks-agent": {
            "metrics": ["correctness", "groundedness"]
        }
    }
)
# correctness judge checks whether expected_facts are present in the response,
# regardless of phrasing — eliminates false negatives from paraphrase sensitivity.
```

---

## Common Pitfalls & Misconceptions

- **Using ROUGE as the sole RAG quality metric** — Beginners reach for ROUGE because it is familiar from summarization literature and requires no LLM judge calls, making it feel "objective." The correct mental model is that ROUGE is a string-similarity proxy, not a semantic accuracy measure; a perfectly correct paraphrase scores 0, while a hallucinated verbatim copy scores 1.

- **Conflating groundedness with correctness** — New practitioners assume that if an answer is grounded (every claim traceable to context), it must be correct. The correct mental model is that groundedness measures faithfulness to context but says nothing about whether the context itself contained the right information; you need context\_sufficiency to catch retrieval gaps and correctness to verify factual accuracy against ground truth.

- **Using a single global metric across all task types** — Teams try to use one "accuracy" number to cover RAG Q&A, summarization, and classification in one dashboard. The correct mental model is that each task type has a different failure mode: RAG needs groundedness + correctness, summarization needs faithfulness + relevance, agents need task completion + tool accuracy; a single metric will be blind to failures in whichever tasks it was not designed for.

- **Ignoring judge model bias in LLM-as-judge pipelines** — Because judge models prefer verbose, confident-sounding outputs, teams see artificially high scores on wordy responses. The correct mental model is that judge bias is systematic (not random) and must be actively mitigated through calibration against human labels, using diverse judge families, and tracking judge accuracy metrics (Cohen's Kappa, F1 against human rater) rather than raw pass rates.

- **Selecting the largest model by default for all use cases** — Engineers assume more parameters always means better performance and choose frontier models for every workload. The correct mental model is that model selection is a multi-constraint optimization: a 70B model with 2s TTFT may fail a latency SLA that a 8B model with 200ms TTFT meets, and for simple classification tasks the small model may match frontier accuracy at 20x lower cost.

- **Running offline evaluation only once at release** — Teams treat evaluation as a one-time gate rather than a continuous process, meaning silent quality regressions in production traffic go undetected. The correct mental model is that offline evaluation and production monitoring should share the same scorer definitions (using MLflow 3's unified harness) so that any divergence between curated eval-set performance and live performance is immediately visible.

---

## Key Definitions

| Term | Definition |
|---|---|
| Groundedness | An LLM-judged binary quality metric assessing whether every claim in a generated response can be directly traced to information in the retrieved context, with no hallucinated additions |
| Chunk relevance | An LLM-judged per-chunk precision metric that scores whether each individual retrieved document chunk is useful in answering the user's request; aggregated into a precision score across all chunks |
| Context sufficiency | An LLM-judged quality metric (requires ground truth) that assesses whether the retrieved context contains enough information to produce the expected response, identifying retrieval-layer failures before blaming the generator |
| Relevance to query | An LLM-judged quality metric assessing whether the generated response directly addresses the user's input question, independent of factual correctness or grounding |
| Correctness | An LLM-judged quality metric (requires ground truth as `expected_facts` or `expected_response`) that compares the generated response to a minimal set of required facts |
| BLEU | Bilingual Evaluation Understudy; a reference-based metric measuring n-gram precision overlap between generated and reference text, with a brevity penalty |
| ROUGE | Recall-Oriented Understudy for Gisting Evaluation; a family of reference-based metrics measuring n-gram and LCS recall against reference text; standard for summarization regression testing |
| LLM-as-judge | An evaluation pattern where a separate, typically more capable, language model scores the output of another model according to a defined rubric, without requiring human annotation per sample |
| Position bias | A systematic tendency of LLM judges to prefer options presented first in pairwise comparisons, independent of actual quality |
| Verbosity bias | A systematic tendency of LLM judges to assign higher scores to longer, more detailed responses, independent of factual accuracy |
| Offline evaluation | Evaluation conducted against a static, curated dataset during development, before deployment, using `mlflow.evaluate()` |
| Online (production) monitoring | Continuous evaluation of live traffic traces using the same scorers defined offline, enabling detection of quality drift in production |
| Foundation Model APIs | Databricks-managed pay-per-token and provisioned-throughput endpoints for accessing state-of-the-art open and commercial models (Llama, Claude, Gemini, GPT families) within the Databricks security perimeter |
| Provisioned throughput | A deployment mode in Databricks Foundation Model APIs that guarantees a reserved rate of tokens-per-second for production workloads, as opposed to pay-per-token shared capacity |

---

## Summary / Quick Recall

- Three metric classes exist: reference-based (BLEU/ROUGE — cheap, string-level, use for regression), LLM-as-judge (reference-free, no ground truth needed, use for production), task-specific (RAG judges, agent trajectory — highest fidelity).
- Databricks built-in judges map to RAG pipeline stages: `chunk_relevance` (retriever precision), `context_sufficiency` (retriever recall vs ground truth), `groundedness` (generator faithfulness), `relevance_to_query` (generator topicality), `correctness` (factual accuracy vs expected facts).
- The entry point for all Databricks evaluation is `mlflow.evaluate(..., model_type="databricks-agent")`.
- LLM judges have known biases (verbosity, position, self-preference) — mitigate by calibrating against human labels and tracking Cohen's Kappa.
- LLM selection is a constraint-satisfaction problem: first eliminate models that violate latency/context-window hard constraints, then optimize capability vs. cost within survivors.
- Offline eval and production monitoring should share the same scorer definitions in MLflow 3 to make quality drift directly measurable.
- `expected_facts[]` is preferred over `expected_response` for correctness evaluation — it tests minimum required information, not exact phrasing.

---

## Self-Check Questions

1. Which Databricks built-in judge requires ground-truth labels as input and assesses whether the retrieved context contains enough information to produce the expected answer?

   <details><summary>Answer</summary>

   **`context_sufficiency`** is the correct answer. It requires `expected_response` or `expected_facts` as ground truth and asks: "Did the retriever return documents from which the expected answer can be derived?" This is distinct from `chunk_relevance` (which tests per-chunk relevance to the query without ground truth) and `groundedness` (which tests whether the response is faithful to retrieved context, also without ground truth). The common wrong answer is `correctness`, which compares the generated response to expected facts — not the retrieved context to expected facts.

   </details>

2. A team notices that 30% of their RAG application's responses are factually wrong, but the `groundedness` score is 95%. What does this combination most likely indicate?

   <details><summary>Answer</summary>

   High groundedness + low correctness indicates a **retrieval failure, not a generation failure**. The generator is faithfully reproducing what it found in the retrieved chunks (hence 95% groundedness), but the retrieved chunks themselves contain wrong or incomplete information (hence 30% factual errors). The appropriate next step is to check `context_sufficiency` or `chunk_relevance` to diagnose the retriever. The wrong interpretation is to conclude the generator is hallucinating — if it were hallucinating, groundedness would be low. This is the key reason why separating retrieval-layer metrics from generation-layer metrics matters.

   </details>

3. **Which TWO** of the following are known systematic biases of LLM-as-judge evaluation that require active mitigation?
   - A. Recall bias (judges prefer high-recall responses)
   - B. Verbosity bias (judges prefer longer responses)
   - C. Position bias (judges prefer options presented first in pairwise comparisons)
   - D. Cost bias (judges rate cheaper model outputs lower)
   - E. Format bias (judges prefer markdown over plain text equally)

   <details><summary>Answer</summary>

   **B (verbosity bias) and C (position bias)** are the correct answers. Verbosity bias is well-documented: LLM judges systematically score longer, more detailed responses higher even when the extra content is irrelevant. Position bias causes judges to prefer the first option in pairwise evaluations regardless of actual quality. Both are mitigated by prompt design (explicit rubrics that penalize padding), calibration against human labels, and running bidirectional comparisons. Option A is not a named bias — LLM judges are not documented to systematically prefer high-recall responses. Option D is fabricated. Option E describes a real sensitivity to formatting but it is not a systematic "bias" in the same directional sense.

   </details>

4. A team must evaluate an LLM application that classifies customer support tickets into 12 categories and must respond in under 200ms. They are evaluating three candidates: `databricks-meta-llama-3-1-8b-instruct`, `databricks-meta-llama-3-3-70b-instruct`, and `databricks-claude-sonnet-5`. Which model is most likely to meet the latency requirement at lowest cost, and why?

   <details><summary>Answer</summary>

   **`databricks-meta-llama-3-1-8b-instruct`** is the correct choice. For a well-defined 12-class classification task, an 8B model provides sufficient capability; the latency advantage of smaller models is substantial (fewer parameters to run inference over), making sub-200ms responses achievable at scale. Claude Sonnet 5 and Llama-3.3-70B are far larger and more expensive — they are appropriate for tasks requiring deep reasoning, long-context understanding, or open-ended generation, none of which are required here. The distractor is Llama-3.3-70B, which teams often default to as a "safe" mid-tier choice, but the hard latency SLA and simple task type eliminate it.

   </details>

5. A team wants to evaluate their RAG application in production using the same quality judges they used during development. They currently use `mlflow.evaluate()` in notebooks during development. What is the recommended Databricks approach to extend this to production monitoring, and what is the key architectural benefit?

   <details><summary>Answer</summary>

   The recommended approach is to define scorers in **MLflow 3's unified evaluation harness** (`mlflow.genai.evaluate()` plus production monitoring) and reuse those same scorer definitions for both offline evaluation and scheduled production monitoring. The key architectural benefit is **directly measurable quality drift**: because the same scorer definitions are applied to both the curated dev eval set and live production traces, any difference in scores between environments signals a real distribution shift — not a difference in how quality was measured. The wrong answer is to build two separate evaluation systems (one for dev, one for prod) using different prompts or judge models, which makes offline and online scores incomparable.

   </details>

---

## Further Reading

- [Built-in AI judges (MLflow 2) — Databricks](https://docs.databricks.com/aws/en/agents/agent-evaluation/llm-judge-reference) — *verified 2026-07-16* — Full reference for each built-in judge: required inputs, output schema, and callable SDK examples
- [Run an evaluation and view the results (MLflow 2) — Databricks](https://docs.databricks.com/aws/en/agents/agent-evaluation/evaluate-agent) — *verified 2026-07-16* — How to call `mlflow.evaluate()` with `model_type="databricks-agent"`, pass models and pre-generated outputs, and interpret results
- [How quality, cost, and latency are assessed by Agent Evaluation (MLflow 2) — Databricks](https://docs.databricks.com/aws/en/agents/agent-evaluation/llm-judge-metrics) — *verified 2026-07-16* — Metric name reference table, aggregate metric formulas, root-cause ordering logic
- [Evaluate and monitor AI agents — Databricks MLflow 3](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/) — *verified 2026-07-16* — Overview of MLflow 3 unified evaluation and production monitoring, migration path from MLflow 2 Agent Evaluation
- [Databricks-hosted foundation models available in Foundation Model APIs](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/supported-models) — *verified 2026-07-16* — Full catalogue of available models with endpoint names, context windows, and capability descriptions
