# Reading Model Hub Cards and Evaluation Metrics for LLM Selection

**Section:** 04 Application Development | **Module:** 02 Model and Embedding Selection | **Est. time:** 2.5 hrs | **Exam mapping:** Application Development — selecting foundation models based on documented capabilities and benchmarks

---

## TL;DR

A model card is a standardized documentation file that accompanies every published ML model, reporting intended use, training data, evaluation results, limitations, and licensing. Public benchmarks — MMLU, HumanEval, MT-Bench, HELM — provide standardized scores that allow cross-model comparison, but each benchmark measures only one narrow slice of capability. On Databricks, model cards are accessible through the HuggingFace Hub integration and the Databricks Marketplace; scores surface in the AI Playground model picker and in provisioned-throughput endpoint descriptions.

**The one thing to remember: a benchmark score is a prior, not a guarantee — it tells you what a model can do under controlled conditions, not what it will do on your specific task and data.**

---

## ELI5 — Explain It Like I'm 5

Think of a model card like the nutrition label on a cereal box. The label tells you what is inside (ingredients = training data), what it is designed for (serving suggestion = intended use), how it performed in standard lab tests (calories per serving = benchmark scores), and warnings about allergens (limitations = bias and failure modes). Just as a cereal that scores high on "fibre content" might taste terrible in your specific recipe, a model that scores 85% on MMLU might still fail on your domain-specific documents — because that standardised test used controlled academic questions, not your data. The most common misconception is that a higher benchmark number means universally better; in reality, each benchmark measures one dimension of capability, and the right question is always "does this benchmark measure the skill my task actually needs?"

---

## Learning Objectives

By the end of this chapter you will be able to:
- [ ] Identify the six standard sections of a HuggingFace model card and explain what each section signals for model selection decisions
- [ ] Interpret MMLU, HumanEval, MT-Bench, and HELM scores to compare candidate models for a given task
- [ ] Apply the `pass@k` formula to reason about code-generation model reliability at different generation budgets
- [ ] Design a model selection workflow for a regulated enterprise use case by reading limitations, bias, and licensing sections of model cards
- [ ] Log model selection rationale as MLflow tags when querying Databricks Foundation Model APIs, enabling auditable governance

---

## Visual Overview

### Model Card Anatomy — Sections and What to Look For

```
┌─────────────────────────────────────────────────────────────────┐
│                      MODEL CARD (README.md)                     │
├───────────────────┬─────────────────────────────────────────────┤
│ SECTION           │ KEY SIGNAL FOR SELECTION                    │
├───────────────────┼─────────────────────────────────────────────┤
│ Model Details     │ Architecture, parameter count, base model   │
│                   │ → sets cost/latency floor                   │
├───────────────────┼─────────────────────────────────────────────┤
│ Intended Use      │ Direct use vs downstream fine-tune          │
│                   │ → confirms task fit without fine-tuning     │
├───────────────────┼─────────────────────────────────────────────┤
│ Out-of-Scope Use  │ Explicit prohibitions                       │
│                   │ → catches compliance blockers early         │
├───────────────────┼─────────────────────────────────────────────┤
│ Bias, Risks &     │ Known demographic skews, failure domains    │
│ Limitations       │ → critical for regulated industries         │
├───────────────────┼─────────────────────────────────────────────┤
│ Training Data     │ Sources, cutoff date, data consent          │
│                   │ → data provenance for governance audit      │
├───────────────────┼─────────────────────────────────────────────┤
│ Evaluation        │ Benchmark scores (MMLU, HumanEval, etc.)   │
│ Results           │ → primary comparison surface                │
├───────────────────┼─────────────────────────────────────────────┤
│ License           │ Commercial use, redistribution rights       │
│                   │ → legal clearance for production deployment │
└───────────────────┴─────────────────────────────────────────────┘
```

### Benchmark Comparison Matrix — What Each Metric Measures

```
Benchmark       │ Task Type              │ Score Format    │ Signals
────────────────┼────────────────────────┼─────────────────┼─────────────────────────────
MMLU            │ 57-subject multiple    │ Accuracy 0–1    │ Broad academic knowledge
(0-shot/5-shot) │ choice Q&A             │ (higher=better) │ Threshold ~0.70 for GPT-4 tier
────────────────┼────────────────────────┼─────────────────┼─────────────────────────────
HumanEval       │ Python code generation │ pass@k (0–1)    │ Coding correctness
                │ 164 problems           │ (higher=better) │ pass@1 ≥ 0.65 → production grade
────────────────┼────────────────────────┼─────────────────┼─────────────────────────────
MT-Bench        │ Multi-turn chat /      │ 1–10 LLM-judge  │ Instruction following,
                │ instruction following  │ (higher=better) │ dialogue coherence
────────────────┼────────────────────────┼─────────────────┼─────────────────────────────
HELM            │ Multi-scenario:        │ Per-metric +    │ Accuracy, calibration,
                │ accuracy, fairness,    │ aggregated rank │ robustness, fairness
                │ robustness, safety     │                 │ together
────────────────┴────────────────────────┴─────────────────┴─────────────────────────────
```

### Model Selection Decision Workflow

```
Task requirement defined
        │
        ▼
Read model card "Intended Use" ──► Does it match your task?
        │                                    │
        │ Yes                                │ No ──► Disqualify or flag for fine-tuning
        ▼
Check "Bias, Risks & Limitations" ──► Is there a known failure in your domain?
        │                                    │
        │ None blocking                      │ Blocking ──► Disqualify
        ▼
Review benchmark scores
        │
        ├──► MMLU: knowledge-breadth tasks (Q&A, summarisation)
        ├──► HumanEval: code generation
        ├──► MT-Bench: chat / instruction following
        └──► HELM: safety-sensitive or regulated environments
        │
        ▼
Check License ──► Commercial use permitted?
        │                    │
        │ Yes                │ No ──► Disqualify or seek enterprise license
        ▼
Run task-specific evaluation on your data
        │
        ▼
Log selection rationale (model name, scores, constraints) as MLflow tags
        │
        ▼
Deploy via Databricks Foundation Model APIs (pay-per-token or provisioned throughput)
```

---

## Key Concepts

### Model Cards — Standardised Documentation for ML Models

Model cards are structured `README.md` files that accompany published ML models, providing standardised documentation of a model's capabilities, limitations, training provenance, and intended use. They were introduced by Mitchell et al. (2018) as a discipline for responsible AI disclosure and have since become the de facto standard on HuggingFace Hub, with the annotated template defining sections for: Model Details, Uses (direct/downstream/out-of-scope), Bias/Risks/Limitations, Training Data, Training Procedure, Evaluation Results, and License.

Under the hood, a model card is parsed at two levels: the YAML metadata block at the top of the file (machine-readable fields: `license`, `datasets`, `language`, `pipeline_tag`, `model-index` with benchmark results) and the Markdown prose body (human-readable, unstructured). The Hub's UI renders both: metadata powers the filter facets at `huggingface.co/models`, while benchmark results specified in the `model-index` YAML block are rendered as a structured widget directly on the model page and indexed by Papers with Code leaderboards.

On Databricks, model cards for Marketplace models appear in the **Databricks Marketplace** listing page; for Foundation Model API endpoints, the per-model description in the documentation (e.g., `docs.databricks.com/.../supported-models`) serves as the functional model card. Teams provisioning custom models should register models in **Unity Catalog** with the model description field populated from the source model card, making governance metadata queryable via `mlflow.search_model_versions()`.

### MMLU — Massive Multitask Language Understanding

MMLU (Hendrycks et al., 2021) measures broad academic knowledge by presenting a model with multiple-choice questions spanning 57 subjects — ranging from elementary mathematics and US history to professional law, medicine, and abstract algebra — across four difficulty levels. Its purpose is to assess a model's ability to encode and reason over a wide academic knowledge base.

Mechanistically, MMLU presents four-answer (A/B/C/D) multiple-choice questions and measures accuracy as the fraction of questions answered correctly. It is evaluated in two standard configurations: **0-shot** (the model receives only the question, with no examples) and **5-shot** (the model receives five solved examples before the question, mimicking in-context learning). The 5-shot score is typically 3–8 percentage points higher than 0-shot. Scores approaching or exceeding **0.70** (70%) indicate GPT-3.5-class capability; scores above **0.86** indicate GPT-4-class capability. A score of ~0.25 is the random-guess baseline for four-option questions.

In the Databricks and HuggingFace ecosystem, MMLU scores appear in the `model-index` YAML block of model cards (e.g., `type: mmlu`, `value: 0.824`) and are surfaced on the Open LLM Leaderboard at `huggingface.co/spaces/open-llm-leaderboard`. When selecting a model for knowledge-intensive tasks — such as summarising regulatory documents, answering policy questions, or classifying support tickets — MMLU is the first benchmark to check for subject-domain alignment.

### HumanEval — Code Generation Correctness

HumanEval (Chen et al., 2021, OpenAI) measures Python code generation quality using 164 hand-crafted programming problems. Each problem provides a function signature, a docstring, and a set of hidden unit tests; the model is asked to complete the function body.

The benchmark computes **pass@k**: given `k` generated code samples per problem, a problem is "solved" if at least one sample passes all unit tests. The formula is:

```
pass@k = 1 - C(n-c, k) / C(n, k)
```

where `n` is the number of generated samples, `c` is the number of samples that pass. In practice, `pass@1` (single sample, no retries) is the most conservative and most reported metric; `pass@10` or `pass@100` is used for ceiling-performance comparisons. A **pass@1 ≥ 0.65** is conventionally associated with production-usable code generation capability. A pass@1 of 0.95+ indicates near-human performance.

On Databricks, HumanEval scores appear in the model cards of code-specialist models such as models in the `CodeLlama` or `DeepSeek-Coder` families available via provisioned throughput. When building a code generation, code review, or SQL generation application, HumanEval pass@1 is the primary benchmark to consult for baseline quality estimation — though domain-specific code (e.g., proprietary APIs, PySpark idioms) will require additional task-specific evaluation.

### MT-Bench — Multi-Turn Instruction Following Quality

MT-Bench (Zheng et al., 2023, LMSYS) evaluates a model's ability to follow multi-turn instructions across eight categories: writing, roleplay, extraction, reasoning, math, coding, knowledge (STEM), and knowledge (humanities/social science). It consists of 80 two-turn conversations specifically designed to probe instruction-following coherence across a dialogue, not just single-response quality.

The mechanism is **LLM-as-judge**: GPT-4 (or a specified judge model) evaluates each response on a **1–10 scale** across dimensions of helpfulness, accuracy, depth, creativity, and instruction adherence. The final score per model is the average across all 160 turns (80 first turns + 80 second turns). A score of **8.0+** indicates strong instruction-following suitable for general-purpose chat and assistant applications; scores below 6.0 indicate the model struggles to maintain coherence across turns. MT-Bench scores are directly comparable across models only when the same judge model is used.

In Databricks workflows, MT-Bench scores are most relevant when building chat assistants, instruction-following agents, or any application where the model must handle follow-up questions, clarifications, or multi-step instructions in a single session. Scores are published on the LMSYS Chatbot Arena Leaderboard and appear in model cards for instruction-tuned models.

### HELM — Holistic Evaluation of Language Models

HELM (Liang et al., 2022, Stanford CRFM) is a comprehensive evaluation framework that goes beyond single-number benchmark scores by measuring models across multiple **scenarios** (task types), **metrics** (accuracy, calibration, robustness, fairness, bias, toxicity, efficiency), and **model sizes** simultaneously. It is designed to expose trade-offs that single benchmarks hide: a model may score high on accuracy while having poor calibration (overconfident wrong answers) or high toxicity.

Mechanistically, HELM defines a **(scenario, metric)** matrix. A scenario is a specific task + dataset combination (e.g., NaturalQuestions for closed-book QA, TweetSentimentExtraction for classification, BoolQ for reading comprehension). Each model is evaluated across all scenarios and all metrics, producing a multi-dimensional profile rather than a single rank. Calibration is measured via Expected Calibration Error (ECE); robustness via accuracy under adversarial rephrasing; fairness via accuracy gap across demographic subgroups.

For safety-sensitive and regulated Databricks deployments, HELM's fairness and toxicity scenario results are the primary governance signal: they expose whether a model amplifies demographic stereotypes or produces harmful completions at measurable rates. The HELM leaderboard is maintained at `crfm.stanford.edu/helm` and is updated as new models are added. On Databricks, HELM metrics are referenced in internal model evaluation runbooks for selecting models that will process sensitive customer data.

### Limitations and Bias Reporting — The Governance Signal in Model Cards

The "Bias, Risks, and Limitations" section of a model card documents known failure modes, demographic skews in model outputs, datasets that were excluded or under-represented, and behaviours that are explicitly out of scope. This is the section most frequently skipped by engineers and most important for compliance teams.

Mechanistically, limitations emerge from the interaction of three factors: training data composition (if finance text was scarce, financial reasoning degrades), RLHF/safety fine-tuning choices (alignment training that reduces toxic outputs may also reduce precision on edge cases), and benchmark distribution (models over-trained on MMLU-style tasks may exhibit benchmark memorisation, inflating scores without corresponding real-world capability). A well-written limitations section will quantify performance degradation on specific demographic groups or domains, not just assert "the model may occasionally make errors."

On Databricks, the limitations section is operationally relevant in three governance scenarios: (1) **data residency** — training data sourced from regions with GDPR or similar restrictions affects whether the model's outputs are compliant artefacts; (2) **financial services regulations** — documented hallucination tendencies or poor calibration scores trigger model risk management (MRM) review requirements; (3) **Unity Catalog data lineage** — the training data citation in the model card should be traceable to a catalogued dataset, enabling downstream lineage audits.


---

## Key Parameters / Configuration Knobs

| Parameter | What it controls | Decision rule |
|---|---|---|
| `pass@k` — value of `k` | Number of code samples generated per problem; higher k catches more correct solutions but increases token cost | Use `pass@1` for latency-sensitive production endpoints; increase to `pass@10` only for batch offline code generation where correctness is paramount |
| MMLU shot count (0-shot vs 5-shot) | Whether in-context examples are provided during benchmark evaluation; 5-shot simulates few-shot prompting | Compare models on the same shot count; if your production prompts will always include examples, prefer the 5-shot score for selection |
| MT-Bench judge model | Which LLM scores the responses (default: GPT-4); different judges produce different absolute scores | When comparing MT-Bench results from different papers, verify they used the same judge model before treating scores as directly comparable |
| HELM scenario type | Which task-dataset combination is used for evaluation; "core scenarios" cover 7 standard tasks; "targeted" scenarios cover specific harms | For regulated use cases, always consult the HELM fairness and toxicity scenarios, not just the accuracy scenario |
| HuggingFace `model-index` YAML | Machine-readable benchmark metadata in a model card, ingested by leaderboards and Hub filter facets | Verify the `source.url` field in the model-index block points to the actual evaluation run, not a self-reported number; cross-check with the Open LLM Leaderboard |


---

## Worked Example: Requirement → Decision

**Given:** A regulated financial services firm is building a Q&A assistant that summarises earnings call transcripts for equity analysts. Two candidate models are shortlisted: Model A (MMLU 0.82, pass@1 0.45, MT-Bench 8.3, license: Apache 2.0) and Model B (MMLU 0.71, pass@1 0.72, MT-Bench 7.1, license: research-only). The application must run in a production Databricks workspace, must not hallucinate financial figures, and must comply with the firm's model risk management policy requiring documented bias assessment.

**Step 1 — Identify the goal:** Select the model that maximises factual accuracy and auditability for earnings transcript summarisation in a regulated production environment.

**Step 2 — Define inputs:** Earnings call transcripts (structured prose with numerical tables), analyst queries in natural language, and the two model cards as the primary information sources.

**Step 3 — Define outputs:** The downstream consumer is an analyst workbench that will display a summary and flag confidence; the output must be auditable, meaning the model selection rationale must be traceable via MLflow.

**Step 4 — Apply constraints:**
- Production deployment rules out Model B's research-only license — commercial use is prohibited.
- MRM policy requires a documented bias assessment, so the model card's Bias/Risks/Limitations section must contain quantitative rather than qualitative-only statements.
- HumanEval (pass@1) is not a primary signal for this text summarisation task — Model B's high code score is irrelevant.
- MMLU's "professional finance" sub-category (one of the 57 subjects) provides a targeted signal; Model A's 0.82 MMLU likely outperforms on financial knowledge.

**Step 5 — Select the approach:** **Choose Model A** and deploy via a Databricks Foundation Model APIs provisioned-throughput endpoint. Rationale: Model A is the only candidate with a compatible commercial license, has a higher MMLU score (indicating stronger financial knowledge encoding), and supports governance requirements through documented bias disclosures; Model B's superior HumanEval score is irrelevant to a summarisation task and cannot offset its licensing disqualification.

---

## Implementation

```python
# Scenario: Deploy a shortlisted model to a Databricks FMAPI endpoint and log the
# selection rationale as auditable MLflow tags so compliance teams can reproduce
# the model choice decision without reading internal Slack threads.

import mlflow
from openai import OpenAI

DATABRICKS_HOST = "https://<workspace>.azuredatabricks.net"
DATABRICKS_TOKEN = dbutils.secrets.get("scope", "databricks-token")

# Log model card selection rationale as MLflow run tags before inference
with mlflow.start_run(run_name="earnings-summary-model-selection") as run:
    mlflow.set_tags({
        "model.name": "databricks-meta-llama-3-3-70b-instruct",
        "model.mmlu_5shot": "0.824",
        "model.mt_bench": "8.3",
        "model.license": "llama3.3-community",
        "model.card_url": "https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/MODEL_CARD.md",
        "selection.rationale": "Highest MMLU among Apache-compatible models; finance sub-category 0.79",
        "selection.disqualified_alternatives": "Model-B: research-only license",
        "governance.bias_assessed": "true",
        "governance.mrm_policy_version": "v2.4",
    })

    # Query the selected model via FMAPI (OpenAI-compatible client)
    client = OpenAI(
        base_url=f"{DATABRICKS_HOST}/serving-endpoints",
        api_key=DATABRICKS_TOKEN,
    )

    transcript_excerpt = "Revenue for Q3 2024 was $4.2B, up 12% year-over-year..."
    response = client.chat.completions.create(
        model="databricks-meta-llama-3-3-70b-instruct",
        messages=[
            {"role": "system", "content": "Summarise the following earnings transcript excerpt for an equity analyst. Be factually precise; do not infer figures not stated."},
            {"role": "user", "content": transcript_excerpt},
        ],
        max_tokens=300,
        temperature=0.1,  # Low temperature for factual summarisation
    )

    summary = response.choices[0].message.content
    mlflow.log_text(summary, artifact_file="sample_summary.txt")
    print(f"MLflow run ID: {run.info.run_id}")
    print(summary)
```

```python
# Anti-pattern: Selecting a model based solely on parameter count
# ("bigger = better") without checking benchmark alignment to the task.
# This fails because parameter count correlates weakly with task-specific
# performance, and larger models incur higher latency and cost without
# guaranteed accuracy gains on your specific domain.

# WRONG — Do not do this:
def select_model_wrong(candidates: list[dict]) -> str:
    """Pick the model with the most parameters."""
    # Anti-pattern: sorts by parameter count, ignores task-relevance of benchmarks,
    # ignores license, ignores domain-specific MMLU sub-category, ignores limitations.
    return max(candidates, key=lambda m: m["parameter_count"])["endpoint_name"]

# WHY IT BREAKS:
# 1. A 70B model may score lower on your specific MMLU sub-category than a 7B model
#    fine-tuned on domain data.
# 2. Larger models double or triple token cost and add 100–400ms latency per call.
# 3. Parameter count is not reported on model cards for proprietary models — you
#    cannot even apply this heuristic consistently.

# CORRECT APPROACH — score by task-relevant benchmark + constraints:
def select_model_correct(
    candidates: list[dict],
    task_benchmark: str = "mmlu_finance",
    required_license: str = "apache-2.0",
) -> str:
    """
    Pick the highest task-relevant benchmark score among license-compatible models.
    Benchmark key must match the field in the model card's model-index YAML.
    """
    eligible = [
        m for m in candidates
        if m.get("license") == required_license
        and task_benchmark in m.get("benchmarks", {})
    ]
    if not eligible:
        raise ValueError(f"No eligible models found for license={required_license}, benchmark={task_benchmark}")
    return max(eligible, key=lambda m: m["benchmarks"][task_benchmark])["endpoint_name"]
```

---

## Common Pitfalls & Misconceptions

- **Treating aggregate MMLU as a domain proxy** — Beginners see "MMLU 0.82" and assume the model is 82% accurate on their finance or legal documents, because MMLU sounds like a general score. MMLU averages 57 subjects; a model can score 0.90 on history and 0.55 on professional law — the aggregate hides sub-domain weakness. Always disaggregate by MMLU subject area when your task is domain-specific.

- **Comparing MT-Bench scores from different papers without verifying the judge model** — Beginners compare two model cards side-by-side and treat their MT-Bench scores as directly equivalent, because both are on a 0–10 scale from the same benchmark. MT-Bench scores shift by 0.5–1.5 points depending on the judge LLM version; scores from a paper using GPT-4-0314 are not directly comparable to scores using GPT-4-1106. Always verify the `judge_model` field or footnote before cross-paper comparison.

- **Skipping the Limitations section because benchmarks look strong** — Beginners move straight to the Evaluation Results section and stop reading once the scores exceed a threshold, because they assume strong benchmarks imply no significant failure modes. The Limitations section documents known failure domains that benchmarks specifically do not cover — e.g., a model may score 0.85 MMLU but the card explicitly states "does not reliably handle numerical reasoning in financial statements." Always read Limitations before finalising a selection.

- **Confusing pass@1 with pass@k for production sizing** — Beginners report pass@10 or pass@100 from a paper's headline figure and plan a production system around it, because the highest k gives the most impressive number. pass@k with k > 1 requires generating multiple samples and selecting the best, which multiplies token cost and latency by k. Production systems almost always need pass@1 performance; confirm which k the reported score uses.

- **Ignoring license restrictions until after development** — Beginners prototype with a research-only or non-commercial model (often the most capable publicly available option), then discover the license prohibits commercial deployment during security review. License is the first hard constraint to check — it is binary (disqualifies a model outright), whereas benchmark scores are continuous trade-offs that can sometimes be remedied with prompting or fine-tuning.

---

## Key Definitions

| Term | Definition |
|---|---|
| Model card | A structured `README.md` file published alongside an ML model that documents intended use, training data, evaluation results, limitations, and license; the de facto standard on HuggingFace Hub, specified in Mitchell et al. (2018) |
| MMLU | Massive Multitask Language Understanding; a multiple-choice benchmark spanning 57 academic subjects, measuring breadth of encoded knowledge; scored as fraction correct (0–1) in 0-shot or 5-shot configurations |
| HumanEval | A code-generation benchmark of 164 Python problems evaluated by executing generated code against hidden unit tests; primary metric is pass@k, the probability that at least one of k generated samples passes all tests |
| pass@k | The probability that at least one of k independently generated code samples for a given problem passes all unit tests; computed as `1 - C(n-c,k)/C(n,k)` where n = total samples generated, c = number that pass |
| MT-Bench | A multi-turn instruction-following benchmark of 80 two-turn dialogues across 8 categories, scored 1–10 by an LLM judge (default GPT-4); measures conversational coherence, not just single-response quality |
| HELM | Holistic Evaluation of Language Models; a Stanford CRFM framework that evaluates models across a (scenario, metric) matrix covering accuracy, calibration, robustness, fairness, and toxicity simultaneously |
| 0-shot evaluation | Model is queried with no in-context examples; measures raw knowledge encoding without prompting assistance |
| 5-shot evaluation | Model is queried with five solved examples preceding the target question; simulates few-shot prompting and typically improves accuracy 3–8 percentage points over 0-shot on MMLU |
| Calibration | The degree to which a model's expressed confidence matches its actual accuracy; a well-calibrated model that says "80% confident" is correct ~80% of the time; measured by Expected Calibration Error (ECE) in HELM |
| model-index YAML | Machine-readable metadata block in a HuggingFace model card specifying benchmark results in a structured format; ingested by the Hub's filter system and leaderboards such as Open LLM Leaderboard |

---

## Summary / Quick Recall

- A model card is a `README.md` with YAML metadata (machine-readable benchmarks, license, datasets) and prose sections (limitations, bias, intended use) — read both levels.
- MMLU measures broad academic knowledge via 57-subject multiple-choice; 0-shot ≠ 5-shot — compare like-for-like; scores above 0.70 indicate GPT-3.5-tier capability.
- HumanEval pass@k measures code correctness by running generated code against unit tests; pass@1 is the production-relevant metric; pass@k with k > 1 multiplies cost by k.
- MT-Bench judges multi-turn instruction quality on a 1–10 scale using an LLM judge — cross-paper comparisons require identical judge models to be valid.
- HELM is the right evaluation framework when safety, fairness, or calibration matter — it exposes trade-offs that single-score benchmarks hide.
- The Limitations section of a model card is the primary governance signal — document bias, failure domains, and data provenance that no benchmark score captures.
- Log model selection rationale as MLflow tags at selection time — this creates an auditable, reproducible record without relying on tribal knowledge.

---

## Self-Check Questions

1. What does an MMLU score of 0.82 in 5-shot configuration specifically measure, and what does it **not** tell you about a model's performance on your financial documents task?

   <details><summary>Answer</summary>

   An MMLU 5-shot score of 0.82 measures the fraction of correctly answered multiple-choice questions across 57 academic subjects when the model is shown five solved examples first (simulating few-shot prompting). It signals that the model has broad academic knowledge encoding at approximately GPT-4-tier capability. It does **not** tell you how the model performs on your specific financial documents because: (a) MMLU questions are standardized academic multiple-choice, not open-ended summarization tasks; (b) the aggregate score masks sub-domain variation — the finance sub-category might be significantly lower than 0.82; (c) the 5-shot setup assumes the production prompt will always include examples; and (d) benchmark datasets may have been partially included in pre-training (benchmark contamination), inflating scores beyond real-world capability. The most tempting wrong answer is "it means the model will be 82% accurate on my financial task" — this confuses a controlled benchmark score with deployment-time performance on out-of-distribution data.

   </details>

2. Your team is building a PySpark code assistant. You compare two models: Model X (HumanEval pass@1 = 0.72) and Model Y (HumanEval pass@10 = 0.91). Which model should you select for a latency-sensitive production endpoint, and why?

   <details><summary>Answer</summary>

   Select **Model X** for the latency-sensitive production endpoint. HumanEval pass@1 is the correct metric for production systems because it measures correctness with a single generation attempt — the real-world constraint for a responsive assistant. Model Y's pass@10 = 0.91 means it needs to generate 10 samples per problem and pick the best one, which multiplies token cost and latency by 10x and requires an additional selection step (e.g., running all 10 samples or a secondary judge). The comparison is not apples-to-apples: Model X's pass@1 of 0.72 is directly deployable, while Model Y's headline number is a ceiling measured under ideal offline conditions. For a fair comparison, you would need Model Y's pass@1 score. The most tempting wrong answer is "Model Y because 0.91 > 0.72" — this ignores that the k values differ, making the numbers incomparable without normalisation.

   </details>

3. **Which TWO** of the following model card sections are the most critical to review when selecting a model for a HIPAA-regulated healthcare application?
   - A. Model Details (architecture, parameter count)
   - B. Bias, Risks, and Limitations
   - C. Training Data (sources, consent, cutoff date)
   - D. Environmental Impact (CO2 emissions)
   - E. Citation / BibTeX

   <details><summary>Answer</summary>

   The correct answers are **B** and **C**.

   **B — Bias, Risks, and Limitations** is required because HIPAA-regulated applications must document known failure modes, demographic skews, and out-of-scope behaviours; an undisclosed bias against certain demographic groups in medical text could constitute a civil rights risk in addition to an accuracy problem.

   **C — Training Data** is required because HIPAA compliance involves data provenance: if the model was trained on patient records or identifiable health data without proper consent or de-identification, deployment may violate HIPAA's Privacy Rule and the model's own training terms.

   **A (Model Details)** provides useful context about architecture but has no direct HIPAA compliance bearing. **D (Environmental Impact)** is irrelevant to regulatory compliance. **E (Citation)** is a research attribution field and carries no governance weight. The most tempting wrong answer is A because parameter count and architecture sound like technical diligence — but they have no regulatory significance without the bias and data provenance checks.

   </details>

4. A colleague argues that HELM and MMLU are redundant — "just use whichever has higher scores." What is the fundamental flaw in this reasoning, and when would you choose HELM over MMLU?

   <details><summary>Answer</summary>

   The fundamental flaw is that HELM and MMLU measure orthogonal dimensions of quality, not the same dimension at different resolutions. MMLU measures a single metric (accuracy) across many knowledge domains. HELM measures multiple metrics (accuracy, calibration, robustness, fairness, toxicity) across multiple task scenarios simultaneously. A model can score extremely high on MMLU accuracy while having poor calibration (it is confidently wrong), producing biased outputs across demographic groups, or generating toxic content — none of which MMLU detects. Choosing "whichever has higher scores" is meaningless because the two produce incomparable outputs: MMLU produces a single 0–1 accuracy number; HELM produces a multi-dimensional profile with per-scenario, per-metric breakdowns. Use HELM over MMLU when the application involves safety-sensitive content, serves diverse demographic populations, or operates in a regulated environment requiring documented fairness assessment — because only HELM exposes those failure modes quantitatively.

   </details>

5. You are evaluating two models for a customer-facing multilingual support chatbot: Model P (MT-Bench 8.6, MMLU 0.79, license: proprietary enterprise) and Model Q (MT-Bench 7.8, MMLU 0.84, license: Apache 2.0). Your organisation's policy prohibits proprietary models in customer-facing production systems. How do you reason through the selection, and what follow-up evaluation would you run before deploying Model Q?

   <details><summary>Answer</summary>

   The selection is straightforward on the licensing constraint: **Model P is disqualified** by the organisational policy prohibiting proprietary models in production customer-facing systems. This is a hard binary gate — no benchmark score can override it. Model Q (Apache 2.0) is the only eligible candidate.

   However, Model Q's lower MT-Bench score (7.8 vs 8.6) represents a meaningful quality gap for a chat application. The follow-up evaluation before deployment should include: (1) **Task-specific MT-Bench equivalent** — run a manual evaluation on 20–30 representative support dialogues from your actual domain (not the generic MT-Bench set) using the same 1–10 LLM-judge approach, to verify that 7.8 translates acceptably to your use case; (2) **Multilingual evaluation** — MT-Bench's standard set is English-only; for a multilingual chatbot, evaluate on translated or native-language test cases for the top 3 supported languages; (3) **Limitations section review** — check Model Q's model card for documented failure modes in customer service or multilingual contexts; (4) **HELM fairness scenarios** — for a customer-facing application, verify there are no documented demographic output skews that could trigger regulatory concern. Only after these checks should Model Q be approved for production.

   </details>

---

## Further Reading

- [HuggingFace Model Cards Documentation](https://huggingface.co/docs/hub/model-cards) — *verified 2026-07-11* — Authoritative guide to model card structure, YAML metadata format, and evaluation results specification
- [HuggingFace Annotated Model Card Template](https://huggingface.co/docs/hub/model-card-annotated) — *verified 2026-07-11* — Section-by-section annotation of the full model card template including role responsibilities (developer, sociotechnic, project organiser)
- [Databricks Foundation Model APIs](https://docs.databricks.com/en/machine-learning/foundation-models/index.html) — *verified 2026-07-11* — Overview of pay-per-token and provisioned throughput modes, OpenAI-compatible API, and workspace integration
- [Databricks Supported Foundation Models](https://docs.databricks.com/en/machine-learning/foundation-model-apis/supported-models.html) — *verified 2026-07-11* — Per-model descriptions including capabilities, limitations, and context window specifications for all FMAPI-hosted models
