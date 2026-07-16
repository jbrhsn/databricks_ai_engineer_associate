# LAB-18 — Masking and PII Mitigation Pipeline

**Lab:** LAB-18 | **Section:** 06 Governance | **Module:** 01 Guardrails and Compliance | **Est. time:** 2 hrs

> **Simulation note:** This is an illustrative walkthrough lab. All code blocks are production-accurate and designed for a real Databricks workspace, but expected outputs are labelled `# Expected output (illustrative)` and reflect realistic results rather than live execution. Work through each step by reading the code and expected output, then answer the Reflection Questions.

---

## Objective

Build a layered PII mitigation pipeline on Databricks: detect PII in a raw dataset using Presidio, apply masking transformations via a PySpark UDF, enforce column-level access control with a Unity Catalog dynamic masking policy, and verify that AI Gateway blocks a PII-bearing request payload.

---

## Prerequisites

- Chapter notes: [Masking and PII Mitigation](notes.md)
- Familiarity with PySpark DataFrames and SQL in Databricks notebooks
- Conceptual understanding of Unity Catalog schemas, tables, and row/column access control
- Conceptual understanding of AI Gateway route configuration

---

## Setup

Install the Presidio libraries and initialise a SparkSession. In a real workspace, run these in the first cell of a Databricks notebook (Runtime 14.3 LTS ML or later).

```python
# Install Presidio analyser and anonymiser into the notebook-scoped environment
%pip install presidio-analyzer presidio-anonymizer spacy
%restart_python

# Download the spaCy English model used by Presidio's NER engine
import subprocess
subprocess.run(["python", "-m", "spacy", "download", "en_core_web_lg"], check=True)
```

```python
# Verify imports succeed before writing any pipeline code
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("lab18-pii-pipeline").getOrCreate()
print("Setup complete")
```

```
# Expected output (illustrative)
Setup complete
```

Create a Unity Catalog schema to hold the raw and masked tables for this lab.

```sql
-- Create an isolated schema for the lab; drop it during teardown
CREATE SCHEMA IF NOT EXISTS main.lab18_governance
COMMENT 'LAB-18 PII mitigation lab — governance section';
```

---

## Steps

### Step 1 — Detect PII in a Raw Dataset Using Presidio + PySpark

**Scenario:** A patient intake system stores free-text notes that contain names, phone numbers, email addresses, and US Social Security Numbers. Before these notes can be used as RAG context, they must be scanned for PII entities so downstream transforms can decide how to handle each field.

Create a small synthetic dataset that represents the raw intake notes.

```python
# Scenario: simulate raw intake notes arriving from an upstream ETL pipeline
from pyspark.sql import Row

raw_data = [
    Row(record_id=1, note="Patient John Smith called 555-867-5309. SSN: 123-45-6789."),
    Row(record_id=2, note="Contact jane.doe@example.com for follow-up on 2024-01-15."),
    Row(record_id=3, note="No PII present. Routine check-in completed without incident."),
    Row(record_id=4, note="Dr. Alice Brown, NPI 1234567890, reviewed the case for Bob Jones."),
]

raw_df = spark.createDataFrame(raw_data)
raw_df.show(truncate=False)
```

```
# Expected output (illustrative)
+---------+------------------------------------------------------------------+
|record_id|note                                                              |
+---------+------------------------------------------------------------------+
|1        |Patient John Smith called 555-867-5309. SSN: 123-45-6789.        |
|2        |Contact jane.doe@example.com for follow-up on 2024-01-15.        |
|3        |No PII present. Routine check-in completed without incident.      |
|4        |Dr. Alice Brown, NPI 1234567890, reviewed the case for Bob Jones. |
+---------+------------------------------------------------------------------+
```

Write a PySpark UDF that runs Presidio's `AnalyzerEngine` on each note and returns a JSON array of detected entity types and their character offsets.

```python
# Scenario: detect PII entity types in free-text notes before deciding on a transformation
# strategy — detection must precede masking so each entity type can be handled differently
import json
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

def detect_pii_entities(text: str) -> str:
    """Return a JSON list of {entity_type, start, end, score} dicts for all PII found."""
    if text is None:
        return "[]"
    analyzer = AnalyzerEngine()
    results = analyzer.analyze(text=text, language="en")
    entities = [
        {
            "entity_type": r.entity_type,
            "start": r.start,
            "end": r.end,
            "score": round(r.score, 3),
        }
        for r in results
    ]
    return json.dumps(entities)

detect_pii_udf = udf(detect_pii_entities, StringType())

detected_df = raw_df.withColumn("pii_entities", detect_pii_udf("note"))
detected_df.select("record_id", "pii_entities").show(truncate=False)
```

```
# Expected output (illustrative)
+---------+--------------------------------------------------------------------------------------------+
|record_id|pii_entities                                                                                |
+---------+--------------------------------------------------------------------------------------------+
|1        |[{"entity_type": "PERSON", "start": 8, "end": 18, "score": 0.85},                          |
|         | {"entity_type": "PHONE_NUMBER", "start": 26, "end": 38, "score": 0.75},                   |
|         | {"entity_type": "US_SSN", "start": 45, "end": 56, "score": 0.85}]                         |
|2        |[{"entity_type": "EMAIL_ADDRESS", "start": 8, "end": 30, "score": 0.85},                   |
|         | {"entity_type": "DATE_TIME", "start": 46, "end": 56, "score": 0.85}]                      |
|3        |[]                                                                                           |
|4        |[{"entity_type": "PERSON", "start": 4, "end": 15, "score": 0.85},                          |
|         | {"entity_type": "PERSON", "start": 54, "end": 62, "score": 0.85}]                         |
+---------+--------------------------------------------------------------------------------------------+
```

**What to notice:** Record 3 produces an empty array — no PII detected. Record 4 detects two `PERSON` entities but misses the NPI (National Provider Identifier), illustrating that Presidio's built-in recognisers do not cover all domain-specific identifiers. A production pipeline would add a custom `PatternRecognizer` for NPI numbers.

---

### Step 2 — Apply Masking Transformations and Write to a Governed Table

**Scenario:** The pipeline must produce a masked version of each note suitable for use as RAG context. SSNs must be fully redacted. Names, phone numbers, and email addresses must be replaced with typed placeholders (pseudonymization) so downstream retrieval can still reason about entity types without exposing real values.

Write a second UDF that uses Presidio's `AnonymizerEngine` to apply operator-specific transformations per entity type.

```python
# Scenario: apply per-entity-type transformations to produce a RAG-safe masked note
# Constraint: SSNs must be replaced with <REDACTED>; other PII replaced with typed placeholders
from presidio_anonymizer.entities import OperatorConfig

def mask_pii(text: str) -> str:
    """Apply entity-type-specific masking operators and return the anonymised text."""
    if text is None:
        return text
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()
    analysis_results = analyzer.analyze(text=text, language="en")
    operators = {
        "US_SSN": OperatorConfig("replace", {"new_value": "<REDACTED_SSN>"}),
        "PERSON": OperatorConfig("replace", {"new_value": "<PERSON>"}),
        "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<PHONE>"}),
        "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL>"}),
        "DATE_TIME": OperatorConfig("replace", {"new_value": "<DATE>"}),
    }
    result = anonymizer.anonymize(text=text, analyzer_results=analysis_results, operators=operators)
    return result.text

mask_pii_udf = udf(mask_pii, StringType())

masked_df = raw_df.withColumn("masked_note", mask_pii_udf("note"))
masked_df.select("record_id", "note", "masked_note").show(truncate=False)
```

```
# Expected output (illustrative)
+---------+-----------------------------------------------------------+-------------------------------------------------------+
|record_id|note                                                       |masked_note                                            |
+---------+-----------------------------------------------------------+-------------------------------------------------------+
|1        |Patient John Smith called 555-867-5309. SSN: 123-45-6789. |Patient <PERSON> called <PHONE>. SSN: <REDACTED_SSN>.  |
|2        |Contact jane.doe@example.com for follow-up on 2024-01-15. |Contact <EMAIL> for follow-up on <DATE>.               |
|3        |No PII present. Routine check-in completed without ...    |No PII present. Routine check-in completed without ... |
|4        |Dr. Alice Brown, NPI 1234567890, reviewed the case for ...|Dr. <PERSON>, NPI 1234567890, reviewed the case for ...|
+---------+-----------------------------------------------------------+-------------------------------------------------------+
```

**What to notice:** Record 4's NPI (1234567890) is not masked because Presidio did not detect it. This confirms the detection gap from Step 1 — masking quality is bounded by detection quality. The NPI leaks into the masked output, demonstrating why custom recognisers matter for domain-specific identifiers.

Write both the raw table (restricted access) and the masked table (RAG-serving access) to Unity Catalog.

```python
# Scenario: persist raw and masked tables to Unity Catalog so access control
# can be enforced centrally rather than relying on each application to mask at query time
raw_df.write.format("delta").mode("overwrite").saveAsTable("main.lab18_governance.intake_raw")
masked_df.select("record_id", "masked_note").write.format("delta").mode("overwrite").saveAsTable("main.lab18_governance.intake_masked")

print("Tables written to Unity Catalog")
```

```
# Expected output (illustrative)
Tables written to Unity Catalog
```

---

### Step 3 — Apply a Unity Catalog Dynamic Masking Policy

**Scenario:** Even after writing the masked table, the raw table must remain readable only by privileged users (data stewards, audit roles). A Unity Catalog column masking policy ensures that non-privileged users who query `intake_raw` receive a masked version of the `note` column automatically — without the application needing to route them to the masked table.

Create a SQL masking function and bind it to the `note` column of the raw table.

```sql
-- Scenario: enforce column-level PII masking at the Unity Catalog layer so that
-- any query against intake_raw by a non-privileged user receives masked output
-- Constraint: users in the `data_stewards` group see the original; all others see the masked value

CREATE OR REPLACE FUNCTION main.lab18_governance.mask_note_column(note STRING)
  RETURNS STRING
  RETURN CASE
    WHEN is_member('data_stewards') THEN note
    ELSE regexp_replace(
           regexp_replace(
             regexp_replace(note, '\\d{3}-\\d{2}-\\d{4}', '<REDACTED_SSN>'),
           '[a-zA-Z0-9._%+\\-]+@[a-zA-Z0-9.\\-]+\\.[a-zA-Z]{2,}', '<EMAIL>'),
         '\\d{3}-\\d{3}-\\d{4}', '<PHONE>')
  END;
```

```sql
-- Bind the masking function to the note column of the raw table
ALTER TABLE main.lab18_governance.intake_raw
  ALTER COLUMN note
  SET MASK main.lab18_governance.mask_note_column;
```

```sql
-- Verify: query the raw table as a non-privileged user (current user is not in data_stewards)
SELECT record_id, note FROM main.lab18_governance.intake_raw ORDER BY record_id;
```

```
# Expected output (illustrative) — current user is NOT in data_stewards group
+---------+-----------------------------------------------------------+
|record_id|note                                                       |
+---------+-----------------------------------------------------------+
|1        |Patient John Smith called <PHONE>. SSN: <REDACTED_SSN>.   |
|2        |Contact <EMAIL> for follow-up on 2024-01-15.               |
|3        |No PII present. Routine check-in completed without ...    |
|4        |Dr. Alice Brown, NPI 1234567890, reviewed the case for ....|
+---------+-----------------------------------------------------------+
```

**What to notice:** The Unity Catalog masking policy applies automatically at query time. The application code does not need to be updated — any future query against `intake_raw` by a non-privileged principal receives masked output. A user in the `data_stewards` group would receive the unmasked original.

---

### Step 4 — Observe AI Gateway PII Blocking Behaviour

**Scenario:** A downstream application sends a prompt that contains a raw SSN to an AI Gateway–proxied LLM endpoint. The gateway must block the request and return a rejection response rather than forwarding the PII to the model.

First, create an AI Gateway route with PII detection enabled (SDK configuration shown for reference).

```python
# Scenario: configure an AI Gateway route that intercepts PII-bearing payloads
# before they reach the model endpoint — AI Gateway is the last-resort backstop,
# not the primary masking layer
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

w.serving_endpoints.create(
    name="lab18-pii-guarded-endpoint",
    config={
        "served_models": [{"model_name": "databricks-dbrx-instruct", "scale_to_zero_enabled": True}],
        "traffic_config": {"routes": [{"served_model_name": "databricks-dbrx-instruct", "traffic_percentage": 100}]},
        "ai_gateway": {
            "guardrails": {
                "input": {
                    "pii": {"behavior": "BLOCK"},       # block request if PII detected in prompt
                    "safety": True,
                },
                "output": {
                    "pii": {"behavior": "BLOCK"},       # block response if model echoes PII
                    "safety": True,
                }
            },
            "usage_tracking_config": {"enabled": True},
            "inference_table_config": {
                "catalog_name": "main",
                "schema_name": "lab18_governance",
                "table_name_prefix": "gateway_inference",
                "enabled": True,
            }
        }
    }
)
print("Endpoint configuration submitted")
```

```
# Expected output (illustrative)
Endpoint configuration submitted
```

Now send two requests: a clean prompt and a PII-bearing prompt.

```python
# Scenario: validate that the PII guardrail correctly distinguishes safe and unsafe payloads
import requests, json

ENDPOINT_URL = "https://<workspace-host>/serving-endpoints/lab18-pii-guarded-endpoint/invocations"
HEADERS = {"Authorization": "Bearer <token>", "Content-Type": "application/json"}

# Request 1 — clean prompt, no PII
clean_payload = {"messages": [{"role": "user", "content": "Summarise the benefits of exercise."}]}
response_clean = requests.post(ENDPOINT_URL, headers=HEADERS, json=clean_payload)
print("Clean prompt status:", response_clean.status_code)
print("Clean prompt response:", response_clean.json())
```

```
# Expected output (illustrative)
Clean prompt status: 200
Clean prompt response: {
  "choices": [{"message": {"role": "assistant", "content": "Regular exercise improves cardiovascular health..."}}]
}
```

```python
# Anti-pattern: sending a prompt with raw PII directly to the model endpoint
# without pre-masking at the application layer — this treats AI Gateway as the
# primary control rather than the backstop, creating a single point of failure
pii_payload = {"messages": [{"role": "user", "content": "The patient SSN 123-45-6789 had a follow-up. Summarise."}]}
response_pii = requests.post(ENDPOINT_URL, headers=HEADERS, json=pii_payload)
print("PII prompt status:", response_pii.status_code)
print("PII prompt response:", response_pii.json())
```

```
# Expected output (illustrative) — AI Gateway blocks the request
PII prompt status: 400
PII prompt response: {
  "error_code": "BAD_REQUEST",
  "message": "Input guardrail triggered: PII detected in request payload. Request blocked."
}
```

**What to notice:** The gateway rejects the PII-bearing request with a 400 and a structured error. This is the correct fail-closed behaviour. However, this is a demonstration of the *backstop* — in a properly designed pipeline, the SSN would have been masked in Step 2 before it ever reached the serving layer, so the gateway would never need to block it.

---

## Validation

Run the following query to confirm that the masked table contains no raw SSNs, phone numbers, or email addresses.

```sql
-- Validation: confirm no PII patterns survive in the masked serving table
SELECT
  record_id,
  masked_note,
  regexp_extract(masked_note, '\\d{3}-\\d{2}-\\d{4}', 0)  AS ssn_leak,
  regexp_extract(masked_note, '[a-zA-Z0-9._%+\\-]+@[a-zA-Z0-9.\\-]+\\.[a-zA-Z]{2,}', 0) AS email_leak,
  regexp_extract(masked_note, '\\d{3}-\\d{3}-\\d{4}', 0)  AS phone_leak
FROM main.lab18_governance.intake_masked
ORDER BY record_id;
```

```
# Expected output (illustrative) — all PII columns are empty strings
+---------+-----------------------------------------------+---------+-----------+-----------+
|record_id|masked_note                                    |ssn_leak |email_leak |phone_leak |
+---------+-----------------------------------------------+---------+-----------+-----------+
|1        |Patient <PERSON> called <PHONE>. SSN: <REDAC...|         |           |           |
|2        |Contact <EMAIL> for follow-up on <DATE>.       |         |           |           |
|3        |No PII present. Routine check-in completed ....|         |           |           |
|4        |Dr. <PERSON>, NPI 1234567890, reviewed the ....|         |           |           |
+---------+-----------------------------------------------+---------+-----------+-----------+
```

All three leak columns are empty — confirming no SSNs, emails, or phone numbers survive in the masked table. The NPI on record 4 remains because it was not in scope for this lab's recogniser set. In a production HIPAA pipeline, a custom `PatternRecognizer` for NPI would extend Presidio's coverage.

---

## Teardown

```sql
-- Drop the lab schema and all tables within it
DROP SCHEMA IF EXISTS main.lab18_governance CASCADE;
```

```python
# Delete the AI Gateway–backed serving endpoint
w.serving_endpoints.delete(name="lab18-pii-guarded-endpoint")
print("Endpoint deleted")
```

```
# Expected output (illustrative)
Endpoint deleted
```

---

## Reflection Questions

1. **Detection gap:** In Step 2, the NPI number `1234567890` on record 4 was not masked because Presidio's default recogniser set does not include NPI. What two changes would you make to the pipeline to close this gap, and at which pipeline stage would each change apply?

2. **Layer responsibility:** This lab placed masking at three distinct layers — application-layer UDF (Step 2), Unity Catalog column masking policy (Step 3), and AI Gateway guardrail (Step 4). If you could only keep one layer due to an organisational constraint, which would you keep and why? What risk does removing each of the other two layers introduce?

3. **Monitoring connection:** The AI Gateway in Step 4 was configured with an inference table. After going live, how would you use the inference table to determine whether the gateway PII guardrail is producing false positives (blocking clean requests) at a rate that is affecting user experience, and what threshold would prompt you to tune the sensitivity down?
