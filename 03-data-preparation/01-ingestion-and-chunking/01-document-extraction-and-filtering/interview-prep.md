# Document Extraction and Filtering — Interview Prep

**Section:** 03 Data Preparation | **Role target:** GenAI Engineer, Data Engineer

## Core Conceptual Questions

These test whether you understand the fundamentals.

| Question | Key points to cover | Common weak-answer trap |
|---|---|---|
| Why not just use PyPDF for all document ingestion? | 1. Strips document structure (tables, lists). 2. Cannot handle scanned documents (lacks OCR). 3. Merges multi-column text improperly. | "It's the standard LangChain loader so I just use it." |
| How do you scale document extraction for 100k PDFs on Databricks? | 1. Store in Unity Catalog Volumes. 2. Read with Spark `binaryFile` format. 3. Apply Pandas UDF to distribute the parsing workload across workers. | Describing a Python `for` loop, which runs entirely on the single driver node and causes OOM errors. |
| What is the danger of leaving headers and footers in your extracted text? | 1. They repeat across chunks, polluting the vector space. 2. They cause irrelevant chunks to be retrieved (e.g. searching for "Confidential" returns every page). | Saying "the LLM is smart enough to ignore them" — the problem is retrieval, not just generation. |

## Applied / Scenario Questions

**Q:** Your RAG application is failing to answer questions that require comparing numbers from financial tables inside quarterly reports. How do you fix the ingestion pipeline?

**Strong answer framework:**
- **Diagnose:** The basic text extractor is likely flattening the tables into garbled, unaligned text strings.
- **Solution:** Swap the extractor for a layout-aware tool like Unstructured.
- **Configuration:** Use the `hi_res` strategy to leverage computer vision models that detect the bounding boxes of tables.
- **Output:** Extract the tables specifically as HTML or Markdown strings, which LLMs can easily interpret, and pass this structured text into the chunking phase.

## System Design / Architecture Questions (if applicable)

**Q:** Design a batch ingestion pipeline on Databricks that processes 50,000 mixed documents daily (some plain text policies, some heavily formatted scanned invoices) into a Vector Search index.

**Approach:**
1. **Clarify requirements:** Define latency constraints and cost budgets.
2. **Propose structure:** 
   - Files land in a Unity Catalog Volume.
   - A Databricks Job triggers a Spark structured streaming or batch job.
   - `spark.read.format("binaryFile")` distributes the files.
   - A lightweight UDF acts as a router: it checks if the document is a scanned image/invoice vs plain text.
   - Plain text goes to a fast parser (PyPDF). Scanned/complex docs go to a heavy parser (Unstructured `hi_res` with OCR).
   - Extracted text is written to a Delta Table.
3. **Justify choices and name tradeoffs explicitly:** Routing saves massive compute costs compared to running `hi_res` parsing on every single document, while still ensuring tables in invoices are captured correctly.

## Vocabulary That Signals Expertise

Use these terms naturally — don't force them:
- **Layout-aware parsing** — Crucial for retaining structure.
- **Bounding boxes** — How advanced parsers identify tables and titles.
- **Pandas UDF** — The correct Databricks mechanism for distributed Python processing.
- **`binaryFile` format** — The Spark paradigm for reading raw unstructured data.

## Vocabulary That Signals Weakness

Avoid these — they signal outdated or shallow understanding:
- **"I just run a for loop over the files"** — Signals zero understanding of distributed computing or Spark.
- **"I use the default loader"** — Signals a lack of critical thinking about data quality.

## STAR Answer Frame

**Situation:** Our enterprise RAG bot couldn't accurately answer queries based on PDF product catalogs.  
**Task:** Improve the ingestion pipeline to capture tabular product specs.  
**Action:** I migrated the ingestion job on Databricks from standard PyPDF to a distributed Spark pipeline using Unstructured. I implemented a Pandas UDF that used the `hi_res` layout strategy to extract tables as Markdown, and applied regex filters to remove repeating headers.  
**Result:** Table-related query accuracy improved by 70%, and by distributing the workload via Spark, the job processed 10,000 catalogs in 30 minutes instead of crashing the driver node.

## Red Flags Interviewers Watch For

- Candidates who treat all PDFs as if they contain identical text structures.
- Complete lack of awareness regarding Databricks compute architecture (driver vs. worker nodes) when discussing large-scale document processing.
- Focusing purely on the LLM or Vector DB while ignoring the "garbage in, garbage out" reality of poor text extraction.
