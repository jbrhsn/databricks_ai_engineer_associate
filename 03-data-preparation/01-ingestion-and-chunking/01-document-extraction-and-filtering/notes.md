# Document Extraction and Filtering

**Section:** 03 Data Preparation | **Module:** 01 Ingestion and Chunking | **Est. time:** 2 hrs | **Exam mapping:** Domain 2: Data Preparation

---

## TL;DR

Document extraction is the process of converting raw, unstructured files (like PDFs, Word docs, and HTML) into clean, machine-readable text augmented with metadata, preparing it for chunking and embedding. The most critical step in a Retrieval-Augmented Generation (RAG) pipeline happens here. **The one thing to remember: The extraction library you choose dictates whether you retain critical structural context like tables, lists, and semantic hierarchy.**

---

## ELI5 — Explain It Like I'm 5

Think of document extraction like taking apart a complex Lego castle so you can use the pieces to build something new. If you just smash the castle with a hammer (basic text extraction), you get a messy pile of loose bricks and completely lose track of what used to be a tower or a wall. If you take it apart carefully and label the pieces as you go (smart, layout-aware extraction), you know exactly which bricks belonged where. In RAG, if you just smash a PDF into text, you lose the tables and headings. If you extract it carefully, the AI knows exactly what information belongs to a specific section, leading to much better answers.

---

## Learning Objectives

By the end of this chapter you will be able to:
- [ ] Implement document extraction pipelines using PyPDF and Unstructured within Databricks.
- [ ] Compare extraction tools based on layout retention, speed, and support for multi-modal elements.
- [ ] Design a distributed document parsing architecture leveraging Databricks Volumes and Apache Spark UDFs.
- [ ] Diagnose and filter common extraction artifacts such as headers, footers, and garbled OCR output.

---

## Visual Overview

### Single Document Extraction Flow

```
Raw File (PDF)
      │
      ├──► Basic Extraction (PyPDF) ─────► Raw Text (Loss of tables)
      │
      └──► Layout-Aware (Unstructured) ──► Elements (Title, Table, Text) + Metadata
```

### Distributed Databricks Extraction Architecture

```
Databricks Unity Catalog Volume (Raw PDFs)
      │
      ▼
spark.read.format("binaryFile").load("/Volumes/.../")
      │
      ▼
Spark DataFrame (path, content as bytes)
      │
      ▼ (Apply Pandas UDF)
Distributed Parsing Nodes (Extract text & metadata)
      │
      ▼
Delta Table (Clean Parsed Documents)
```

---

## Key Concepts

### Basic Text Extraction (e.g., PyPDF)

Basic text extraction relies on parsing the internal text streams of a document format like PDF. Tools like `PyPDF` or `pdfminer` read the underlying text characters encoded in the file. Mechanistically, they map the byte streams to characters but do not understand the visual rendering or spatial relationship of the text on the page. In the Databricks ecosystem, these are typically lightweight Python libraries used inside a pandas UDF for rapid, large-scale extraction where layout (like complex tables) is not present or not important.

### Layout-Aware Extraction (e.g., Unstructured)

Layout-aware extraction preserves the structural integrity of a document. Libraries like `Unstructured` use computer vision models (like YOLOX) or advanced heuristic parsing to detect elements such as titles, narrative text, lists, and tables. Under the hood, the document is rendered as an image or heavily analyzed spatially to classify every bounding box before extracting the text within it. In LangChain or Databricks GenAI workflows, these tools output rich JSON objects containing both the text and its classification, which is critical for table-heavy enterprise documents.

### Distributed Processing with Databricks Volumes

Reading thousands of PDFs sequentially on a single node is a massive bottleneck. Distributed processing leverages Apache Spark to parallelize the workload. Mechanistically, files are stored in Unity Catalog Volumes. Spark reads these files as a binary stream (`binaryFile` format), distributing the byte data across worker nodes. A user-defined function (UDF) containing the extraction logic (like a PyPDF parser) is then applied to each partition in parallel. This manifests in Databricks as a standard PySpark DataFrame transformation pipeline that writes the extracted output directly into a Delta Lake table for downstream chunking.

### Key Parameters / Configuration Knobs

| Parameter | What it controls | Decision rule |
|---|---|---|
| `strategy` (Unstructured) | Fast text extraction vs. AI-driven layout parsing | Use `fast` for plain text documents to save compute; use `hi_res` when tables, charts, and complex layouts must be preserved. |
| `pdf_image_dpi` | Resolution of the document when converted to image for OCR | Increase to 300+ if OCR is missing small text; keep at 200 for faster processing of standard fonts. |
| `infer_table_structure` | Whether to extract tables as HTML/Markdown rather than raw text | Set to `True` if the document contains data tables that LLMs will need to answer analytical questions. |

### Worked Example: Requirement → Decision

**Given:** An enterprise needs to ingest 50,000 quarterly financial reports (PDFs) to build an internal Q&A bot for financial analysts. 
- **Step 1 — Identify the goal:** Extract readable text from the reports to populate a Vector Database.
- **Step 2 — Define inputs:** 50,000 PDF files stored in an AWS S3-backed Databricks Volume.
- **Step 3 — Define outputs:** Clean text, explicitly keeping tabular financial data intact so the LLM can compare Q1 vs Q2 revenues.
- **Step 4 — Apply constraints:** The extraction must finish within a 2-hour batch window, and tabular data cannot be jumbled into a single unreadable string.
- **Step 5 — Select the approach:** Use Spark `binaryFile` read combined with a Pandas UDF wrapping the `Unstructured` library with `strategy="hi_res"`. *Rationale: Spark provides the parallelization needed to hit the 2-hour constraint, while Unstructured `hi_res` ensures the critical financial tables are preserved as structured text (HTML/Markdown) rather than garbled strings, which PyPDF would do.*

---

## Implementation

```python
# Scenario: Distributing PDF extraction across a Databricks cluster for massive scale.
from pyspark.sql.functions import pandas_udf
import pandas as pd
import io
from pypdf import PdfReader

@pandas_udf("string")
def extract_text_pypdf(content_series: pd.Series) -> pd.Series:
    def extract(content_bytes):
        try:
            reader = PdfReader(io.BytesIO(content_bytes))
            return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        except Exception:
            return ""
    return content_series.apply(extract)

# Read binary files from a Unity Catalog Volume
df_raw = spark.read.format("binaryFile").load("/Volumes/main/default/raw_pdfs/*.pdf")

# Apply distributed extraction
df_parsed = df_raw.withColumn("extracted_text", extract_text_pypdf("content"))
display(df_parsed.select("path", "extracted_text"))
```

```python
# Anti-pattern: Extracting thousands of files in a single-threaded Python loop on the driver node.
import os
from pypdf import PdfReader

# This will take hours and eventually cause an OutOfMemory error on the driver.
results = []
for file in os.listdir("/Volumes/main/default/raw_pdfs/"):
    reader = PdfReader(f"/Volumes/main/default/raw_pdfs/{file}")
    text = "\n".join([page.extract_text() for page in reader.pages])
    results.append({"file": file, "text": text})

# Correct approach:
# Leverage Spark to distribute the workload across the cluster using spark.read.format("binaryFile")
# as shown in the previous snippet, ensuring horizontal scalability.
```

---

## Common Pitfalls & Misconceptions

- **Using basic extractors for everything** — Beginners often use PyPDF because it is fast and easy to install. However, this strips out tables and document structures, making it impossible for the LLM to answer data-driven questions. You must use layout-aware extractors for complex documents.
- **Ignoring headers, footers, and page numbers** — Beginners index the raw extracted text as-is. This pollutes the vector space with repeating, useless text (e.g., "Company Confidential - Page 14"), degrading search relevance. You must apply filtering steps post-extraction to strip these out.
- **Single-node processing bottlenecks** — Developers often write a standard Python `for` loop to process documents on a Databricks notebook. This runs solely on the driver node, ignoring the cluster's compute power. Always use Spark UDFs to parallelize document parsing.

---

## Key Definitions

| Term | Definition |
|---|---|
| Layout-Aware Parsing | The process of identifying structural elements (tables, titles, narrative) in a document before extracting the text, preserving the visual context. |
| OCR (Optical Character Recognition) | Technology that converts images of typed, handwritten, or printed text into machine-encoded text. Required for scanned PDFs without embedded text layers. |
| Pandas UDF | A user-defined function in PySpark executed using Apache Arrow to vectorize operations, allowing efficient application of Python libraries (like Unstructured) across a distributed dataset. |

---

## Summary / Quick Recall

- Basic extraction (PyPDF) is fast but destroys layout and tables.
- Layout-aware extraction (Unstructured) uses AI/heuristics to preserve tables, titles, and lists.
- Process documents at scale in Databricks by reading from Volumes using `spark.read.format("binaryFile")`.
- Distribute extraction logic across worker nodes using Spark Pandas UDFs.
- Always filter headers, footers, and OCR garbage before chunking to maintain vector DB quality.

---

## Self-Check Questions

1. Which library is best suited for quickly extracting raw, unstructured text from a purely text-based PDF where layout does not matter?

   <details><summary>Answer</summary>
   
   **PyPDF (or pdfminer).** This is the correct answer because it directly reads the text streams without the overhead of layout parsing or OCR, making it extremely fast for plain text. Unstructured with hi_res would be overkill and much slower.
   
   </details>

2. You are processing scanned invoices and need to ensure the data within the tables is accurately passed to an LLM. Which approach should you take?

   <details><summary>Answer</summary>
   
   **Use a layout-aware extractor (like Unstructured) with a high-resolution strategy and OCR enabled.** This is correct because scanned invoices lack an embedded text layer (requiring OCR) and contain tabular data that requires structural preservation. Using PyPDF would fail because it cannot perform OCR, and basic extraction would destroy the table alignment.
   
   </details>

3. **Which TWO** techniques are critical for optimizing the processing of 100,000 PDF documents in a Databricks environment?
   - A. Looping through the files using a Python `for` loop on the driver node.
   - B. Storing the files in a Unity Catalog Volume.
   - C. Using `spark.read.format("binaryFile")` to load the data into a DataFrame.
   - D. Converting all PDFs to Word documents before extraction.
   - E. Using a single-node cluster to avoid network shuffle.

   <details><summary>Answer</summary>
   
   **B and C.** Storing files in a Volume (B) provides governed, scalable storage accessible to the cluster. Using `binaryFile` (C) allows Spark to distribute the byte streams across worker nodes for parallel processing. A is a massive anti-pattern that will cause an OOM error. D is an unnecessary intermediate step. E prevents distributed processing, defeating the purpose of Spark.
   
   </details>

4. After running a basic extraction pipeline, your RAG application frequently returns answers containing "Page 42" and "Confidential Internal Use Only". What step was missed in the data preparation phase?

   <details><summary>Answer</summary>
   
   **Filtering and post-processing.** The pipeline successfully extracted text but failed to filter out structural artifacts like headers and footers. These repeating elements degrade retrieval quality and must be removed using regex, heuristics, or layout-aware element filtering before the text is chunked.
   
   </details>

5. You need to balance compute costs with extraction quality. Your dataset consists of 80% plain text employee handbooks and 20% complex financial statements with tables. What is the most cost-effective architectural approach?

   <details><summary>Answer</summary>
   
   **Implement a routing step: use a fast extractor (PyPDF or Unstructured 'fast') for the handbooks, and a heavy layout-aware extractor (Unstructured 'hi_res') for the financial statements.** Applying a heavy vision-based layout parser to all documents wastes expensive compute on simple text files. Routing documents based on complexity ensures you only pay for layout preservation where it is strictly required.
   
   </details>

---

## Further Reading

- [Databricks: Read binary files](https://docs.databricks.com/en/query/formats/binary.html) — *verified 2026-07-11* — Official documentation on reading binary files (like PDFs) into Spark DataFrames.
- [LangChain: Document Loaders](https://python.langchain.com/v0.2/docs/concepts/#document-loaders) — *verified 2026-07-11* — Overview of loading documents from unstructured formats.
- [Unstructured Documentation](https://unstructured-io.github.io/unstructured/) — *verified 2026-07-11* — Core concepts on document partitioning and layout strategies.
