# Document Extraction and Filtering

**Lab:** LAB-02 | **Section:** 03 Data Preparation | **Module:** 01 Ingestion and Chunking | **Est. time:** 1.5 hrs

## Objective

Build a distributed document extraction pipeline on Databricks that reads raw PDFs from a Volume, extracts text while preserving table structures using a Pandas UDF, and writes the clean output to a Delta table.

## Prerequisites

- A Databricks workspace with Unity Catalog enabled.
- A cluster running DBR 14.3 LTS ML or higher.
- Basic understanding of PySpark DataFrames.

## Setup

First, install the required libraries for layout-aware extraction and create a Volume to hold our sample files.

```python
# Setup commands
%pip install unstructured[pdf] pypdf
dbutils.library.restartPython()

# Create a catalog, schema, and volume for the lab
spark.sql("CREATE CATALOG IF NOT EXISTS genai_lab")
spark.sql("CREATE SCHEMA IF NOT EXISTS genai_lab.ingestion")
spark.sql("CREATE VOLUME IF NOT EXISTS genai_lab.ingestion.raw_docs")
```

Upload a sample PDF (containing a table) to the volume using the Databricks UI, or generate a dummy one programmatically. For this lab, assume a file named `sample_report.pdf` exists in `/Volumes/genai_lab/ingestion/raw_docs/`.

## Steps

### Step 1 — Read Binary Files at Scale

Load the raw PDF files into a Spark DataFrame using the `binaryFile` format. This distributes the raw byte streams across the cluster.

```python
# Step 1 code/config
df_raw = spark.read.format("binaryFile").load("/Volumes/genai_lab/ingestion/raw_docs/*.pdf")
display(df_raw.select("path", "length"))
```

### Step 2 — Create the Extraction UDF

Wrap the `unstructured` library in a Pandas UDF. We will configure it to use the `fast` strategy for speed, but fall back to layout parsing if needed.

```python
# Step 2 code/config
from pyspark.sql.functions import pandas_udf
import pandas as pd
import io
from unstructured.partition.pdf import partition_pdf

@pandas_udf("string")
def extract_text_udf(content_series: pd.Series) -> pd.Series:
    def extract(content_bytes):
        try:
            # Using unstructured to parse the PDF bytes
            elements = partition_pdf(file=io.BytesIO(content_bytes), strategy="fast")
            # Combine the text of all extracted elements
            return "\n\n".join([str(el) for el in elements])
        except Exception as e:
            return f"Error: {str(e)}"
            
    return content_series.apply(extract)
```

### Step 3 — Apply UDF and Filter Artifacts

Apply the UDF to extract the text, and use Spark SQL functions to filter out common extraction artifacts (like a repeating header).

```python
# Step 3 code/config
from pyspark.sql.functions import col, regexp_replace

df_parsed = df_raw.withColumn("extracted_text", extract_text_udf("content"))

# Filter out a hypothetical repeating header: "CONFIDENTIAL INTERNAL DOCUMENT"
df_cleaned = df_parsed.withColumn(
    "clean_text", 
    regexp_replace(col("extracted_text"), "CONFIDENTIAL INTERNAL DOCUMENT", "")
)

# Write to Delta
df_cleaned.select("path", "clean_text").write.mode("overwrite").saveAsTable("genai_lab.ingestion.parsed_documents")
```

## Validation

Verify that the Delta table contains the extracted text and that the artifacts were successfully removed.

```python
# Validation command / expected output
df_final = spark.table("genai_lab.ingestion.parsed_documents")
display(df_final)

# Assert that the table has rows and the extracted text is not empty
assert df_final.count() > 0, "No documents were parsed."
assert df_final.filter(col("clean_text").contains("CONFIDENTIAL")).count() == 0, "Artifact filtering failed."
print("Validation passed!")
```

## Teardown

Clean up the resources created during the lab to avoid unnecessary storage costs.

```sql
DROP TABLE IF EXISTS genai_lab.ingestion.parsed_documents;
DROP VOLUME IF EXISTS genai_lab.ingestion.raw_docs;
```

## Reflection Questions

1. What would happen if you tried to process 1,000,000 PDFs by iterating through the volume path with a standard Python `for` loop on the driver?
2. If your source PDFs are scanned images without a text layer, how would you modify the UDF in Step 2?
3. How does the choice of `strategy="fast"` vs `strategy="hi_res"` in Unstructured impact the downstream chunking strategy?
