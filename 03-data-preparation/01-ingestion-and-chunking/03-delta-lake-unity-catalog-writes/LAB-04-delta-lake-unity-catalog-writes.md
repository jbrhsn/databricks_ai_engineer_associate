# Write Chunked Data to a Unity Catalog Delta Table

**Lab:** LAB-04 | **Section:** Ingestion and Chunking | **Module:** Data Preparation | **Est. time:** 1 hr

## Objective

By the end of this lab, you will have simulated the extraction of text chunks from a document, converted them into a Spark DataFrame, and written them to a Unity Catalog Delta table with Change Data Feed enabled—preparing the table for a downstream Vector Search index.

## Prerequisites

- Access to a Databricks workspace.
- A cluster running Databricks Runtime 14.3 LTS ML or higher.
- `USE CATALOG` and `CREATE SCHEMA` privileges on a Unity Catalog catalog (e.g., `main` or your personal dev catalog).

## Setup

First, let's create a dedicated schema for our GenAI data preparation work.

```python
# Setup commands
catalog_name = "main" # Replace with your catalog if needed
schema_name = "genai_prep_lab"

spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog_name}")
spark.sql(f"USE CATALOG {catalog_name}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")
spark.sql(f"USE SCHEMA {schema_name}")

print(f"Using catalog: {catalog_name}, schema: {schema_name}")
```

## Steps

### Step 1 — Simulate Chunked Document Generation

In a real pipeline, you would use a library like LangChain or PyMuPDF to read a file from a UC Volume and split it into chunks. Here, we will simulate the output of that process: a list of Python dictionaries representing text chunks.

```python
# Step 1 code/config
import uuid

doc_1_id = str(uuid.uuid4())
doc_2_id = str(uuid.uuid4())

# Simulating chunks generated from an ingestion process
raw_chunks = [
    {"chunk_id": str(uuid.uuid4()), "doc_id": doc_1_id, "text": "Databricks provides a unified data platform.", "source": "intro.pdf"},
    {"chunk_id": str(uuid.uuid4()), "doc_id": doc_1_id, "text": "It includes Unity Catalog for governance.", "source": "intro.pdf"},
    {"chunk_id": str(uuid.uuid4()), "doc_id": doc_2_id, "text": "Vector Search requires Change Data Feed to be enabled.", "source": "vector_docs.txt"}
]

print(f"Generated {len(raw_chunks)} chunks.")
```

### Step 2 — Convert to a Spark DataFrame

To write to Delta Lake efficiently, we must convert our Python dictionaries into a native Spark DataFrame.

```python
# Step 2 code/config
from pyspark.sql.types import StructType, StructField, StringType

# Define explicit schema for safety
schema = StructType([
    StructField("chunk_id", StringType(), False),
    StructField("doc_id", StringType(), False),
    StructField("text", StringType(), True),
    StructField("source", StringType(), True)
])

# Create the DataFrame
df_chunks = spark.createDataFrame(raw_chunks, schema=schema)

display(df_chunks)
```

### Step 3 — Write to Delta with Change Data Feed Enabled

This is the critical step. We must enable `delta.enableChangeDataFeed` so that a downstream Databricks Vector Search index can incrementally sync new rows.

```python
# Step 3 code/config
table_name = "chunked_documents"

(df_chunks.write
    .format("delta")
    .mode("append")
    .option("delta.enableChangeDataFeed", "true")
    .saveAsTable(table_name))

print(f"Successfully wrote chunks to {catalog_name}.{schema_name}.{table_name}")
```

### Step 4 — Simulate an Update (Append new chunks)

Let's append a new chunk to simulate a pipeline running the next day. Because CDF is enabled, Delta will log this specific insert event.

```python
# Step 4 code/config
new_chunk = [{"chunk_id": str(uuid.uuid4()), "doc_id": str(uuid.uuid4()), "text": "This chunk was added later.", "source": "update.txt"}]
df_new = spark.createDataFrame(new_chunk, schema=schema)

(df_new.write
    .format("delta")
    .mode("append")
    .saveAsTable(table_name))

print("Appended new chunk.")
```

## Validation

Verify that the table exists, contains our data, and—most importantly—that the Change Data Feed recorded our transactions. We can query the `table_changes` function to see the CDF log.

```python
# Validation command / expected output
# 1. Verify standard table query
display(spark.sql(f"SELECT * FROM {table_name}"))

# 2. Verify Change Data Feed is actively tracking row-level events
# We query the changes from version 0 (table creation) onwards
display(spark.sql(f"SELECT * FROM table_changes('{table_name}', 0)"))
```
*Expected output: The `table_changes` query should return 4 rows total. You will see columns like `_change_type` (which will be 'insert' for all rows) and `_commit_version`, proving that Vector Search has the metadata it needs to sync incrementally.*

## Teardown

Clean up the resources created during this lab.

```python
# Drop the table and schema
spark.sql(f"DROP TABLE IF EXISTS {catalog_name}.{schema_name}.{table_name}")
spark.sql(f"DROP SCHEMA IF EXISTS {catalog_name}.{schema_name}")
print("Teardown complete.")
```

## Reflection Questions

1. If you ran `df.write.mode("overwrite").saveAsTable(table_name)` in Step 4 instead of `append`, how would that affect a downstream Vector Search continuous sync?
2. Why do we explicitly define a `StructType` schema in Step 2 rather than letting Spark infer the schema from the dictionaries?
3. How does this architecture (Volume → Spark DataFrame → Delta Table) compare to saving chunked JSON files back into a Volume?