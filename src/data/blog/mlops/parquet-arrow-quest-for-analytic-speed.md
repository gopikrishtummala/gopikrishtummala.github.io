---
author: Gopi Krishna Tummala
pubDatetime: 2025-12-03T00:00:00Z
modDatetime: 2025-12-03T00:00:00Z
title: "The DNA of Data: Parquet, Arrow, and the Quest for Analytic Speed"
slug: parquet-arrow-quest-for-analytic-speed
featured: true
draft: false
tags:
  - mlops
  - data-engineering
  - parquet
  - arrow
  - data-formats
  - performance
description: "The unsung hero of modern data processing is how we structure data itself. Learn how Apache Parquet and Apache Arrow solve the fundamental trade-off between storage efficiency and compute speed in large-scale analytics and ML pipelines."
track: MLOps & Production
difficulty: Advanced
interview_relevance:
  - System Design
  - ML-Infra
estimated_read_time: 45
---

*By Gopi Krishna Tummala*

---

## Introduction: The Library Analogy

Welcome to the cutting edge of data systems. While algorithms and hardware often take center stage, the unsung hero—or villain—of modern data processing is the way we structure data itself.

Imagine you have a massive library (your dataset). If the books are organized randomly, finding a single sentence takes forever. If they are perfectly shelved by topic (**columnar storage**), but the pages are written in a foreign language (unoptimized format), you still hit a wall.

This is the problem that formats like **Apache Parquet**, **Apache Arrow**, and **ORC** solve. They are essentially competing standards for organizing the "pages" and "shelves" of our massive data libraries to maximize analytic performance.

---

## 1. The Core Duo: Disk vs. Memory 💾/🧠

The fundamental trade-off in modern analytics is the split between **data at rest** (on disk) and **data in motion** (in memory).

### A. Apache Parquet: The Master of the Disk 🏆 (Data at Rest)

Parquet is a columnar storage format optimized for space efficiency and minimizing I/O operations. Think of Parquet as an **archivist** meticulously compressing and indexing data on permanent storage (like S3 or HDFS).

**How it works (The Columnar Advantage):**

Instead of storing a full row (e.g., all details for a single customer) together, Parquet stores all values for a single column (e.g., all customer names) together.

**Benefit:** Queries often only read a few columns. Parquet allows the system to skip reading irrelevant columns entirely (**column pruning**).

**Key Feature: Smart Encoding & Skipping:**

Parquet employs highly effective encoding techniques like:

- **Dictionary Encoding (DICT)**: Replaces repeated values with dictionary indices
- **Run-Length Encoding (RLE)**: Compresses sequences of identical values
- **Bit-Packed Encoding (BP)**: Efficiently stores small integer values

Crucially, it stores statistics like min/max values (**zone maps**) in its metadata, allowing the engine to skip large sections of the file (**data skipping**) when filtering.

**Trade-off:** Parquet is space-efficient, achieving the **best overall compression ratio** (typically around 0.13 compression ratio). However, getting the data ready for processing requires CPU-intensive **decoding** and **decompression**.

**Example:**
```
Row-oriented (CSV-like):
Row 1: [Name: "Alice", Age: 30, City: "NYC", Salary: 100000]
Row 2: [Name: "Bob", Age: 25, City: "SF", Salary: 120000]

Columnar (Parquet):
Column Name: ["Alice", "Bob"]
Column Age: [30, 25]
Column City: ["NYC", "SF"]
Column Salary: [100000, 120000]
```

When querying for "average salary", Parquet only reads the Salary column, skipping Name, Age, and City entirely.

### B. Apache Arrow: The Speed Demon of RAM ⚡ (Data in Motion)

Arrow is a **language-agnostic, in-memory columnar data format** optimized for blazing-fast computation. If Parquet is the archivist, Arrow is the **zero-overhead desk organizer** for the CPU.

**How it works (Zero-Copy):**

Arrow defines a standardized memory layout that is ready for modern CPU architectures (like SIMD, or vectorization) and GPUs. Since all components of a query engine can understand this *exact* memory layout, data can be passed between different systems (Python/Pandas, Spark, C++, R) without costly **serialization** and **deserialization** (the **zero-copy** concept).

**The Arrow vs. Database Dilemma (Querying):**

Arrow is designed for high-speed interoperability, but by default, it provides **no encoding support** for numeric types, or only limited support for strings. This means the data is unencoded, which is why it has a **poor compression ratio** (typically around 1.07 compression ratio by default). The perceived difficulty in "querying" Arrow stems from this unencoded, **plain-memory** nature; traditional database systems prefer to query data that is already **encoded** to save space and enable *direct querying* in the encoded domain.

**Trade-off:** Arrow offers the **best decompression/transcoding throughput** (speed) but has the **worst compression ratio** without explicit encoding (size).

**Zero-Copy Example:**
```python
# Traditional approach (expensive):
pandas_df → serialize to JSON → network transfer → deserialize → numpy array
# Cost: Serialization + Network + Deserialization

# Arrow approach (zero-copy):
pandas_df → Arrow format → network transfer → Arrow format → numpy array
# Cost: Only network transfer (same memory layout everywhere)
```

---

## 2. The Trade-Offs: Performance vs. Storage Depth ⚖️

The core tension: you cannot have maximum compression *and* maximum speed simultaneously. This fundamental trade-off has been systematically evaluated in recent research [[1]](https://www.vldb.org/pvldb/vol16/p3044-liu.pdf), which provides the empirical foundation for understanding these format characteristics.

| Dimension | Parquet (On-Disk Optimized) | Arrow (In-Memory Optimized) | ORC (Balanced Storage) | Key Learning |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Goal** | Minimize disk size and I/O | Maximize in-memory compute speed | Read-heavy analytical storage | Different formats for different stages |
| **Compression Ratio** | **Best** (0.13 CR) | **Worst** (1.07 CR by default) | Good (0.27 CR) | Compression is essential for I/O efficiency [[1]](https://www.vldb.org/pvldb/vol16/p3044-liu.pdf) |
| **Transcoding Speed** | Slower due to intensive decoding | **Fastest** (Zero-copy read capability) | Worse than Parquet (especially for Zstd/zlib) | Zero-copy is faster than decoding |
| **Data Skipping** | **Fine-Grained (Record-level)** | Only Chunk-level, requiring full chunk reading | Chunk-level, but with smaller batches/stripes, offering better opportunity than Arrow | Skipping only necessary data is critical for low-selectivity queries |
| **"Point Query" Access** | **Best** for very low selectivity (finding a few records) because it decodes only what's needed | Worst by default, as it needs to load entire row batches | Better than Parquet at slightly higher selectivity due to efficient bulk loading | Format choice depends on query pattern |

### The Opportunity: A Unified Co-Design

Recent research by Liu et al. [[1]](https://www.vldb.org/pvldb/vol16/p3044-liu.pdf) demonstrates that the systems of the future will need to **co-design a unified in-memory and on-disk representation**. This means:

**Parquet needs an in-memory companion:**

Leveraging Parquet's encoded data *directly* in memory, avoiding the expensive conversion to Arrow when possible. An augmented Parquet variant showed up to **100x speedup** over the streaming baseline by adding an in-memory representation, direct query, and SIMD.

**Key Innovation:**
- Keep data in encoded format in memory
- Enable direct querying on encoded dictionaries
- Use SIMD operations for vectorized processing
- Avoid decompression until absolutely necessary

**Arrow needs more encoding:**

Implementing optimizations like **Direct Querying** (running predicates on the encoded dictionary) on Arrow files, which can lead to significant speedups (2x to 4x) by skipping the data decoding step.

**Key Innovation:**
- Add dictionary encoding support to Arrow
- Enable predicate pushdown on encoded data
- Maintain zero-copy benefits while gaining compression
- Bridge the gap between storage and compute efficiency

---

## 3. Large-Scale Data Loading and Inference Pipelines 🤖

The true value of this design lies in distributed systems and machine learning (ML) where **data movement** is the primary bottleneck.

### A. The Parquet ➡️ Arrow Pipeline (Loading)

In large-scale data loaders (like **Apache Spark**, **Dask**, or **dlt**), the combined format strategy is key:

**1. Storage (Parquet):**

The dataset is stored as **Parquet files** in a data lake for its compression and efficient columnar pruning capabilities. This minimizes cloud storage costs and I/O time.

**Benefits:**
- Reduced storage costs (10x compression typical)
- Faster data transfer from object storage
- Column pruning reduces I/O by 90%+ for selective queries
- Zone maps enable data skipping at file level

**2. Ingestion/Transformation (Arrow):**

When a data loader (e.g., PySpark) needs the data, it reads the Parquet file and converts it into an **Arrow Table** in memory.

**The Conversion Process:**
```
Parquet File (Disk)
  ↓ (Read + Decompress + Decode)
Arrow Table (Memory)
  ↓ (Zero-Copy Operations)
Pandas DataFrame / NumPy Array / Spark DataFrame
```

**3. Zero-Copy Execution:**

The data remains in the Arrow format while moving between Python (Pandas/NumPy) and the underlying C++/Java execution engines (like Spark or Ray). This **zero-copy interoperability** eliminates the serialization/deserialization bottleneck at every step of ETL/feature engineering.

**Real-World Example:**

Tools like **dlt** serialize data directly into Parquet from Arrow and deserialize directly back to Arrow, avoiding Python row-by-row processing and leveraging fast C++ libraries like `pyarrow`.

```python
# Traditional ETL (slow):
for row in csv_reader:
    process_row(row)  # Python loop, row-by-row
    write_to_parquet(row)

# Arrow-based ETL (fast):
arrow_table = pyarrow.read_csv(file)  # Bulk read
arrow_table = transform(arrow_table)  # Vectorized operations
pyarrow.write_table(arrow_table, "output.parquet")  # Bulk write
```

**Performance Gains:**
- 10-100x faster than row-by-row processing
- Memory efficient (no intermediate copies)
- Leverages SIMD for vectorized operations

### B. Arrow and Distributed Inference (Inference)

For large-scale ML inference pipelines, Arrow provides two crucial advantages for distributed computing:

**1. Distributed Data Transfer (Arrow Flight):**

In a cluster (like **Ray** or **Spark**) where models run across many nodes, data must be transferred quickly. **Arrow Flight** is an RPC framework specifically designed to stream large amounts of Arrow data across the network with **minimal overhead**. This is essential for low-latency, real-time analytics and distributed ML.

**How Arrow Flight Works:**

```
Client                    Server
  |                         |
  |-- Arrow Flight Request->|
  |                         |
  |<-- Arrow Batch Stream --|
  |<-- Arrow Batch Stream --|
  |<-- Arrow Batch Stream --|
```

**Key Features:**
- **Streaming**: Data flows as it's processed, not all at once
- **Zero-Copy**: Network protocol uses Arrow's memory layout directly
- **Parallel**: Multiple streams can run concurrently
- **Low Latency**: Optimized for real-time scenarios

**Use Cases:**
- Real-time feature serving
- Distributed model inference
- Stream processing pipelines
- Federated learning data transfer

**2. GPU Acceleration (RAPIDS):**

Arrow's standardized columnar memory format is optimized for **SIMD** and is the core format for GPU-accelerated libraries like **RAPIDS cuDF**. This allows data to be moved directly to the GPU for feature engineering and inference without expensive conversions, maximizing hardware utilization.

**The GPU Pipeline:**

```
CPU Memory (Arrow)
  ↓ (Direct Memory Transfer)
GPU Memory (cuDF)
  ↓ (GPU Processing)
GPU Results (cuDF)
  ↓ (Direct Memory Transfer)
CPU Memory (Arrow)
```

**Benefits:**
- **No CPU-GPU Conversion**: Arrow format is GPU-ready
- **SIMD Optimization**: Vectorized operations on GPU
- **Memory Efficiency**: Direct memory mapping
- **High Throughput**: Process billions of rows per second

**ML Framework Integration:**

ML frameworks like **TensorFlow I/O** and **PyTorch** also leverage Arrow for faster data ingestion into the training/inference loops.

**Example: TensorFlow with Arrow**
```python
import tensorflow_io as tfio

# Read Parquet directly into TensorFlow
dataset = tfio.IODataset.from_parquet("data.parquet")
# Arrow format is used internally for zero-copy transfer
```

**Example: PyTorch with Arrow**
```python
import pyarrow.parquet as pq

# Read Parquet to Arrow
table = pq.read_table("data.parquet")
# Convert to PyTorch tensor with minimal overhead
tensor = torch.from_numpy(table.column("features").to_numpy())
```

---

## 4. Encoding Techniques: The Compression Arsenal 🗜️

Understanding the encoding techniques is crucial for optimizing data pipelines.

### Dictionary Encoding (DICT)

**How it works:**
- Build a dictionary of unique values
- Replace values with dictionary indices
- Store dictionary separately

**Example:**
```
Original: ["NYC", "SF", "NYC", "LA", "SF", "NYC"]
Dictionary: {0: "NYC", 1: "SF", 2: "LA"}
Encoded: [0, 1, 0, 2, 1, 0]
```

**Benefits:**
- Excellent compression for low-cardinality columns
- Enables direct querying on encoded values
- Fast lookups via dictionary

**Trade-offs:**
- Dictionary must fit in memory
- Less effective for high-cardinality columns

### Run-Length Encoding (RLE)

**How it works:**
- Compress sequences of identical values
- Store as (value, count) pairs

**Example:**
```
Original: [1, 1, 1, 1, 2, 2, 3, 3, 3]
Encoded: [(1, 4), (2, 2), (3, 3)]
```

**Benefits:**
- Excellent for sorted or nearly-sorted data
- Very fast encoding/decoding
- Minimal CPU overhead

**Use Cases:**
- Time-series data
- Sorted columns
- Boolean columns

### Bit-Packed Encoding (BP)

**How it works:**
- Store small integers using minimal bits
- Pack multiple values into single bytes

**Example:**
```
Values: [3, 5, 2, 7] (all < 8, need 3 bits each)
Packed: 011 101 010 111 (12 bits = 1.5 bytes vs 16 bytes original)
```

**Benefits:**
- Maximum compression for small integers
- Direct bit manipulation
- SIMD-friendly operations

**Use Cases:**
- Integer IDs
- Enumerated types
- Small numeric ranges

### Delta Encoding

**How it works:**
- Store differences between consecutive values
- Often combined with RLE for sorted data

**Example:**
```
Original: [100, 102, 105, 109, 114]
Deltas: [100, +2, +3, +4, +5]
```

**Benefits:**
- Excellent for sorted sequences
- Smaller values compress better
- Works well with RLE

---

## 5. Zone Maps and Data Skipping: The Query Optimizer's Best Friend 🗺️

**Zone Maps** are metadata structures that store min/max values for data chunks, enabling powerful query optimizations.

### How Zone Maps Work

For each column chunk, Parquet stores:
- Minimum value
- Maximum value
- Null count
- Distinct count (optional)

**Example:**
```
Query: WHERE age > 50

Zone Map for age column:
Chunk 1: min=18, max=35  → Skip (all values < 50)
Chunk 2: min=25, max=45  → Skip (all values < 50)
Chunk 3: min=40, max=60  → Read (may contain values > 50)
Chunk 4: min=55, max=75  → Read (all values > 50)
```

**Result:** Only 2 out of 4 chunks need to be read, reducing I/O by 50%.

### Fine-Grained vs. Chunk-Level Skipping

**Parquet (Fine-Grained):**
- Record-level statistics
- Can skip individual row groups
- Most granular skipping capability

**Arrow (Chunk-Level):**
- Only chunk-level statistics
- Must read entire chunk if any value matches
- Less efficient for selective queries

**ORC (Balanced):**
- Stripe-level statistics
- Smaller stripes than Arrow chunks
- Better than Arrow, worse than Parquet

### Real-World Impact

For a query selecting 1% of rows:
- **Parquet**: Reads ~1-5% of data (excellent skipping)
- **Arrow**: Reads ~10-20% of data (chunk-level only)
- **ORC**: Reads ~5-10% of data (stripe-level)

---

## 6. The Future: Unified Co-Design 🚀

The next generation of data systems will bridge the gap between Parquet and Arrow.

### Augmented Parquet: In-Memory Representation

**Key Innovations:**

1. **Direct Query on Encoded Data:**
   - Keep data in dictionary-encoded format in memory
   - Run predicates directly on dictionary indices
   - Avoid decompression until final output

2. **SIMD-Optimized Operations:**
   - Vectorized operations on encoded data
   - Leverage CPU SIMD instructions
   - Process multiple values in parallel

3. **Hybrid Format:**
   - Parquet on disk (compressed)
   - Augmented Parquet in memory (encoded but queryable)
   - Arrow for inter-system transfer (when needed)

**Performance Gains:**
- Up to **100x speedup** over streaming baseline [[1]](https://www.vldb.org/pvldb/vol16/p3044-liu.pdf)
- Maintains Parquet's compression benefits
- Gains Arrow's zero-copy advantages

### Enhanced Arrow: Encoding Support

**Key Innovations:**

1. **Dictionary Encoding in Arrow:**
   - Add dictionary type support
   - Enable direct querying on dictionaries
   - Maintain zero-copy semantics

2. **Predicate Pushdown:**
   - Run filters on encoded data
   - Skip decoding for filtered-out rows
   - 2x to 4x speedup demonstrated [[1]](https://www.vldb.org/pvldb/vol16/p3044-liu.pdf)

3. **Compression Options:**
   - Optional compression for Arrow files
   - Trade speed for storage when needed
   - Flexible compression levels

---

## 7. Practical Recommendations 💡

### When to Use Parquet

✅ **Best for:**
- Long-term storage (data lakes, archives)
- High compression requirements
- Selective queries (low selectivity)
- Cost-sensitive storage (cloud object storage)
- Batch processing pipelines

**Example Use Cases:**
- Data warehouse storage
- ML training data archives
- Log file storage
- Historical data retention

### When to Use Arrow

✅ **Best for:**
- In-memory analytics
- Real-time processing
- Inter-system data transfer
- GPU acceleration
- Low-latency queries

**Example Use Cases:**
- Feature engineering pipelines
- Real-time dashboards
- Distributed ML inference
- Stream processing
- Interactive analytics

### Hybrid Approach: The Best of Both Worlds

**Recommended Pattern:**

```
Storage Layer:     Parquet (compressed, efficient)
  ↓
Ingestion Layer:   Parquet → Arrow (one-time conversion)
  ↓
Processing Layer:  Arrow (zero-copy, fast)
  ↓
Output Layer:      Arrow → Parquet (for persistence)
```

**Tools That Support This:**
- **Apache Spark**: Native Parquet/Arrow support
- **Dask**: Seamless Parquet/Arrow integration
- **Polars**: Built on Arrow, writes Parquet
- **DuckDB**: Optimized for both formats

---

## 8. Performance Benchmarks and Real-World Impact 📊

### Compression Ratios

**Typical Compression Ratios (lower is better):**

| Format | Compression Ratio | Use Case |
|--------|------------------|----------|
| Parquet (Snappy) | 0.13-0.20 | Balanced compression/speed |
| Parquet (Zstd) | 0.10-0.15 | Maximum compression |
| Arrow (uncompressed) | 1.00-1.10 | Maximum speed |
| Arrow (compressed) | 0.20-0.30 | Balanced option |
| ORC (Zlib) | 0.25-0.35 | Hive/Spark ecosystem |

### Query Performance

**Selective Query (1% selectivity):**

- **Parquet**: 10-50x faster than row-oriented
- **Arrow**: 5-20x faster (if data fits in memory)
- **ORC**: 10-30x faster

**Full Table Scan:**

- **Parquet**: 2-5x faster than row-oriented
- **Arrow**: 10-100x faster (zero-copy, SIMD)
- **ORC**: 3-8x faster

### Storage Cost Impact

**Example: 1TB dataset**

| Format | Storage Size | Monthly Cost (S3) | Savings |
|--------|-------------|-------------------|---------|
| CSV | 1 TB | $23 | Baseline |
| Parquet | 130 GB | $3 | 87% savings |
| Arrow | 1.07 TB | $24.61 | No savings |
| ORC | 270 GB | $6.21 | 73% savings |

**Annual Savings with Parquet: $240 per TB**

---

## 9. Integration with Modern Data Stacks 🔗

### Apache Spark

**Parquet Integration:**
```python
# Read Parquet with column pruning
df = spark.read.parquet("s3://bucket/data/")
df.select("column1", "column2").filter(df.column1 > 100)
# Only column1 and column2 are read from disk
```

**Arrow Integration:**
```python
# Convert to Arrow for zero-copy operations
arrow_df = df.toPandas()  # Uses Arrow internally
# Fast interop with Python/Pandas
```

### Dask

**Seamless Format Support:**
```python
import dask.dataframe as dd

# Read Parquet (lazy loading)
df = dd.read_parquet("data/*.parquet")
# Automatically uses Arrow for in-memory operations
result = df.groupby("category").sum().compute()
```

### Polars

**Built on Arrow:**
```python
import polars as pl

# Read Parquet
df = pl.read_parquet("data.parquet")
# All operations use Arrow format internally
# Zero-copy operations throughout
result = df.filter(pl.col("age") > 30)
```

### DuckDB

**Optimized for Both:**
```sql
-- Read Parquet directly
SELECT * FROM 'data.parquet' WHERE age > 30;

-- Export to Arrow
COPY (SELECT * FROM table) TO 'output.arrow' (FORMAT ARROW);
```

---

## 10. Common Pitfalls and Best Practices ⚠️

### Common Mistakes

**1. Over-Partitioning Parquet Files:**
```python
# Bad: Too many small files
df.write.partitionBy("year", "month", "day", "hour", "minute").parquet("data/")
# Result: Millions of tiny files, poor performance

# Good: Balanced partitioning
df.write.partitionBy("year", "month").parquet("data/")
# Result: Reasonable file sizes, good performance
```

**2. Ignoring Column Order:**
```python
# Bad: Frequently queried columns last
schema = ["id", "metadata1", "metadata2", "name", "age"]  # name, age queried often

# Good: Frequently queried columns first
schema = ["name", "age", "id", "metadata1", "metadata2"]
# Enables better column pruning
```

**3. Wrong Compression Codec:**
```python
# Bad: Gzip (slow) for frequently accessed data
df.write.parquet("data/", compression="gzip")

# Good: Snappy (fast) for hot data, Zstd for cold data
df.write.parquet("data/", compression="snappy")  # Hot data
df.write.parquet("archive/", compression="zstd")  # Cold data
```

### Best Practices

**1. Optimal Row Group Size:**
- **Parquet**: 128MB - 1GB row groups
- Balance between skipping granularity and read efficiency
- Larger row groups = better compression, less skipping

**2. Column Ordering:**
- Put frequently queried columns first
- Put high-cardinality columns early
- Put filtering columns early

**3. Compression Selection:**
- **Snappy**: Fast, moderate compression (hot data)
- **Zstd**: Slower, excellent compression (cold data)
- **LZ4**: Very fast, good compression (real-time)

**4. Arrow Chunk Sizes:**
- **Small chunks**: Better for streaming, more overhead
- **Large chunks**: Better for batch processing, less overhead
- Typical: 64KB - 1MB per chunk

---

## Conclusion: The Path Forward

The core lesson for data engineers and ML practitioners is: **Parquet** wins the battle of **storage efficiency and I/O**, while **Arrow** is the **high-speed rail system** for data in motion, making both indispensable in modern large-scale pipelines.

**Key Takeaways:**

1. **Use Parquet for Storage**: Best compression, excellent I/O efficiency, fine-grained skipping
2. **Use Arrow for Processing**: Zero-copy operations, SIMD optimization, language interoperability
3. **Hybrid Approach**: Store in Parquet, process in Arrow, write back to Parquet
4. **Future is Unified**: Next-gen systems will combine Parquet's encoding with Arrow's speed
5. **Choose Wisely**: Format selection depends on your query patterns and access patterns

**The Future:**

The next generation of data systems will bridge the gap, creating formats that offer:
- Parquet-level compression
- Arrow-level speed
- Unified in-memory and on-disk representation
- Direct querying on encoded data
- SIMD-optimized operations

As we build the data systems of tomorrow, understanding these fundamental trade-offs and design principles will be crucial for creating efficient, scalable, and cost-effective pipelines.

---

**Further Reading:**

- **Primary Research Paper**: Liu, Chunwei, et al. ["A Deep Dive into Common Open Formats for Analytical DBMSs"](https://www.vldb.org/pvldb/vol16/p3044-liu.pdf). Proceedings of the VLDB Endowment 16.11 (2023): 3044-3056. *This comprehensive evaluation of Parquet, Arrow, and ORC provides the empirical foundation for understanding format trade-offs and performance characteristics.*

- ["Apache Arrow and the Future of Data Frames" with Wes McKinney](https://www.youtube.com/watch?v=fyj4FyH3XdU)
- [Apache Parquet Documentation](https://parquet.apache.org/)
- [Apache Arrow Documentation](https://arrow.apache.org/)
- [Columnar Storage and Vectorization](https://www.cidrdb.org/cidr2023/papers/p69-li.pdf)

---

*Understanding data formats is fundamental to building efficient ML systems. In production, the choice between Parquet and Arrow can mean the difference between a pipeline that costs thousands per month versus one that costs hundreds.*

