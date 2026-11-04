# vchord

[![Website: vchord.pgsty.com](https://img.shields.io/badge/Website-vchord.pgsty.com-slategray?style=flat&logo=cilium&logoColor=white)](https://vchord.pgsty.com/)
[![Docs: vchord](https://img.shields.io/badge/Docs-vchord-slategray?style=flat)](https://vchord.pgsty.com/docs/vchord/)
[![Catalog: vchord](https://img.shields.io/badge/Catalog-vchord-%233E668F?style=flat&logo=postgresql&logoColor=white&labelColor=3E668F)](https://pgext.cloud/ext/vchord)
[![Version: v1.2.0](https://img.shields.io/badge/Version-v1.2.0-slategray?style=flat)](https://github.com/pgsty/vchord/releases)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-slategray?style=flat&logo=gnu&logoColor=white)](https://github.com/pgsty/vchord/blob/main/LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/pgsty/vchord?style=flat&logo=github&logoColor=black&color=slategray)](https://star-history.com/#pgsty/vchord&Date)
[![PostgreSQL: 14-19](https://img.shields.io/badge/PostgreSQL-14--19-%233E668F?style=flat&logo=postgresql&logoColor=white&labelColor=3E668F)](https://pgext.cloud/ext/vchord)

**`vchord`** is a PostgreSQL extension engineered for scalable, high-performance, and disk-friendly vector similarity search, serving as the successor to `pgvecto.rs`.

[Website](https://vchord.pgsty.com/) | [Documentation](https://vchord.pgsty.com/docs/vchord/) | [vchord_bm25](https://github.com/pgsty/vchord_bm25) | [pg_tokenizer](https://github.com/pgsty/pg_tokenizer) | [Blog](https://vchord.pgsty.com/blog/) | [PGEXT](https://pgext.cloud/ext/vchord) | [Releases](https://github.com/pgsty/vchord/releases) | [Issues](https://github.com/pgsty/vchord/issues) | [Upstream](https://github.com/supervc-stack/VectorChord)

---

To store vectors compactly while preserving recall, `vchord` combines **RaBitQ**[^1] quantization with inverted-file (`vchordrq`) and single-layer graph (`vchordg`) indexes plus in-database reranking. It stores up to 400,000 vectors per $1—roughly 6x more vectors per dollar than Pinecone's storage-optimized tier and 26x more than in-memory `pgvector` / `pgvecto.rs` HNSW indexes.

![vchord Cost and Performance Comparison](https://github.com/user-attachments/assets/2d985f1e-7093-4c3a-8bf3-9f0b92c0e7e7)

[^1]: Gao, Jianyang, and Cheng Long. "RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search." *Proceedings of the ACM on Management of Data* 2.3 (2024): 1–27.

---

## Vchord Suite

`vchord` is the core extension of the **Vchord** family maintained by [PGSTY](https://pgsty.com):

- **[`vchord`](https://github.com/pgsty/vchord)** ([Docs](https://vchord.pgsty.com/docs/vchord/) / [Meta](https://pgext.cloud/ext/vchord)): Scalable, high-performance, and disk-friendly vector similarity search.
- **[`vchord_bm25`](https://github.com/pgsty/vchord_bm25)** ([Docs](https://vchord.pgsty.com/docs/bm25/) / [Meta](https://pgext.cloud/ext/vchord_bm25)): Native BM25 full-text ranking index (`Block-WeakAnd`).
- **[`pg_tokenizer`](https://github.com/pgsty/pg_tokenizer)** ([Docs](https://vchord.pgsty.com/docs/tokenizer/) / [Meta](https://pgext.cloud/ext/pg_tokenizer)): Standalone multilingual text analyzer and vocabulary tokenizer.

---

## Features

- **Cost-Effective Storage**: Host 100M × 768-dim vectors on a single AWS `i4i.xlarge` ($247/mo)[^2], or [scale to 1B × 96-dim vectors](https://vchord.pgsty.com/blog/post/scaling-vector-search-to-1-billion-on-postgresql/) on an `i7ie.6xlarge` ($2,246/mo)[^3] by keeping only compact centroids in RAM and streaming quantized lists from NVMe SSD.
- **Fast Index Construction**: Build a 100M-vector index in ~20 minutes using [hierarchical K-means](https://vchord.pgsty.com/docs/vchord/usage/partitioning-tuning/#hierarchical-k-means)[^4] or [external index precomputation](https://vchord.pgsty.com/docs/vchord/usage/external-build/).
- **Bounded Memory Growth**: Control index build memory via [dimensionality reduction and sampling](https://vchord.pgsty.com/docs/vchord/usage/partitioning-tuning/#reduce-sampling-factor)[^5], enabling billion-scale index builds on 128 GB RAM nodes.
- **Drop-In `pgvector` Compatibility**: Directly indexes [`pgvector`](https://vchord.pgsty.com/docs/vector/) data types (`vector`, `halfvec`) and supports standard distance operators (`<->`, `<#>`, `<=>`) alongside multi-vector MaxSim (`@#`) retrieval.
- **Native Low-Bit Quantization Types**: Provides [native 4-bit (`rabitq4`) and 8-bit (`rabitq8`) vector types](https://vchord.pgsty.com/docs/vchord/usage/quantization-types/) with <1% recall loss on `rabitq8`.

[^2]: See [Store 400k Vectors for $1 in PostgreSQL](https://vchord.pgsty.com/blog/post/vectorchord-store-400k-vectors-for-1-in-postgresql/).
[^3]: See [Scaling Vector Search to 1 Billion on PostgreSQL](https://vchord.pgsty.com/blog/post/scaling-vector-search-to-1-billion-on-postgresql/).
[^4]: See [How We Made 100M Vector Indexing in 20 Minutes Possible on PostgreSQL](https://vchord.pgsty.com/blog/post/how-we-made-100m-vector-indexing-in-20-minutes-possible-on-postgresql/#hierarchical-k-means) and [Partitioning Tuning](https://vchord.pgsty.com/docs/vchord/usage/partitioning-tuning/#hierarchical-k-means).
[^5]: See [Dimensionality Reduction](https://vchord.pgsty.com/blog/post/how-we-made-100m-vector-indexing-in-20-minutes-possible-on-postgresql/#dimensionality-reduction) and [Sampling Factor Tuning](https://vchord.pgsty.com/docs/vchord/usage/partitioning-tuning/#reduce-sampling-factor).

---

## Quick Start

For full setup instructions (Debian/Ubuntu packages, PGXN, Kubernetes, and source builds), see the [Installation Guide](https://vchord.pgsty.com/docs/vchord/getting-started/installation/).

### 1. Install via Native Linux Packages ([`pig`](https://pig.pgsty.com/))

On EL 8/9/10, Debian 12/13, and Ubuntu 22.04/24.04 (PostgreSQL 14–18), you can install pre-built native packages from the [Pigsty repository](https://pigsty.io/ext/e/vchord) (see package matrix on [pgext.cloud](https://pgext.cloud/ext/vchord)):

```bash
curl -fsSL https://repo.pigsty.io/pig | bash
pig repo add pgsql -u
pig ext add vchord
```

Then add `vchord` to `shared_preload_libraries` and restart PostgreSQL:

```bash
psql -U postgres -c 'ALTER SYSTEM SET shared_preload_libraries = "vchord"'
sudo systemctl restart postgresql
```

### 2. Run with Docker

To spin up a ready-to-use PostgreSQL 18 container with `vchord` preloaded:

```bash
docker run \
  --name vchord-demo \
  -e POSTGRES_PASSWORD=mysecretpassword \
  -p 5432:5432 \
  -d ghcr.io/tensorchord/vchord-postgres:pg18-v1.1.1
```

> [!NOTE]
> For hybrid vector + full-text search, use the all-in-one [`tensorchord/vchord-suite:pg18-latest`](https://vchord.pgsty.com/docs/vchord/getting-started/vectorchord-suite/) image, which bundles [`vchord`](https://github.com/pgsty/vchord), [`vchord_bm25`](https://github.com/pgsty/vchord_bm25), and [`pg_tokenizer`](https://github.com/pgsty/pg_tokenizer).

Connect to the instance with `psql`:

```bash
psql -h localhost -p 5432 -U postgres
```

### 3. Basic SQL Usage

Enable the extension (this automatically enables `pgvector` as well):

```sql
CREATE EXTENSION IF NOT EXISTS vchord CASCADE;
```

Create a table with a `vector` column and insert sample data:

```sql
CREATE TABLE items (id bigserial PRIMARY KEY, embedding vector(3));

INSERT INTO items (embedding)
SELECT ARRAY[random(), random(), random()]::real[]
FROM generate_series(1, 1000);
```

Build a `vchordrq` index and run a nearest-neighbor query:

```sql
CREATE INDEX ON items USING vchordrq (embedding vector_l2_ops);

SELECT * FROM items ORDER BY embedding <-> '[3, 1, 2]' LIMIT 5;
```

When tuning `vchordrq`, use one comma-separated `probes` value per level in
`build.internal.lists`. Each value must be in `1..=4294967295`; empty entries
and spaces inside a value are invalid, and one trailing comma is ignored for
compatibility with 1.1.1. An empty string is valid for `lists = []`.
For `vchordg`, `alpha` must contain 1–8 ascending values starting with `1.0`,
all below `2.0`.

### PostgreSQL 19 RC1 validation

PostgreSQL 19 is a prerelease. The `pg19` feature and C API compatibility
code are available in `main` for development validation; official support
and packages remain PostgreSQL 14–18. RC1 source builds have been validated
on Linux x86_64 and macOS arm64 with Rust 1.99.1, pgrx 0.19.3, and
pgvector 0.8.7, including PostgreSQL assertions and the full SQL regression
suite.

Install pgvector for PostgreSQL 19, then build and install `vchord` against
the RC1 installation's `pg_config`:

```bash
git clone https://github.com/pgsty/vchord.git
cd vchord
PG_CONFIG=/path/to/pg19/bin/pg_config make build
make PG_CONFIG=/path/to/pg19/bin/pg_config install # use sudo if required
```

Add `vchord` to that server's `shared_preload_libraries`, restart it, and run
`CREATE EXTENSION IF NOT EXISTS vchord CASCADE;`. PostgreSQL 19 is not
included in the CI or release matrices. Formal release support will be
considered after PostgreSQL 19 GA and a fresh compatibility check.

---

## Documentation

Full documentation is available at **[vchord.pgsty.com/docs/vchord](https://vchord.pgsty.com/docs/vchord/)**:

- **Getting Started**
  - [Overview](https://vchord.pgsty.com/docs/vchord/getting-started/overview/)
  - [Installation](https://vchord.pgsty.com/docs/vchord/getting-started/installation/)
  - [Vchord Suite](https://vchord.pgsty.com/docs/vchord/getting-started/vectorchord-suite/)
- **Indexing & Querying**
  - [Indexing (`vchordrq`)](https://vchord.pgsty.com/docs/vchord/usage/indexing/)
  - [Graph Index (`vchordg`)](https://vchord.pgsty.com/docs/vchord/usage/graph-index/)
  - [Quantization Types (`rabitq4` / `rabitq8`)](https://vchord.pgsty.com/docs/vchord/usage/quantization-types/)
  - [Multi-Vector Retrieval (MaxSim)](https://vchord.pgsty.com/docs/vchord/usage/multi-vector-retrieval/)
  - [Similarity Filter (Range Query)](https://vchord.pgsty.com/docs/vchord/usage/similarity-filter/)
  - [Prefilter](https://vchord.pgsty.com/docs/vchord/usage/prefilter/) & [Rerank in Table](https://vchord.pgsty.com/docs/vchord/usage/rerank-in-table/)
- **Tuning & Operations**
  - [Partitioning Tuning](https://vchord.pgsty.com/docs/vchord/usage/partitioning-tuning/)
  - [External Index Build](https://vchord.pgsty.com/docs/vchord/usage/external-build/)
  - [PostgreSQL Tuning](https://vchord.pgsty.com/docs/vchord/usage/postgresql-tuning/)
  - [Prewarm](https://vchord.pgsty.com/docs/vchord/usage/prewarm/) & [Prefetch](https://vchord.pgsty.com/docs/vchord/usage/prefetch/)
  - [Monitoring](https://vchord.pgsty.com/docs/vchord/usage/monitoring/), [Measure Recall](https://vchord.pgsty.com/docs/vchord/usage/measure-recall/), & [Fallback Parameters](https://vchord.pgsty.com/docs/vchord/usage/fallback-parameters/)
  - [Scalability](https://vchord.pgsty.com/docs/vchord/admin/scalability/), [Kubernetes](https://vchord.pgsty.com/docs/vchord/admin/kubernetes/), & [Migration](https://vchord.pgsty.com/docs/vchord/admin/migration/)
- **Benchmarks & Use Cases**
  - [Performance](https://vchord.pgsty.com/docs/vchord/benchmark/performance/) & [Price Benchmarks](https://vchord.pgsty.com/docs/vchord/benchmark/price/)
  - [Hybrid Search](https://vchord.pgsty.com/docs/vchord/use-case/hybrid-search/) & [ColBERT Rerank](https://vchord.pgsty.com/docs/vchord/use-case/colbert-rerank/)
  - [Customer Stories (Earth Genome)](https://vchord.pgsty.com/docs/vchord/customer-stories/earthgenome/)

---

## License

This software is licensed under the **GNU Affero General Public License v3.0 (AGPLv3)**. You may use, modify, and distribute this software under the terms of the [AGPLv3](LICENSE).

- Copyright (c) 2025-2026 TensorChord Inc.
- Copyright (c) 2026 PGSTY PTE. LTD.
- Maintained by [Ruohang Feng](mailto:rh@vonng.com) @ [PGSTY](https://pgsty.com)
