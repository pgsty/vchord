# vchord

[![Website: vchord.pgsty.com](https://img.shields.io/badge/Website-vchord.pgsty.com-slategray?style=flat&logo=cilium&logoColor=white)](https://vchord.pgsty.com/)
[![Docs: vchord](https://img.shields.io/badge/Docs-vchord-slategray?style=flat)](https://vchord.pgsty.com/docs/vchord/)
[![Catalog: vchord](https://img.shields.io/badge/Catalog-vchord-%233E668F?style=flat&logo=postgresql&logoColor=white&labelColor=3E668F)](https://pgext.cloud/ext/vchord)
[![Version: 1.4.0](https://img.shields.io/badge/Version-1.4.0-slategray?style=flat)](docs/release-notes/1.4.0.md)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-slategray?style=flat&logo=gnu&logoColor=white)](https://github.com/pgsty/vchord/blob/main/LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/pgsty/vchord?style=flat&logo=github&logoColor=black&color=slategray)](https://star-history.com/#pgsty/vchord&Date)
[![PostgreSQL: 15-19](https://img.shields.io/badge/PostgreSQL-15--19-%233E668F?style=flat&logo=postgresql&logoColor=white&labelColor=3E668F)](#build-from-source)

vchord 1.4.0 provides disk-friendly vector similarity search for PostgreSQL 15–19 on `x86_64` and `aarch64`. It requires the `vector` extension from [pgvector](https://github.com/pgvector/pgvector).

[Documentation](https://vchord.pgsty.com/docs/vchord/) · [Releases](https://github.com/pgsty/vchord/releases) · [Issues](https://github.com/pgsty/vchord/issues)

## Features

- **`vchordrq`**: IVF/RaBitQ indexing for large datasets, with filtering, reranking, external builds, and recall evaluation.
- **`vchordg`**: graph indexing for low-latency vector search.
- L2, inner-product, cosine, range search, and multi-vector MaxSim operators.
- `vector`, `halfvec`, and quantized `rabitq4` / `rabitq8` types.

ANN routing can trade small distance errors for speed. On ARM CPUs with FP16 and on x86 CPUs with AVX-512 FP16, `halfvec` L2 centroid selection and graph pruning round differences to `f16` and accumulate in `f32`, retrying in `f32` if a difference overflows. Final reranking, sphere predicates, and the residual-quantization lower bounds that decide reranking order keep `f32` differences and accumulation. Tune probes and search breadth against recall on your own data.

Combine with [vchord_bm25](https://github.com/pgsty/vchord_bm25) and [pg_tokenizer](https://github.com/pgsty/pg_tokenizer) for hybrid vector and full-text search.

## Installation

Install from the [Pigsty package repository](https://pigsty.io/ext/e/vchord):

```bash
pig repo add pgsql -u
pig ext add vchord
```

The GitHub release workflow builds Linux `.deb` and `.zip` packages on Ubuntu 22.04 for PostgreSQL 15–19 on both architectures. The `.deb` packages require `libc6 >= 2.35`. Choose the package for your PostgreSQL major version and CPU architecture. See [GitHub Releases](https://github.com/pgsty/vchord/releases) and the [installation guide](https://vchord.pgsty.com/docs/vchord/getting-started/installation/).

Add `vchord` to `shared_preload_libraries`, restart PostgreSQL, and enable the extension:

```sql
CREATE EXTENSION IF NOT EXISTS vchord CASCADE;
```

## Usage

```sql
CREATE TABLE items (id bigserial PRIMARY KEY, embedding vector(3));

INSERT INTO items (embedding)
SELECT ARRAY[random(), random(), random()]::real[]
FROM generate_series(1, 1000);

CREATE INDEX ON items USING vchordrq (embedding vector_l2_ops);

SELECT * FROM items ORDER BY embedding <-> '[3, 1, 2]' LIMIT 5;
```

Use `USING vchordg` for a graph index. See [indexing](https://vchord.pgsty.com/docs/vchord/usage/indexing/), [graph search](https://vchord.pgsty.com/docs/vchord/usage/graph-index/), and [parameter tuning](https://vchord.pgsty.com/docs/vchord/usage/partitioning-tuning/) for configuration.

For a query such as `WHERE tenant_id = 42 ORDER BY embedding <-> $1 LIMIT 10`, an RQ index on `(embedding vector_l2_ops, tenant_id)` can filter candidates inside each IVF list. `vchordrq.probes` selects the initial lists, and the scan streams and reranks lazily from those lists first. Only when the current round runs out of candidates does a pushed-down filter key expand to further lists, doubling probes each round up to `vchordrq.max_probes` (default `-1`, no cap; initial probes are always honored).

Because further lists are read lazily only after earlier rounds are exhausted, rows from later rounds can be nearer than rows already returned, and RQ ranks candidates using `f32` distances. If strict SQL distance order is required, overfetch in a materialized CTE or subquery and re-sort by the SQL distance operator. Set `vchordrq.max_probes` to cap worst-case expansion on very selective or absent keys (or to `0` to disable expansion beyond the initial probes).

Sphere searches through either ANN index can miss matching rows. PostgreSQL rechecks the strict radius for candidates the index returns. For an exhaustive range query, use `EXPLAIN` to verify that a sequential scan evaluates the sphere predicate on the table. See [issue #29](https://github.com/pgsty/vchord/issues/29).

## Build from Source

Requires PostgreSQL 15–19 development headers, pgvector, Rust 1.100.0+, GNU Make, and matching Clang, libclang, and LLVM tools on `x86_64` or `aarch64`. Built with pgrx 0.20.1.

```bash
git clone --branch 1.4.0 --depth 1 https://github.com/pgsty/vchord.git
cd vchord
PG_CONFIG=/path/to/pg_config make build
sudo make PG_CONFIG=/path/to/pg_config install
```

Set `CLANG_PATH` and `LIBCLANG_PATH` if multiple LLVM versions are installed. For Clang installations that load a default configuration file, set `CLANG_NO_DEFAULT_CONFIG=1`.

## Upgrading

Install the new library and SQL files, restart PostgreSQL, then run in each database:

```sql
ALTER EXTENSION vchord UPDATE;
```

1.4.0 reads ordinary 1.3.0 RQ indexes and writes the same format (`1001`). Adding filter key columns requires a new RQ index, which uses format `1003`. Before restoring a 1.3.0 binary, remove or replace every keyed index while 1.4.0 is still running, and restore a compatible 1.3.0 extension catalog or database backup. The distribution provides a forward SQL upgrade. PostgreSQL 14 installations must also move to a supported server major. See the [1.4.0 release notes](docs/release-notes/1.4.0.md).

## Testing

```bash
cargo test --locked --workspace --exclude vchord
PG_CONFIG=/path/to/pg_config make clippy
sqllogictest --db "$USER" --user "$USER" --label pg19 './tests/general/*.slt' './tests/vchordg/*.slt' './tests/vchordrq/*.slt' './tests/vchordrq/pg17/*.slt'
```

Pass the server major version to `--label` (`pg15` through `pg19`), and omit `./tests/vchordrq/pg17/*.slt` on PostgreSQL 15–16. The SQL tests require a running PostgreSQL instance with `vchord` preloaded and installed; additional recovery checks are documented in [devtools/README.md](devtools/README.md).

## License

[GNU Affero General Public License v3.0](LICENSE).

Copyright (c) 2025–2026 TensorChord Inc. and 2026 PGSTY PTE. LTD. Maintained by [Ruohang Feng](mailto:rh@vonng.com) at [PGSTY](https://pgsty.com).
