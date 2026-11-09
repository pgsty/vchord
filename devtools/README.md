# Recovery and recorder regression checks

The regular SQLLogicTest suite covers sampling privileges (including RLS and
role revocation), rollback/SAVEPOINT cleanup, and external-build validation.
These additional checks inspect external effects that SQL alone cannot verify.
Both Python scripts require `psycopg` 3.

Run `test-recovery.py` as a non-root user with PostgreSQL 18+, pgvector,
pageinspect, pg_freespacemap and pg_walinspect installed:

```sh
python3 devtools/test-recovery.py \
  --pg-bin /path/to/postgresql/bin --work /tmp/vchord-recovery-check
```

The work directory must not exist. The script builds an instrumented **private
copy** of the current tracked source, starts a private cluster and stops it on
completion. Keep its build directory separate from production build caches.
Production code contains no fault-injection hooks. The script preserves logs,
the private cluster and `results.json` for inspection.

It interrupts new-vector-page publication before/after the paired WAL record
and after the FSM update, with concurrent insertion. It also persists just the
FSM images before crashing to model stale hints surviving ahead of WAL. Checks
cover committed IDs, two-page WAL records, vector-tape reachability, VACUUM and
concurrent REINDEX. These are controlled process-crash tests, not power-loss or
storage-hardware certification. Complete maintenance-page reclamation remains
outside this check and is not implemented by this change.

Run the recorder test against a **private local cluster**, using libpq variables
(`PGHOST`, `PGPORT`, `PGUSER`) and `max_prepared_transactions > 0`:

```sh
python3 devtools/test-recorder.py
```

It creates its own database, checks the SQLite files after failed database
deletion, ordinary commit and two-phase transactions, then drops the database.
The calling OS user needs access to that server's data directory. Prepared
transaction cleanup deliberately retains diagnostics; only ordinary committed
drops are expected to delete samples immediately.
