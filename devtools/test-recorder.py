#!/usr/bin/env python3
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License v3 (AGPLv3).
# See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
#
# Copyright (c) 2026 PGSTY PTE. LTD.
# Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

"""Test recorder external effects on a PRIVATE local server via libpq env.

Requires psycopg3, superuser and max_prepared_transactions > 0. Creates and drops
one dedicated database; reads its SQLite files to check post-commit cleanup.
"""

import json
import os
from pathlib import Path
import sqlite3

import psycopg
from psycopg import sql

name = 'vchord_recorder_check_' + str(os.getpid())
prepared = name + '_prepared'
results = {}

with psycopg.connect(dbname='postgres', autocommit=True, options='-c client_encoding=UTF8') as admin:
    assert int(admin.execute('SHOW max_prepared_transactions').fetchone()[0]) > 0
    admin.execute(sql.SQL('CREATE DATABASE {}').format(sql.Identifier(name)))
    try:
        with psycopg.connect(dbname=name, autocommit=True, options='-c client_encoding=UTF8') as c:
            c.execute('CREATE EXTENSION vchord CASCADE')
            c.execute('CREATE TABLE t(v vector(3))')
            c.execute("INSERT INTO t VALUES('[1,2,3]')")
            c.execute('CREATE INDEX ti ON t USING vchordrq(v vector_l2_ops)')
            c.execute('SET enable_seqscan=off')
            c.execute('SET vchordrq.query_sampling_rate=1')
            c.execute('SET vchordrq.query_sampling_max_records=10')
            oid = c.execute("SELECT 'ti'::regclass::oid").fetchone()[0]
            dbid = c.execute('SELECT oid FROM pg_database WHERE datname=current_database()').fetchone()[0]
            file = Path(c.execute('SHOW data_directory').fetchone()[0]) / 'pgsql_tmp_vchord_sampling' / f'database_{dbid}.sqlite'

            def sample():
                c.execute('SET vchordrq.query_sampling_enable=on')
                c.execute("SELECT * FROM t ORDER BY v <-> '[4,5,6]' LIMIT 1").fetchall()
                c.execute('SET vchordrq.query_sampling_enable=off')

            def stored(index_oid):
                with sqlite3.connect(f'file:{file}?mode=ro', uri=True) as db:
                    return db.execute('SELECT count(*) FROM sqlite_master WHERE type=\'table\' AND name=?', (f'index_{index_oid}',)).fetchone()[0] == 1

            sample()
            assert stored(oid)
            try:
                admin.execute(sql.SQL('DROP DATABASE {}').format(sql.Identifier(name)))
                raise AssertionError('DROP DATABASE unexpectedly succeeded')
            except psycopg.errors.ObjectInUse:
                pass
            assert file.exists()
            with psycopg.connect(dbname=name, autocommit=True, options='-c client_encoding=UTF8') as fresh:
                assert fresh.execute("SELECT * FROM vchordrq_sampled_values('ti')").fetchall() == [('[4,5,6]',)]
            results['failed_drop_database_preserves_new_connection_samples'] = True

            for action in ['ROLLBACK', 'COMMIT']:
                c.execute('BEGIN')
                c.execute('DROP INDEX ti')
                c.execute(sql.SQL('PREPARE TRANSACTION {}').format(sql.Literal(prepared)))
                assert stored(oid), 'PREPARE deleted diagnostic data'
                c.execute(sql.SQL(action + ' PREPARED {}').format(sql.Literal(prepared)))
                assert stored(oid)
                if action == 'ROLLBACK':
                    assert c.execute("SELECT count(*) FROM vchordrq_sampled_values('ti')").fetchone()[0] == 1
                else:
                    assert c.execute("SELECT to_regclass('ti')").fetchone()[0] is None
                results[action.lower() + '_prepared_retains_samples'] = True

            c.execute('CREATE INDEX ti ON t USING vchordrq(v vector_l2_ops)')
            current_oid = c.execute("SELECT 'ti'::regclass::oid").fetchone()[0]
            sample()
            c.execute('BEGIN')
            c.execute('SAVEPOINT inner_save')
            c.execute('DROP INDEX ti')
            assert stored(current_oid), 'DROP deleted before commit'
            c.execute('RELEASE inner_save')
            c.execute('COMMIT')
            assert not stored(current_oid), 'committed DROP did not clean samples'
            assert stored(oid), 'old prepared cleanup leaked into a later commit'
            results['ordinary_commit_cleans_only_committed_drop'] = True
    finally:
        # Test owns this DB only. Prepared transactions are resolved on failure.
        with psycopg.connect(dbname=name, autocommit=True, options='-c client_encoding=UTF8') as cleanup:
            if cleanup.execute('SELECT 1 FROM pg_prepared_xacts WHERE gid=%s', (prepared,)).fetchone():
                cleanup.execute(sql.SQL('ROLLBACK PREPARED {}').format(sql.Literal(prepared)))
        admin.execute(sql.SQL('DROP DATABASE {}').format(sql.Identifier(name)))
    assert not file.exists(), 'committed DROP DATABASE left its SQLite file'
    results['ordinary_drop_database_cleans_file'] = True
print(json.dumps(results, indent=2))
