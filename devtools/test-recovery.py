#!/usr/bin/env python3
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License v3 (AGPLv3).
# See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
#
# Copyright (c) 2026 PGSTY PTE. LTD.
# Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

"""Fault-inject a private source copy; never install it into a shared server.

Requires PG18+, pgvector, pageinspect, pg_freespacemap, pg_walinspect and psycopg3.
Run as a non-root user with --work pointing at a NEW directory. The work directory
keeps the instrumented source, private cluster, logs and JSON for inspection.
No fault hook is compiled into the production extension.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import time

import psycopg

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--pg-bin', type=Path, required=True)
parser.add_argument('--work', type=Path, required=True)
parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
args = parser.parse_args()
root = args.work.resolve()
root.mkdir(exist_ok=False)
src = root / 'source'
src.mkdir()
control, marker = root / 'control', root / 'marker'
data = root / 'pgdata'
pg = args.pg_bin.resolve()
env = dict(os.environ, PG_CONFIG=str(pg / 'pg_config'))


def run(name, cmd):
    with (root / (name + '.log')).open('a') as log:
        subprocess.run(list(map(str, cmd)), env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)


for name in subprocess.check_output(['git', 'ls-files'], cwd=args.source, text=True).splitlines():
    target = src / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.source / name, target)

# Markers stop only the backend explicitly selected by the controller.
hook = r'''
fn recovery_probe(stage: &str, first: u32, fresh: u32) {
    let expected = format!("{}:{}", stage, std::process::id());
    if std::fs::read_to_string(CONTROL).ok().as_deref() != Some(&expected) { return; }
    std::fs::write(MARKER, format!("{first},{fresh}")).unwrap();
    let start = std::time::Instant::now();
    while std::fs::read_to_string(CONTROL).ok().as_deref() == Some(&expected) {
        assert!(start.elapsed().as_secs() < 60, "recovery probe timed out");
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    if std::fs::read_to_string(CONTROL).ok().as_deref() == Some("error") {
        panic!("injected recovery error");
    }
}
'''.replace('CONTROL', json.dumps(str(control))).replace('MARKER', json.dumps(str(marker)))
p = src / 'src/index/storage.rs'
s = p.read_text()
anchor = '            let state = std::mem::replace(&mut first.state, std::ptr::null_mut());'
assert s.count(anchor) == 1
s = s.replace(anchor, '''            if std::fs::read_to_string(''' + json.dumps(str(control)) + ''').ok().as_deref()
                == Some(format!("stale:{}", std::process::id()).as_str()) {
                // Model an FSM hint surviving ahead of page/link WAL. The real
                // page is still zero; the controller persists just the FSM.
                pgrx::pg_sys::RecordPageWithFreeSpace(second.raw, second.id, second_space as _);
                pgrx::pg_sys::FreeSpaceMapVacuumRange(second.raw, second.id, second.id + 1);
                recovery_probe("stale", first.id, second.id);
            }
            recovery_probe("before", first.id, second.id);
''' + anchor)
anchor = '            // The images above are freed by Finish/Abort.'
assert s.count(anchor) == 1
s = s.replace(anchor, '            recovery_probe("after", first.id, second.id);\n' + anchor)
anchor = '            second.record_freespace(second_space);\n        }'
assert s.count(anchor) == 1
s = s.replace(anchor, '            second.record_freespace(second_space);\n            recovery_probe("fsm", first.id, second.id);\n        }')
anchor = '        drop(second);\n        drop(first);'
assert s.count(anchor) == 1
s = s.replace(anchor, '        let recovery_ids = (first.id, second.id);\n' + anchor +
              '\n        recovery_probe("unlocked", recovery_ids.0, recovery_ids.1);')
p.write_text(s + hook)
run('build', ['make', '-C', src, 'build'])
lib = root / 'lib'
lib.mkdir()
for file in (src / 'build/pkglibdir').iterdir():
    shutil.copy2(file, lib / file.name)
run('initdb', [pg / 'initdb', '-D', data, '-U', 'postgres', '-A', 'trust', '--locale=C', '--data-checksums'])
with (data / 'postgresql.conf').open('a') as f:
    f.write(f"\nport=57037\nlisten_addresses=''\nunix_socket_directories='{root}'\n"
            f"shared_preload_libraries='vchord'\ndynamic_library_path='{lib}:$libdir'\n"
            f"extension_control_path='{src}/build/sharedir:$system'\n"
            "shared_buffers='128MB'\nwal_keep_size='1GB'\nautovacuum=off\nmax_parallel_maintenance_workers=0\n"
            "fsync=on\nfull_page_writes=on\nsynchronous_commit=on\n")


def start():
    run('start', [pg / 'pg_ctl', '-D', data, '-l', root / 'server.log', '-w', 'start'])


def stop(mode='fast'):
    run('stop', [pg / 'pg_ctl', '-D', data, '-m', mode, '-w', 'stop'])
    control.unlink(missing_ok=True)


def connect():
    return psycopg.connect(host=str(root), port=57037, user='postgres', dbname='postgres',
                           autocommit=True, options='-c statement_timeout=30000 -c client_encoding=UTF8')


def execute(c, sql):
    try:
        c.execute(sql)
        return True
    except psycopg.Error:
        return False


def page(c, idx, block):
    return bytes(c.execute('SELECT get_raw_page(%s,%s)', (idx, block)).fetchone()[0])


def vector_pages(c, idx):
    raw = page(c, idx, 0)
    word = struct.unpack_from('<I', raw, 24)[0]
    meta = raw[word & 32767:]
    assert meta[:8] == b'vchordrq'
    a, b = struct.unpack_from('<HH', meta, 36)
    roots = struct.unpack_from('<' + 'I' * ((b-a)//4), meta, a)
    pages = []
    for block in roots:
        while block != 2**32-1:
            assert block not in pages, 'cycle in vector tape'
            pages.append(block)
            raw = page(c, idx, block)
            special = struct.unpack_from('<H', raw, 16)[0]
            block = struct.unpack_from('<I', raw, special)[0]
    return pages


def vector_ctids(c, idx, blocks=None):
    ctids = []
    for block in (vector_pages(c, idx) if blocks is None else blocks):
        raw = page(c, idx, block)
        lower = struct.unpack_from('<H', raw, 12)[0]
        for pos in range(24, lower, 4):
            word = struct.unpack_from('<I', raw, pos)[0]
            if (word >> 15) & 3 != 1:
                continue
            tag, payload = struct.unpack_from('<QQ', raw, word & 32767)
            if tag == 0 and payload:
                ctids.append(f'({payload >> 32},{(payload >> 16) & 65535})')
    return sorted(ctids)


def graph_chain(c, idx):
    """Return the vchordg vertex-page chain starting at the meta page."""
    raw = page(c, idx, 0)
    special = struct.unpack_from('<H', raw, 16)[0]
    current = struct.unpack_from('<II', raw, special)[1]
    pages = []
    while current != 2**32 - 1:
        assert current not in pages, 'cycle in graph vertex chain'
        pages.append(current)
        raw = page(c, idx, current)
        special = struct.unpack_from('<H', raw, 16)[0]
        current = struct.unpack_from('<II', raw, special)[0]
    return pages


def graph_query(c, table):
    c.execute('SET enable_seqscan=off')
    c.execute('SET vchordg.ef_search=200')
    return c.execute(
        f'SELECT id FROM {table} '
        'ORDER BY v <-> array_fill(0.9::real, ARRAY[128])::vector LIMIT 10'
    ).fetchall()


def ids(c, table):
    c.execute('SET enable_seqscan=off')
    c.execute('SET vchordrq.epsilon=4')
    sql = f'SELECT id FROM {table} ORDER BY v <-> array_fill(0.5::real,ARRAY[256])::vector LIMIT 10000'
    plan = c.execute('EXPLAIN ' + sql).fetchall()
    assert any('Index Scan' in row[0] for row in plan), plan
    return sorted(row[0] for row in c.execute(sql))


results = []
start()
try:
    with connect() as c:
        for extension in ['vchord CASCADE', 'pageinspect', 'pg_freespacemap', 'pg_walinspect']:
            c.execute('CREATE EXTENSION IF NOT EXISTS ' + extension)
    cases = [(stage, mode) for stage in ['before', 'after', 'fsm']
             for mode in ['success', 'error', 'crash']] + [('stale', 'crash'), ('fsm', 'unflushed'), ('unlocked', 'crash')]
    for n, (stage, mode) in enumerate(cases):
        table, idx = f'recovery_{n}', f'recovery_{n}_idx'
        with connect() as c:
            c.execute(f'DROP TABLE IF EXISTS {table}')
            c.execute(f'CREATE TABLE {table}(id int,v vector(256))')
            c.execute(f"CREATE INDEX {idx} ON {table} USING vchordrq(v vector_l2_ops) WITH(options='degree_of_parallelism=1')")
            c.execute(f'INSERT INTO {table} SELECT i,array_fill(i::real/100,ARRAY[256])::vector FROM generate_series(1,40)i')
            c.execute('CHECKPOINT')
            wal_start = c.execute('SELECT pg_current_wal_insert_lsn()::text').fetchone()[0]
            with connect() as a, connect() as b, ThreadPoolExecutor() as pool:
                marker.unlink(missing_ok=True)
                control.write_text(f'{stage}:{a.info.backend_pid}')
                af = pool.submit(execute, a, f'INSERT INTO {table} SELECT i,array_fill(i::real/100,ARRAY[256])::vector FROM generate_series(1001,1020)i')
                deadline = time.monotonic() + 30
                while not marker.exists():
                    assert not af.done(), 'fault hook did not fire'
                    assert time.monotonic() < deadline, 'fault hook timeout'
                    time.sleep(.02)
                tail, fresh = map(int, marker.read_text().split(','))
                free = c.execute('SELECT pg_freespace(%s,%s)', (idx, fresh)).fetchone()[0]
                assert (free > 0) == (stage in ['fsm', 'stale', 'unlocked']), (stage, free)
                bf = pool.submit(execute, b, f'INSERT INTO {table} VALUES(2000,array_fill(0.5::real,ARRAY[256])::vector)')
                deadline = time.monotonic() + 5
                if stage == 'unlocked':
                    assert bf.result(timeout=5), 'independent writer failed to commit'
                    b_ctid = c.execute(f'SELECT ctid::text FROM {table} WHERE id=2000').fetchone()[0]
                    assert b_ctid in vector_ctids(c, idx, [fresh]), 'writer did not use the newly published page'
                while stage != 'unlocked':
                    wait = c.execute('SELECT wait_event_type FROM pg_stat_activity WHERE pid=%s', (b.info.backend_pid,)).fetchone()
                    if wait == ('LWLock',):
                        break
                    assert not bf.done() and time.monotonic() < deadline, 'writer bypassed unpublished page'
                    time.sleep(.02)
                if mode in ['crash', 'unflushed']:
                    saved_fsm = None
                    if stage == 'stale':
                        # Persist ONLY the hint pages, modeling non-WAL FSM
                        # reaching disk before page/link WAL. A checkpoint
                        # would wait on the deliberately held main-page locks.
                        path = c.execute('SELECT pg_relation_filepath(%s::regclass)', (idx,)).fetchone()[0]
                        count = c.execute("SELECT pg_relation_size(%s::regclass,'fsm')/8192", (idx,)).fetchone()[0]
                        images = []
                        for block in range(count):
                            image = bytearray(c.execute("SELECT get_raw_page(%s,'fsm',%s)", (idx, block)).fetchone()[0])
                            checksum = c.execute('SELECT page_checksum(%s,%s)', (bytes(image), block)).fetchone()[0]
                            struct.pack_into('<H', image, 8, checksum & 0xffff)
                            images.append(image)
                        saved_fsm = (data / (path + '_fsm'), images)
                    elif mode != 'unflushed' and stage != 'unlocked':
                        c.execute('SELECT pg_switch_wal()')
                    stop('immediate')
                    if saved_fsm:
                        path, images = saved_fsm
                        with path.open('r+b') as f:
                            for image in images:
                                f.write(image)
                            f.flush()
                            os.fsync(f.fileno())
                elif mode == 'error':
                    control.write_text('error')
                else:
                    control.unlink()
                a_ok, b_ok = af.result(timeout=30), bf.result(timeout=30)
                control.unlink(missing_ok=True)
        if mode in ['crash', 'unflushed']:
            start()
        with connect() as c:
            if not b_ok:
                c.execute(f'INSERT INTO {table} VALUES(2000,array_fill(0.5::real,ARRAY[256])::vector)')
            assert a_ok == (mode == 'success')
            expected = sorted(row[0] for row in c.execute(f'SELECT id FROM {table}'))
            assert ids(c, table) == expected
            reachable = vector_pages(c, idx)
            assert (fresh in reachable) == (stage in ['after', 'fsm', 'unlocked'] or mode == 'success') or mode == 'unflushed'
            records = c.execute("SELECT block_ref FROM pg_get_wal_records_info(%s,pg_current_wal_lsn()) WHERE resource_manager='Generic'", (wal_start,)).fetchall()
            pair = [row[0] for row in records if re.search(rf'blk {tail}\b', row[0] or '') and re.search(rf'blk {fresh}\b', row[0] or '')]
            if stage in ['after', 'fsm', 'unlocked'] and mode != 'unflushed' or mode == 'success':
                assert pair, 'initialization and link not found in one WAL record'
            c.execute(f'DELETE FROM {table} WHERE id=2000')
            c.execute(f'VACUUM {table}')
            assert ids(c, table) == [i for i in expected if i != 2000]
            heap_ctids = sorted(row[0] for row in c.execute(f'SELECT ctid::text FROM {table}'))
            assert vector_ctids(c, idx) == heap_ctids, 'VACUUM missed a vector or a committed vector is outside the tape'
            c.execute(f'REINDEX INDEX CONCURRENTLY {idx}')
            assert ids(c, table) == [i for i in expected if i != 2000]
            results.append(dict(stage=stage, mode=mode, rows=len(expected), fsm=free,
                                tail=tail, fresh=fresh, vector_pages=reachable, paired_wal=pair,
                                recovered_ids_match=True, vacuum_and_reindex=True))
            (root / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
            print(stage, mode, 'passed', flush=True)

    # Cover both durable paired WAL and an FSM hint surviving without it.
    for graph_stage in ['after', 'stale']:
        graph_table, graph_idx = 'graph_recovery', 'graph_recovery_idx'
        with connect() as c:
            c.execute(f'DROP TABLE IF EXISTS {graph_table}')
            c.execute(f'CREATE TABLE {graph_table}(id int, v vector(128))')
            c.execute(f'CREATE INDEX {graph_idx} ON {graph_table} USING vchordg(v vector_l2_ops)')
            c.execute(
                f'INSERT INTO {graph_table} '
                "SELECT i, array_fill((i::real / 1000), ARRAY[128])::vector "
                'FROM generate_series(1, 400) i'
            )
            before = graph_chain(c, graph_idx)
            c.execute('CHECKPOINT')
            wal_start = c.execute('SELECT pg_current_wal_insert_lsn()::text').fetchone()[0]

        marker.unlink(missing_ok=True)
        with connect() as a, ThreadPoolExecutor() as pool:
            control.write_text(f'{graph_stage}:{a.info.backend_pid}')
            af = pool.submit(
                execute, a, f'INSERT INTO {graph_table} '
                "SELECT i, array_fill((i::real / 1000), ARRAY[128])::vector "
                'FROM generate_series(401, 3000) i',
            )
            deadline = time.monotonic() + 30
            while not marker.exists():
                assert not af.done(), 'vchordg pair publication hook did not fire'
                assert time.monotonic() < deadline, 'vchordg pair publication hook timeout'
                time.sleep(.02)
            tail, fresh = map(int, marker.read_text().split(','))
            assert tail in before, (tail, before)
            saved_fsm = None
            with connect() as c:
                if graph_stage == 'after':
                    c.execute('SELECT pg_switch_wal()')
                else:
                    assert c.execute('SELECT pg_freespace(%s,%s)', (graph_idx, fresh)).fetchone()[0] > 0
                    path = c.execute('SELECT pg_relation_filepath(%s::regclass)', (graph_idx,)).fetchone()[0]
                    count = c.execute("SELECT pg_relation_size(%s::regclass,'fsm')/8192", (graph_idx,)).fetchone()[0]
                    images = []
                    for block in range(count):
                        image = bytearray(c.execute("SELECT get_raw_page(%s,'fsm',%s)", (graph_idx, block)).fetchone()[0])
                        checksum = c.execute('SELECT page_checksum(%s,%s)', (bytes(image), block)).fetchone()[0]
                        struct.pack_into('<H', image, 8, checksum & 0xffff)
                        images.append(image)
                    saved_fsm = (data / (path + '_fsm'), images)
            stop('immediate')
            assert not af.result(timeout=30), 'crashed publisher unexpectedly committed'
            if saved_fsm:
                path, images = saved_fsm
                with path.open('r+b') as f:
                    for image in images:
                        f.write(image)
                    f.flush()
                    os.fsync(f.fileno())

        start()
        with connect() as c:
            assert sorted(row[0] for row in c.execute(f'SELECT id FROM {graph_table}')) == list(range(1, 401))
            chain = graph_chain(c, graph_idx)
            assert (fresh in chain) == (graph_stage == 'after'), (fresh, before, chain)
            paired_wal = []
            if graph_stage == 'after':
                records = c.execute("SELECT block_ref FROM pg_get_wal_records_info(%s,pg_current_wal_lsn()) WHERE resource_manager='Generic'", (wal_start,)).fetchall()
                paired_wal = [r[0] for r in records if re.search(rf'blk {tail}\b', r[0] or '') and re.search(rf'blk {fresh}\b', r[0] or '')]
                assert paired_wal, 'graph initialization/link not found in one WAL record'
            else:
                assert page(c, graph_idx, fresh) == bytes(8192), 'unpublished graph page was initialized'

            c.execute(
                f'INSERT INTO {graph_table} '
                "SELECT i, array_fill(0.9::real, ARRAY[128])::vector "
                'FROM generate_series(10001, 10060) i'
            )
            old_ctids = {r[0] for r in c.execute(f'SELECT ctid::text FROM {graph_table} WHERE id >= 10001')}
            c.execute(f'DELETE FROM {graph_table} WHERE id BETWEEN 10001 AND 10060')
            c.execute(f'VACUUM (INDEX_CLEANUP ON) {graph_table}')
            c.execute(
                f'INSERT INTO {graph_table} '
                "SELECT i, array_fill((-0.9)::real, ARRAY[128])::vector "
                'FROM generate_series(20001, 20060) i'
            )
            new_ctids = {r[0] for r in c.execute(f'SELECT ctid::text FROM {graph_table} WHERE id >= 20001')}
            reused = len(old_ctids & new_ctids)
            assert reused > 0, 'heap-TID reuse was not exercised'
            c.execute('SET enable_seqscan=off')
            plan = c.execute(f'EXPLAIN SELECT id FROM {graph_table} ORDER BY v <-> array_fill(0.9::real, ARRAY[128])::vector LIMIT 10').fetchall()
            assert any('Index Scan' in r[0] for r in plan), plan
            got = graph_query(c, graph_table)
            assert len(got) == 10 and not any(row[0] >= 20001 for row in got), got
            if graph_stage == 'stale':
                assert c.execute('SELECT pg_freespace(%s,%s)', (graph_idx, fresh)).fetchone()[0] == 0
            results.append(dict(
                stage='vchordg-' + graph_stage, tail=tail, fresh=fresh,
                vertex_pages_before=before, vertex_pages_after=chain,
                recovered_rows=400, recovered_ids_match=True, paired_wal=paired_wal,
                reused_heap_tids=reused, impostors_in_top10=0,
            ))
            (root / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
            print('vchordg', graph_stage, 'passed', flush=True)

finally:
    control.unlink(missing_ok=True)
    stop()
