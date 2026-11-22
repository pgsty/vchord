#!/usr/bin/env python3
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License v3 (AGPLv3).
# See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
#
# Copyright (c) 2026 PGSTY PTE. LTD.
# Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

"""Crash, and fail over to a standby, a vchordrq VACUUM where its free runs move; cancel a one-list build while it writes.

As test-recovery.py: a private, instrumented source copy and private clusters in a NEW --work directory, never
installed into a shared server. Requires PG18+ and psycopg3.

A VACUUM is stopped at: a kept run's page taken (`take`); a list's tapes swapped, its old pages not yet kept
(`swap`); the free loop, after runs that do not fit were evicted (`evict`). Then the primary is shut down
immediately (`crash`) or its standby promoted (`promote`). On the server that survives: every page is in at most one
of a list's tapes, the free list, the kept runs and the index's other pages, and at most a list's pages are in none;
every row is found; a second VACUUM succeeds and the same holds after it.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import time

import psycopg

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--pg-bin', type=Path, required=True)
parser.add_argument('--work', type=Path, required=True)
parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
# a lifecycle (assets.npz, life.npz, emb.npy, as stdbench's life-b16-530k): the `take` cases run on a slice of it
parser.add_argument('--life', type=Path)
args = parser.parse_args()
root = args.work.resolve()
root.mkdir(exist_ok=False)
src = root / 'source'
src.mkdir()
control, marker = root / 'control', root / 'marker'
primary, standby = root / 'primary', root / 'standby'
pg = args.pg_bin.resolve()
env = dict(os.environ, PG_CONFIG=str(pg / 'pg_config'))
PORTS = {primary: 57041, standby: 57042}
PAGE_MAP = r'''
use crate::closure_lifetime_binder::{id_0, id_1, id_2};
use crate::tape::by_next;
use crate::tuples::*;
use crate::{Opaque, tape};
use index::relation::{Page, PageGuard, RelationRead};
use index_accessor::FunctionalAccessor;

pub fn page_map<R: RelationRead>(index: &R) -> Vec<(&'static str, i64, u32)>
where
    R::Page: Page<Opaque = Opaque>,
{
    let mut out = Vec::new();
    let meta_guard = index.read(0);
    let meta_tuple = MetaTuple::deserialize_ref(meta_guard.get(1).unwrap());
    let height_of_root = meta_tuple.height_of_root();
    let centroids_first = meta_tuple.centroids_first();
    let vectors_first = meta_tuple.vectors_first().to_vec();
    let freepages_first = meta_tuple.freepages_first();
    let root = meta_tuple.first();
    drop(meta_guard);
    out.push(("meta", -1, 0));
    out.push(("freepages", -1, freepages_first));
    for g in by_next(index, centroids_first) {
        out.push(("centroid", -1, g.id()));
    }
    for (i, &first) in vectors_first.iter().enumerate() {
        for g in by_next(index, first) {
            out.push(("vtape", i as i64, g.id()));
        }
    }
    {
        let g = index.read(freepages_first);
        let bytes = g.get(1).unwrap();
        let mut current = u32::from_ne_bytes(bytes[..4].try_into().unwrap());
        drop(g);
        while current != u32::MAX {
            out.push(("free", -1, current));
            current = index.read(current).get_opaque().next;
        }
        let g = index.read(freepages_first);
        if let Some(bytes) = g.get(2) {
            let len = u32::from_ne_bytes(bytes[..4].try_into().unwrap()) as usize;
            for i in 0..len {
                let at = 8 + 8 * i;
                let s = u32::from_ne_bytes(bytes[at..at + 4].try_into().unwrap());
                let e = u32::from_ne_bytes(bytes[at + 4..at + 8].try_into().unwrap());
                out.extend((s..e).map(|id| ("free", -2, id)));
            }
        }
    }
    let mut state = vec![root];
    for _ in (1..height_of_root).rev() {
        let mut results = Vec::new();
        for first in state {
            tape::read_h1_tape::<R, _, _>(
                by_next(index, first).inspect(|g| out.push(("h1", -1, g.id()))),
                || FunctionalAccessor::new((), id_0(|_, _| ()), id_1(|_, _| [(); _])),
                |(), _, _, first, _| results.push(first),
            );
        }
        state = results;
    }
    let mut words = Vec::new();
    for (list, first) in state.into_iter().enumerate() {
        let list = list as i64;
        out.push(("jump", list, first));
        let jump_guard = index.read(first);
        let jump_tuple = JumpTuple::deserialize_ref(jump_guard.get(1).unwrap());
        let directory = by_next(index, jump_tuple.directory_first())
            .inspect(|g| out.push(("directory", list, g.id())));
        let pages = tape::read_directory_words::<R>(directory, &mut words, true);
        let mut vectors = Vec::new();
        tape::read_frozen_tape::<R, _, _>(
            words[..pages].iter().map(|&id| index.read(id)).inspect(|g| out.push(("frozen", list, g.id()))),
            tape::All,
            || FunctionalAccessor::new((), id_0(|_, _| ()), id_1(|_, _| [(); _])),
            (),
            id_2(|row| vectors.extend_from_slice(row.prefetch)),
        );
        tape::read_appendable_tape::<R, _>(
            by_next(index, jump_tuple.appendable_first()).inspect(|g| out.push(("appendable", list, g.id()))),
            0,
            |_, _, _| (),
            (),
            id_2(|row| vectors.extend_from_slice(row.prefetch)),
        );
        out.extend(vectors.into_iter().map(|id| ("vector", list, id)));
    }
    out
}
'''


def run(name, cmd, check=True):
    with (root / (name + '.log')).open('a') as log:
        return subprocess.run(list(map(str, cmd)), env=env, stdout=log, stderr=subprocess.STDOUT, check=check)


for name in subprocess.check_output(['git', 'ls-files'], cwd=args.source, text=True).splitlines():
    target = src / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.source / name, target)

# a probe stops only the backend the controller selects, until it is killed or let go
probe = r'''
pub fn recovery_probe(stage: &str) {
    let expected = format!("{}:{}", stage, std::process::id());
    if std::fs::read_to_string(CONTROL).ok().as_deref() != Some(&expected) {
        return;
    }
    std::fs::write(MARKER, stage).unwrap();
    let start = std::time::Instant::now();
    while std::fs::read_to_string(CONTROL).ok().as_deref() == Some(&expected) {
        assert!(start.elapsed().as_secs() < 120, "recovery probe timed out");
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}
'''.replace('CONTROL', json.dumps(str(control))).replace('MARKER', json.dumps(str(marker)))
p = src / 'crates/vchordrq/src/maintain.rs'
s = p.read_text()
for anchor, stage in [
    ('                Source::Runs(id) => match freepages::take(index, freepages_first, *id) {\n                    Some(guard) => {\n',
     'take'),
    ('        drop(jump_guard);\n        *pending.0.borrow_mut() = trace;\n', 'swap'),
    ('            freepages::free(index, freepages_first, id);\n            between();\n', 'evict'),
]:
    assert s.count(anchor) == 1, stage
    s = s.replace(anchor, anchor + f'        recovery_probe("{stage}");\n')
p.write_text(s + probe)
# and a build, as it writes a list, at the first page its vectors move on to, just after it served interrupts
p = src / 'crates/vchordrq/src/build.rs'
s = p.read_text()
anchor = '                if tape.last() != last || full {\n                    tape.unlock();\n                    index::interrupt::check();\n'
assert s.count(anchor) == 1, 'write'
p.write_text(s.replace(anchor, anchor + '                    crate::maintain::recovery_probe("write");\n') )
# every page that a scan, a VACUUM or an allocation can reach, as (kind, list, page)
(src / 'crates/vchordrq/src/page_map.rs').write_text(PAGE_MAP)
p = src / 'crates/vchordrq/src/lib.rs'
s = p.read_text()
assert 'mod maintain;\n' in s
p.write_text(s.replace('mod maintain;\n', 'mod maintain;\nmod page_map;\npub use page_map::page_map;\n', 1))
p = src / 'src/index/functions.rs'
s = p.read_text()
assert 'struct Index {\n' in s
p.write_text(s.replace('struct Index {\n', '''#[pgrx::pg_extern(sql = "")]
fn _vchordrq_page_map(indexrelid: Oid) -> SetOfIterator<'static, String> {
    let relation = open_checked(indexrelid, IndexAccess::Prewarm);
    let index = unsafe { PostgresRelation::<vchordrq::Opaque>::new(relation.raw()) };
    let map = vchordrq::page_map(&index);
    SetOfIterator::new(map.into_iter().map(|(k, l, b)| format!("{k}\\t{l}\\t{b}")))
}

struct Index {
''', 1))
run('build', ['make', '-C', src, 'build'])
lib = root / 'lib'
lib.mkdir()
for file in (src / 'build/pkglibdir').iterdir():
    shutil.copy2(file, lib / file.name)
run('initdb', [pg / 'initdb', '-D', primary, '-U', 'postgres', '-A', 'trust', '--locale=C', '--data-checksums'])
with (primary / 'postgresql.conf').open('a') as f:
    f.write(f"\nlisten_addresses=''\nunix_socket_directories='{root}'\n"
            f"shared_preload_libraries='vchord'\ndynamic_library_path='{lib}:$libdir'\n"
            f"extension_control_path='{src}/build/sharedir:$system'\n"
            "shared_buffers='256MB'\nautovacuum=off\nmax_parallel_maintenance_workers=0\nwal_level=replica\n"
            "max_wal_senders=4\nwal_keep_size='4GB'\nfsync=on\nfull_page_writes=on\nsynchronous_commit=on\n"
            "maintenance_work_mem='64MB'\n")


def start(data):
    run('start', [pg / 'pg_ctl', '-D', data, '-o', f'-p {PORTS[data]}', '-l', root / f'{data.name}.log', '-w', 'start'])


def stop(data, mode='fast'):
    run('stop', [pg / 'pg_ctl', '-D', data, '-m', mode, '-w', 'stop'], check=False)


def connect(data, **kwargs):
    return psycopg.connect(host=str(root), port=PORTS[data], user='postgres', dbname='postgres', autocommit=True,
                           options='-c statement_timeout=600000 -c client_encoding=UTF8', **kwargs)


def standby_up():
    stop(standby)
    shutil.rmtree(standby, ignore_errors=True)
    run('basebackup', [pg / 'pg_basebackup', '-h', root, '-p', PORTS[primary], '-U', 'postgres', '-D', standby,
                       '-R', '-X', 'stream'])
    start(standby)


def pages(c, idx):
    """the page map's pages by kind, the vectors a list's rows point to left out, and the index's size in pages"""
    rows = [line[0].split('\t') for line in c.execute('SELECT _vchordrq_page_map(%s::regclass::oid)', (idx,))]
    count = c.execute("SELECT pg_relation_size(%s::regclass) / current_setting('block_size')::int", (idx,)).fetchone()[0]
    where = {}
    for kind, l, block in rows:
        if kind == 'vector':
            continue
        assert int(block) not in where, f'page {block} is {where.get(int(block))} and {kind} {l}'
        where[int(block)] = (kind, int(l))
    lists = {}
    for block, (kind, l) in where.items():
        if kind in ('directory', 'frozen', 'appendable'):
            lists[l] = lists.get(l, 0) + 1
    return where, count, max(lists.values(), default=0)


def found(c, table, lists=None, op='<->'):
    c.execute('SET enable_seqscan = off')
    if lists:
        c.execute(f"SET vchordrq.probes = '{lists}'")
    sql = f'SELECT count(*), sum(id) FROM (SELECT id FROM {table} ORDER BY v {op} (SELECT v FROM {table} LIMIT 1) LIMIT 100000000) s'
    assert any('Index Scan' in r[0] for r in c.execute('EXPLAIN ' + sql)), 'not an index scan'
    return c.execute(sql).fetchone() == c.execute(f'SELECT count(*), sum(id) FROM {table}').fetchone()


def copy_rows(c, table, ids, owners, vectors):
    """binary COPY of (id int4, owner int8, v halfvec) rows"""
    import numpy as np
    out = bytearray(b'PGCOPY\n\xff\r\n\0' + struct.pack('>ii', 0, 0))
    halves = np.asarray(vectors, np.float32).astype('>f2')
    head = struct.pack('>hh', halves.shape[1], 0)
    for i, o, h in zip(ids, owners, halves):
        data = head + h.tobytes()
        out += struct.pack('>hiiiq', 3, 4, int(i), 8, int(o)) + struct.pack('>i', len(data)) + data
    out += struct.pack('>h', -1)
    with c.cursor().copy(f'COPY {table} FROM STDIN (FORMAT binary)') as cp:
        cp.write(bytes(out))


def life_slice(c, table, idx):
    """A few-list index of the lifecycle's first 15 sessions' uploads; then sessions until it grows by a third, their
    permanent deletes, and a VACUUM, which keeps the lists' old pages as runs; then as many sessions again, for the
    VACUUM under test to place each list in a run another left. (A single list cannot take its own old run, freed
    only after its swap, and the run from the VACUUM before is less than half of a list grown by a third twice.)"""
    import numpy as np
    a, s = np.load(args.life / 'assets.npz'), np.load(args.life / 'life.npz')
    emb = np.load(args.life / 'emb.npy', mmap_mode='r')
    sd, st, up = s['sess_day'], s['sess_start'], s['up_order']
    smart = (a['vis'] != 2) & (a['emb_row'] >= 0)
    c.execute(f'DROP TABLE IF EXISTS {table}, {table}_stage')
    c.execute(f'CREATE TABLE {table} (id int PRIMARY KEY, owner int8, v halfvec(768))')
    c.execute(f'CREATE UNLOGGED TABLE {table}_stage (id int, owner int8, v halfvec(768))')

    def apply(x0, x1):
        ups = up[st[x0]:st[x1]]
        ups = ups[smart[ups]]
        for k in range(0, len(ups), 20000):
            r = ups[k:k + 20000]
            copy_rows(c, f'{table}_stage', r, a['owner'][r], emb[a['emb_row'][r]])
            c.execute(f'INSERT INTO {table} SELECT * FROM {table}_stage ON CONFLICT (id) DO UPDATE SET v = excluded.v')
            c.execute(f'TRUNCATE {table}_stage')
        gone = np.flatnonzero(np.isin(s['del_day'], np.unique(sd[x0:x1])) & ~s['user_del'])
        c.execute(f'DELETE FROM {table} WHERE id = ANY(%s)', (gone.tolist(),))
        return c.execute(f'SELECT count(*) FROM {table}').fetchone()[0]

    def grown(x0, rows):
        x = x0 + 1
        while x < len(sd) and len(np.flatnonzero(smart[up[st[x0]:st[x]]])) < rows // 3:
            x += 1
        return x
    rows = apply(0, 15)
    c.execute(f"CREATE INDEX {idx} ON {table} USING vchordrq (v halfvec_cosine_ops, owner) WITH (options = $$\n[build.internal]\nlists = [4]\n$$)")
    x = grown(15, rows)
    rows = apply(15, x)
    c.execute(f'VACUUM {table}')
    apply(x, grown(x, rows))


LISTS, DIM = 1100, 16
results = []
start(primary)
try:
    with connect(primary) as c:
        c.execute('CREATE EXTENSION IF NOT EXISTS vchord CASCADE')
        c.execute("CREATE FUNCTION _vchordrq_page_map(oid) RETURNS SETOF text STRICT LANGUAGE c AS 'vchord', '_vchordrq_page_map_wrapper'")
    standby_up()
    stages = ['swap', 'evict'] + (['take'] if args.life else [])
    for n, (stage, mode) in enumerate([(s, m) for s in stages for m in ['crash', 'promote']]):
        table, idx = f'runs_{n}', f'runs_{n}_idx'
        op = '<=>' if stage == 'take' else '<->'
        with connect(primary) as c:
            if stage == 'take':
                life_slice(c, table, idx)
            else:
                # a list's rows near its own centroid. For `evict`, one row each at the build, then lists that outgrow
                # their pages, so that the VACUUM extends for each and keeps more freed runs than fit; else all rows at
                # the build, a tenth deleted, so that the VACUUM places each list in the run the one before it left
                c.execute(f'DROP TABLE IF EXISTS {table}, {table}_c')
                c.execute(f'CREATE TABLE {table}_c (id int PRIMARY KEY, parent int, vector vector({DIM}))')
                c.execute(f"SELECT setseed({0.1 + n / 100})")
                c.execute(f'INSERT INTO {table}_c SELECT i - 1, NULL, ARRAY(SELECT random() FROM generate_series(1, {DIM}))::real[]::vector FROM generate_series(1, {LISTS}) i')
                c.execute(f'CREATE TABLE {table} (id int PRIMARY KEY, v vector({DIM}))')
                near = f'unnest((SELECT vector::real[] FROM {table}_c WHERE id = g % {LISTS}))'
                fill = lambda a, b: c.execute(f'INSERT INTO {table} SELECT g, ARRAY(SELECT x + 0.0001 * (g % 97) FROM {near} x)::vector FROM generate_series({a}, {b}) g')
                fill(1, LISTS if stage == 'evict' else LISTS * 200)
                c.execute(f"CREATE INDEX {idx} ON {table} USING vchordrq (v vector_l2_ops) WITH (options = $$\nbuild.external.table = 'public.{table}_c'\n$$)")
                if stage == 'evict':
                    fill(LISTS + 1, LISTS * 200)
                else:
                    c.execute(f'DELETE FROM {table} WHERE id % 10 = 3')
            before = pages(c, idx)

            c.execute('CHECKPOINT')
        with connect(primary) as a, ThreadPoolExecutor() as pool:
            marker.unlink(missing_ok=True)
            control.write_text(f'{stage}:{a.info.backend_pid}')
            vacuum = pool.submit(lambda: a.execute(f'VACUUM {table}'))
            deadline = time.monotonic() + 600
            while not marker.exists() and not vacuum.done():
                assert time.monotonic() < deadline, 'probe timeout'
                time.sleep(.02)
            if not marker.exists():
                results.append(dict(stage=stage, mode=mode, fired=False, vacuum=str(vacuum.exception())))
                (root / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
                print(stage, mode, 'probe did not fire', flush=True)
                control.unlink(missing_ok=True)
                continue
            if mode == 'promote':
                time.sleep(2)  # the WAL writer sends what the VACUUM logged
                with connect(primary) as c, connect(standby) as s:
                    lsn = c.execute('SELECT pg_current_wal_flush_lsn()').fetchone()[0]
                    while s.execute('SELECT pg_last_wal_replay_lsn() < %s', (lsn,)).fetchone()[0]:
                        time.sleep(.05)
                run('promote', [pg / 'pg_ctl', '-D', standby, '-w', 'promote'])
            stop(primary, 'immediate')
            control.unlink(missing_ok=True)
            try:
                vacuum.result(timeout=60)
                raise AssertionError('the stopped VACUUM finished')
            except psycopg.Error:
                pass
        survivor = standby if mode == 'promote' else primary
        if mode == 'crash':
            start(primary)
        with connect(survivor) as c:
            where, count, largest = pages(c, idx)
            lost = count - len(where)
            assert lost <= 2 * before[2], f'{lost} pages lost, more than a list ({before[2]})'
            assert found(c, table, LISTS if stage != 'take' else 4, op), 'a row is not found'
            c.execute(f'DELETE FROM {table} WHERE id % 10 = 7 AND id <> (SELECT id FROM {table} LIMIT 1)')
            c.execute(f'VACUUM {table}')
            where2, count2, _ = pages(c, idx)
            assert count2 - len(where2) <= lost, 'the second VACUUM lost pages'
            assert found(c, table, LISTS if stage != 'take' else 4, op), 'a row is not found after a second VACUUM'
            kept = sum(1 for kind, l in where.values() if kind == 'free' and l == -2)
            results.append(dict(stage=stage, mode=mode, pages=count, lost=lost, largest_list=before[2],
                                kept_run_pages=kept, second_vacuum_lost=count2 - len(where2)))
            (root / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
            print(stage, mode, 'passed', results[-1], flush=True)
        if mode == 'promote':
            stop(standby, 'fast')
            start(primary)
        standby_up()

    # a one-list build cancelled while it writes: an error, no temporary file left, no index, and it builds after
    stop(standby)
    with connect(primary) as c:
        c.execute('DROP TABLE IF EXISTS cancelled')
        c.execute('CREATE TABLE cancelled (id int, owner int8, v vector(768))')
        c.execute("INSERT INTO cancelled SELECT g, g % 7, ARRAY(SELECT sin(g * 0.001 + j) FROM generate_series(1, 768) j)::real[]::vector FROM generate_series(1, 20000) g")
    ddl = "CREATE INDEX cancelled_idx ON cancelled USING vchordrq (v vector_l2_ops, owner) WITH (options = $$\n[build.internal]\nlists = []\n$$)"
    with connect(primary) as a, connect(primary) as c, ThreadPoolExecutor() as pool:
        marker.unlink(missing_ok=True)
        control.write_text(f'write:{a.info.backend_pid}')
        build = pool.submit(lambda: a.execute(ddl))
        deadline = time.monotonic() + 600
        while not marker.exists():
            assert not build.done(), f'the build ended before it wrote: {build.exception()}'
            assert time.monotonic() < deadline, 'probe timeout'
            time.sleep(.02)
        c.execute('SELECT pg_cancel_backend(%s)', (a.info.backend_pid,))
        control.unlink()
        try:
            build.result(timeout=60)
            raise AssertionError('the cancelled build finished')
        except psycopg.errors.QueryCanceled:
            pass
        assert c.execute('SELECT count(*) FROM pg_ls_tmpdir()').fetchone()[0] == 0, 'a temporary file is left'
        assert c.execute("SELECT to_regclass('cancelled_idx')").fetchone()[0] is None
        c.execute(ddl)
        assert found(c, 'cancelled')
        results.append(dict(stage='write', mode='cancel'))
        (root / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
        print('write cancel passed', flush=True)
finally:
    control.unlink(missing_ok=True)
    stop(standby)
    stop(primary)
