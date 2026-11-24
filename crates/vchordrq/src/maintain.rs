// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use crate::closure_lifetime_binder::{id_0, id_1, id_2, id_3};
use crate::keys::{Columns, Keys};
use crate::operator::Operator;
use crate::tape_writer::{DirectoryTapeWriter, FrozenTapeWriter};
use crate::tuples::*;
use crate::{Branch, Opaque, freepages, tape};
use index::prefetcher::PrefetcherSequenceFamily;
use index::relation::{
    Page, PageGuard, Relation, RelationRead, RelationReadTypes, RelationWrite, RelationWriteTypes,
};
use index_accessor::FunctionalAccessor;
use rabitq::packing::unpack;
use std::cell::RefCell;
use std::num::NonZero;

pub trait MaintainChooser {
    fn choose(&mut self, i: usize) -> bool;
    fn participants(&self) -> usize {
        1
    }
    /// Whether a chosen list whose compaction would reproduce its tapes
    /// unchanged may be skipped (VACUUM); parallel build compaction always
    /// rewrites.
    fn skip_unchanged(&self) -> bool {
        false
    }
    /// Whether the caller owns the index exclusively (CREATE INDEX build
    /// compaction), so no concurrent readers or writers access the lists.
    fn is_exclusive(&self) -> bool {
        false
    }
}

pub struct Maintain {
    pub number_of_formerly_allocated_pages: usize,
    pub number_of_freshly_allocated_pages: usize,
    pub number_of_freed_pages: usize,
    /// Live index tuples in the lists that were chosen and compacted.
    pub number_of_tuples: u64,
}

#[derive(Default)]
struct PendingPages {
    /// Unpublished pages allocated for the list currently being compacted.
    in_flight: Vec<u32>,
    /// Retired pages from finished lists, kept locally for reuse by later lists.
    reusable: Vec<u32>,
}

/// The pages a maintain holds unpublished or locally reusable before returning to `freepages`.
#[derive(Default)]
pub struct Pending(RefCell<PendingPages>);

impl Pending {
    /// Frees all held pages when maintain aborts with no page locked, as in its `check`.
    pub fn give_back<R: RelationRead + RelationWrite>(&self, index: &R)
    where
        R::Page: Page<Opaque = Opaque>,
    {
        let meta_guard = index.read(0);
        let meta_bytes = meta_guard.get(1).expect("data corruption");
        let freepages_first = MetaTuple::deserialize_ref(meta_bytes).freepages_first();
        drop(meta_guard);
        self.free_in_flight(index, freepages_first);
        self.free_reusable(index, freepages_first, || ());
    }
    fn free_in_flight<R: RelationWrite>(&self, index: &R, freepages_first: u32)
    where
        R::Page: Page<Opaque = Opaque>,
    {
        loop {
            let Some(id) = self.0.borrow_mut().in_flight.pop() else {
                return;
            };
            freepages::free(index, freepages_first, id);
        }
    }
    fn free_reusable<R: RelationWrite>(
        &self,
        index: &R,
        freepages_first: u32,
        mut between: impl FnMut(),
    ) where
        R::Page: Page<Opaque = Opaque>,
    {
        loop {
            let Some(id) = self.0.borrow_mut().reusable.pop() else {
                return;
            };
            freepages::free(index, freepages_first, id);
            between();
        }
    }
}

/// `memory`: the most bytes of a list's rows that are sorted at once by their keys.
/// `check`: a delay point, where maintain locks no page and `pending` may be given back.
pub fn maintain<'b, R: RelationRead + RelationWrite, O: Operator>(
    index: &'b R,
    mut prefetch_h0_tuples: impl PrefetcherSequenceFamily<'b, R>,
    chooser: &mut impl MaintainChooser,
    memory: usize,
    pending: &Pending,
    check: impl Fn(),
) -> Maintain
where
    R::Page: Page<Opaque = Opaque>,
{
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
    let dim = meta_tuple.dim();
    let height_of_root = meta_tuple.height_of_root();
    let freepages_first = meta_tuple.freepages_first();
    let keys = Keys::of_meta(meta_tuple);
    let keyed = !keys.is_empty();
    let summarized = keys.summaries() != 0;
    let exclusive = chooser.is_exclusive();
    let memory = memory / chooser.participants();

    type State = Vec<u32>;
    let mut state: State = vec![meta_tuple.first()];

    drop(meta_guard);

    struct Buffers {
        number_of_formerly_allocated_pages: usize,
        number_of_freshly_allocated_pages: usize,
    }

    let buffers = RefCell::new(Buffers {
        number_of_formerly_allocated_pages: 0,
        number_of_freshly_allocated_pages: 0,
    });

    let step = |state: State| {
        let mut results = Vec::new();
        for first in state {
            tape::read_h1_tape::<R, _, _>(
                tape::by_next(index, first),
                || FunctionalAccessor::new((), id_0(|_, _| ()), id_1(|_, _| [(); _])),
                |(), _, _, first, _| results.push(first),
            );
            // `check` may sleep or serve interrupts, which LWLocks hold off: never under a buffer lock (#30)
            check();
        }
        results
    };

    for _ in (1..height_of_root).rev() {
        state = step(state);
    }

    let hooked_index = RelationHooked(index, {
        id_3(|index: &R, opaque: Opaque, tracking_freespace: bool| {
            let mut buffers = buffers.borrow_mut();
            let guard =
                if !tracking_freespace && let Some(id) = pending.0.borrow_mut().reusable.pop() {
                    buffers.number_of_formerly_allocated_pages += 1;
                    let mut guard = index.write(id, false);
                    guard.clear(opaque);
                    guard
                } else if !tracking_freespace
                    && let Some(mut guard) = freepages::alloc(index, freepages_first)
                {
                    buffers.number_of_formerly_allocated_pages += 1;
                    guard.clear(opaque);
                    guard
                } else {
                    buffers.number_of_freshly_allocated_pages += 1;
                    index.extend(opaque, tracking_freespace)
                };
            pending.0.borrow_mut().in_flight.push(guard.id());
            guard
        })
    });

    let mut number_of_tuples = 0_u64;
    let mut number_of_freed_pages = 0_usize;

    let mut words = Vec::new();
    for (idx, first) in state.into_iter().enumerate() {
        if !chooser.choose(idx) {
            continue;
        }

        if chooser.skip_unchanged()
            && let Some(tuples) = inspect_unchanged::<R>(index, first)
        {
            number_of_tuples += tuples;
            check();
            continue;
        }

        // only a maintain moves a list's tapes, and no other visits this list: they stay until the swap
        let (directory_first, appendable_first) = {
            let jump_guard = index.read(first);
            let jump_bytes = jump_guard.get(1).expect("data corruption");
            let jump_tuple = JumpTuple::deserialize_ref(jump_bytes);
            (jump_tuple.directory_first(), jump_tuple.appendable_first())
        };

        let mut trace = Vec::new();
        let mut tuples = 0_u64;
        let frozen_accessor = || {
            FunctionalAccessor::new(
                Vec::<[u8; 16]>::new(),
                Vec::<[u8; 16]>::extend_from_slice,
                id_1(
                    |elements: Vec<_>, (metadata, delta): (&[[f32; 32]; 4], &[f32; 32])| {
                        let unpacked = unpack(&elements);
                        std::array::from_fn(|i| {
                            let f = |&x| [x & 1 != 0, x & 2 != 0, x & 4 != 0, x & 8 != 0];
                            let signs = unpacked[i].iter().flat_map(f).collect::<Vec<_>>();
                            (
                                (
                                    rabitq::bit::CodeMetadata {
                                        dis_u_2: metadata[0][i],
                                        factor_cnt: metadata[1][i],
                                        factor_ip: metadata[2][i],
                                        factor_err: metadata[3][i],
                                    },
                                    signs,
                                ),
                                delta[i],
                            )
                        })
                    },
                ),
            )
        };

        let mut tape: Option<FrozenTapeWriter<'_, _>> = None;
        let mut pages_written = 0_usize;
        const UNLOCK_INTERVAL_PAGES: usize = 16;
        let mut freeze = |tape: &mut Option<FrozenTapeWriter<'_, _>>, branch, columns: &[u8]| {
            let writer =
                tape.get_or_insert_with(|| FrozenTapeWriter::create(&hooked_index, &keys, false));
            let last = writer.last();
            writer.push(branch, columns);
            if !exclusive && writer.last() != last {
                pages_written += 1;
                if pages_written >= UNLOCK_INTERVAL_PAGES {
                    pages_written = 0;
                    writer.unlock();
                    check();
                }
            }
        };

        let mut at = (appendable_first, 0);
        let mut pass = 0_usize;
        let rows = RefCell::new(Rows::new(dim as _));
        let stream_batch: RefCell<Vec<Branch<NonZero<u64>>>> = RefCell::new(Vec::new());
        const STREAM_BATCH_ROWS: usize = 256;

        if exclusive {
            if !keyed {
                let mut push = id_2(|row| {
                    freeze(&mut tape, branch_of_row(row, dim as _), &[]);
                    tuples += 1;
                });
                let pages = tape::read_directory_words::<R>(
                    tape::by_next(index, directory_first).inspect(|guard| trace.push(guard.id())),
                    &mut words,
                    true,
                );
                tape::read_frozen_tape::<R, _, _>(
                    tape::by_directory(&mut prefetch_h0_tuples, words[..pages].iter().copied())
                        .inspect(|guard| trace.push(guard.id())),
                    tape::All,
                    frozen_accessor,
                    (),
                    &mut push,
                );
                let appendable_guards =
                    tape::by_next(index, appendable_first).inspect(|guard| trace.push(guard.id()));
                let access = |metadata, elements: &[u64], delta| {
                    (unpack_code(metadata, elements, dim as _), delta)
                };
                tape::read_appendable_tape::<R, _>(appendable_guards, 0, access, (), &mut push);
            } else {
                let mut push = id_2(|row| {
                    let mut r = rows.borrow_mut();
                    r.push(row, &keys);
                    tuples += 1;
                    if r.bytes() > memory {
                        r.flush(&keys, |branch, columns| freeze(&mut tape, branch, columns));
                    }
                });
                let pages = tape::read_directory_words::<R>(
                    tape::by_next(index, directory_first).inspect(|guard| trace.push(guard.id())),
                    &mut words,
                    true,
                );
                tape::read_frozen_tape::<R, _, _>(
                    tape::by_directory(&mut prefetch_h0_tuples, words[..pages].iter().copied())
                        .inspect(|guard| trace.push(guard.id())),
                    tape::All,
                    frozen_accessor,
                    (),
                    &mut push,
                );
                let appendable_guards =
                    tape::by_next(index, appendable_first).inspect(|guard| trace.push(guard.id()));
                let access = |metadata, elements: &[u64], delta| {
                    (unpack_code(metadata, elements, dim as _), delta)
                };
                tape::read_appendable_tape::<R, _>(appendable_guards, 0, access, (), &mut push);
                rows.borrow_mut()
                    .flush(&keys, |branch, columns| freeze(&mut tape, branch, columns));
            }
        } else if !keyed {
            let mut push = id_2(|row| {
                stream_batch.borrow_mut().push(branch_of_row(row, dim as _));
                tuples += 1;
            });
            let mut between = || {
                if stream_batch.borrow().len() >= STREAM_BATCH_ROWS {
                    for branch in stream_batch.borrow_mut().drain(..) {
                        freeze(&mut tape, branch, &[]);
                    }
                    if let Some(writer) = tape.as_mut() {
                        writer.unlock();
                    }
                }
                check();
            };
            let pages = tape::read_directory_words::<R>(
                paced(tape::by_next(index, directory_first), &mut between)
                    .inspect(|guard| trace.push(guard.id())),
                &mut words,
                true,
            );
            tape::read_frozen_tape::<R, _, _>(
                paced(
                    tape::by_directory(&mut prefetch_h0_tuples, words[..pages].iter().copied()),
                    &mut between,
                )
                .inspect(|guard| trace.push(guard.id())),
                tape::All,
                frozen_accessor,
                (),
                &mut push,
            );
            trace.push(appendable_first);
            pass = read_appendable(
                index,
                &mut at,
                &mut trace,
                ReadLimit::InitialTail,
                &mut between,
                &mut push,
                dim,
            );
            for branch in stream_batch.borrow_mut().drain(..) {
                freeze(&mut tape, branch, &[]);
            }
        } else {
            let mut push = id_2(|row| {
                rows.borrow_mut().push(row, &keys);
                tuples += 1;
            });
            let mut between = || {
                // a list larger than `memory` is written as sorted runs
                if rows.borrow().bytes() > memory {
                    rows.borrow_mut()
                        .flush(&keys, |branch, columns| freeze(&mut tape, branch, columns));
                    if let Some(writer) = tape.as_mut() {
                        writer.unlock();
                    }
                }
                check();
            };
            let pages = tape::read_directory_words::<R>(
                paced(tape::by_next(index, directory_first), &mut between)
                    .inspect(|guard| trace.push(guard.id())),
                &mut words,
                true,
            );
            tape::read_frozen_tape::<R, _, _>(
                paced(
                    tape::by_directory(&mut prefetch_h0_tuples, words[..pages].iter().copied()),
                    &mut between,
                )
                .inspect(|guard| trace.push(guard.id())),
                tape::All,
                frozen_accessor,
                (),
                &mut push,
            );
            // the appendable tape in passes, each through the tail it starts with: into the sort, then copied
            trace.push(appendable_first);
            pass = read_appendable(
                index,
                &mut at,
                &mut trace,
                ReadLimit::InitialTail,
                &mut between,
                &mut push,
                dim,
            );
            rows.borrow_mut()
                .flush(&keys, |branch, columns| freeze(&mut tape, branch, columns));
        }

        let tape = tape.unwrap_or_else(|| FrozenTapeWriter::create(&hooked_index, &keys, false));
        let (frozen_tape, branches, columns) = tape.into_inner();
        let frozen_first = { frozen_tape }.first();

        let mut directory = tape::Layout::new(keys.summaries());
        if exclusive {
            for guard in tape::by_next(index, frozen_first) {
                directory.page(guard.id());
                if summarized {
                    for i in 1..=guard.len() {
                        let bytes = guard.get(i).expect("data corruption");
                        directory.tuple(match FrozenTuple::deserialize_ref(bytes) {
                            FrozenTupleReader::_0(x) => {
                                Some(keys.summarize(x.payload(), x.columns()))
                            }
                            FrozenTupleReader::_1(_) => None,
                        });
                    }
                }
            }
        } else {
            for guard in paced(tape::by_next(index, frozen_first), &check) {
                directory.page(guard.id());
                if summarized {
                    for i in 1..=guard.len() {
                        let bytes = guard.get(i).expect("data corruption");
                        directory.tuple(match FrozenTuple::deserialize_ref(bytes) {
                            FrozenTupleReader::_0(x) => {
                                Some(keys.summarize(x.payload(), x.columns()))
                            }
                            FrozenTupleReader::_1(_) => None,
                        });
                    }
                }
            }
        }

        let mut directory_tape = DirectoryTapeWriter::create(&hooked_index, false);
        directory_tape.push(directory.ids.as_slice());
        if summarized {
            directory_tape.push_layout(&directory.words());
        }
        let directory_first = { directory_tape.into_inner() }.first();

        let mut appendable_tape = tape::TapeWriter::create(&hooked_index, false);
        let width = keys.columns();
        for (i, branch) in branches.into_iter().enumerate() {
            appendable_tape.push(appendable(branch, &columns[i * width..][..width]));
        }

        if exclusive {
            let appendable_first = { appendable_tape }.first();
            let mut jump_guard = index.write(first, false);
            let jump_bytes = jump_guard.get_mut(1).expect("data corruption");
            let mut jump_tuple = JumpTuple::deserialize_mut(jump_bytes);
            *jump_tuple.directory_first() = directory_first;
            *jump_tuple.frozen_first() = frozen_first;
            *jump_tuple.appendable_first() = appendable_first;
            *jump_tuple.tuples() = tuples;
            number_of_tuples += tuples;
            drop(jump_guard);
            number_of_freed_pages += trace.len();
            {
                let mut held = pending.0.borrow_mut();
                held.in_flight.clear();
                held.reusable.extend(trace);
            }
            check();
        } else {
            let mut push_tail = id_2(|row| {
                if keyed {
                    rows.borrow_mut().push(row, &keys);
                } else {
                    stream_batch.borrow_mut().push(branch_of_row(row, dim as _));
                }
                tuples += 1;
            });
            let copy = |tape: &mut tape::TapeWriter<_, _>| -> bool {
                if keyed {
                    if rows.borrow().len() == 0 {
                        return false;
                    }
                    rows.borrow_mut().flush(&keys, |branch, columns| {
                        tape.push(appendable(branch, columns));
                    });
                    true
                } else {
                    if stream_batch.borrow().is_empty() {
                        return false;
                    }
                    for branch in stream_batch.borrow_mut().drain(..) {
                        tape.push(appendable(branch, &[]));
                    }
                    true
                }
            };
            appendable_tape.unlock();
            // another pass only while they shrink: the lock waits for what is appended during the last
            loop {
                let n = read_appendable(
                    index,
                    &mut at,
                    &mut trace,
                    ReadLimit::MaxPages(pass),
                    || {
                        if copy(&mut appendable_tape) {
                            appendable_tape.unlock();
                        }
                        check();
                    },
                    &mut push_tail,
                    dim,
                );
                if n >= pass {
                    break;
                }
                pass = n;
            }
            if copy(&mut appendable_tape) {
                appendable_tape.unlock();
            }

            // inserts append under the jump page's share lock, so none is under way
            let mut jump_guard = index.write(first, false);
            read_appendable(
                index,
                &mut at,
                &mut trace,
                ReadLimit::ToEnd,
                || (),
                &mut push_tail,
                dim,
            );
            copy(&mut appendable_tape);
            let jump_bytes = jump_guard.get_mut(1).expect("data corruption");
            let mut jump_tuple = JumpTuple::deserialize_mut(jump_bytes);
            *jump_tuple.directory_first() = directory_first;
            *jump_tuple.frozen_first() = frozen_first;
            *jump_tuple.appendable_first() = { appendable_tape }.first();
            *jump_tuple.tuples() = tuples;
            number_of_tuples += tuples;

            // dropping the guard publishes the new tapes; readers of the old ones hold its share lock
            drop(jump_guard);
            number_of_freed_pages += trace.len();
            {
                let mut held = pending.0.borrow_mut();
                held.in_flight.clear();
                held.reusable.extend(trace);
            }
            pending.free_reusable(index, freepages_first, &check);
            check();
        }
    }

    if exclusive {
        pending.free_reusable(index, freepages_first, || ());
    }

    let buffers = RefCell::into_inner(buffers);

    Maintain {
        number_of_formerly_allocated_pages: buffers.number_of_formerly_allocated_pages,
        number_of_freshly_allocated_pages: buffers.number_of_freshly_allocated_pages,
        number_of_freed_pages,
        number_of_tuples,
    }
}

fn branch_of_row(row: tape::Row<'_, (rabitq::bit::Code, f32)>, dim: usize) -> Branch<NonZero<u64>> {
    let ((metadata, mut signs), delta) = row.value;
    signs.truncate(dim);
    Branch {
        code: (metadata, signs),
        delta,
        prefetch: row.prefetch.to_vec(),
        head: row.head,
        norm: 0.0,
        extra: row.payload,
    }
}

/// `pages`, with `between` run before each is read.
fn paced<I: Iterator>(mut pages: I, mut between: impl FnMut()) -> impl Iterator<Item = I::Item> {
    std::iter::from_fn(move || {
        between();
        pages.next()
    })
}

#[derive(Clone, Copy)]
enum ReadLimit {
    InitialTail,
    MaxPages(usize),
    ToEnd,
}

/// Reads the appendable tape on from `at`, a page and its tuples read; returns its pages read.
fn read_appendable<'b, R: RelationRead + 'b>(
    index: &'b R,
    at: &mut (u32, u16),
    trace: &mut Vec<u32>,
    limit: ReadLimit,
    mut between: impl FnMut(),
    callback: &mut impl for<'a> FnMut(tape::Row<'a, (rabitq::bit::Code, f32)>),
    dim: u32,
) -> usize
where
    R::Page: Page<Opaque = Opaque>,
{
    let (first, skip) = *at;
    let stop_at = match limit {
        ReadLimit::InitialTail => {
            let guard = index.read(first);
            let opaque = *guard.get_opaque();
            drop(guard);
            if opaque.next == u32::MAX {
                Some(first)
            } else {
                let hint = if opaque.skip == first {
                    opaque.next
                } else {
                    opaque.skip
                };
                Some(tape::by_next(index, hint).last().expect("a page").id())
            }
        }
        ReadLimit::MaxPages(_) | ReadLimit::ToEnd => None,
    };
    let max_pages = match limit {
        ReadLimit::MaxPages(n) => Some(n),
        ReadLimit::InitialTail | ReadLimit::ToEnd => None,
    };
    let mut pages = 0;
    let mut next = Some(first);
    let guards = std::iter::from_fn(|| {
        let id = next?;
        if pages != 0 {
            between();
            trace.push(id);
        }
        let guard = index.read(id);
        let next_id = guard.get_opaque().next;
        *at = (id, guard.len());
        pages += 1;
        let stop =
            next_id == u32::MAX || stop_at == Some(id) || max_pages.is_some_and(|cap| pages >= cap);
        next = (!stop).then_some(next_id);
        Some(guard)
    });
    let access =
        |metadata, elements: &[u64], delta| (unpack_code(metadata, elements, dim as _), delta);
    tape::read_appendable_tape::<R, _>(guards, skip, access, (), callback);
    pages
}

/// A row as an appendable tuple stores it.
fn appendable(branch: Branch<NonZero<u64>>, columns: &[u8]) -> AppendableTuple {
    AppendableTuple {
        metadata: [
            branch.code.0.dis_u_2,
            branch.code.0.factor_cnt,
            branch.code.0.factor_ip,
            branch.code.0.factor_err,
        ],
        elements: rabitq::bit::binary::pack_code(&branch.code.1),
        delta: branch.delta,
        prefetch: branch.prefetch,
        head: branch.head,
        payload: Some(branch.extra),
        columns: columns.to_vec(),
    }
}

/// Decides, under the list's jump page share lock, whether compacting the
/// list would reproduce its tapes unchanged: no frozen or appendable entry is
/// deleted (raw payload slots are inspected; the tape readers skip deleted
/// entries) and the appendable tape holds fewer tuples than one frozen block,
/// so compaction would write the same frozen blocks and the same remainder.
/// Returns the live tuple count of an unchanged list (#24).
fn inspect_unchanged<R: RelationRead>(index: &R, first: u32) -> Option<u64>
where
    R::Page: Page<Opaque = Opaque>,
{
    let jump_guard = index.read(first);
    let jump_bytes = jump_guard.get(1).expect("data corruption");
    let jump_tuple = JumpTuple::deserialize_ref(jump_bytes);
    // Check the short appendable tape first so insert-heavy lists bail out
    // without scanning the entire frozen tape.
    let mut appendable = 0_u64;
    for guard in tape::by_next(index, jump_tuple.appendable_first()) {
        for i in 1..=guard.len() {
            let bytes = guard.get(i).expect("data corruption");
            // a deleted entry: the list needs compacting
            AppendableTuple::deserialize_ref(bytes).payload()?;
            appendable += 1;
        }
        if appendable >= 32 {
            return None;
        }
    }
    let mut frozen = 0_u64;
    let mut words = Vec::new();
    let directory = tape::by_next(index, jump_tuple.directory_first());
    let pages = tape::read_directory_words::<R>(directory, &mut words, false);
    for &id in &words[..pages] {
        let guard = index.read(id);
        for i in 1..=guard.len() {
            let bytes = guard.get(i).expect("data corruption");
            if let FrozenTupleReader::_0(tuple) = FrozenTuple::deserialize_ref(bytes) {
                if tuple.payload().iter().any(Option::is_none) {
                    return None;
                }
                frozen += 32;
            }
        }
    }
    drop(jump_guard);
    Some(frozen + appendable)
}

/// A list's row, but for its signs, pages and columns.
#[derive(Clone, Copy)]
struct Fixed {
    metadata: [f32; 4],
    delta: f32,
    head: u16,
    payload: NonZero<u64>,
}

/// A list's rows, packed, to be sorted by their keys.
struct Rows {
    dim: usize,
    fixed: Vec<Fixed>,
    /// per row: the signs of the code, a bit each
    signs: Vec<u64>,
    /// per row: the same number of pages
    prefetch: Vec<u32>,
    /// per row: the filter key columns, as an appendable tuple stores them
    columns: Vec<u8>,
}

impl Rows {
    fn new(dim: usize) -> Self {
        Self {
            dim,
            fixed: Vec::new(),
            signs: Vec::new(),
            prefetch: Vec::new(),
            columns: Vec::new(),
        }
    }
    fn len(&self) -> usize {
        self.fixed.len()
    }
    fn bytes(&self) -> usize {
        let fixed = size_of::<Fixed>() + size_of::<u32>();
        let words = self.signs.len() * size_of::<u64>() + self.prefetch.len() * 4;
        self.len() * fixed + words + self.columns.len()
    }
    fn push(&mut self, row: tape::Row<'_, (rabitq::bit::Code, f32)>, keys: &Keys) {
        let ((m, signs), delta) = row.value;
        let metadata = [m.dis_u_2, m.factor_cnt, m.factor_ip, m.factor_err];
        let (head, payload) = (row.head, row.payload);
        self.fixed.push(Fixed {
            metadata,
            delta,
            head,
            payload,
        });
        // a frozen code is padded to a multiple of 4 signs, with zeros
        let signs = rabitq::bit::binary::pack_code(&signs[..self.dim]);
        self.signs.extend_from_slice(&signs);
        self.prefetch.extend_from_slice(row.prefetch);
        row.columns.extend(keys, &mut self.columns);
    }
    /// Writes the rows to `sink` in the order of their keys, stable, and clears them.
    fn flush(&mut self, keys: &Keys, mut sink: impl FnMut(Branch<NonZero<u64>>, &[u8])) {
        let width = keys.columns();
        let row = |i: u32| Columns::row(&self.columns[i as usize * width..][..width]);
        let mut order = (0..self.len() as u32).collect::<Vec<_>>();
        order.sort_unstable_by(|&i, &j| keys.cmp(row(i), row(j)).then(i.cmp(&j)));
        for i in order {
            let (branch, columns) = self.branch(i as _);
            sink(branch, columns);
        }
        *self = Self::new(self.dim);
    }
    fn branch(&self, i: usize) -> (Branch<NonZero<u64>>, &[u8]) {
        let Fixed {
            metadata,
            delta,
            head,
            payload,
            ..
        } = self.fixed[i];
        let words = self.dim.div_ceil(64);
        let n = self.prefetch.len() / self.len();
        let width = self.columns.len() / self.len();
        let branch = Branch {
            code: unpack_code(metadata, &self.signs[i * words..][..words], self.dim),
            delta,
            prefetch: self.prefetch[i * n..][..n].to_vec(),
            head,
            norm: 0.0,
            extra: payload,
        };
        (branch, &self.columns[i * width..][..width])
    }
}

/// A code from its metadata and its first `dim` signs, as an appendable tuple stores it.
fn unpack_code(metadata: [f32; 4], words: &[u64], dim: usize) -> rabitq::bit::Code {
    let signs = words
        .iter()
        .flat_map(|x| std::array::from_fn::<_, 64, _>(|j| x & (1 << j) != 0))
        .take(dim)
        .collect();
    (rabitq::bit::CodeMetadata::from_array(metadata), signs)
}

#[derive(Clone)]
struct RelationHooked<'b, R, E>(&'b R, E);

impl<'b, R, E> Relation for RelationHooked<'b, R, E>
where
    R: Relation,
    E: Clone,
{
    type Page = R::Page;
}

impl<'b, R, E> RelationReadTypes for RelationHooked<'b, R, E>
where
    R: RelationRead,
    E: Clone,
{
    type ReadGuard<'a> = R::ReadGuard<'a>;
}

impl<'b, R, E> RelationRead for RelationHooked<'b, R, E>
where
    R: RelationRead,
    E: Clone,
{
    fn read(&self, id: u32) -> Self::ReadGuard<'_> {
        self.0.read(id)
    }
}

impl<'b, R, E> RelationWriteTypes for RelationHooked<'b, R, E>
where
    R: RelationWrite,
    E: Clone + for<'a> Fn(&'a R, <Self::Page as Page>::Opaque, bool) -> R::WriteGuard<'a>,
{
    type WriteGuard<'a> = R::WriteGuard<'a>;
}

impl<'b, R, E> RelationWrite for RelationHooked<'b, R, E>
where
    R: RelationWrite,
    E: Clone + for<'a> Fn(&'a R, <Self::Page as Page>::Opaque, bool) -> R::WriteGuard<'a>,
{
    fn commit_pair(&self, first: Self::WriteGuard<'_>, second: Self::WriteGuard<'_>) {
        self.0.commit_pair(first, second);
    }

    fn write(&self, id: u32, tracking_freespace: bool) -> Self::WriteGuard<'_> {
        self.0.write(id, tracking_freespace)
    }

    fn extend(
        &self,
        opaque: <Self::Page as Page>::Opaque,
        tracking_freespace: bool,
    ) -> Self::WriteGuard<'_> {
        (self.1)(self.0, opaque, tracking_freespace)
    }

    fn search(&self, freespace: usize) -> Option<Self::WriteGuard<'_>> {
        self.0.search(freespace)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keys::{Summary, Width};
    use crate::operator::Op;
    use crate::types::{DistanceKind, Structure, VchordrqIndexOptions, VectorKind, VectorOptions};
    use index::test_support::{MakePlainPrefetcher, MemoryRelation};
    use index_accessor::L2S;
    use std::cell::Cell;
    use std::panic::AssertUnwindSafe;
    use vector::vect::VectOwned;

    type M = MemoryRelation<Opaque>;
    type O = Op<VectOwned<f32>, L2S>;
    const DIM: usize = 100;

    struct All;

    impl MaintainChooser for All {
        fn choose(&mut self, _: usize) -> bool {
            true
        }
    }

    fn keys() -> [Keys; 2] {
        [
            Keys::new([]),
            Keys::new([(Width::W8, false, Some(Summary::Bounds))]),
        ]
    }

    fn meta(relation: &M) -> (u32, u32) {
        let guard = relation.read(0);
        let meta = MetaTuple::deserialize_ref(guard.get(1).unwrap());
        (meta.first(), meta.freepages_first())
    }

    /// One list as a build writes it, with rows 1 to `n` appended; returns its jump page.
    fn index(relation: &M, keys: &Keys, n: u64) -> u32 {
        let (dim, v, d) = (DIM as _, VectorKind::Vecf32, DistanceKind::L2S);
        let options = VchordrqIndexOptions {
            degree_of_parallelism: 1,
            ..Default::default()
        };
        let centroids = vec![VectOwned::new(vec![0.0; DIM])];
        let structure = Structure {
            centroids,
            children: vec![Vec::new()],
        };
        crate::build::<_, O>(
            VectorOptions { dim, v, d },
            options,
            keys,
            relation,
            vec![structure],
        );
        let first = meta(relation).0;
        (1..=n).for_each(|payload| append(relation, keys, first, payload));
        first
    }

    /// As an insert appends, under the jump page's share lock.
    fn append(relation: &M, keys: &Keys, first: u32, payload: u64) {
        let key = (payload % 13).to_le_bytes();
        let tuple = AppendableTuple {
            metadata: [1.0; 4],
            delta: 0.0,
            payload: NonZero::new(payload),
            prefetch: vec![7],
            head: 0,
            elements: vec![payload; DIM.div_ceil(64)],
            columns: if keys.is_empty() {
                Vec::new()
            } else {
                [&key[..], &[0; 4]].concat()
            },
        };
        let jump_guard = relation.read(first);
        let appendable_first =
            JumpTuple::deserialize_ref(jump_guard.get(1).unwrap()).appendable_first();
        let bytes = AppendableTuple::serialize(&tuple);
        tape::append(
            relation,
            appendable_first,
            &bytes,
            false,
            Some(meta(relation).1),
        );
    }

    /// The list's payloads, sorted, its tuple count, and the pages neither in its tapes nor free.
    fn list(relation: &M, first: u32) -> (Vec<u64>, u64, u32) {
        let jump_guard = relation.read(first);
        let jump_tuple = JumpTuple::deserialize_ref(jump_guard.get(1).unwrap());
        let (mut payloads, mut pages) = (Vec::new(), 0);
        let accessor = || FunctionalAccessor::new((), id_0(|_, _| ()), id_1(|_, _| [(); _]));
        pages += tape::by_next(relation, jump_tuple.directory_first()).count() as u32;
        let frozen = tape::by_next(relation, jump_tuple.frozen_first()).inspect(|_| pages += 1);
        tape::read_frozen_tape::<M, _, _>(frozen, tape::All, accessor, (), &mut |row| {
            payloads.push(row.payload.get())
        });
        let appendable =
            tape::by_next(relation, jump_tuple.appendable_first()).inspect(|_| pages += 1);
        tape::read_appendable_tape::<M, _>(appendable, 0, |_, _, _| (), (), &mut |row| {
            payloads.push(row.payload.get())
        });
        let mut guard = relation.write(meta(relation).1, false);
        let mut id = *FreepagesTuple::deserialize_mut(guard.get_mut(1).unwrap()).first();
        drop(guard);
        while id != u32::MAX {
            pages += 1;
            id = relation.read(id).get_opaque().next;
        }
        payloads.sort_unstable();
        (payloads, jump_tuple.tuples(), relation.pages() - pages)
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn appended_meanwhile() {
        // `burst` rows at every delay point until the swap: with 120 they outpace each pass
        for (keys, memory, burst) in keys()
            .into_iter()
            .flat_map(|k| [(1 << 12, 1), (1 << 30, 1), (1 << 30, 120)].map(|(m, b)| (k, m, b)))
        {
            let relation = M::new([]);
            let first = index(&relation, &keys, 3000);
            let (next, calls, old) = (Cell::new(3001), Cell::new(0), Cell::new(0));
            let check = || {
                for id in 0..relation.pages() {
                    assert!(
                        relation.page(id).try_borrow_mut().is_ok(),
                        "page {id} is locked"
                    );
                }
                calls.set(calls.get() + 1);
                assert!(calls.get() < 100_000, "maintain does not end");
                let jump_guard = relation.read(first);
                let jump_tuple = JumpTuple::deserialize_ref(jump_guard.get(1).unwrap());
                if jump_tuple.appendable_first() == old.get() {
                    drop(jump_guard);
                    for _ in 0..burst {
                        append(&relation, &keys, first, next.get());
                        next.set(next.get() + 1);
                    }
                }
            };
            for _ in 0..2 {
                let (_, _, other) = list(&relation, first);
                let jump_guard = relation.read(first);
                old.set(JumpTuple::deserialize_ref(jump_guard.get(1).unwrap()).appendable_first());
                drop(jump_guard);
                let prefetch = MakePlainPrefetcher { index: &relation };
                let maintain = maintain::<_, O>(
                    &relation,
                    prefetch,
                    &mut All,
                    memory,
                    &Pending::default(),
                    check,
                );
                let (payloads, tuples, now) = list(&relation, first);
                assert_eq!(payloads, (1..next.get()).collect::<Vec<_>>());
                assert_eq!(
                    (tuples, maintain.number_of_tuples),
                    (payloads.len() as u64, tuples)
                );
                assert_eq!(now, other, "a page is lost");
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn abandoned() {
        for keys in keys() {
            let calls = Cell::new(0);
            let relation = M::new([]);
            index(&relation, &keys, 3000);
            let prefetch = MakePlainPrefetcher { index: &relation };
            maintain::<_, O>(
                &relation,
                prefetch,
                &mut All,
                1 << 12,
                &Pending::default(),
                || calls.set(calls.get() + 1),
            );
            for k in 1..=calls.get() {
                let relation = M::new([]);
                let first = index(&relation, &keys, 3000);
                let (_, _, other) = list(&relation, first);
                let (pending, calls) = (Pending::default(), Cell::new(0));
                let check = || {
                    calls.set(calls.get() + 1);
                    if calls.get() == k {
                        append(&relation, &keys, first, 3001);
                        std::panic::panic_any(k);
                    }
                };
                let prefetch = MakePlainPrefetcher { index: &relation };
                let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
                    maintain::<_, O>(&relation, prefetch, &mut All, 1 << 12, &pending, check)
                }));
                assert_eq!(*result.err().unwrap().downcast::<u32>().unwrap(), k);
                pending.give_back(&relation);
                let (payloads, _, now) = list(&relation, first);
                assert_eq!(payloads, (1..=3001).collect::<Vec<_>>());
                assert_eq!(now, other, "a page is lost at delay point {k}");
            }
        }
    }
}
