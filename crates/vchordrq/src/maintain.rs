// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use crate::closure_lifetime_binder::{id_1, id_2, id_3};
use crate::keys::Keys;
use crate::list::{ListWriter, Rows, appendable_tuple, firsts, unpack_code};
use crate::operator::Operator;
use crate::tape::paced;
use crate::tuples::*;
use crate::{Opaque, freepages, tape};
use index::prefetcher::PrefetcherSequenceFamily;
use index::relation::{
    Page, PageGuard, Relation, RelationRead, RelationReadTypes, RelationWrite, RelationWriteTypes,
};
use index_accessor::FunctionalAccessor;
use rabitq::packing::unpack;
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::ops::Range;

pub struct Maintain {
    pub number_of_formerly_allocated_pages: usize,
    pub number_of_freshly_allocated_pages: usize,
    pub number_of_freed_pages: usize,
    /// Live index tuples in the lists that were chosen and compacted.
    pub number_of_tuples: u64,
}

/// The pages a maintain holds: a list's new ones while unpublished, then its old ones, and the free runs that lists are placed on.
#[derive(Default)]
pub struct Pending {
    list: RefCell<Vec<u32>>,
    runs: RefCell<Runs>,
}

impl Pending {
    /// Frees them, where the maintain holding them locks no page, as in its `check`.
    pub fn give_back<R: RelationRead + RelationWrite>(&self, index: &R)
    where
        R::Page: Page<Opaque = Opaque>,
    {
        let meta_guard = index.read(0);
        let meta_bytes = meta_guard.get(1).expect("data corruption");
        let freepages_first = MetaTuple::deserialize_ref(meta_bytes).freepages_first();
        drop(meta_guard);
        self.free(index, freepages_first, || ());
    }
    /// The runs are kept for the next maintain, the longest first, and the pages of those that are not kept go to
    /// the free list, which hands out what is freed last first: they are freed from the highest, so they are handed
    /// out in order.
    fn free<R: RelationRead + RelationWrite>(
        &self,
        index: &R,
        freepages_first: u32,
        mut between: impl FnMut(),
    ) -> usize
    where
        R::Page: Page<Opaque = Opaque>,
    {
        let mut runs = std::mem::take(&mut *self.runs.borrow_mut());
        self.list
            .borrow_mut()
            .drain(..)
            .for_each(|id| runs.insert(id));
        let mut runs = runs.0.into_iter().map(|(s, e)| [s, e]).collect::<Vec<_>>();
        let n = runs.iter().map(|&[s, e]| (e - s) as usize).sum();
        runs.sort_by_key(|&[s, e]| std::cmp::Reverse(e - s));
        let kept = freepages::put_runs(index, freepages_first, &runs);
        let mut rest = runs[kept..]
            .iter()
            .flat_map(|&[s, e]| s..e)
            .collect::<Vec<_>>();
        rest.sort_unstable();
        *self.list.borrow_mut() = rest;
        while let Some(id) = { self.list.borrow_mut().pop() } {
            freepages::free(index, freepages_first, id);
            between();
        }
        n
    }
}

/// Free pages, as runs of pages that follow one another: each run's first page and its end.
#[derive(Default)]
struct Runs(BTreeMap<u32, u32>);

impl Runs {
    fn insert(&mut self, id: u32) {
        self.insert_run(id, id + 1);
    }
    /// Pages `start..end`, none of which it holds.
    fn insert_run(&mut self, start: u32, end: u32) {
        let (mut start, mut end) = (start, end);
        if let Some((&s, &e)) = self.0.range(..start).next_back()
            && e == start
        {
            start = s;
        }
        if let Some(e) = self.0.remove(&end) {
            end = e;
        }
        self.0.insert(start, end);
    }
    /// The front `n` pages of the shortest run of at least `n`, or else the longest run of at least `least`.
    fn take(&mut self, n: usize, least: usize) -> Option<Range<u32>> {
        let len = |(&s, &e): (&u32, &u32)| (e - s) as usize;
        let fits = self
            .0
            .iter()
            .filter(|&x| len(x) >= n)
            .min_by_key(|&x| len(x));
        let (&start, &end) = fits.or_else(|| {
            self.0
                .iter()
                .filter(|&x| len(x) >= least)
                .max_by_key(|&x| len(x))
        })?;
        self.0.remove(&start);
        let taken = start..end.min(start + n as u32);
        if taken.end < end {
            self.0.insert(taken.end, end);
        }
        Some(taken)
    }
    /// Up to `n` pages of the run that begins at `start`.
    fn take_from(&mut self, start: u32, n: usize) -> Option<Range<u32>> {
        let end = self.0.remove(&start)?;
        let taken = start..end.min(start + n as u32);
        if taken.end < end {
            self.0.insert(taken.end, end);
        }
        Some(taken)
    }
}

/// Where the pages of the list being written come from.
enum Source {
    /// Not chosen yet.
    None,
    /// Free pages that this maintain holds.
    Runs(Range<u32>),
    /// The pages at the head of the free list, as they were when chosen.
    Free(Range<u32>),
    Extend,
}

/// `memory`: the most bytes of a list's rows that are sorted at once by their keys.
/// `check`: a delay point, where maintain locks no page and `pending` may be given back.
pub fn maintain<'b, R: RelationRead + RelationWrite, O: Operator>(
    index: &'b R,
    mut prefetch_h0_tuples: impl PrefetcherSequenceFamily<'b, R>,
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
    let freepages_first = meta_tuple.freepages_first();
    let keys = Keys::of_meta(meta_tuple);

    drop(meta_guard);

    struct Buffers {
        number_of_formerly_allocated_pages: usize,
        number_of_freshly_allocated_pages: usize,
    }

    let buffers = RefCell::new(Buffers {
        number_of_formerly_allocated_pages: 0,
        number_of_freshly_allocated_pages: 0,
    });

    let state = firsts(index, &check);
    // the runs that earlier maintains left free; given back with the rest of the pending pages on an error
    for [start, end] in freepages::take_runs(index, freepages_first) {
        pending.runs.borrow_mut().insert_run(start, end);
    }

    // a list's `n` pages, in the order it is written: a free run that fits, else the longest free run if it takes
    // at least half of them, else an extension
    let choose = |index: &R, n: usize| {
        let mut runs = pending.runs.borrow_mut();
        if let Some(run) = runs.take(n, usize::MAX) {
            return Source::Runs(run);
        }
        let head = freepages::head(index, freepages_first, n);
        let free = head.first().map_or(0..0, |&id| id..id + head.len() as u32);
        if free.len() == n {
            return Source::Free(free);
        }
        let half = n.div_ceil(2);
        match runs.take(n, half.max(free.len() + 1)) {
            Some(run) => Source::Runs(run),
            None if free.len() >= half => Source::Free(free),
            None => Source::Extend,
        }
    };
    let placement = RefCell::new((0_usize, Source::None));
    let hooked_index = RelationHooked(index, {
        id_3(|index: &R, opaque: Opaque, tracking_freespace: bool| {
            let mut buffers = buffers.borrow_mut();
            let mut placement = placement.borrow_mut();
            let (want, source) = &mut *placement;
            *want = want.saturating_sub(1);
            let guard = loop {
                match source {
                    _ if tracking_freespace => break None,
                    Source::None => *source = choose(index, *want + 1),
                    Source::Runs(run) => match run.next() {
                        Some(id) => break Some(index.write(id, false)),
                        None => match pending.runs.borrow_mut().take_from(run.end, *want + 1) {
                            Some(run) => *source = Source::Runs(run),
                            None => *source = Source::None,
                        },
                    },
                    Source::Free(run) => match run.next() {
                        Some(id) => match freepages::alloc(index, freepages_first) {
                            Some(guard) => {
                                if guard.id() != id {
                                    *source = Source::None;
                                }
                                break Some(guard);
                            }
                            None => *source = Source::Extend,
                        },
                        None => *source = Source::None,
                    },
                    Source::Extend => break None,
                }
            };
            let guard = match guard {
                Some(mut guard) => {
                    buffers.number_of_formerly_allocated_pages += 1;
                    guard.clear(opaque);
                    guard
                }
                None => {
                    buffers.number_of_freshly_allocated_pages += 1;
                    index.extend(opaque, tracking_freespace)
                }
            };
            pending.list.borrow_mut().push(guard.id());
            guard
        })
    });

    let mut number_of_tuples = 0_u64;

    let mut words = Vec::new();
    for first in state {
        // lists with nothing to compact are left alone (#24)
        if let Some(tuples) = inspect_unchanged::<R>(index, first) {
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
        // the pages that the new list is placed for, as many as the old one took until its rows are counted
        let estimate = Cell::new(0);
        let mut tuples = 0_u64;
        let rows = RefCell::new(Rows::new(dim as _));
        let mut push = id_2(|row| {
            rows.borrow_mut().push(row, &keys);
            tuples += 1;
        });
        // the new list is begun at its first rows, once the old one's size is known
        let start = || {
            *placement.borrow_mut() = (estimate.get(), Source::None);
            ListWriter::create(&hooked_index, &keys)
        };
        // at each page the frozen tape moves on to, it is unlocked for `check`
        let freeze = |list: &mut Option<_>, branch, columns: &[u8]| {
            let list = list.get_or_insert_with(start);
            if list.push(branch, columns) {
                list.unlock();
                check();
            }
        };
        let mut list = None;
        let mut between = || {
            // a list larger than `memory` is written as sorted runs
            if rows.borrow().bytes() > memory {
                rows.borrow_mut()
                    .flush(&keys, |branch, columns| freeze(&mut list, branch, columns));
            }
            if let Some(list) = list.as_mut() {
                list.unlock();
            }
            check();
        };
        let pages = tape::read_directory_words::<R>(
            paced(tape::by_next(index, directory_first), &mut between)
                .inspect(|guard| trace.push(guard.id())),
            &mut words,
            true,
        );
        let directory_pages = trace.len();
        estimate.set(directory_pages + pages + 1);
        tape::read_frozen_tape::<R, _, _>(
            paced(
                tape::by_directory(&mut prefetch_h0_tuples, words[..pages].iter().copied()),
                &mut between,
            )
            .inspect(|guard| trace.push(guard.id())),
            tape::All,
            || {
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
            },
            (),
            &mut push,
        );
        let frozen_rows = rows.borrow().len();
        // the appendable tape in passes, each through the tail it starts with: into the sort, then copied
        let mut at = (appendable_first, 0);
        trace.push(appendable_first);
        let mut pass = read_appendable(index, &mut at, &mut trace, &mut between, &mut push, dim);

        // the new frozen tape holds the rows as densely as the old one did
        estimate.set(match frozen_rows {
            0 => trace.len() + 1,
            _ => (rows.borrow().len() * pages).div_ceil(frozen_rows) + 1 + directory_pages,
        });
        rows.borrow_mut()
            .flush(&keys, |branch, columns| freeze(&mut list, branch, columns));
        let written = list.unwrap_or_else(start).finish(&check);
        let (frozen_first, directory_first) = (written.frozen_first, written.directory_first);
        let mut appendable_tape = written.appendable;
        let copy = |tape: &mut tape::TapeWriter<_, _>| {
            rows.borrow_mut().flush(&keys, |branch, columns| {
                tape.push(appendable_tuple(branch, columns));
            });
        };
        // another pass only while they shrink: the lock waits for what is appended during the last
        loop {
            let n = read_appendable(
                index,
                &mut at,
                &mut trace,
                || {
                    copy(&mut appendable_tape);
                    appendable_tape.unlock();
                    check();
                },
                &mut push,
                dim,
            );
            if n >= pass {
                break;
            }
            pass = n;
        }
        copy(&mut appendable_tape);
        appendable_tape.unlock();

        // inserts append under the jump page's share lock, so none is under way
        let mut jump_guard = index.write(first, false);
        read_appendable(index, &mut at, &mut trace, || (), &mut push, dim);
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
        pending.list.borrow_mut().clear();
        let mut runs = pending.runs.borrow_mut();
        if let (_, Source::Runs(rest)) = placement.replace((0, Source::None)) {
            rest.for_each(|id| runs.insert(id));
        }
        trace.into_iter().for_each(|id| runs.insert(id));
    }
    let number_of_freed_pages = pending.free(index, freepages_first, &check);

    let buffers = RefCell::into_inner(buffers);

    Maintain {
        number_of_formerly_allocated_pages: buffers.number_of_formerly_allocated_pages,
        number_of_freshly_allocated_pages: buffers.number_of_freshly_allocated_pages,
        number_of_freed_pages,
        number_of_tuples,
    }
}

/// Reads the appendable tape on from `at`, a page and its tuples read, through the tail it starts with; returns its pages.
fn read_appendable<'b, R: RelationRead + 'b>(
    index: &'b R,
    at: &mut (u32, u16),
    trace: &mut Vec<u32>,
    mut between: impl FnMut(),
    callback: impl for<'a> FnMut(tape::Row<'a, (rabitq::bit::Code, f32)>),
    dim: u32,
) -> usize
where
    R::Page: Page<Opaque = Opaque>,
{
    let tail = tape::by_next(index, at.0).last().expect("a page").id();
    let (first, skip) = *at;
    let mut pages = 0;
    let mut next = Some(first);
    let guards = std::iter::from_fn(|| {
        let id = next?;
        if pages != 0 {
            between();
            trace.push(id);
        }
        let guard = index.read(id);
        next = (id != tail).then(|| guard.get_opaque().next);
        *at = (id, guard.len());
        pages += 1;
        Some(guard)
    });
    let access =
        |metadata, elements: &[u64], delta| (unpack_code(metadata, elements, dim as _), delta);
    tape::read_appendable_tape::<R, _>(guards, skip, access, (), callback);
    pages
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
    drop(jump_guard);
    Some(frozen + appendable)
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
    use crate::closure_lifetime_binder::id_0;
    use crate::keys::{Summary, Width};
    use crate::operator::Op;
    use crate::types::{DistanceKind, Structure, VchordrqIndexOptions, VectorKind, VectorOptions};
    use index::test_support::{MakePlainPrefetcher, MemoryRelation};
    use index_accessor::L2S;
    use std::cell::Cell;
    use std::num::NonZero;
    use std::panic::AssertUnwindSafe;
    use vector::vect::VectOwned;

    type M = MemoryRelation<Opaque>;
    type O = Op<VectOwned<f32>, L2S>;
    const DIM: usize = 100;

    struct NoRows;

    impl crate::Sorted for NoRows {
        fn next(&mut self) -> Option<&[u8]> {
            None
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

    /// One empty list as a build writes it, with rows 1 to `n` appended; returns its jump page.
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
        crate::write::<_, O>(relation, NoRows, usize::MAX);
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
        tape::read_frozen_tape::<M, _, _>(frozen, tape::All, accessor, (), |row| {
            payloads.push(row.payload.get())
        });
        let appendable =
            tape::by_next(relation, jump_tuple.appendable_first()).inspect(|_| pages += 1);
        tape::read_appendable_tape::<M, _>(
            appendable,
            0,
            |_, _, _| (),
            (),
            |row| payloads.push(row.payload.get()),
        );
        pages += free_list(relation).len() as u32;
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
                // a block's worth more, so that the list is compacted, not left alone
                for _ in 0..32 {
                    append(&relation, &keys, first, next.get());
                    next.set(next.get() + 1);
                }
                let (_, _, other) = list(&relation, first);
                let jump_guard = relation.read(first);
                old.set(JumpTuple::deserialize_ref(jump_guard.get(1).unwrap()).appendable_first());
                drop(jump_guard);
                let prefetch = MakePlainPrefetcher { index: &relation };
                let maintain =
                    maintain::<_, O>(&relation, prefetch, memory, &Pending::default(), check);
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
            maintain::<_, O>(&relation, prefetch, 1 << 12, &Pending::default(), || {
                calls.set(calls.get() + 1)
            });
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
                    maintain::<_, O>(&relation, prefetch, 1 << 12, &pending, check)
                }));
                assert_eq!(*result.err().unwrap().downcast::<u32>().unwrap(), k);
                pending.give_back(&relation);
                let (payloads, _, now) = list(&relation, first);
                assert_eq!(payloads, (1..=3001).collect::<Vec<_>>());
                assert_eq!(now, other, "a page is lost at delay point {k}");
            }
        }
    }

    /// Lists of `sizes` rows, appended one list after another; returns their jump pages.
    fn lists(relation: &M, keys: &Keys, sizes: &[u64]) -> Vec<u32> {
        let (dim, v, d) = (DIM as _, VectorKind::Vecf32, DistanceKind::L2S);
        let options = VchordrqIndexOptions {
            degree_of_parallelism: 1,
            ..Default::default()
        };
        let centroids = (0..sizes.len())
            .map(|i| VectOwned::new(vec![i as f32; DIM]))
            .collect::<Vec<_>>();
        let structures = vec![
            Structure {
                centroids,
                children: vec![Vec::new(); sizes.len()],
            },
            Structure {
                centroids: vec![VectOwned::new(vec![0.0; DIM])],
                children: vec![(0..sizes.len() as u32).collect()],
            },
        ];
        let vector_options = VectorOptions { dim, v, d };
        crate::build::<_, O>(vector_options, options, keys, relation, structures);
        crate::write::<_, O>(relation, NoRows, usize::MAX);
        let mut firsts = firsts(relation, || ());
        firsts.sort_unstable();
        let mut payload = 0;
        for (&first, &size) in std::iter::zip(&firsts, sizes) {
            for _ in 0..size {
                payload += 1;
                append(relation, keys, first, payload);
            }
        }
        firsts
    }

    /// A list's pages in the order the search reads them, but for its directory, then its directory.
    fn pages(relation: &M, first: u32) -> (Vec<u32>, Vec<u32>) {
        let jump_guard = relation.read(first);
        let jump_tuple = JumpTuple::deserialize_ref(jump_guard.get(1).unwrap());
        let mut words = Vec::new();
        let directory = tape::by_next(relation, jump_tuple.directory_first());
        let n = tape::read_directory_words::<M>(directory, &mut words, false);
        let appendable = tape::by_next(relation, jump_tuple.appendable_first());
        let tape = (words[..n].iter().copied()).chain(appendable.map(|g| g.id()));
        let directory = tape::by_next(relation, jump_tuple.directory_first()).map(|g| g.id());
        (tape.collect(), directory.collect())
    }

    /// The free pages: those the free list hands out, in order, then those of the kept runs.
    fn free_list(relation: &M) -> Vec<u32> {
        let guard = relation.read(meta(relation).1);
        let mut id = FreepagesTuple::deserialize_ref(guard.get(1).unwrap()).first();
        let kept = guard
            .get(2)
            .map(|x| FreeRunsTuple::deserialize_ref(x).runs().to_vec());
        drop(guard);
        let mut ids = Vec::new();
        while id != u32::MAX {
            ids.push(id);
            id = relation.read(id).get_opaque().next;
        }
        ids.extend(kept.unwrap_or_default().into_iter().flat_map(|[s, e]| s..e));
        ids
    }

    fn runs(pages: &[u32]) -> usize {
        1 + pages.windows(2).filter(|x| x[1] != x[0] + 1).count()
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn placed_in_one_run() {
        for keys in keys() {
            let relation = M::new([]);
            let firsts = lists(&relation, &keys, &[3000, 2600, 2200, 1800, 1400, 1000]);
            let prefetch = MakePlainPrefetcher { index: &relation };
            let pass = maintain::<_, O>(&relation, prefetch, 1 << 30, &Pending::default(), || ());
            let (first_tape, first_directory) = pages(&relation, firsts[0]);
            // only the first list is placed by extension; each other is placed in the run its predecessor left
            let first = first_tape.len() + first_directory.len();
            assert_eq!(pass.number_of_freshly_allocated_pages, first);
            for &list in &firsts {
                let (tape, directory) = pages(&relation, list);
                let run = [&tape[..], &directory[..]].concat();
                assert_eq!(runs(&run), 1, "{run:?}");
            }
            let guard = relation.read(meta(&relation).1);
            let kept = FreeRunsTuple::deserialize_ref(guard.get(2).unwrap())
                .runs()
                .to_vec();
            assert!(
                kept.windows(2)
                    .all(|x| x[0][1] - x[0][0] >= x[1][1] - x[1][0]),
                "the longest runs first: {kept:?}"
            );
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn reuses_freed_runs() {
        for keys in keys() {
            let relation = M::new([]);
            let firsts = lists(&relation, &keys, &[1000, 3000, 1500, 2500, 2000]);
            let (mut payload, mut frees) = (100_000, Vec::new());
            for _ in 0..6 {
                let prefetch = MakePlainPrefetcher { index: &relation };
                maintain::<_, O>(&relation, prefetch, 1 << 30, &Pending::default(), || ());
                let mut largest = 0;
                for &list in &firsts {
                    let (tape, directory) = pages(&relation, list);
                    let run = [&tape[..], &directory[..]].concat();
                    // where no free run fits, the longest takes at least half of the list
                    assert!(runs(&run) <= 2, "{run:?}");
                    largest = largest.max(run.len());
                    for _ in 0..200 {
                        payload += 1;
                        append(&relation, &keys, list, payload);
                    }
                }
                // a maintain extends the index only where no free run takes half of a list
                frees.push(free_list(&relation).len());
                assert!(
                    frees.iter().max() <= Some(&(frees[0] + largest)),
                    "{frees:?}"
                );
            }
        }
    }
}
