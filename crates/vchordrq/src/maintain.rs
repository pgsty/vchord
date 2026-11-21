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
use std::cell::RefCell;

pub struct Maintain {
    pub number_of_formerly_allocated_pages: usize,
    pub number_of_freshly_allocated_pages: usize,
    pub number_of_freed_pages: usize,
    /// Live index tuples in the lists that were chosen and compacted.
    pub number_of_tuples: u64,
}

/// The pages a maintain holds unpublished or not yet freed: a list's new ones, then its old ones.
#[derive(Default)]
pub struct Pending(RefCell<Vec<u32>>);

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
    fn free<R: RelationWrite>(
        &self,
        index: &R,
        freepages_first: u32,
        mut between: impl FnMut(),
    ) -> usize
    where
        R::Page: Page<Opaque = Opaque>,
    {
        let mut n = 0;
        loop {
            let Some(id) = self.0.borrow_mut().pop() else {
                return n;
            };
            freepages::free(index, freepages_first, id);
            n += 1;
            between();
        }
    }
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

    let hooked_index = RelationHooked(index, {
        id_3(|index: &R, opaque: Opaque, tracking_freespace: bool| {
            let mut buffers = buffers.borrow_mut();
            let guard = if !tracking_freespace
                && let Some(mut guard) = freepages::alloc(index, freepages_first)
            {
                buffers.number_of_formerly_allocated_pages += 1;
                guard.clear(opaque);
                guard
            } else {
                buffers.number_of_freshly_allocated_pages += 1;
                index.extend(opaque, tracking_freespace)
            };
            pending.0.borrow_mut().push(guard.id());
            guard
        })
    });

    let mut number_of_tuples = 0_u64;
    let mut number_of_freed_pages = 0_usize;

    let mut words = Vec::new();
    for first in state {
        // lists with nothing to compact are left alone: VACUUM used to rewrite and WAL-log every list (#24)
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
        let mut tuples = 0_u64;
        let rows = RefCell::new(Rows::new(dim as _));
        let mut push = id_2(|row| {
            rows.borrow_mut().push(row, &keys);
            tuples += 1;
        });
        // at each page the frozen tape moves on to, it is unlocked for `check`
        let freeze = |list: &mut ListWriter<'_, _>, branch, columns: &[u8]| {
            if list.push(branch, columns) {
                list.unlock();
                check();
            }
        };
        let mut list = ListWriter::create(&hooked_index, &keys);
        let mut between = || {
            // a list larger than `memory` is written as sorted runs
            if rows.borrow().bytes() > memory {
                rows.borrow_mut()
                    .flush(&keys, |branch, columns| freeze(&mut list, branch, columns));
            }
            list.unlock();
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
        // the appendable tape in passes, each through the tail it starts with: into the sort, then copied
        let mut at = (appendable_first, 0);
        trace.push(appendable_first);
        let mut pass = read_appendable(index, &mut at, &mut trace, &mut between, &mut push, dim);

        rows.borrow_mut()
            .flush(&keys, |branch, columns| freeze(&mut list, branch, columns));
        let written = list.finish(&check);
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
        *pending.0.borrow_mut() = trace;
        number_of_freed_pages += pending.free(index, freepages_first, &check);
    }

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
}
