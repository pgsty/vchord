// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use crate::closure_lifetime_binder::id_4;
use crate::operator::*;
use crate::tuples::{MetaTuple, WithReader};
use crate::{RerankMethod, vectors};
use always_equal::AlwaysEqual;
use distance::Distance;
use index::fetch::BorrowedIter;
use index::packed::PackedRefMut;
use index::prefetcher::{Prefetcher, Sequence};
use index::relation::{Page, RelationRead};
use index_accessor::{Accessor2, DefaultWithDimension, LTryAccess};
use std::borrow::Borrow;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque, vec_deque};
use std::marker::PhantomData;
use std::num::NonZero;
use vector::{VectorBorrowed, VectorOwned};

type Result = (Reverse<Distance>, AlwaysEqual<NonZero<u64>>);

pub fn how(index: &impl RelationRead) -> RerankMethod {
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
    let rerank_in_heap = meta_tuple.rerank_in_heap();
    if rerank_in_heap {
        RerankMethod::Heap
    } else {
        RerankMethod::Index
    }
}

/// The table that returned rows are fetched from, read ahead of them.
pub trait Table {
    /// The most rows to read ahead; zero reads none.
    fn window(&self) -> usize;
    fn prefetch(&mut self, payload: NonZero<u64>);
}

impl Table for () {
    fn window(&self) -> usize {
        0
    }

    fn prefetch(&mut self, _: NonZero<u64>) {}
}

/// Rows to read ahead after `n`, doubling so that a scan ended early reads few.
#[inline]
fn ramp(n: usize, window: usize) -> usize {
    (1 << (n / 2 + 1).ilog2()).min(window)
}

/// Candidates whose table pages are read ahead of the table rerank, in the order it takes them.
pub struct TableAhead<S: Sequence, H> {
    sequence: S,
    ahead: VecDeque<S::Item>,
    taken: usize,
    table: H,
}

impl<S: Sequence, H> TableAhead<S, H> {
    pub fn new(sequence: S, table: H) -> Self {
        Self {
            sequence,
            ahead: VecDeque::new(),
            taken: 0,
            table,
        }
    }
}

impl<'b, S, T, W, H> TableAhead<S, H>
where
    S: Sequence<Item = ((Reverse<Distance>, AlwaysEqual<T>), AlwaysEqual<W>)>,
    W: PackedRefMut<T = (NonZero<u64>, u16, BorrowedIter<'b>)>,
    H: Table,
{
    #[inline]
    fn fill(&mut self) {
        while self.ahead.len() < ramp(self.taken, self.table.window())
            && let Some(candidate) = self.sequence.next()
        {
            self.table.prefetch(candidate.1.0.get().0);
            self.ahead.push_back(candidate);
        }
    }
}

impl<'b, S, T, W, H> Sequence for TableAhead<S, H>
where
    S: Sequence<Item = ((Reverse<Distance>, AlwaysEqual<T>), AlwaysEqual<W>)>,
    W: PackedRefMut<T = (NonZero<u64>, u16, BorrowedIter<'b>)>,
    H: Table,
{
    type Item = S::Item;
    type Inner = std::iter::Chain<vec_deque::IntoIter<S::Item>, S::Inner>;

    #[inline]
    fn next(&mut self) -> Option<S::Item> {
        let candidate = self.ahead.pop_front().or_else(|| self.sequence.next())?;
        self.taken += 1;
        self.fill();
        Some(candidate)
    }

    #[inline]
    fn peek(&mut self) -> Option<&S::Item> {
        if self.ahead.is_empty() {
            self.sequence.peek()
        } else {
            self.ahead.front()
        }
    }

    fn into_inner(self) -> Self::Inner {
        self.ahead.into_iter().chain(self.sequence.into_inner())
    }

    fn refill(&mut self) -> bool {
        self.sequence.refill()
    }
}

pub struct Reranker<T, F, P, W, H> {
    prefetcher: P,
    cache: BinaryHeap<Result>,
    /// The smallest cached rows, in order, with their table pages read ahead.
    ahead: VecDeque<Result>,
    returned: usize,
    f: F,
    table: H,
    _phantom: PhantomData<fn(T, W) -> (T, W)>,
}

impl<'b, T, F, P, W, H> Iterator for Reranker<T, F, P, W, H>
where
    F: FnMut(NonZero<u64>, P::Guards, u16) -> Option<Distance>,
    P: Prefetcher<'b, Item = ((Reverse<Distance>, AlwaysEqual<T>), AlwaysEqual<W>)>,
    W: 'b + PackedRefMut<T = (NonZero<u64>, u16, BorrowedIter<'b>)>,
    H: Table,
{
    type Item = (Distance, NonZero<u64>);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // The previous candidate's guards are released here (#28).
            index::interrupt::check();
            let top = self.ahead.front().or(self.cache.peek()).map(|(d, ..)| *d);
            let Some(((_, AlwaysEqual(mut w)), prefetch)) =
                self.prefetcher.next_if(|((d, _), ..)| Some(*d) > top)
            else {
                // With nothing cached, every candidate was returned: the prefetcher may read more.
                if top.is_none() && self.prefetcher.refill() {
                    continue;
                }
                break;
            };
            let &mut (payload, head, ..) = w.get_mut();
            if let Some(distance) = (self.f)(payload, prefetch, head) {
                self.push((Reverse(distance), AlwaysEqual(payload)));
            };
        }
        let row = self.ahead.pop_front().or_else(|| self.cache.pop())?;
        self.returned += 1;
        while self.ahead.len() < self.window()
            && let Some(row) = self.cache.pop()
        {
            self.table.prefetch(row.1.0);
            self.ahead.push_back(row);
        }
        let (Reverse(distance), AlwaysEqual(payload)) = row;
        Some((distance, payload))
    }
}

impl<T, F, P, W, H: Table> Reranker<T, F, P, W, H> {
    fn window(&self) -> usize {
        ramp(self.returned, self.table.window())
    }

    fn push(&mut self, row: Result) {
        let i = self.ahead.partition_point(|x| x.0 >= row.0);
        if i < self.window() {
            self.table.prefetch(row.1.0);
            self.ahead.insert(i, row);
            if self.ahead.len() > self.window() {
                self.cache.extend(self.ahead.pop_back());
            }
        } else {
            self.cache.push(row);
        }
    }
}

impl<T, F, P, W, H> Reranker<T, F, P, W, H> {
    pub fn finish(self) -> (P, impl Iterator<Item = Result>) {
        (self.prefetcher, self.ahead.into_iter().chain(self.cache))
    }
}

pub fn rerank_index<
    'b,
    O: Operator,
    T,
    P: Prefetcher<'b, Item = ((Reverse<Distance>, AlwaysEqual<T>), AlwaysEqual<W>)>,
    W: 'b + PackedRefMut<T = (NonZero<u64>, u16, BorrowedIter<'b>)>,
>(
    vector: impl Borrow<O::Vector>,
    prefetcher: P,
    table: impl Table,
) -> Reranker<T, impl FnMut(NonZero<u64>, P::Guards, u16) -> Option<Distance>, P, W, impl Table> {
    let dim = vector.borrow().as_borrowed().dim();
    Reranker {
        prefetcher,
        cache: BinaryHeap::new(),
        ahead: VecDeque::new(),
        returned: 0,
        table,
        f: id_4::<_, P, _, _, _>(move |payload, prefetch, head| {
            vectors::read::<P::R, O, _>(
                prefetch,
                head,
                payload,
                LTryAccess::new(
                    O::Vector::unpack(vector.borrow().as_borrowed()),
                    O::DistanceAccessor::default_with_dimension(dim),
                ),
            )
        }),
        _phantom: PhantomData,
    }
}

pub fn rerank_heap<
    'b,
    O: Operator,
    T,
    P: Prefetcher<'b, Item = ((Reverse<Distance>, AlwaysEqual<T>), AlwaysEqual<W>)>,
    W: 'b + PackedRefMut<T = (NonZero<u64>, u16, BorrowedIter<'b>)>,
>(
    vector: impl Borrow<O::Vector>,
    prefetcher: P,
    mut fetch: impl FnMut(NonZero<u64>) -> Option<O::Vector> + 'b,
) -> Reranker<T, impl FnMut(NonZero<u64>, P::Guards, u16) -> Option<Distance>, P, W, ()> {
    let dim = vector.borrow().as_borrowed().dim();
    Reranker {
        prefetcher,
        cache: BinaryHeap::new(),
        ahead: VecDeque::new(),
        returned: 0,
        table: (),
        f: id_4::<_, P, _, _, _>(move |payload, _, _| {
            let unpack = O::Vector::unpack(vector.borrow().as_borrowed());
            let vector = fetch(payload)?;
            let vector = O::Vector::unpack(vector.as_borrowed());
            let mut accessor = O::DistanceAccessor::default_with_dimension(dim);
            accessor.push(unpack.0, vector.0);
            let distance = accessor.finish(unpack.1, vector.1);
            Some(distance)
        }),
        _phantom: PhantomData,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use index::packed::PackedRefMut4;
    use index::prefetcher::PlainPrefetcher;
    use index::test_support::MemoryRelation;
    use std::cell::RefCell;

    struct Recorder<'a>(usize, &'a RefCell<Vec<NonZero<u64>>>);

    impl Table for Recorder<'_> {
        fn window(&self) -> usize {
            self.0
        }

        fn prefetch(&mut self, payload: NonZero<u64>) {
            self.1.borrow_mut().push(payload);
        }
    }

    // (lower bound, exact distance); the fourth row's distance is below its bound
    const ROWS: [(f32, f32); 6] = [
        (0.1, 0.3),
        (0.2, 0.25),
        (0.3, 0.9),
        (0.35, 0.33),
        (0.4, 0.5),
        (0.6, 0.7),
    ];

    type Candidate<'b> = (
        (Reverse<Distance>, AlwaysEqual<()>),
        AlwaysEqual<PackedRefMut4<'b, (NonZero<u64>, u16, BorrowedIter<'b>)>>,
    );

    fn candidates<'b>() -> BinaryHeap<Candidate<'b>> {
        let candidate = |(i, &(bound, _)): (usize, &(f32, f32))| -> Candidate<'b> {
            let payload = NonZero::new(i as u64 + 1).unwrap();
            let w = Box::leak(Box::new((
                payload,
                0,
                BorrowedIter::from_slice(&[], |_| &[]),
            )));
            (
                (Reverse(Distance::from_f32(bound)), AlwaysEqual(())),
                AlwaysEqual(PackedRefMut4(w)),
            )
        };
        ROWS.iter().enumerate().map(candidate).collect()
    }

    fn distance(payload: NonZero<u64>) -> Option<Distance> {
        Some(Distance::from_f32(ROWS[payload.get() as usize - 1].1))
    }

    fn read(x: &RefCell<Vec<NonZero<u64>>>) -> Vec<u64> {
        x.take().into_iter().map(NonZero::get).collect()
    }

    #[test]
    fn reading_ahead_keeps_the_order() {
        let index = MemoryRelation::<crate::Opaque>::new([]);
        let run = |table: Recorder| {
            let reranker = Reranker {
                prefetcher: PlainPrefetcher::new(&index, candidates()),
                cache: BinaryHeap::new(),
                ahead: VecDeque::new(),
                returned: 0,
                table,
                f: |payload, _, _| distance(payload),
                _phantom: PhantomData::<fn((), _) -> ((), _)>,
            };
            reranker
                .map(|(_, payload)| payload.get())
                .collect::<Vec<_>>()
        };
        let (none, one, four) = Default::default();
        let expected = run(Recorder(0, &none));
        assert_eq!(expected, [2, 1, 4, 5, 6, 3]);
        assert_eq!(run(Recorder(1, &one)), expected);
        assert_eq!(run(Recorder(4, &four)), expected);
        assert!(read(&none).is_empty());
        // the window ramps from one row, so the first rows are read as with a window of one
        assert_eq!(read(&one), [1, 2, 1, 3, 4, 3, 5, 3, 6, 3]);
        assert_eq!(read(&four), [1, 2, 1, 3, 4, 5, 6]);
    }

    #[test]
    fn table_rerank_reads_candidates_ahead() {
        let index = MemoryRelation::<crate::Opaque>::new([]);
        // requests as payloads, table reads as payloads + 100
        let run = |window: usize, log: &RefCell<Vec<NonZero<u64>>>| {
            let reranker = Reranker {
                prefetcher: PlainPrefetcher::new(
                    &index,
                    TableAhead::new(candidates(), Recorder(window, log)),
                ),
                cache: BinaryHeap::new(),
                ahead: VecDeque::new(),
                returned: 0,
                table: (),
                f: |payload: NonZero<u64>, _, _| {
                    log.borrow_mut().push(payload.checked_add(100).unwrap());
                    distance(payload)
                },
                _phantom: PhantomData::<fn((), _) -> ((), _)>,
            };
            reranker
                .map(|(_, payload)| payload.get())
                .collect::<Vec<_>>()
        };
        let (none, one, four) = Default::default();
        assert_eq!(run(0, &none), [2, 1, 4, 5, 6, 3]);
        assert_eq!(run(1, &one), [2, 1, 4, 5, 6, 3]);
        assert_eq!(run(4, &four), [2, 1, 4, 5, 6, 3]);
        assert_eq!(read(&none), [101, 102, 103, 104, 105, 106]);
        assert_eq!(read(&one), [2, 101, 3, 102, 4, 103, 5, 104, 6, 105, 106]);
        assert_eq!(read(&four), [2, 101, 3, 4, 102, 5, 103, 6, 104, 105, 106]);
    }
}
