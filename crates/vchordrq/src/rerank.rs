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
use crate::{KHint, KKind, RerankMethod, vectors};
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
use vector::vect::MIN_FAST_X2;
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
    /// How many rows will be fetched, as the executor hints it.
    fn k(&self) -> KHint {
        KHint::UNKNOWN
    }
    fn prefetch(&mut self, payload: NonZero<u64>);
}

impl Table for () {
    fn window(&self) -> usize {
        0
    }

    fn prefetch(&mut self, _: NonZero<u64>) {}
}

/// A table with the executor's hint, chosen per scan so that a scan without one reads as before.
pub struct Hinted<T>(pub T, pub KHint);

impl<T: Table> Table for Hinted<T> {
    fn window(&self) -> usize {
        self.0.window()
    }

    fn k(&self) -> KHint {
        self.1
    }

    fn prefetch(&mut self, payload: NonZero<u64>) {
        self.0.prefetch(payload)
    }
}

/// Rows to read ahead after `n`: as many as `k` says are to come, else doubling.
#[inline]
fn ramp(n: usize, window: usize, k: KHint) -> usize {
    let doubling = 1 << (n / 2 + 1).ilog2();
    let left = usize::try_from(k.k).unwrap_or(usize::MAX).saturating_sub(n);
    match k.kind {
        KKind::Unknown => doubling,
        KKind::Exact => left,
        KKind::AtLeast => doubling.max(left),
        KKind::AtMost => doubling.min(left),
    }
    .min(window)
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
        // Candidates the rerank drops come on top of the rows, so `k` bounds them only from below.
        let k = self.table.k().and(KKind::AtLeast);
        while self.ahead.len() < ramp(self.taken, self.table.window(), k)
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
        ramp(self.returned, self.table.window(), self.table.k())
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
    let accessor = move || O::DistanceAccessor::default_with_dimension(dim);
    rerank_index_by::<O, T, P, W, _>(vector, prefetcher, table, accessor)
}

/// [`rerank_index`] with each row's distance from a new `accessor()`.
pub fn rerank_index_by<
    'b,
    O: Operator,
    T,
    P: Prefetcher<'b, Item = ((Reverse<Distance>, AlwaysEqual<T>), AlwaysEqual<W>)>,
    W: 'b + PackedRefMut<T = (NonZero<u64>, u16, BorrowedIter<'b>)>,
    A: Accessor2<
            <O::Vector as Vector>::Element,
            <O::Vector as Vector>::Element,
            <O::Vector as Vector>::Metadata,
            <O::Vector as Vector>::Metadata,
            Output = Distance,
        >,
>(
    vector: impl Borrow<O::Vector>,
    prefetcher: P,
    table: impl Table,
    accessor: impl Fn() -> A,
) -> Reranker<T, impl FnMut(NonZero<u64>, P::Guards, u16) -> Option<Distance>, P, W, impl Table> {
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
                LTryAccess::new(O::Vector::unpack(vector.borrow().as_borrowed()), accessor()),
            )
        }),
        _phantom: PhantomData,
    }
}

pub fn rerank_heap<
    'b,
    T,
    P: Prefetcher<'b, Item = ((Reverse<Distance>, AlwaysEqual<T>), AlwaysEqual<W>)>,
    W: 'b + PackedRefMut<T = (NonZero<u64>, u16, BorrowedIter<'b>)>,
>(
    prefetcher: P,
    mut distance: impl FnMut(NonZero<u64>) -> Option<Distance> + 'b,
) -> Reranker<T, impl FnMut(NonZero<u64>, P::Guards, u16) -> Option<Distance>, P, W, ()> {
    Reranker {
        prefetcher,
        cache: BinaryHeap::new(),
        ahead: VecDeque::new(),
        returned: 0,
        table: (),
        f: id_4::<_, P, _, _, _>(move |payload, _, _| distance(payload)),
        _phantom: PhantomData,
    }
}

/// The distance from `query` to `row`.
#[inline]
pub fn exact_distance<O: Operator>(
    query: <O::Vector as VectorOwned>::Borrowed<'_>,
    row: <O::Vector as VectorOwned>::Borrowed<'_>,
) -> Distance {
    let mut accessor = O::DistanceAccessor::default_with_dimension(query.dim());
    let (query, row) = (O::Vector::unpack(query), O::Vector::unpack(row));
    accessor.push(query.0, row.0);
    accessor.finish(query.1, row.1)
}

/// The cosine operator's distance from `query`, as given, to `row`, less one; `y2` is `|query|²`.
#[inline]
pub fn cosine_distance<O: Operator>(
    query: <O::Vector as VectorOwned>::Borrowed<'_>,
    row: <O::Vector as VectorOwned>::Borrowed<'_>,
    y2: f32,
) -> Distance {
    let x2 = O::Vector::squared_norm(row);
    let p = x2 * y2;
    if x2 >= MIN_FAST_X2 && p.is_normal() {
        Distance::from_f32(exact_distance::<O>(query, row).to_f32() / p.sqrt())
    } else {
        operator_cosine::<O>(query, row)
    }
}

/// [`cosine_distance`] off its fast path, kept out of line so that each scan inlines the rest.
#[cold]
#[inline(never)]
fn operator_cosine<O: Operator>(
    query: <O::Vector as VectorOwned>::Borrowed<'_>,
    row: <O::Vector as VectorOwned>::Borrowed<'_>,
) -> Distance {
    let (query, row) = (query.own(), row.own());
    Distance::from_f32(row.as_borrowed().operator_cos(query.as_borrowed()).to_f32() - 1.0)
}

/// The dot product distance from `query` to `row` scaled to unit length, as a cosine index stores it.
#[inline]
pub fn unit_distance<O: Operator>(
    query: <O::Vector as VectorOwned>::Borrowed<'_>,
    row: <O::Vector as VectorOwned>::Borrowed<'_>,
) -> Distance {
    let x2 = O::Vector::squared_norm(row);
    if (MIN_FAST_X2..f32::INFINITY).contains(&x2) {
        Distance::from_f32(exact_distance::<O>(query, row).to_f32() / x2.sqrt())
    } else {
        normalized_distance::<O>(query, row)
    }
}

/// [`unit_distance`] off its fast path, kept out of line so that each scan inlines the rest.
#[cold]
#[inline(never)]
fn normalized_distance<O: Operator>(
    query: <O::Vector as VectorOwned>::Borrowed<'_>,
    row: <O::Vector as VectorOwned>::Borrowed<'_>,
) -> Distance {
    exact_distance::<O>(query, row.function_normalize().as_borrowed())
}

#[cfg(test)]
mod tests {
    use super::*;
    use index::packed::PackedRefMut4;
    use index::prefetcher::PlainPrefetcher;
    use index::test_support::MemoryRelation;
    use index_accessor::Dot;
    use std::cell::RefCell;

    struct Recorder<'a>(usize, KHint, &'a RefCell<Vec<NonZero<u64>>>);

    impl Table for Recorder<'_> {
        fn window(&self) -> usize {
            self.0
        }

        fn k(&self) -> KHint {
            self.1
        }

        fn prefetch(&mut self, payload: NonZero<u64>) {
            self.2.borrow_mut().push(payload);
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

    type Row<'b> = (NonZero<u64>, u16, BorrowedIter<'b>);

    type Candidate<'b> = (
        (Reverse<Distance>, AlwaysEqual<()>),
        AlwaysEqual<PackedRefMut4<'b, Row<'b>>>,
    );

    fn rows<'b>() -> [Row<'b>; 6] {
        std::array::from_fn(|i| {
            let payload = NonZero::new(i as u64 + 1).unwrap();
            (payload, 0, BorrowedIter::from_slice(&[], |_| &[]))
        })
    }

    fn candidates<'b>(rows: &'b mut [Row<'b>; 6]) -> BinaryHeap<Candidate<'b>> {
        std::iter::zip(rows, ROWS)
            .map(|(w, (bound, _))| {
                (
                    (Reverse(Distance::from_f32(bound)), AlwaysEqual(())),
                    AlwaysEqual(PackedRefMut4(w)),
                )
            })
            .collect()
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
        let run = |table: Recorder, n: usize| {
            let mut rows = rows();
            let reranker = Reranker {
                prefetcher: PlainPrefetcher::new(&index, candidates(&mut rows)),
                cache: BinaryHeap::new(),
                ahead: VecDeque::new(),
                returned: 0,
                table,
                f: |payload, _, _| distance(payload),
                _phantom: PhantomData::<fn((), _) -> ((), _)>,
            };
            reranker
                .map(|(_, payload)| payload.get())
                .take(n)
                .collect::<Vec<_>>()
        };
        let unknown = KHint::UNKNOWN;
        let (none, one, four) = Default::default();
        let expected = run(Recorder(0, unknown, &none), 6);
        assert_eq!(expected, [2, 1, 4, 5, 6, 3]);
        assert_eq!(run(Recorder(1, unknown, &one), 6), expected);
        assert_eq!(run(Recorder(4, unknown, &four), 6), expected);
        assert!(read(&none).is_empty());
        // the window ramps from one row, so the first rows are read as with a window of one
        assert_eq!(read(&one), [1, 2, 1, 3, 4, 3, 5, 3, 6, 3]);
        assert_eq!(read(&four), [1, 2, 1, 3, 4, 5, 6]);
        let log = RefCell::default();
        let reads = |k, kind, n| {
            assert_eq!(run(Recorder(4, KHint { k, kind }, &log), n), expected[..n]);
            read(&log)
        };
        assert_eq!(reads(0, KKind::Unknown, 3), [1, 2, 1, 3, 4]);
        assert_eq!(reads(3, KKind::Exact, 3), [1, 2, 3, 4]);
        assert_eq!(reads(3, KKind::AtLeast, 6), [1, 2, 3, 4, 5, 6]);
        assert_eq!(reads(1, KKind::AtMost, 1), [1, 2]);
    }

    #[test]
    fn table_rerank_reads_candidates_ahead() {
        let index = MemoryRelation::<crate::Opaque>::new([]);
        // requests as payloads, table reads as payloads + 100
        let run = |window: usize, k: KHint, log: &RefCell<Vec<NonZero<u64>>>| {
            let (table, mut rows) = (Recorder(window, k, log), rows());
            let candidates = TableAhead::new(candidates(&mut rows), table);
            let prefetcher = PlainPrefetcher::new(&index, candidates);
            rerank_heap(prefetcher, |payload: NonZero<u64>| {
                log.borrow_mut().push(payload.checked_add(100).unwrap());
                distance(payload)
            })
            .map(|(_, payload)| payload.get())
            .collect::<Vec<_>>()
        };
        let unknown = KHint::UNKNOWN;
        let (none, one, four) = Default::default();
        assert_eq!(run(0, unknown, &none), [2, 1, 4, 5, 6, 3]);
        assert_eq!(run(1, unknown, &one), [2, 1, 4, 5, 6, 3]);
        assert_eq!(run(4, unknown, &four), [2, 1, 4, 5, 6, 3]);
        assert_eq!(read(&none), [101, 102, 103, 104, 105, 106]);
        assert_eq!(read(&one), [2, 101, 3, 102, 4, 103, 5, 104, 6, 105, 106]);
        assert_eq!(read(&four), [2, 101, 3, 4, 102, 5, 103, 6, 104, 105, 106]);
        let log = RefCell::default();
        let reads = |k, kind| {
            assert_eq!(run(4, KHint { k, kind }, &log), [2, 1, 4, 5, 6, 3]);
            read(&log)
        };
        assert_eq!(
            reads(4, KKind::Exact),
            [2, 3, 4, 101, 102, 5, 103, 6, 104, 105, 106]
        );
        assert_eq!(
            reads(4, KKind::AtMost),
            [2, 101, 3, 4, 102, 5, 103, 6, 104, 105, 106]
        );
    }

    fn floats(seed: u64, scale: f32) -> Vec<f32> {
        let mut state = seed;
        (0..300)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 40) as f32 / (1 << 24) as f32 - 0.5) * scale
            })
            .collect()
    }

    fn unit(x: Vec<f32>) -> Vec<f32> {
        let norm = x.iter().map(|x| x * x).sum::<f32>().sqrt();
        x.into_iter().map(|x| x / norm).collect()
    }

    /// `unit_distance` against the distance to a normalised copy of the row, within the copy's rounding.
    fn compare<V: Vector>(make: impl Fn(&[f32]) -> V, tolerance: f32)
    where
        Op<V, Dot>: Operator<Vector = V>,
    {
        let (query, row) = (make(&unit(floats(0, 1.0))), make(&floats(99, 3.0)));
        let (query, row) = (query.as_borrowed(), row.as_borrowed());
        let scaled = unit_distance::<Op<V, Dot>>(query, row).to_f32();
        let copied =
            exact_distance::<Op<V, Dot>>(query, row.function_normalize().as_borrowed()).to_f32();
        assert!((scaled - copied).abs() <= tolerance, "{scaled} {copied}");
    }

    #[test]
    fn unit_distance_scales_by_the_norm() {
        use simd::{F16, Floating, f16};
        use vector::rabitq4::Rabitq4Owned;
        use vector::rabitq8::Rabitq8Owned;
        use vector::vect::VectOwned;
        compare(|x| VectOwned::new(x.to_vec()), 1e-6);
        compare(
            |x| VectOwned::new(x.iter().map(|&x| f16::_from_f32(x)).collect()),
            2e-3,
        );
        compare(
            |x| {
                let (m, code) = rabitq::byte::ugly_code(x);
                let code = rabitq::byte::pack_code(&code);
                let abs = f32::reduce_sum_of_abs_x(x);
                Rabitq8Owned::new(
                    x.len() as u32,
                    m.dis_u_2,
                    m.norm_of_lattice,
                    m.sum_of_code,
                    abs,
                    code,
                )
            },
            1e-4,
        );
        compare(
            |x| {
                let (m, code) = rabitq::halfbyte::ugly_code(x);
                let code = rabitq::halfbyte::pack_code(&code);
                let abs = f32::reduce_sum_of_abs_x(x);
                Rabitq4Owned::new(
                    x.len() as u32,
                    m.dis_u_2,
                    m.norm_of_lattice,
                    m.sum_of_code,
                    abs,
                    code,
                )
            },
            1e-4,
        );
        let query = VectOwned::new(unit(vec![1.0; 300]));
        let row = VectOwned::new(vec![1e-22f32; 300]);
        let d = unit_distance::<Op<VectOwned<f32>, Dot>>(query.as_borrowed(), row.as_borrowed());
        assert!((d.to_f32() + 1.0).abs() <= 1e-6, "{}", d.to_f32());
    }
}
