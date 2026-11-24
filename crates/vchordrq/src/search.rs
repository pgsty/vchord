// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use crate::closure_lifetime_binder::{id_0, id_1, id_2};
use crate::keys::KeyFilter;
use crate::linked_vec::LinkedVec;
use crate::operator::*;
use crate::tape::{All, Filter, Row, Selected, by_next};
use crate::tuples::*;
use crate::{Opaque, centroids, tape};
use always_equal::AlwaysEqual;
use dary_heap::QuaternaryHeap as Heap;
use distance::Distance;
use index::bump::Bump;
use index::fetch::{BorrowedIter, Fetch};
use index::packed::{PackedRefMut4, PackedRefMut8};
use index::prefetcher::{Prefetcher, PrefetcherHeapFamily, PrefetcherSequenceFamily, Sequence};
use index::relation::{Page, Relation, RelationRead, RelationReadTypes};
use index_accessor::{DefaultWithDimension, FunctionalAccessor, LAccess};
use rabitq::bit::binary::BinaryLut;
use rabitq::bit::block::BlockLut;
use std::borrow::Borrow;
use std::cell::{Cell, RefCell};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet, VecDeque};
use std::marker::PhantomData;
use std::num::NonZero;
use std::rc::Rc;
use vector::{VectorBorrowed, VectorOwned};

type Extra1<'b> = &'b mut (u32, f32, u16, BorrowedIter<'b>);

type List = (Reverse<Distance>, AlwaysEqual<f32>, AlwaysEqual<u32>);

pub type Candidate<'b> = (
    (Reverse<Distance>, AlwaysEqual<()>),
    AlwaysEqual<PackedRefMut4<'b, (NonZero<u64>, u16, BorrowedIter<'b>)>>,
);

pub type MaxsimCandidate<'b> = (
    (Reverse<Distance>, AlwaysEqual<Distance>),
    AlwaysEqual<PackedRefMut8<'b, (NonZero<u64>, u16, BorrowedIter<'b>)>>,
);

type Refill<'b> = Box<dyn FnMut() -> Option<Heap<Candidate<'b>>> + 'b>;

#[derive(Debug)]
pub struct SearchParams {
    pub probes: Vec<u32>,
    pub epsilon: f32,
    pub filter: Option<KeyFilter>,
    /// The most lists a search with a filter reads ([`default_search`]).
    pub max_probes: Option<u32>,
}

struct Searcher<'b, R, O, V, B, H1, H0> {
    index: &'b R,
    bump: &'b B,
    vector: V,
    dim: u32,
    is_residual: bool,
    /// Bottom first.
    cells: Vec<u32>,
    root: List,
    lut: (BlockLut, BinaryLut),
    epsilon: f32,
    prefetch_h1_vectors: H1,
    prefetch_h0_tuples: H0,
    /// Scratch space for directories, kept across lists.
    words: Vec<u32>,
    _operator: PhantomData<fn() -> O>,
}

struct Route<I> {
    lists: Vec<List>,
    complete: bool,
    /// The other lists of the bottom level, nearest first, if `sorted`.
    rest: I,
}

impl<'b, R, O, V, B, H1, H0> Searcher<'b, R, O, V, B, H1, H0>
where
    R: RelationRead,
    R::Page: Page<Opaque = Opaque>,
    O: Operator,
    V: Borrow<O::Vector> + Clone,
    B: Bump,
    H1: PrefetcherHeapFamily<'b, R>,
    H0: PrefetcherSequenceFamily<'b, R>,
{
    fn new(
        index: &'b R,
        query: V,
        params: &SearchParams,
        bump: &'b B,
        prefetch_h1_vectors: H1,
        prefetch_h0_tuples: H0,
    ) -> Self {
        let vector = query.borrow().as_borrowed();
        let meta_guard = index.read(0);
        let meta_bytes = meta_guard.get(1).expect("data corruption");
        let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
        let dim = meta_tuple.dim();
        let is_residual = meta_tuple.is_residual();
        let height_of_root = meta_tuple.height_of_root();
        let cells = meta_tuple.cells().to_vec();
        assert_eq!(dim, vector.dim(), "unmatched dimensions");
        if height_of_root as usize != 1 + params.probes.len() {
            panic!(
                "usage: need {} probes, but {} probes provided",
                height_of_root - 1,
                params.probes.len()
            );
        }
        debug_assert_eq!(cells[(height_of_root - 1) as usize], 1);
        let norm = meta_tuple.centroid_norm();
        let first = meta_tuple.first();
        let root = if is_residual {
            let prefetch =
                BorrowedIter::from_slice(meta_tuple.centroid_prefetch(), |x| bump.alloc_slice(x));
            let head = meta_tuple.centroid_head();
            let distance = centroids::read::<R, O, _>(
                prefetch.map(|id| index.read(id)),
                head,
                LAccess::new(
                    O::Vector::unpack(vector),
                    O::DistanceAccessor::default_with_dimension(dim),
                ),
            );
            (Reverse(distance), AlwaysEqual(norm), AlwaysEqual(first))
        } else {
            // fast path
            (
                Reverse(Distance::ZERO),
                AlwaysEqual(norm),
                AlwaysEqual(first),
            )
        };
        drop(meta_guard);
        let lut = O::Vector::preprocess(vector);
        Self {
            index,
            bump,
            vector: query,
            dim,
            is_residual,
            cells,
            root,
            lut,
            epsilon: params.epsilon,
            prefetch_h1_vectors,
            prefetch_h0_tuples,
            words: Vec::new(),
            _operator: PhantomData,
        }
    }

    fn route(
        &mut self,
        probes: &[u32],
        sorted: bool,
    ) -> Route<impl Iterator<Item = List> + use<'b, R, O, V, B, H1, H0>> {
        let Self {
            ref vector,
            index,
            bump,
            dim,
            is_residual,
            ref cells,
            root,
            ref lut,
            epsilon,
            ref mut prefetch_h1_vectors,
            ..
        } = *self;
        let mut step = |lists: Vec<List>| {
            let mut results = LinkedVec::<(_, AlwaysEqual<Extra1<'b>>)>::new();
            for (Reverse(dis_f), AlwaysEqual(norm), AlwaysEqual(first)) in lists {
                let residual =
                    is_residual.then(|| O::routing_residual(&lut.0, dis_f.to_f32(), norm));
                tape::read_h1_tape::<R, _, _>(
                    by_next(index, first),
                    || O::block_access(&lut.0, residual),
                    |(rough, err), head, norm, first, prefetch| {
                        let lowerbound = Distance::from_f32(rough - err * epsilon);
                        results.push((
                            Reverse(lowerbound),
                            AlwaysEqual(bump.alloc((
                                first,
                                norm,
                                head,
                                BorrowedIter::from_slice(prefetch, |x| bump.alloc_slice(x)),
                            ))),
                        ));
                    },
                );
            }
            let mut heap = prefetch_h1_vectors.prefetch(results.into_vec());
            let mut cache = BinaryHeap::<(_, _, _)>::new();
            let vector = vector.clone();
            std::iter::from_fn(move || {
                while let Some((
                    (Reverse(_), AlwaysEqual(&mut (first, norm, head, ..))),
                    prefetch,
                )) = heap.next_if(|(d, _)| Some(*d) > cache.peek().map(|(d, ..)| *d))
                {
                    let distance = centroids::read::<R, O, _>(
                        prefetch,
                        head,
                        LAccess::new(
                            O::Vector::unpack(vector.borrow().as_borrowed()),
                            O::DistanceAccessor::default_with_dimension(dim),
                        ),
                    );
                    cache.push((Reverse(distance), AlwaysEqual(norm), AlwaysEqual(first)));
                }
                cache.pop()
            })
        };
        let levels = probes.len();
        let mut lists = vec![root];
        let mut rest = None;
        let mut complete = true;
        for (i, (&probes, &cells)) in
            std::iter::zip(probes, cells[..levels].iter().rev()).enumerate()
        {
            let partial_scan = probes < cells;
            complete &= !partial_scan;
            if partial_scan || is_residual || (sorted && i + 1 == levels) {
                lists = rest.insert(step(lists)).take(probes as _).collect();
            } else {
                // fast path
                let mut results = LinkedVec::new();
                for (Reverse(_), AlwaysEqual(_), AlwaysEqual(first)) in lists {
                    tape::read_h1_tape::<R, _, _>(
                        by_next(index, first),
                        || FunctionalAccessor::new((), id_0(|_, _| ()), id_1(|_, _| [(); _])),
                        |(), _, norm, first, _| {
                            results.push((
                                Reverse(Distance::ZERO),
                                AlwaysEqual(norm),
                                AlwaysEqual(first),
                            ));
                        },
                    );
                }
                lists = results.into_vec();
            }
        }
        Route {
            lists,
            complete,
            rest: rest.into_iter().flatten(),
        }
    }

    /// Returns how many rows the lists hold.
    // Inlined with its per-row callback; tapes take it by `&mut` so `#[inline(always)]` holds.
    #[inline(always)]
    fn read_lists(
        &mut self,
        lists: Vec<List>,
        filter: impl Filter,
        mut callback: impl for<'a> FnMut(Row<'a, (f32, f32)>),
    ) -> u64 {
        let Self {
            index,
            is_residual,
            ref lut,
            ref mut prefetch_h0_tuples,
            ref mut words,
            ..
        } = *self;
        let is_not_plain = prefetch_h0_tuples.is_not_plain();
        let firsts = if is_not_plain {
            lists.iter().map(|list| list.2.0).collect()
        } else {
            Vec::new()
        };
        let ahead = Ahead {
            index,
            pages: RefCell::new(schedule(index, prefetch_h0_tuples, firsts, filter)),
            list: Cell::new(0),
        };
        let mut tuples = 0;
        for (i, (Reverse(dis_f), AlwaysEqual(norm), AlwaysEqual(first))) in
            lists.into_iter().enumerate()
        {
            // Between lists nothing is locked (#28).
            index::interrupt::check();
            ahead.begin(i);
            let residual = is_residual.then(|| O::residual(&lut.0, dis_f.to_f32(), norm));
            let jump_guard = ahead.read(first);
            let jump_bytes = jump_guard.get(1).expect("data corruption");
            let jump_tuple = JumpTuple::deserialize_ref(jump_bytes);
            let block_access = || O::block_access(&lut.0, residual);
            let directory = (filter.selects_blocks() || is_not_plain).then(|| {
                let directory = by_next(&ahead, jump_tuple.directory_first());
                tape::read_directory::<R>(directory, words, filter)
            });
            match directory {
                Some((pages, Some(selected))) => {
                    let (ids, layout) = words.split_at(pages);
                    let (firsts, masks) = layout.split_at(pages);
                    if selected != 0 {
                        tape::read_frozen_tape::<R, _, _>(
                            ids[..selected].iter().map(|&id| ahead.read(id)),
                            Selected::new(&firsts[..selected], masks),
                            block_access,
                            filter,
                            &mut callback,
                        );
                    }
                }
                Some((pages, None)) if is_not_plain => {
                    tape::read_frozen_tape::<R, _, _>(
                        words[..pages].iter().map(|&id| ahead.read(id)),
                        All,
                        block_access,
                        filter,
                        &mut callback,
                    );
                }
                _ => {
                    tape::read_frozen_tape::<R, _, _>(
                        by_next(&ahead, jump_tuple.frozen_first()),
                        All,
                        block_access,
                        filter,
                        &mut callback,
                    );
                }
            }
            tape::read_appendable_tape::<R, _>(
                by_next(&ahead, jump_tuple.appendable_first()),
                0,
                O::binary_access(&lut.1, residual),
                filter,
                &mut callback,
            );
            tuples += jump_tuple.tuples();
        }
        tuples
    }

    fn read(
        &mut self,
        lists: Vec<List>,
        filter: impl Filter,
        bound: impl Bound,
    ) -> Heap<Candidate<'b>> {
        let (bump, epsilon) = (self.bump, self.epsilon);
        let mut results = LinkedVec::new();
        self.read_lists(
            lists,
            filter,
            id_2(
                #[inline(always)]
                |row| {
                    let (rough, err) = row.value;
                    let lowerbound = Distance::from_f32(rough - err * epsilon);
                    if !bound.admits(lowerbound) {
                        return;
                    }
                    results.push((
                        (Reverse(lowerbound), AlwaysEqual(())),
                        AlwaysEqual(PackedRefMut4(bump.alloc((
                            row.payload,
                            row.head,
                            BorrowedIter::from_slice(row.prefetch, |x| bump.alloc_slice(x)),
                        )))),
                    ));
                },
            ),
        );
        Heap::from(results.into_vec())
    }

    /// The rest of [`default_search`], with the bound chosen at compile time.
    #[inline(always)]
    fn search(mut self, params: SearchParams, bound: impl Bound) -> Candidates<'b>
    where
        Self: 'b,
    {
        let SearchParams {
            mut probes,
            filter,
            max_probes,
            ..
        } = params;
        let Route {
            lists,
            mut complete,
            rest,
        } = self.route(&probes, false);
        let Some(filter) = filter else {
            return Candidates::new(self.read(lists, (), bound), None);
        };
        let max_probes = max_probes.map_or(usize::MAX, |x| x as usize);
        if complete || lists.len() >= max_probes {
            return Candidates::new(self.read(lists, &filter, bound), None);
        }
        let mut read = lists.iter().map(|list| list.2.0).collect::<HashSet<_>>();
        let heap = self.read(lists, &filter, bound);
        // Below a root alone, routing's ranking of every list gives each round its further lists.
        let mut rest = (probes.len() == 1).then_some(rest);
        let refill = move || {
            while !complete && read.len() < max_probes {
                for x in probes.iter_mut() {
                    *x = x.saturating_mul(2).max(1);
                }
                let mut lists = if let Some(rest) = rest.as_mut() {
                    complete = probes[0] >= self.cells[0];
                    rest.take(probes[0] as usize - read.len()).collect()
                } else {
                    let route = self.route(&probes, false);
                    complete = route.complete;
                    let mut lists = route.lists;
                    lists.retain(|list| !read.contains(&list.2.0));
                    lists
                };
                lists.truncate(max_probes - read.len());
                read.extend(lists.iter().map(|list| list.2.0));
                if !lists.is_empty() {
                    return Some(self.read(lists, &filter, bound));
                }
            }
            None
        };
        Candidates::new(heap, Some(Box::new(refill)))
    }
}

/// A page of the `list`-th list, scheduled ahead of reading it.
struct Scheduled {
    list: usize,
    id: u32,
}

impl Fetch<'_> for Scheduled {
    type Iter = std::iter::Once<u32>;

    #[inline(always)]
    fn fetch(&self) -> Self::Iter {
        std::iter::once(self.id)
    }
}

/// Reads ahead the jump pages of the lists at `firsts`, then the tape pages their directories list.
/// A directory is read, not read ahead, under its jump page, as its list is.
fn schedule<'b, R>(
    index: &'b R,
    p: &mut impl PrefetcherSequenceFamily<'b, R>,
    firsts: Vec<u32>,
    filter: impl Filter,
) -> impl Prefetcher<'b, R = R, Item = Scheduled>
where
    R: RelationRead + 'b,
    R::Page: Page<Opaque = Opaque>,
{
    let jumps = firsts
        .into_iter()
        .enumerate()
        .map(|(list, id)| Scheduled { list, id });
    let mut jumps = p.prefetch(jumps.collect::<VecDeque<_>>());
    let (mut words, mut queue) = (Vec::new(), VecDeque::new());
    p.prefetch(
        std::iter::from_fn(move || {
            loop {
                if let Some(page) = queue.pop_front() {
                    return Some(page);
                }
                let (Scheduled { list, .. }, mut guards) = jumps.next()?;
                let jump_guard = guards.next().expect("should be at least one element");
                let jump_tuple =
                    JumpTuple::deserialize_ref(jump_guard.get(1).expect("data corruption"));
                let directory = by_next(index, jump_tuple.directory_first());
                let (pages, selected) = tape::read_directory::<R>(directory, &mut words, filter);
                let ids = words[..selected.unwrap_or(pages)].iter().copied();
                let ids = ids.chain([jump_tuple.appendable_first()]);
                queue.extend(ids.map(|id| Scheduled { list, id }));
            }
        })
        .peekable(),
    )
}

/// Reads the pages of the `list`-th list, taking those its schedule read ahead.
/// Buffers are locked only when taken, under the list's jump page: a stale schedule costs reads.
struct Ahead<'b, R, P> {
    index: &'b R,
    pages: RefCell<P>,
    list: Cell<usize>,
}

impl<'b, R: RelationRead + 'b, P: Prefetcher<'b, R = R, Item = Scheduled>> Ahead<'b, R, P> {
    /// Begins the `list`-th list, dropping the pages scheduled for earlier lists.
    fn begin(&self, list: usize) {
        self.list.set(list);
        let mut pages = self.pages.borrow_mut();
        while pages.next_if(|page| page.list < list).is_some() {}
    }
}

impl<R: RelationRead, P> Relation for Ahead<'_, R, P> {
    type Page = R::Page;
}

impl<'b, R: RelationRead, P> RelationReadTypes for Ahead<'b, R, P> {
    type ReadGuard<'a> = R::ReadGuard<'b>;
}

impl<'b, R: RelationRead + 'b, P: Prefetcher<'b, R = R, Item = Scheduled>> RelationRead
    for Ahead<'b, R, P>
{
    fn read(&self, id: u32) -> R::ReadGuard<'b> {
        let list = self.list.get();
        let page = self
            .pages
            .borrow_mut()
            .next_if(|page| page.list == list && page.id == id);
        match page {
            Some((_, mut guards)) => guards.next().expect("should be at least one element"),
            None => self.index.read(id),
        }
    }
}

/// With a filter, an empty heap reads further lists, whose rows can be nearer than those returned.
pub struct Candidates<'b> {
    heap: Heap<Candidate<'b>>,
    refill: Option<Refill<'b>>,
}

impl<'b> Candidates<'b> {
    fn new(heap: Heap<Candidate<'b>>, refill: Option<Refill<'b>>) -> Self {
        Self { heap, refill }
    }

    pub fn can_refill(&self) -> bool {
        self.refill.is_some()
    }
}

impl<'b> Sequence for Candidates<'b> {
    type Item = Candidate<'b>;
    type Inner = std::vec::IntoIter<Candidate<'b>>;

    #[inline]
    fn next(&mut self) -> Option<Candidate<'b>> {
        self.heap.pop()
    }

    #[inline]
    fn peek(&mut self) -> Option<&Candidate<'b>> {
        self.heap.peek()
    }

    fn into_inner(self) -> Self::Inner {
        self.heap.into_vec().into_iter()
    }

    fn refill(&mut self) -> bool {
        while self.heap.is_empty()
            && let Some(refill) = self.refill.as_mut()
        {
            match refill() {
                Some(heap) => self.heap = heap,
                None => self.refill = None,
            }
        }
        !self.heap.is_empty()
    }
}

/// A sphere's radius, which a candidate's lower bound must stay under, or `()` for no sphere.
trait Bound: Copy + 'static {
    fn admits(self, lowerbound: Distance) -> bool;
}

impl Bound for () {
    #[inline(always)]
    fn admits(self, _: Distance) -> bool {
        true
    }
}

impl Bound for Distance {
    #[inline(always)]
    fn admits(self, lowerbound: Distance) -> bool {
        lowerbound < self
    }
}

/// Candidates past `bound` are dropped; with a filter, refills double the probes to `max_probes`.
// Inlined into each scanner arm, which compiles the halfvec search to fewer instructions.
#[inline(always)]
pub fn default_search<'b, R: RelationRead, O: Operator>(
    index: &'b R,
    vector: Rc<O::Vector>,
    bound: Option<Distance>,
    params: SearchParams,
    bump: &'b impl Bump,
    prefetch_h1_vectors: impl PrefetcherHeapFamily<'b, R> + 'b,
    prefetch_h0_tuples: impl PrefetcherSequenceFamily<'b, R> + 'b,
) -> Candidates<'b>
where
    R::Page: Page<Opaque = Opaque>,
{
    let searcher = Searcher::<_, O, _, _, _, _>::new(
        index,
        vector,
        &params,
        bump,
        prefetch_h1_vectors,
        prefetch_h0_tuples,
    );
    match bound {
        Some(radius) => searcher.search(params, radius),
        None => searcher.search(params, ()),
    }
}

pub fn maxsim_search<'b, R: RelationRead, O: Operator>(
    index: &'b R,
    vector: &O::Vector,
    params: &SearchParams,
    mut threshold: u32,
    bump: &'b impl Bump,
    prefetch_h1_vectors: impl PrefetcherHeapFamily<'b, R>,
    prefetch_h0_tuples: impl PrefetcherSequenceFamily<'b, R>,
) -> (Vec<MaxsimCandidate<'b>>, Distance)
where
    R::Page: Page<Opaque = Opaque>,
{
    let mut searcher = Searcher::<_, O, _, _, _, _>::new(
        index,
        vector,
        params,
        bump,
        prefetch_h1_vectors,
        prefetch_h0_tuples,
    );
    let epsilon = params.epsilon;
    let Route { lists, rest, .. } = searcher.route(&params.probes, threshold != 0);
    let mut results = LinkedVec::<(_, AlwaysEqual<_>)>::new();
    let tuples = searcher.read_lists(
        lists,
        (),
        id_2(|row| {
            let (rough, err) = row.value;
            let lowerbound = Distance::from_f32(rough - err * epsilon);
            let rough = Distance::from_f32(rough);
            results.push((
                (Reverse(lowerbound), AlwaysEqual(rough)),
                AlwaysEqual(PackedRefMut8(bump.alloc((
                    row.payload,
                    row.head,
                    BorrowedIter::from_slice(row.prefetch, |x| bump.alloc_slice(x)),
                )))),
            ));
        }),
    );
    threshold = threshold.saturating_sub(tuples.min(u32::MAX as _) as _);
    let mut estimation_by_threshold = Distance::NEG_INFINITY;
    // Remaining routing distances only estimate missing MaxSim terms; they are
    // not lower bounds and never reach the reranker.
    for (Reverse(distance), AlwaysEqual(_), AlwaysEqual(first)) in rest {
        if threshold == 0 {
            break;
        }
        index::interrupt::check();
        let jump_guard = index.read(first);
        let jump_bytes = jump_guard.get(1).expect("data corruption");
        let jump_tuple = JumpTuple::deserialize_ref(jump_bytes);
        threshold = threshold.saturating_sub(jump_tuple.tuples().min(u32::MAX as _) as _);
        estimation_by_threshold = distance;
    }
    (results.into_vec(), estimation_by_threshold)
}
