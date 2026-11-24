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
use index_accessor::{Accessor1, DefaultWithDimension, FunctionalAccessor, LAccess};
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

/// (distance, norm, first, K): K is the centroid while routing ranks, then whether it is skipped.
type List<K = ()> = (
    Reverse<Distance>,
    AlwaysEqual<f32>,
    AlwaysEqual<u32>,
    AlwaysEqual<K>,
);

pub type Candidate<'b> = (
    (Reverse<Distance>, AlwaysEqual<()>),
    AlwaysEqual<PackedRefMut4<'b, (NonZero<u64>, u16, BorrowedIter<'b>)>>,
);

pub type MaxsimCandidate<'b> = (
    (Reverse<Distance>, AlwaysEqual<Distance>),
    AlwaysEqual<PackedRefMut8<'b, (NonZero<u64>, u16, BorrowedIter<'b>)>>,
);

type Refill<'b> = Box<dyn FnMut() -> Option<Heap<Candidate<'b>>> + 'b>;

/// The nearest centroids whose planes bound the lists beyond them.
const NEAREST: usize = 4;

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

struct Route<K, I> {
    lists: Vec<List<K>>,
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
            (
                Reverse(distance),
                AlwaysEqual(norm),
                AlwaysEqual(first),
                AlwaysEqual(()),
            )
        } else {
            // fast path
            (
                Reverse(Distance::ZERO),
                AlwaysEqual(norm),
                AlwaysEqual(first),
                AlwaysEqual(()),
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

    fn route<S: Scope<'b, O>>(
        &mut self,
        probes: &[u32],
        sorted: bool,
        scope: &S,
    ) -> Route<S::Skipped, impl Iterator<Item = List<S::Skipped>> + use<'b, R, O, V, B, H1, H0, S>>
    {
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
        let mut step = |lists: Vec<List<S::Skipped>>| {
            let mut results = LinkedVec::<(_, AlwaysEqual<Extra1<'b>>)>::new();
            for (Reverse(dis_f), AlwaysEqual(norm), AlwaysEqual(first), _) in lists {
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
            let mut cache = BinaryHeap::<List<S::Centroid>>::new();
            let vector = vector.clone();
            let mut scope = scope.clone();
            std::iter::from_fn(move || {
                while let Some((
                    (Reverse(_), AlwaysEqual(&mut (first, norm, head, ..))),
                    prefetch,
                )) = heap.next_if(|(d, _)| Some(*d) > cache.peek().map(|(d, ..)| *d))
                {
                    let (distance, centroid) = centroids::read::<R, O, _>(
                        prefetch,
                        head,
                        (
                            LAccess::new(
                                O::Vector::unpack(vector.borrow().as_borrowed()),
                                O::DistanceAccessor::default_with_dimension(dim),
                            ),
                            scope.keep(bump, dim),
                        ),
                    );
                    cache.push((
                        Reverse(distance),
                        AlwaysEqual(norm),
                        AlwaysEqual(first),
                        AlwaysEqual(centroid),
                    ));
                }
                let (distance, norm, first, AlwaysEqual(centroid)) = cache.pop()?;
                let skipped = scope.offer(distance.0, centroid);
                Some((distance, norm, first, AlwaysEqual(skipped)))
            })
        };
        let levels = probes.len();
        let (distance, norm, first, _) = root;
        let mut lists = vec![(distance, norm, first, AlwaysEqual(S::Skipped::default()))];
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
                for (Reverse(_), AlwaysEqual(_), AlwaysEqual(first), _) in lists {
                    tape::read_h1_tape::<R, _, _>(
                        by_next(index, first),
                        || FunctionalAccessor::new((), id_0(|_, _| ()), id_1(|_, _| [(); _])),
                        |(), _, norm, first, _| {
                            results.push((
                                Reverse(Distance::ZERO),
                                AlwaysEqual(norm),
                                AlwaysEqual(first),
                                AlwaysEqual(S::Skipped::default()),
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
        for (i, (Reverse(dis_f), AlwaysEqual(norm), AlwaysEqual(first), _)) in
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

    /// The rest of [`default_search`], with what it reads chosen at compile time.
    #[inline(always)]
    fn search<S: Scope<'b, O> + 'b>(mut self, params: SearchParams, scope: S) -> Candidates<'b>
    where
        Self: 'b,
    {
        let bound = scope.rows();
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
        } = self.route(&probes, scope.ranks(), &scope);
        let Some(filter) = filter else {
            return Candidates::new(self.read(S::unskipped(lists), (), bound), None);
        };
        let max_probes = max_probes.map_or(usize::MAX, |x| x as usize);
        if complete || lists.len() >= max_probes {
            return Candidates::new(self.read(S::unskipped(lists), &filter, bound), None);
        }
        let mut read = lists.iter().map(|list| list.2.0).collect::<HashSet<_>>();
        let heap = self.read(S::unskipped(lists), &filter, bound);
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
                    let route = self.route(&probes, scope.ranks(), &scope);
                    complete = route.complete;
                    let mut lists = route.lists;
                    lists.retain(|list| !read.contains(&list.2.0));
                    lists
                };
                lists.truncate(max_probes - read.len());
                read.extend(lists.iter().map(|list| list.2.0));
                if !lists.is_empty() {
                    return Some(self.read(S::unskipped(lists), &filter, bound));
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

/// A sphere around the search vector.
#[derive(Debug, Clone, Copy)]
pub struct Ball {
    /// Rows at or past it are dropped.
    pub bound: Distance,
    /// Its reach past the plane between two centroids, per unit of the distance between them.
    pub reach: Option<f32>,
}

impl Ball {
    /// For rows nearest their own centroid: the planes midway to nearer centroids.
    pub fn reach_l2(radius: f32) -> Option<f32> {
        Some(2.0 * radius)
    }

    /// For unit rows nearest their own centroid by inner product: the planes through the origin.
    pub fn reach_cosine(radius: f32) -> Option<f32> {
        // A unit query is within 1 of every plane through the origin: such a reach skips no list.
        Some((2.0 * radius).sqrt()).filter(|reach| *reach < 1.0)
    }
}

type Element<O> = <<O as Operator>::Vector as Vector>::Element;
type Metadata<O> = <<O as Operator>::Vector as Vector>::Metadata;

/// What a search reads of the lists routing offers and of their rows: all of them, or a sphere's.
trait Scope<'b, O: Operator>: Clone {
    type Rows: Bound;
    /// What routing keeps of a centroid it reads until it offers the list.
    type Centroid;
    /// Whether an offered list is skipped.
    type Skipped: Default;

    fn rows(&self) -> Self::Rows;

    /// Whether routing ranks the lists even where it would read them all.
    fn ranks(&self) -> bool;

    fn keep(
        &mut self,
        bump: &'b impl Bump,
        dim: u32,
    ) -> impl Accessor1<Element<O>, Metadata<O>, Output = Self::Centroid>;

    fn offer(&mut self, distance: Distance, centroid: Self::Centroid) -> Self::Skipped;

    fn unskipped(lists: Vec<List<Self::Skipped>>) -> Vec<List>;
}

impl<'b, O: Operator, B: Bound> Scope<'b, O> for B {
    type Rows = B;
    type Centroid = ();
    type Skipped = ();

    #[inline(always)]
    fn rows(&self) -> B {
        *self
    }

    #[inline(always)]
    fn ranks(&self) -> bool {
        false
    }

    #[inline(always)]
    fn keep(
        &mut self,
        _: &'b impl Bump,
        _: u32,
    ) -> impl Accessor1<Element<O>, Metadata<O>, Output = ()> {
    }

    #[inline(always)]
    fn offer(&mut self, _: Distance, (): ()) {}

    #[inline(always)]
    fn unskipped(lists: Vec<List>) -> Vec<List> {
        lists
    }
}

/// Below a root alone: skips a list whose rows, each nearest its own centroid, lie past the sphere.
#[derive(Clone)]
struct Planes<'b, O: Operator> {
    bound: Distance,
    reach: f32,
    /// A centroid's elements while routing reads it.
    scratch: Vec<Element<O>>,
    /// The nearest centroids and their distances: the first lists offered.
    nearest: Vec<(<O::Vector as VectorOwned>::Borrowed<'b>, f32)>,
}

impl<'b, O: Operator> Scope<'b, O> for Planes<'b, O> {
    type Rows = Distance;
    type Centroid = <O::Vector as VectorOwned>::Borrowed<'b>;
    type Skipped = bool;

    fn rows(&self) -> Distance {
        self.bound
    }

    fn ranks(&self) -> bool {
        true
    }

    fn keep(
        &mut self,
        bump: &'b impl Bump,
        dim: u32,
    ) -> impl Accessor1<Element<O>, Metadata<O>, Output = Self::Centroid> {
        self.scratch.clear();
        FunctionalAccessor::new(
            &mut self.scratch,
            id_0(|elements: &mut &mut Vec<_>, slice| elements.extend_from_slice(slice)),
            move |elements: &mut Vec<_>, metadata| {
                O::Vector::borrowed(dim, bump.alloc_slice(elements), metadata)
            },
        )
    }

    // Insertion routes approximately: a row past its plane by more than the reach is missed.
    fn offer(&mut self, distance: Distance, centroid: Self::Centroid) -> bool {
        let distance = distance.to_f32();
        let skipped = self.nearest.iter().any(|&(other, nearer)| {
            distance - nearer > self.reach * centroid.operator_l2s(other).to_f32().sqrt()
        });
        if self.nearest.len() < NEAREST {
            self.nearest.push((centroid, distance));
        }
        skipped
    }

    fn unskipped(lists: Vec<List<bool>>) -> Vec<List> {
        let lists = lists.into_iter().filter(|list| !list.3.0);
        lists
            .map(|(distance, norm, first, _)| (distance, norm, first, AlwaysEqual(())))
            .collect()
    }
}

/// Candidates past `ball` are dropped; with a filter, refills double the probes to `max_probes`.
// Inlined into each scanner arm, which compiles the halfvec search to fewer instructions.
#[inline(always)]
pub fn default_search<'b, R: RelationRead, O: Operator>(
    index: &'b R,
    vector: Rc<O::Vector>,
    ball: Option<Ball>,
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
    let Some(Ball { bound, reach }) = ball else {
        return searcher.search(params, ());
    };
    match reach.filter(|_| searcher.cells.len() == 2) {
        Some(reach) => {
            let (scratch, nearest) = (Vec::new(), Vec::new());
            searcher.search(
                params,
                Planes {
                    bound,
                    reach,
                    scratch,
                    nearest,
                },
            )
        }
        None => searcher.search(params, bound),
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
    let Route { lists, rest, .. } = searcher.route(&params.probes, threshold != 0, &());
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
    for (Reverse(distance), AlwaysEqual(_), AlwaysEqual(first), _) in rest {
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

#[cfg(test)]
mod tests {
    use super::*;
    use index::prefetcher::PlainPrefetcher;
    use index::test_support::MemoryRelation;
    use index_accessor::{Dot, L2S};
    use std::cell::Cell;
    use vector::vect::{VectBorrowed, VectOwned};

    #[test]
    fn rerank_refills_only_after_the_initial_rows_are_consumed() {
        let index = MemoryRelation::<Opaque>::new([]);
        let reads = Cell::new(0);
        let row = |id| {
            (
                NonZero::new(id).unwrap(),
                0,
                BorrowedIter::from_slice(&[], |_| &[]),
            )
        };
        let mut first = row(1);
        let mut second = row(2);
        let candidate = |r| {
            (
                (Reverse(Distance::from_f32(0.0)), AlwaysEqual(())),
                AlwaysEqual(PackedRefMut4(r)),
            )
        };
        let mut later = Some(Heap::from(vec![candidate(&mut second)]));
        let refill = || {
            reads.set(reads.get() + 1);
            later.take()
        };
        let candidates = Candidates::new(
            Heap::from(vec![candidate(&mut first)]),
            Some(Box::new(refill)),
        );
        let mut rows = crate::rerank_heap(PlainPrefetcher::new(&index, candidates), |_| {
            Some(Distance::from_f32(1.0))
        });
        assert_eq!(rows.next().unwrap().1.get(), 1);
        assert_eq!(reads.get(), 0);
        assert_eq!(rows.next().unwrap().1.get(), 2);
        assert_eq!(reads.get(), 1);
    }

    /// Whether a sphere of `reach` skips the last of the lists around `centroids`, offered in turn.
    fn skips<O: Operator<Vector = VectOwned<f32>>>(
        reach: Option<f32>,
        query: [f32; 2],
        centroids: &[[f32; 2]],
    ) -> bool {
        let (bound, scratch, nearest) = (Distance::ZERO, Vec::new(), Vec::new());
        let mut planes = Planes::<O> {
            bound,
            reach: reach.unwrap(),
            scratch,
            nearest,
        };
        let offer = |centroid| {
            let centroid = VectBorrowed::new(centroid);
            let distance = crate::exact_distance::<O>(VectBorrowed::new(&query), centroid);
            planes.offer(distance, centroid)
        };
        centroids.iter().map(|x| x.as_slice()).map(offer).last() == Some(true)
    }

    #[test]
    fn a_sphere_skips_the_lists_past_its_reach() {
        type L2 = Op<VectOwned<f32>, L2S>;
        let centroids = [[0.0, 0.0], [10.0, 0.0]];
        assert!(skips::<L2>(Ball::reach_l2(3.9), [1.0, 0.0], &centroids));
        assert!(!skips::<L2>(Ball::reach_l2(4.1), [1.0, 0.0], &centroids));
        type Ip = Op<VectOwned<f32>, Dot>;
        let centroids = [[1.0, 0.0], [0.0, 1.0]];
        assert!(skips::<Ip>(
            Ball::reach_cosine(0.008),
            [0.8, 0.6],
            &centroids
        ));
        assert!(!skips::<Ip>(
            Ball::reach_cosine(0.012),
            [0.8, 0.6],
            &centroids
        ));
        assert_eq!(Ball::reach_cosine(0.5), None);
    }
}
