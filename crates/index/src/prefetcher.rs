// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use crate::fetch::Fetch;
use crate::relation::{
    Hints, ReadStream, RelationPrefetch, RelationRead, RelationReadStream, RelationReadTypes,
};
use dary_heap::DaryHeap;
use std::cell::RefCell;
use std::collections::{BinaryHeap, VecDeque};
use std::rc::Rc;

pub const WINDOW_SIZE: usize = 32;
const _: () = assert!(WINDOW_SIZE > 0);

/// A `MixedPrefetcher` streams while at least half of the next `STREAM_SPAN` pages follow the page before them.
const STREAM_SPAN: usize = 16;
const _: () = assert!(STREAM_SPAN > 1);

pub trait Sequence {
    type Item;
    type Inner: Iterator<Item = Self::Item>;
    #[must_use]
    fn next(&mut self) -> Option<Self::Item>;
    #[must_use]
    fn peek(&mut self) -> Option<&Self::Item>;
    fn into_inner(self) -> Self::Inner;
    /// Asked once every item is taken: whether the sequence gained new items.
    fn refill(&mut self) -> bool {
        false
    }
}

impl<T: Ord> Sequence for BinaryHeap<T> {
    type Item = T;
    type Inner = std::vec::IntoIter<T>;
    #[inline]
    fn next(&mut self) -> Option<T> {
        self.pop()
    }
    #[inline]
    fn peek(&mut self) -> Option<&T> {
        (self as &Self).peek()
    }
    #[inline]
    fn into_inner(self) -> Self::Inner {
        self.into_vec().into_iter()
    }
}

impl<const N: usize, T: Ord> Sequence for DaryHeap<T, N> {
    type Item = T;
    type Inner = std::vec::IntoIter<T>;
    #[inline]
    fn next(&mut self) -> Option<T> {
        self.pop()
    }
    #[inline]
    fn peek(&mut self) -> Option<&T> {
        (self as &Self).peek()
    }
    #[inline]
    fn into_inner(self) -> Self::Inner {
        self.into_vec().into_iter()
    }
}

impl<I: Iterator> Sequence for std::iter::Peekable<I> {
    type Item = I::Item;
    type Inner = std::iter::Peekable<I>;
    #[inline]
    fn next(&mut self) -> Option<I::Item> {
        Iterator::next(self)
    }
    #[inline]
    fn peek(&mut self) -> Option<&I::Item> {
        self.peek()
    }
    #[inline]
    fn into_inner(self) -> Self::Inner {
        self
    }
}

impl<T> Sequence for VecDeque<T> {
    type Item = T;
    type Inner = std::collections::vec_deque::IntoIter<T>;
    #[inline]
    fn next(&mut self) -> Option<T> {
        self.pop_front()
    }
    #[inline]
    fn peek(&mut self) -> Option<&T> {
        self.front()
    }
    #[inline]
    fn into_inner(self) -> Self::Inner {
        self.into_iter()
    }
}

pub trait Prefetcher<'b>: IntoIterator
where
    Self::Item: Fetch<'b>,
{
    type R: RelationRead + 'b;
    type Guards: ExactSizeIterator<Item = <Self::R as RelationReadTypes>::ReadGuard<'b>>;

    #[must_use]
    fn next(&mut self) -> Option<(Self::Item, Self::Guards)>;
    #[must_use]
    fn next_if(
        &mut self,
        predicate: impl FnOnce(&Self::Item) -> bool,
    ) -> Option<(Self::Item, Self::Guards)>;
    fn refill(&mut self) -> bool;
}

pub struct PlainPrefetcher<'b, R, S: Sequence> {
    relation: &'b R,
    sequence: S,
}

impl<'b, R, S: Sequence> PlainPrefetcher<'b, R, S> {
    #[inline]
    pub fn new(relation: &'b R, sequence: S) -> Self {
        Self { relation, sequence }
    }
}

impl<'b, R, S: Sequence> IntoIterator for PlainPrefetcher<'b, R, S> {
    type Item = S::Item;

    type IntoIter = S::Inner;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.sequence.into_inner()
    }
}

impl<'b, R: RelationRead, S: Sequence> Prefetcher<'b> for PlainPrefetcher<'b, R, S>
where
    S::Item: Fetch<'b>,
{
    type R = R;
    type Guards = PlainPrefetcherGuards<'b, R, <S::Item as Fetch<'b>>::Iter>;

    #[inline]
    fn next(
        &mut self,
    ) -> Option<(
        Self::Item,
        PlainPrefetcherGuards<'b, R, <S::Item as Fetch<'b>>::Iter>,
    )> {
        let e = self.sequence.next()?;
        let list = e.fetch();
        Some((
            e,
            PlainPrefetcherGuards {
                relation: self.relation,
                list,
            },
        ))
    }

    #[inline]
    fn next_if(
        &mut self,
        predicate: impl FnOnce(&Self::Item) -> bool,
    ) -> Option<(
        S::Item,
        PlainPrefetcherGuards<'b, R, <S::Item as Fetch<'b>>::Iter>,
    )> {
        if !predicate(self.sequence.peek()?) {
            return None;
        }
        let e = self.sequence.next()?;
        let list = e.fetch();
        Some((
            e,
            PlainPrefetcherGuards {
                relation: self.relation,
                list,
            },
        ))
    }

    #[inline]
    fn refill(&mut self) -> bool {
        self.sequence.refill()
    }
}

pub struct PlainPrefetcherGuards<'b, R, L> {
    relation: &'b R,
    list: L,
}

impl<'b, R: RelationRead, L: Iterator<Item = u32>> Iterator for PlainPrefetcherGuards<'b, R, L> {
    type Item = R::ReadGuard<'b>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let id = self.list.next()?;
        Some(self.relation.read(id))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.list.size_hint()
    }
}

impl<'b, R: RelationRead, L: Iterator<Item = u32>> ExactSizeIterator
    for PlainPrefetcherGuards<'b, R, L>
{
}

pub struct SimplePrefetcher<'b, R, S: Sequence> {
    relation: &'b R,
    window: VecDeque<S::Item>,
    sequence: S,
}

impl<'b, R, S: Sequence> SimplePrefetcher<'b, R, S> {
    #[inline]
    pub fn new(relation: &'b R, sequence: S) -> Self {
        Self {
            relation,
            window: VecDeque::new(),
            sequence,
        }
    }
}

impl<'b, R, S: Sequence> IntoIterator for SimplePrefetcher<'b, R, S> {
    type Item = S::Item;

    type IntoIter = std::iter::Chain<std::collections::vec_deque::IntoIter<S::Item>, S::Inner>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.window.into_iter().chain(self.sequence.into_inner())
    }
}

impl<'b, R: RelationRead + RelationPrefetch, S: Sequence> Prefetcher<'b>
    for SimplePrefetcher<'b, R, S>
where
    S::Item: Fetch<'b>,
{
    type R = R;
    type Guards = SimplePrefetcherGuards<'b, R, <S::Item as Fetch<'b>>::Iter>;

    #[inline]
    fn next(
        &mut self,
    ) -> Option<(
        Self::Item,
        SimplePrefetcherGuards<'b, R, <S::Item as Fetch<'b>>::Iter>,
    )> {
        while self.window.len() < WINDOW_SIZE
            && let Some(e) = self.sequence.next()
        {
            for id in e.fetch() {
                self.relation.prefetch(id);
            }
            self.window.push_back(e);
        }
        let e = self.window.pop_front()?;
        let list = e.fetch();
        Some((
            e,
            SimplePrefetcherGuards {
                relation: self.relation,
                list,
            },
        ))
    }

    #[inline]
    fn next_if(
        &mut self,
        predicate: impl FnOnce(&S::Item) -> bool,
    ) -> Option<(
        S::Item,
        SimplePrefetcherGuards<'b, R, <S::Item as Fetch<'b>>::Iter>,
    )> {
        while self.window.len() < WINDOW_SIZE
            && let Some(e) = self.sequence.next()
        {
            for id in e.fetch() {
                self.relation.prefetch(id);
            }
            self.window.push_back(e);
        }
        let e = self.window.pop_front_if(move |x| predicate(x))?;
        let list = e.fetch();
        Some((
            e,
            SimplePrefetcherGuards {
                relation: self.relation,
                list,
            },
        ))
    }

    #[inline]
    fn refill(&mut self) -> bool {
        self.sequence.refill()
    }
}

pub struct SimplePrefetcherGuards<'b, R, L> {
    relation: &'b R,
    list: L,
}

impl<'b, R: RelationRead, L: Iterator<Item = u32>> Iterator for SimplePrefetcherGuards<'b, R, L> {
    type Item = R::ReadGuard<'b>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let id = self.list.next()?;
        Some(self.relation.read(id))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.list.size_hint()
    }
}

impl<'b, R: RelationRead, L: ExactSizeIterator<Item = u32>> ExactSizeIterator
    for SimplePrefetcherGuards<'b, R, L>
{
}

/// Reads spans of pages that mostly follow one another through a read stream, the rest as `SimplePrefetcher` does.
pub struct MixedPrefetcher<'b, R: RelationReadStream + 'b, S: Sequence>
where
    S::Item: Fetch<'b>,
{
    relation: &'b R,
    hints: Hints,
    shared: Rc<RefCell<Mixed<S>>>,
    stream: Option<R::ReadStream<'b, Run<S>>>,
}

struct Mixed<S: Sequence> {
    sequence: S,
    window: VecDeque<S::Item>,
    /// The last page given to the open stream, `None` once it has ended.
    last: Option<u32>,
    /// Items given to the stream and not yet taken.
    pending: usize,
    /// Items at the front of the window that are prefetched.
    advised: usize,
}

/// The items that a `MixedPrefetcher` gives to its stream while they stay dense.
pub struct Run<S: Sequence>(Rc<RefCell<Mixed<S>>>);

/// Whether a page follows `last`.
#[inline]
fn follows(last: u32, page: u32) -> bool {
    page == last.wrapping_add(1)
}

impl<'b, S: Sequence> Mixed<S>
where
    S::Item: Fetch<'b>,
{
    /// Fills the window to `STREAM_SPAN - 1` items beyond the prefetch window, so that a span can be judged at each of them.
    #[inline]
    fn top_up(&mut self) {
        while self.window.len() < WINDOW_SIZE + STREAM_SPAN - 1
            && let Some(e) = self.sequence.next()
        {
            self.window.push_back(e);
        }
    }

    /// How many of the `STREAM_SPAN` items from the `i`-th on begin at a page that follows the last page before them.
    #[inline]
    fn adjacent(&self, i: usize, mut last: Option<u32>) -> usize {
        let mut n = 0;
        for e in self.window.range(i..).take(STREAM_SPAN) {
            let mut pages = e.fetch();
            if let Some(first) = pages.next() {
                n += usize::from(last.is_some_and(|last| follows(last, first)));
                last = Some(pages.last().unwrap_or(first));
            }
        }
        n
    }

    /// Whether a span dense enough to stream starts at the `i`-th item of the window; a shorter run is not worth a stream.
    #[inline]
    fn starts(&self, i: usize) -> bool {
        i + STREAM_SPAN <= self.window.len() && self.adjacent(i, None) >= STREAM_SPAN / 2
    }

    /// Whether the open stream goes on to the next item.
    #[inline]
    fn dense(&self) -> bool {
        let (Some(last), Some(e)) = (self.last, self.window.front()) else {
            return false;
        };
        e.fetch().next().is_some_and(|first| follows(last, first))
            || self.adjacent(0, Some(last)) >= STREAM_SPAN / 2
    }
}

impl<'b, S: Sequence> Iterator for Run<S>
where
    S::Item: Fetch<'b>,
{
    type Item = S::Item;

    #[inline]
    fn next(&mut self) -> Option<S::Item> {
        let mut m = self.0.borrow_mut();
        m.top_up();
        if !m.dense() {
            m.last = None;
            return None;
        }
        let e = m.window.pop_front()?;
        m.advised = m.advised.saturating_sub(1);
        m.last = e.fetch().last().or(m.last);
        m.pending += 1;
        Some(e)
    }
}

impl<'b, R: RelationReadStream, S: Sequence> MixedPrefetcher<'b, R, S>
where
    S::Item: Fetch<'b>,
{
    #[inline]
    pub fn new(relation: &'b R, sequence: S, hints: Hints) -> Self {
        let mixed = Mixed {
            sequence,
            window: VecDeque::new(),
            last: None,
            pending: 0,
            advised: 0,
        };
        Self {
            relation,
            hints,
            shared: Rc::new(RefCell::new(mixed)),
            stream: None,
        }
    }
}

impl<'b, R: RelationPrefetch + RelationReadStream, S: Sequence> MixedPrefetcher<'b, R, S>
where
    S::Item: Fetch<'b>,
{
    /// Keeps a stream while its span lasts, starts one on a span at the front, and prefetches up to the next span.
    #[inline]
    fn fill(&mut self) {
        let mut m = self.shared.borrow_mut();
        m.top_up();
        if self.stream.is_some() {
            if m.dense() {
                return;
            }
            if m.pending == 0 {
                self.stream = None;
            }
        }
        if self.stream.is_none()
            && m.advised == 0
            && m.starts(0)
            && let Some(first) = m.window[0].fetch().next()
        {
            m.last = Some(first.wrapping_sub(1));
            drop(m);
            let run = Run(self.shared.clone());
            self.stream = Some(self.relation.read_stream(run, self.hints));
            return;
        }
        while m.advised < m.window.len().min(WINDOW_SIZE) && !m.starts(m.advised) {
            for id in m.window[m.advised].fetch() {
                self.relation.prefetch(id);
            }
            m.advised += 1;
        }
    }
}

impl<'b, R: RelationReadStream, S: Sequence> IntoIterator for MixedPrefetcher<'b, R, S>
where
    S::Item: Fetch<'b>,
{
    type Item = S::Item;

    type IntoIter = std::iter::Chain<
        std::vec::IntoIter<S::Item>,
        std::iter::Chain<std::collections::vec_deque::IntoIter<S::Item>, S::Inner>,
    >;

    #[inline]
    fn into_iter(mut self) -> Self::IntoIter {
        self.shared.borrow_mut().last = None;
        let streamed: Vec<_> = self
            .stream
            .take()
            .map_or_else(Vec::new, |s| s.into_inner().collect());
        let m = Rc::into_inner(self.shared)
            .expect("the stream is gone")
            .into_inner();
        let rest = m.window.into_iter().chain(m.sequence.into_inner());
        streamed.into_iter().chain(rest)
    }
}

impl<'b, R: RelationRead + RelationPrefetch + RelationReadStream, S: Sequence> Prefetcher<'b>
    for MixedPrefetcher<'b, R, S>
where
    S::Item: Fetch<'b>,
{
    type R = R;

    type Guards = MixedGuards<
        SimplePrefetcherGuards<'b, R, <S::Item as Fetch<'b>>::Iter>,
        <R::ReadStream<'b, Run<S>> as ReadStream<'b>>::Guards,
    >;

    #[inline]
    fn next(&mut self) -> Option<(S::Item, Self::Guards)> {
        self.next_if(|_| true)
    }

    #[inline]
    fn next_if(
        &mut self,
        predicate: impl FnOnce(&S::Item) -> bool,
    ) -> Option<(S::Item, Self::Guards)> {
        self.fill();
        if let Some(stream) = self.stream.as_mut() {
            let (e, guards) = stream.next_if(predicate)?;
            self.shared.borrow_mut().pending -= 1;
            return Some((e, MixedGuards::Stream(guards)));
        }
        let mut m = self.shared.borrow_mut();
        let e = m.window.pop_front_if(move |x| predicate(x))?;
        m.advised = m.advised.saturating_sub(1);
        drop(m);
        let list = e.fetch();
        let relation = self.relation;
        Some((
            e,
            MixedGuards::Simple(SimplePrefetcherGuards { relation, list }),
        ))
    }

    #[inline]
    fn refill(&mut self) -> bool {
        self.shared.borrow_mut().sequence.refill()
    }
}

pub enum MixedGuards<A, B> {
    Simple(A),
    Stream(B),
}

impl<T, A: Iterator<Item = T>, B: Iterator<Item = T>> Iterator for MixedGuards<A, B> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        match self {
            MixedGuards::Simple(a) => a.next(),
            MixedGuards::Stream(b) => b.next(),
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            MixedGuards::Simple(a) => a.size_hint(),
            MixedGuards::Stream(b) => b.size_hint(),
        }
    }
}

impl<T, A: ExactSizeIterator<Item = T>, B: ExactSizeIterator<Item = T>> ExactSizeIterator
    for MixedGuards<A, B>
{
}

pub struct StreamPrefetcherSequence<S>(S);

impl<S: Sequence> Iterator for StreamPrefetcherSequence<S> {
    type Item = S::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

pub struct StreamPrefetcher<'b, R: RelationReadStream + 'b, S: Sequence>
where
    S::Item: Fetch<'b>,
{
    stream: R::ReadStream<'b, StreamPrefetcherSequence<S>>,
}

impl<'b, R: RelationReadStream, S: Sequence> StreamPrefetcher<'b, R, S>
where
    S::Item: Fetch<'b>,
{
    #[inline]
    pub fn new(relation: &'b R, sequence: S, hints: Hints) -> Self {
        let stream = relation.read_stream(StreamPrefetcherSequence(sequence), hints);
        Self { stream }
    }
}

impl<'b, R: RelationReadStream, S: Sequence> IntoIterator for StreamPrefetcher<'b, R, S>
where
    S::Item: Fetch<'b>,
{
    type Item = S::Item;

    type IntoIter = <R::ReadStream<'b, StreamPrefetcherSequence<S>> as ReadStream<'b>>::Inner;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.stream.into_inner()
    }
}

impl<'b, R: RelationRead + RelationReadStream, S: Sequence> Prefetcher<'b>
    for StreamPrefetcher<'b, R, S>
where
    S::Item: Fetch<'b>,
{
    type R = R;

    type Guards = <R::ReadStream<'b, StreamPrefetcherSequence<S>> as ReadStream<'b>>::Guards;

    #[inline]
    fn next(&mut self) -> Option<(S::Item, Self::Guards)> {
        self.stream.next()
    }

    #[inline]
    fn next_if(
        &mut self,
        predicate: impl FnOnce(&Self::Item) -> bool,
    ) -> Option<(S::Item, Self::Guards)> {
        self.stream.next_if(predicate)
    }

    #[inline]
    fn refill(&mut self) -> bool {
        self.stream
            .refill(|StreamPrefetcherSequence(sequence)| sequence.refill())
    }
}

pub trait PrefetcherSequenceFamily<'b, R> {
    type P<S: Sequence>: Prefetcher<'b, R = R, Item = S::Item>
    where
        S::Item: Fetch<'b>;

    fn prefetch<S: Sequence>(&mut self, seq: S) -> Self::P<S>
    where
        S::Item: Fetch<'b>;

    fn is_not_plain(&self) -> bool;
}

pub trait PrefetcherHeapFamily<'b, R> {
    type P<T>: Prefetcher<'b, R = R, Item = T>
    where
        T: Ord + Fetch<'b>;

    fn prefetch<T>(&mut self, seq: Vec<T>) -> Self::P<T>
    where
        T: Ord + Fetch<'b>;

    fn is_not_plain(&self) -> bool;
}

#[cfg(all(test, feature = "test-support"))]
mod tests {
    use super::*;
    use crate::relation::{PageGuard, Relation, RelationReadStreamTypes};
    use crate::test_support::{MemoryPage, MemoryRelation};
    use std::cell::Cell;
    use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

    #[repr(transparent)]
    #[derive(Clone, Copy, FromBytes, IntoBytes, Immutable, KnownLayout)]
    struct Opaque(u32);

    #[allow(unsafe_code)]
    unsafe impl crate::relation::Opaque for Opaque {}

    #[derive(Debug, Clone, Copy, PartialEq)]
    enum By {
        Prefetch,
        Stream,
    }

    /// Logs each page read ahead, counts streams and checks that a stream is alone and reads its own pages.
    struct Logged {
        pages: MemoryRelation<Opaque>,
        log: RefCell<Vec<(u32, By)>>,
        live: Cell<usize>,
        streams: Cell<usize>,
    }

    impl Relation for Logged {
        type Page = MemoryPage<Opaque>;
    }

    impl RelationReadTypes for Logged {
        type ReadGuard<'b> = <MemoryRelation<Opaque> as RelationReadTypes>::ReadGuard<'b>;
    }

    impl RelationRead for Logged {
        fn read(&self, id: u32) -> Self::ReadGuard<'_> {
            assert_eq!(self.live.get(), 0, "a plain read while a stream is open");
            self.pages.read(id)
        }
    }

    impl RelationPrefetch for Logged {
        fn prefetch(&self, id: u32) {
            self.log.borrow_mut().push((id, By::Prefetch));
        }
    }

    impl RelationReadStreamTypes for Logged {
        type ReadStream<'b, I: Iterator>
            = Stream<'b, I>
        where
            I::Item: Fetch<'b>;
    }

    impl RelationReadStream for Logged {
        fn read_stream<'b, I: Iterator>(&'b self, iter: I, _: Hints) -> Stream<'b, I>
        where
            I::Item: Fetch<'b>,
        {
            assert_eq!(self.live.get(), 0, "two streams open");
            self.live.set(1);
            self.streams.set(self.streams.get() + 1);
            Stream(self, Some(iter), VecDeque::new())
        }
    }

    /// Reads two items ahead and, as PostgreSQL does, never asks its iterator again once it has ended.
    struct Stream<'b, I: Iterator>(&'b Logged, Option<I>, VecDeque<I::Item>);

    impl<I: Iterator> Drop for Stream<'_, I> {
        fn drop(&mut self) {
            self.0.live.set(0);
        }
    }

    impl<'b, I: Iterator> ReadStream<'b> for Stream<'b, I>
    where
        I::Item: Fetch<'b>,
    {
        type Relation = Logged;
        type Guards = std::vec::IntoIter<<Logged as RelationReadTypes>::ReadGuard<'b>>;
        type Item = I::Item;
        type Inner = std::iter::Chain<
            std::collections::vec_deque::IntoIter<I::Item>,
            std::iter::Flatten<std::option::IntoIter<I>>,
        >;
        type Iter = I;

        fn next(&mut self) -> Option<(I::Item, Self::Guards)> {
            self.next_if(|_| true)
        }

        fn next_if<P: FnOnce(&I::Item) -> bool>(
            &mut self,
            predicate: P,
        ) -> Option<(I::Item, Self::Guards)> {
            while self.2.len() < 2
                && let Some(iter) = self.1.as_mut()
            {
                let Some(e) = iter.next() else {
                    self.1 = None;
                    break;
                };
                self.0
                    .log
                    .borrow_mut()
                    .extend(e.fetch().map(|id| (id, By::Stream)));
                self.2.push_back(e);
            }
            let e = self.2.pop_front_if(|e| predicate(e))?;
            let guards = e
                .fetch()
                .map(|id| self.0.pages.read(id))
                .collect::<Vec<_>>();
            Some((e, guards.into_iter()))
        }

        fn into_inner(mut self) -> Self::Inner {
            let (queue, iter) = (std::mem::take(&mut self.2), self.1.take());
            queue.into_iter().chain(iter.into_iter().flatten())
        }

        fn refill(&mut self, refill: impl FnOnce(&mut I) -> bool) -> bool {
            self.1.as_mut().is_some_and(refill)
        }
    }

    fn relation() -> Logged {
        Logged {
            pages: MemoryRelation::new((0..1000).map(|_| MemoryPage::new(Opaque(0)))),
            log: RefCell::default(),
            live: Cell::new(0),
            streams: Cell::new(0),
        }
    }

    /// Takes every item, checking each guard reads its own page, and returns the log.
    fn take_all(relation: &Logged, pages: &[u32]) -> Vec<(u32, By)> {
        let mut prefetcher = MixedPrefetcher::new(
            relation,
            pages.iter().copied().collect::<VecDeque<_>>(),
            Hints::default(),
        );
        let mut taken = Vec::new();
        while let Some((e, guards)) = prefetcher.next() {
            assert_eq!(guards.map(|g| g.id()).collect::<Vec<_>>(), [e]);
            taken.push(e);
        }
        assert_eq!(taken, pages);
        relation.log.take()
    }

    #[test]
    fn mixed_prefetcher() {
        let sparse = (0..20).map(|i| 3 * i);
        let pages: Vec<u32> = sparse
            .chain(100..120)
            .chain(125..145)
            .chain([300, 310, 320])
            .collect();
        let relation = relation();
        let mut prefetcher = MixedPrefetcher::new(
            &relation,
            pages.iter().copied().collect::<VecDeque<_>>(),
            Hints::default(),
        );
        assert!(prefetcher.next_if(|&e| e != 0).is_none());
        drop(prefetcher);
        relation.log.take();
        // the stream starts once half of the next `STREAM_SPAN` pages follow one another and goes on over the gap at 125
        let streamed = 13..pages.len() - 3;
        let expected = pages.iter().enumerate().map(|(i, &e)| {
            (
                e,
                if streamed.contains(&i) {
                    By::Stream
                } else {
                    By::Prefetch
                },
            )
        });
        assert_eq!(take_all(&relation, &pages), expected.collect::<Vec<_>>());
        assert_eq!(relation.streams.get(), 1);
        let mut prefetcher = MixedPrefetcher::new(
            &relation,
            pages.iter().copied().collect::<VecDeque<_>>(),
            Hints::default(),
        );
        let head = std::iter::from_fn(|| prefetcher.next_if(|&e| e < 105)).count();
        assert!(prefetcher.next_if(|&e| e < 105).is_none());
        assert_eq!(prefetcher.into_iter().collect::<Vec<_>>(), pages[head..]);
    }

    #[test]
    fn mixed_prefetcher_second_stream() {
        let scattered = (0..40).map(|i| 200 + 3 * i);
        let pages: Vec<u32> = (100..130).chain(scattered).chain(500..530).collect();
        let relation = relation();
        // a span ends at its last adjacent page; the next starts as soon as its pages are half of `STREAM_SPAN`
        let prefetched = 30..30 + 40 - (STREAM_SPAN / 2 - 1);
        let by = |i| {
            if prefetched.contains(&i) {
                By::Prefetch
            } else {
                By::Stream
            }
        };
        let expected = pages.iter().enumerate().map(|(i, &e)| (e, by(i)));
        assert_eq!(take_all(&relation, &pages), expected.collect::<Vec<_>>());
        assert_eq!(relation.streams.get(), 2);
    }
}
