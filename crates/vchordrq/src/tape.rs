// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use crate::keys::Columns;
use crate::tuples::*;
use crate::{Opaque, freepages};
use index::prefetcher::{Prefetcher, PrefetcherSequenceFamily};
use index::relation::{Page, PageGuard, RelationRead, RelationWrite};
use index_accessor::Accessor1;
use std::marker::PhantomData;
use std::num::NonZero;
use std::ops::Range;

/// Writes a new tape page by page.
///
/// When the head page is full, it is linked to a freshly extended page and its
/// WAL record is written before the new page's record. So the tape must stay
/// unreachable from readers until the writer is dropped: either the index is
/// still being built, or (in `maintain`) the tape is published by updating the
/// jump tuple only after every writer of it has been dropped. Appending to a
/// reachable tape goes through [`append`] instead.
pub struct TapeWriter<'a, R, T>
where
    R: RelationWrite + 'a,
{
    /// The last page, unless unlocked.
    head: Option<R::WriteGuard<'a>>,
    last: u32,
    first: u32,
    index: &'a R,
    tracking_freespace: bool,
    _phantom: PhantomData<fn(T) -> T>,
}

impl<'a, R, T> TapeWriter<'a, R, T>
where
    R: RelationWrite + 'a,
    R::Page: Page<Opaque = Opaque>,
{
    pub fn create(index: &'a R, tracking_freespace: bool) -> Self {
        let mut head = index.extend(
            Opaque {
                next: u32::MAX,
                skip: u32::MAX,
            },
            tracking_freespace,
        );
        head.get_opaque_mut().skip = head.id();
        let first = head.id();
        Self {
            head: Some(head),
            last: first,
            first,
            index,
            tracking_freespace,
            _phantom: PhantomData,
        }
    }
    pub fn first(&self) -> u32 {
        self.first
    }
    pub fn last(&self) -> u32 {
        self.last
    }
    /// Unlocks the last page; the next write locks it again.
    pub fn unlock(&mut self) {
        self.head = None;
    }
    fn head(&mut self) -> &mut R::WriteGuard<'a> {
        self.head
            .get_or_insert_with(|| self.index.write(self.last, self.tracking_freespace))
    }
    pub fn freespace(&mut self) -> u16 {
        self.head().freespace()
    }
    pub fn tape_move(&mut self) {
        if self.head().len() == 0 {
            panic!("implementation: a clear page cannot accommodate a single tuple");
        }
        let next = self.index.extend(
            Opaque {
                next: u32::MAX,
                skip: u32::MAX,
            },
            self.tracking_freespace,
        );
        self.head().get_opaque_mut().next = next.id();
        self.last = next.id();
        self.head = Some(next);
    }
}

impl<'a, R, T> TapeWriter<'a, R, T>
where
    R: RelationWrite + 'a,
    R::Page: Page<Opaque = Opaque>,
    T: Tuple,
{
    pub fn push(&mut self, x: T) -> (u32, u16) {
        let bytes = T::serialize(&x);
        if let Some(i) = self.head().alloc(&bytes) {
            (self.last, i)
        } else {
            self.tape_move();
            if let Some(i) = self.head().alloc(&bytes) {
                (self.last, i)
            } else {
                panic!("implementation: a free page cannot accommodate a single tuple")
            }
        }
    }
    pub fn tape_put(&mut self, x: T) -> (u32, u16) {
        let bytes = T::serialize(&x);
        if let Some(i) = self.head().alloc(&bytes) {
            (self.last, i)
        } else {
            panic!("implementation: a free page cannot accommodate a single tuple")
        }
    }
}

impl<'a, R, T> TapeWriter<'a, R, T>
where
    R: RelationWrite + 'a,
    R::Page: Page<Opaque = Opaque>,
    T: Tuple,
{
    /// Writes `last`, its elements in parts on the pages it does not fit.
    pub fn push_parts(&mut self, mut last: impl Parts<Tuple = T>) {
        loop {
            let freespace = self.freespace();
            if last.size() <= freespace as usize {
                self.tape_put(last.tuple());
                return;
            }
            let n = last.elements().len();
            match last.fit(freespace).map(|w| w.min(n)) {
                // A part without elements would only fill the page's tail.
                Some(w) if w > 0 => {
                    let rest = last.elements().split_off(w);
                    let part = std::mem::replace(last.elements(), rest);
                    self.tape_put(last.part(part));
                }
                _ => self.tape_move(),
            }
        }
    }
}

/// `pages`, with `between` run before each is read.
pub fn paced<I: Iterator>(
    mut pages: I,
    mut between: impl FnMut(),
) -> impl Iterator<Item = I::Item> {
    std::iter::from_fn(move || {
        between();
        pages.next()
    })
}

pub fn by_directory<'b, R>(
    p: &mut impl PrefetcherSequenceFamily<'b, R>,
    iter: impl Iterator<Item = u32>,
) -> impl Iterator<Item = R::ReadGuard<'b>>
where
    R: RelationRead + 'b,
{
    let mut t = p.prefetch(iter.peekable());
    std::iter::from_fn(move || {
        let (_, mut x) = t.next()?;
        let ret = x.next().expect("should be at least one element");
        assert!(x.next().is_none(), "should be at most one element");
        Some(ret)
    })
}

pub fn by_next<'b, R>(index: &'b R, first: u32) -> impl Iterator<Item = R::ReadGuard<'b>>
where
    R: RelationRead + 'b,
    R::Page: Page<Opaque = Opaque>,
{
    let mut current = first;
    std::iter::from_fn(move || {
        if current != u32::MAX {
            let guard = index.read(current);
            current = guard.get_opaque().next;
            Some(guard)
        } else {
            None
        }
    })
}

pub fn read_h1_tape<'b, R, A, T>(
    iter: impl Iterator<Item = R::ReadGuard<'b>>,
    accessor: impl Fn() -> A,
    mut callback: impl for<'a> FnMut(T, u16, f32, u32, &'a [u32]),
) where
    R: RelationRead + 'b,
    A: for<'a> Accessor1<[u8; 16], (&'a [[f32; 32]; 4], &'a [f32; 32]), Output = [T; 32]>,
{
    let mut x = None;
    for guard in iter {
        for i in 1..=guard.len() {
            let bytes = guard.get(i).expect("data corruption");
            let tuple = H1Tuple::deserialize_ref(bytes);
            match tuple {
                H1TupleReader::_0(tuple) => {
                    let mut x = x.take().unwrap_or_else(&accessor);
                    x.push(tuple.elements());
                    let values = x.finish((tuple.metadata(), tuple.delta()));
                    let prefetch = tuple.prefetch();
                    let flattened = prefetch.as_flattened();
                    let step = prefetch.len();
                    for (j, value) in values.into_iter().enumerate() {
                        if j < tuple.len() as usize {
                            callback(
                                value,
                                tuple.head()[j],
                                tuple.norm()[j],
                                tuple.first()[j],
                                &flattened[j * step..][..step],
                            );
                        }
                    }
                }
                H1TupleReader::_1(tuple) => {
                    x.get_or_insert_with(&accessor).push(tuple.elements());
                }
            }
        }
    }
}

/// Selects the rows a tape reader passes on; `()` selects every row.
pub trait Filter: Copy {
    /// Whether to read only the blocks that [`Filter::meets`] selects ([`plan`]).
    fn selects_blocks(self) -> bool;
    /// The blocks that may meet the filter: no other does.
    fn span(self, summaries: Summaries<'_>) -> Range<usize>;
    /// The blocks of `blocks`, within one word, whose summaries meet the filter, a bit each.
    fn meets(self, summaries: Summaries<'_>, blocks: Range<usize>) -> u32;
    /// The rows of a frozen block to pass on.
    fn block(self, columns: &[u8]) -> u32;
    fn row(self, columns: Columns<'_>) -> bool;
}

impl Filter for () {
    #[inline(always)]
    fn selects_blocks(self) -> bool {
        false
    }

    #[inline(always)]
    fn span(self, summaries: Summaries<'_>) -> Range<usize> {
        0..summaries.blocks()
    }

    #[inline(always)]
    fn meets(self, _: Summaries<'_>, _: Range<usize>) -> u32 {
        u32::MAX
    }

    #[inline(always)]
    fn block(self, _: &[u8]) -> u32 {
        u32::MAX
    }

    #[inline(always)]
    fn row(self, _: Columns<'_>) -> bool {
        true
    }
}

/// A row that a tape reader passes on.
pub struct Row<'a, T> {
    pub value: T,
    pub head: u16,
    pub payload: NonZero<u64>,
    pub prefetch: &'a [u32],
    pub columns: Columns<'a>,
}

/// The blocks of a frozen tape that a reader reads, over the pages it is given.
pub trait Blocks {
    /// The block that the first tuple of the next page belongs to.
    fn page(&mut self) -> u32;
    fn contains(&self, b: u32) -> bool;
}

pub struct All;

impl Blocks for All {
    #[inline(always)]
    fn page(&mut self) -> u32 {
        0
    }

    #[inline(always)]
    fn contains(&self, _: u32) -> bool {
        true
    }
}

pub struct Selected<'a> {
    firsts: std::slice::Iter<'a, u32>,
    masks: &'a [u32],
}

impl<'a> Selected<'a> {
    pub fn new(firsts: &'a [u32], masks: &'a [u32]) -> Self {
        Self {
            firsts: firsts.iter(),
            masks,
        }
    }
}

impl Blocks for Selected<'_> {
    #[inline(always)]
    fn page(&mut self) -> u32 {
        *self.firsts.next().unwrap()
    }

    #[inline(always)]
    fn contains(&self, b: u32) -> bool {
        self.masks[b as usize / 32] >> (b % 32) & 1 != 0
    }
}

/// Reads the selected rows; `pages` must hold every part of a selected block.
pub fn read_frozen_tape<'b, R, A, T>(
    pages: impl Iterator<Item = R::ReadGuard<'b>>,
    mut blocks: impl Blocks,
    accessor: impl Fn() -> A,
    filter: impl Filter,
    mut callback: impl for<'a> FnMut(Row<'a, T>),
) where
    R: RelationRead + 'b,
    A: for<'a> Accessor1<[u8; 16], (&'a [[f32; 32]; 4], &'a [f32; 32]), Output = [T; 32]>,
{
    let mut x = None;
    for guard in pages {
        let mut b = blocks.page();
        for i in 1..=guard.len() {
            let selected = blocks.contains(b);
            let bytes = guard.get(i).expect("data corruption");
            let tuple = FrozenTuple::deserialize_ref(bytes);
            match tuple {
                FrozenTupleReader::_0(tuple) => {
                    b += 1;
                    let mask = if selected {
                        filter.block(tuple.columns())
                    } else {
                        0
                    };
                    if mask == 0 {
                        // Skip the block, with the parts of it pushed so far.
                        x = None;
                        continue;
                    }
                    let columns = tuple.columns();
                    let mut x = x.take().unwrap_or_else(&accessor);
                    x.push(tuple.elements());
                    let values = x.finish((tuple.metadata(), tuple.delta()));
                    let prefetch = tuple.prefetch();
                    let flattened = prefetch.as_flattened();
                    let step = prefetch.len();
                    for (j, value) in values.into_iter().enumerate() {
                        if mask & (1 << j) != 0
                            && let Some(payload) = tuple.payload()[j]
                        {
                            callback(Row {
                                value,
                                head: tuple.head()[j],
                                payload,
                                prefetch: &flattened[j * step..][..step],
                                columns: Columns::block(columns, j),
                            });
                        }
                    }
                }
                FrozenTupleReader::_1(tuple) => {
                    if selected {
                        x.get_or_insert_with(&accessor).push(tuple.elements());
                    }
                }
            }
        }
    }
}

/// Marks a page's first block as begun on an earlier page.
pub const CONTINUED: u32 = 1 << 31;

/// Marks a layout whose first summary's words never decrease from block to block.
pub const SORTED: u32 = 1 << 31;

/// A summary's word, from its low and its high half.
#[inline(always)]
pub fn wide([lo, hi]: [u32; 2]) -> u64 {
    lo as u64 | (hi as u64) << 32
}

/// Per summarized key, every block's first summary word, then every block's second.
#[derive(Debug, Clone, Copy)]
pub struct Summaries<'a> {
    words: &'a [[u32; 2]],
    blocks: usize,
    sorted: bool,
}

impl<'a> Summaries<'a> {
    pub fn new(words: &'a [u32], blocks: usize, sorted: bool) -> Self {
        let words = words.as_chunks().0;
        Self {
            words,
            blocks,
            sorted,
        }
    }
    #[inline(always)]
    pub fn blocks(self) -> usize {
        self.blocks
    }
    /// Whether the first summary's words never decrease from block to block.
    #[inline(always)]
    pub fn sorted(self) -> bool {
        self.sorted
    }
    /// Word `i` of every block's summary of the `k`-th summarized key.
    #[inline(always)]
    pub fn words(self, k: usize, i: usize) -> &'a [[u32; 2]] {
        &self.words[(2 * k + i) * self.blocks..][..self.blocks]
    }
}

/// The page ids and the layout of a frozen tape, as its pages and tuples are read.
#[derive(Debug, Default)]
pub struct Layout {
    pub ids: Vec<u32>,
    firsts: Vec<u32>,
    /// Per summarized key, per block, its summary.
    summaries: Vec<Vec<[u64; 2]>>,
    blocks: u32,
    continued: bool,
}

impl Layout {
    pub fn new(summaries: usize) -> Self {
        Self {
            summaries: vec![Vec::new(); summaries],
            ..Default::default()
        }
    }
    pub fn page(&mut self, id: u32) {
        let continued = if self.continued { CONTINUED } else { 0 };
        self.ids.push(id);
        self.firsts.push(self.blocks | continued);
    }
    /// A block's last part with its summaries, or `None` for a part before it.
    pub fn tuple(&mut self, summaries: Option<impl IntoIterator<Item = [u64; 2]>>) {
        self.continued = summaries.is_none();
        if let Some(summaries) = summaries {
            std::iter::zip(&mut self.summaries, summaries).for_each(|(x, y)| x.push(y));
            self.blocks += 1;
        }
    }
    /// The words after the page ids: per page its first block, the block count, then the summaries.
    pub fn words(&self) -> Vec<u32> {
        let sorted = (self.summaries.first())
            .is_some_and(|x| x.is_sorted_by(|a, b| a[0] <= b[0] && a[1] <= b[1]));
        let mut words = self.firsts.clone();
        words.push(self.blocks | if sorted { SORTED } else { 0 });
        for summaries in &self.summaries {
            let word =
                |i: usize| (summaries.iter()).flat_map(move |x| [x[i] as u32, (x[i] >> 32) as u32]);
            words.extend(word(0).chain(word(1)));
        }
        words
    }
}

/// Reads the page ids, and with `layout` the layout, into `words`; returns the page count.
pub fn read_directory_words<'b, R>(
    iter: impl Iterator<Item = R::ReadGuard<'b>>,
    words: &mut Vec<u32>,
    layout: bool,
) -> usize
where
    R: RelationRead + 'b,
{
    words.clear();
    let mut pages = 0;
    for guard in iter {
        for i in 1..=guard.len() {
            let bytes = guard.get(i).expect("data corruption");
            let (elements, is_id) = match DirectoryTuple::deserialize_ref(bytes) {
                DirectoryTupleReader::_0(tuple) => (tuple.elements(), true),
                DirectoryTupleReader::_1(tuple) => (tuple.elements(), true),
                DirectoryTupleReader::_2(tuple) => (tuple.elements(), false),
            };
            if !is_id && !layout {
                // the layout follows every page id
                return pages;
            }
            words.extend_from_slice(elements);
            pages += if is_id { elements.len() } else { 0 };
        }
    }
    pages
}

/// Reads a directory into `words`; returns its page count and, if `filter` selects blocks, `plan`.
pub fn read_directory<'b, R>(
    iter: impl Iterator<Item = R::ReadGuard<'b>>,
    words: &mut Vec<u32>,
    filter: impl Filter,
) -> (usize, Option<usize>)
where
    R: RelationRead + 'b,
{
    let pages = read_directory_words::<R>(iter, words, filter.selects_blocks());
    let selected = filter.selects_blocks().then(|| plan(words, pages, filter));
    (pages, selected.flatten())
}

/// The pages to read and their selected blocks, or `None` to read every page.
fn plan(words: &mut [u32], pages: usize, filter: impl Filter) -> Option<usize> {
    let (ids, layout) = words.split_at_mut(pages);
    let (firsts, rest) = layout.split_at_mut_checked(pages)?;
    let &header = rest.first()?;
    let blocks = (header & !SORTED) as usize;
    let masks = select_blocks(filter, rest, blocks, header & SORTED != 0);
    if masks.iter().map(|x| x.count_ones() as usize).sum::<usize>() == blocks {
        return None;
    }
    Some(select_pages(ids, firsts, masks, blocks))
}

/// Replaces the block count and the summaries with the bitmask of the blocks `filter` selects.
fn select_blocks(filter: impl Filter, rest: &mut [u32], blocks: usize, sorted: bool) -> &[u32] {
    let span = filter.span(Summaries::new(&rest[1..], blocks, sorted));
    let n = blocks.div_ceil(32);
    for i in 0..n {
        let word = span.start.max(32 * i)..span.end.min(32 * i + 32);
        let mask = if word.is_empty() {
            0
        } else {
            let bits = u32::MAX >> (32 - word.len()) << (word.start % 32);
            filter.meets(Summaries::new(&rest[1..], blocks, sorted), word) & bits
        };
        // word `i` is written once its blocks are read; later blocks' summaries lie past it
        rest[i] = mask;
    }
    &rest[..n]
}

/// Keeps the pages that hold a selected block's part, in place; returns how many.
fn select_pages(ids: &mut [u32], firsts: &mut [u32], masks: &[u32], blocks: usize) -> usize {
    let n = ids.len();
    let mut kept = 0;
    for i in 0..n {
        let first = firsts[i] & !CONTINUED;
        // the blocks with a part on page `i` are `first..end`
        let end = match firsts.get(i + 1) {
            Some(&next) => (next & !CONTINUED) + (next >> 31),
            None => blocks as u32,
        };
        if (first..end).any(|j| masks[j as usize / 32] >> (j % 32) & 1 != 0) {
            ids[kept] = ids[i];
            firsts[kept] = first;
            kept += 1;
        }
    }
    kept
}

/// `skip`: how many tuples of the first page are read already.
pub fn read_appendable_tape<'b, R, T>(
    iter: impl Iterator<Item = R::ReadGuard<'b>>,
    mut skip: u16,
    mut access: impl for<'a> FnMut([f32; 4], &'a [u64], f32) -> T,
    filter: impl Filter,
    mut callback: impl for<'a> FnMut(Row<'a, T>),
) where
    R: RelationRead + 'b,
{
    for guard in iter {
        for i in std::mem::take(&mut skip) + 1..=guard.len() {
            let bytes = guard.get(i).expect("data corruption");
            let tuple = AppendableTuple::deserialize_ref(bytes);
            let columns = Columns::row(tuple.columns());
            if let Some(payload) = tuple.payload()
                && filter.row(columns)
            {
                let value = access(tuple.metadata(), tuple.elements(), tuple.delta());
                callback(Row {
                    value,
                    head: tuple.head(),
                    payload,
                    prefetch: tuple.prefetch(),
                    columns,
                });
            }
        }
    }
}

#[allow(clippy::collapsible_else_if)]
pub fn append<R: RelationRead + RelationWrite>(
    index: &R,
    first: u32,
    bytes: &[u8],
    tracking_freespace: bool,
    freepages_first: Option<u32>,
) -> (u32, u16)
where
    R::Page: Page<Opaque = Opaque>,
{
    assert!(!tracking_freespace || freepages_first.is_none());
    assert!(first != u32::MAX);
    let mut current = first;
    loop {
        let read = index.read(current);
        if read.freespace() as usize >= bytes.len() || read.get_opaque().next == u32::MAX {
            drop(read);
            let mut write = index.write(current, tracking_freespace);
            if write.get_opaque().next == u32::MAX {
                if let Some(i) = write.alloc(bytes) {
                    return (current, i);
                }
                let mut extend = {
                    if let Some(freepages_first) = freepages_first {
                        if let Some(mut guard) = freepages::alloc(index, freepages_first) {
                            guard.clear(Opaque {
                                next: u32::MAX,
                                skip: u32::MAX,
                            });
                            guard
                        } else {
                            index.extend(
                                Opaque {
                                    next: u32::MAX,
                                    skip: u32::MAX,
                                },
                                tracking_freespace,
                            )
                        }
                    } else {
                        index.extend(
                            Opaque {
                                next: u32::MAX,
                                skip: u32::MAX,
                            },
                            tracking_freespace,
                        )
                    }
                };
                let fresh = extend.id();
                let Some(i) = extend.alloc(bytes) else {
                    panic!("implementation: a clear page cannot accommodate a single tuple");
                };
                if tracking_freespace {
                    // FSM allocation bypasses the tail lock. Publish the new
                    // vector page and its incoming link in one WAL record,
                    // before making its free space available to other writers.
                    write.get_opaque_mut().next = fresh;
                    index.commit_pair(write, extend);
                } else {
                    // These pages are not allocated through the FSM. Log the
                    // new page before exposing it through the locked tail.
                    drop(extend);
                    write.get_opaque_mut().next = fresh;
                    drop(write);
                }
                let mut past = index.write(first, tracking_freespace);
                past.get_opaque_mut().skip = fresh.max(past.get_opaque().skip);
                return (fresh, i);
            }
            if current == first && write.get_opaque().skip != first {
                current = write.get_opaque().skip;
            } else {
                current = write.get_opaque().next;
            }
        } else {
            if current == first && read.get_opaque().skip != first {
                current = read.get_opaque().skip;
            } else {
                current = read.get_opaque().next;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keys::{Condition, Intervals, KeyFilter, Keys, Summary, Width};

    fn filter(points: Vec<u64>) -> KeyFilter {
        let keys = Keys::new([(Width::W8, false, Some(Summary::Bounds))]);
        let values = Intervals::points(points.into_iter().map(u128::from).collect());
        KeyFilter::new(
            &keys,
            vec![Some(Condition {
                values,
                null: false,
                exact: true,
            })],
        )
        .unwrap()
    }

    /// Per page, its tuples as (block, whether the block's last), and the blocks' zone maps.
    fn tape(sorted: bool) -> (Vec<Vec<(usize, bool)>>, Vec<[u64; 2]>) {
        let blocks = rand::random::<u32>() as usize % 100;
        let mut pages = vec![Vec::new()];
        for j in 0..blocks {
            let n = 1 + rand::random::<u32>() as usize % 4;
            for k in 0..n {
                if !pages.last().unwrap().is_empty() && rand::random::<u8>() % 3 == 0 {
                    pages.push(Vec::new());
                }
                pages.last_mut().unwrap().push((j, k == n - 1));
            }
        }
        let mut bounds = (0..2 * blocks)
            .map(|_| rand::random::<u64>() % 1000)
            .collect::<Vec<_>>();
        if sorted {
            bounds.sort_unstable();
        }
        let zones = (bounds.chunks(2))
            .map(|x| [x[0].min(x[1]), x[0].max(x[1])])
            .collect();
        (pages, zones)
    }

    fn directory(pages: &[Vec<(usize, bool)>], zones: &[[u64; 2]], layout: bool) -> Vec<u32> {
        let mut x = Layout::new(1);
        for (i, page) in pages.iter().enumerate() {
            x.page(7000 + i as u32);
            for &(j, last) in page {
                x.tuple(last.then(|| [zones[j]]));
            }
        }
        let mut words = x.ids.clone();
        if layout {
            words.extend(x.words());
        }
        words
    }

    #[test]
    fn layout() {
        for i in 0..if cfg!(not(miri)) { 512 } else { 2 } {
            let (pages, zones) = tape(i % 2 == 0);
            let wanted = (0..1 + rand::random::<u32>() % 12)
                .map(|_| rand::random::<u64>() % 1000)
                .collect::<Vec<_>>();
            let filter = filter(wanted.clone());
            let meets = |[min, max]: [u64; 2]| wanted.iter().any(|&t| min <= t && t <= max);
            let mut words = directory(&pages, &zones, true);
            let n = pages.len();
            if i % 2 == 0 {
                assert!(words[2 * n] & SORTED != 0);
            }
            let Some(kept) = plan(&mut words, n, &filter) else {
                assert!(zones.iter().all(|&r| meets(r)));
                continue;
            };
            let (ids, layout) = words.split_at(n);
            let (firsts, masks) = layout.split_at(n);
            for (j, &r) in zones.iter().enumerate() {
                assert_eq!(masks[j / 32] >> (j % 32) & 1 != 0, meets(r));
            }
            let expected = (0..n)
                .filter(|&i| pages[i].iter().any(|&(j, _)| meets(zones[j])))
                .map(|i| 7000 + i as u32)
                .collect::<Vec<_>>();
            assert_eq!(&ids[..kept], expected);
            for (&id, &first) in std::iter::zip(&ids[..kept], &firsts[..kept]) {
                let mut j = first as usize;
                for &(block, last) in &pages[(id - 7000) as usize] {
                    assert_eq!(j, block);
                    j += last as usize;
                }
            }
        }
    }

    #[test]
    fn layout_all() {
        for _ in 0..if cfg!(not(miri)) { 64 } else { 1 } {
            let (pages, zones) = tape(true);
            let n = pages.len();
            let some = filter(vec![rand::random::<u64>() % 1000]);
            let mut words = directory(&pages, &zones, false);
            assert_eq!(plan(&mut words, n, &some), None);
            let all = filter((0..1000).collect());
            let mut words = directory(&pages, &zones, true);
            assert_eq!(plan(&mut words, n, &all), None);
        }
    }
}
