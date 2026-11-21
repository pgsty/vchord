// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use crate::relation::{
    Opaque, Page, PageGuard, Relation, RelationRead, RelationReadTypes, RelationWrite,
    RelationWriteTypes,
};
use std::cell::{Ref, RefCell, RefMut};
use std::ops::{Deref, DerefMut};
use zerocopy::IntoBytes;

/// A page that counts its free space as PostgreSQL's do.
pub struct MemoryPage<O> {
    opaque: O,
    items: Vec<Option<Vec<u64>>>,
    /// Whether the page has no free space.
    pub full: bool,
}

impl<O> MemoryPage<O> {
    pub fn new(opaque: O) -> Self {
        Self {
            opaque,
            items: Vec::new(),
            full: false,
        }
    }
}

impl<O: Opaque> Page for MemoryPage<O> {
    type Opaque = O;
    fn get_opaque(&self) -> &O {
        &self.opaque
    }
    fn get_opaque_mut(&mut self) -> &mut O {
        &mut self.opaque
    }
    fn len(&self) -> u16 {
        self.items.len() as u16
    }
    fn get(&self, i: u16) -> Option<&[u8]> {
        let item = self.items.get(i as usize - 1)?.as_deref();
        item.map(IntoBytes::as_bytes)
    }
    fn get_mut(&mut self, i: u16) -> Option<&mut [u8]> {
        let item = self.items.get_mut(i as usize - 1)?.as_deref_mut();
        item.map(IntoBytes::as_mut_bytes)
    }
    fn alloc(&mut self, data: &[u8]) -> Option<u16> {
        if data.len() > self.freespace() as usize {
            return None;
        }
        let mut item = vec![0_u64; data.len().div_ceil(8)];
        item.as_mut_bytes()[..data.len()].copy_from_slice(data);
        self.items.push(Some(item));
        Some(self.len())
    }
    fn free(&mut self, i: u16) {
        self.items[i as usize - 1] = None;
    }
    fn freespace(&self) -> u16 {
        if self.full {
            return 0;
        }
        let lower = 24 + 4 * self.items.len();
        let used = self
            .items
            .iter()
            .flatten()
            .map(|x| 8 * x.len())
            .sum::<usize>();
        (8192 - size_of::<O>() - used).saturating_sub(lower + 4) as u16
    }
    fn clear(&mut self, opaque: O) {
        self.opaque = opaque;
        self.items.clear();
    }
}

pub struct Guard<T>(u32, T);

impl<T> PageGuard for Guard<T> {
    fn id(&self) -> u32 {
        self.0
    }
}

impl<O, T: Deref<Target = MemoryPage<O>>> Deref for Guard<T> {
    type Target = MemoryPage<O>;
    fn deref(&self) -> &MemoryPage<O> {
        &self.1
    }
}

impl<O, T: DerefMut<Target = MemoryPage<O>>> DerefMut for Guard<T> {
    fn deref_mut(&mut self) -> &mut MemoryPage<O> {
        &mut self.1
    }
}

/// Boxed, so that pages stay put as more are added.
#[allow(clippy::vec_box)]
pub struct MemoryRelation<O>(RefCell<Vec<Box<RefCell<MemoryPage<O>>>>>);

impl<O> MemoryRelation<O> {
    pub fn new(pages: impl IntoIterator<Item = MemoryPage<O>>) -> Self {
        let pages = pages.into_iter().map(|x| Box::new(RefCell::new(x)));
        Self(RefCell::new(pages.collect()))
    }
    pub fn pages(&self) -> u32 {
        self.0.borrow().len() as u32
    }
    pub fn page(&self, id: u32) -> &RefCell<MemoryPage<O>> {
        let page: *const RefCell<MemoryPage<O>> = &*self.0.borrow()[id as usize];
        // Safety: the box lives as long as the relation
        #[allow(unsafe_code)]
        unsafe {
            &*page
        }
    }
}

impl<O: Opaque> Relation for MemoryRelation<O> {
    type Page = MemoryPage<O>;
}

impl<O: Opaque> RelationReadTypes for MemoryRelation<O> {
    type ReadGuard<'b> = Guard<Ref<'b, MemoryPage<O>>>;
}

impl<O: Opaque> RelationRead for MemoryRelation<O> {
    fn read(&self, id: u32) -> Self::ReadGuard<'_> {
        Guard(id, self.page(id).borrow())
    }
}

impl<O: Opaque> RelationWriteTypes for MemoryRelation<O> {
    type WriteGuard<'b> = Guard<RefMut<'b, MemoryPage<O>>>;
}

impl<O: Opaque> RelationWrite for MemoryRelation<O> {
    fn write(&self, id: u32, _: bool) -> Self::WriteGuard<'_> {
        Guard(id, self.page(id).borrow_mut())
    }
    fn extend(&self, opaque: O, _: bool) -> Self::WriteGuard<'_> {
        let id = {
            let mut pages = self.0.borrow_mut();
            pages.push(Box::new(RefCell::new(MemoryPage::new(opaque))));
            pages.len() as u32 - 1
        };
        self.write(id, false)
    }
    fn search(&self, _: usize) -> Option<Self::WriteGuard<'_>> {
        None
    }
    fn commit_pair(&self, _: Self::WriteGuard<'_>, _: Self::WriteGuard<'_>) {}
}

/// A plain prefetcher family, for tests.
pub struct MakePlainPrefetcher<'b, R> {
    pub index: &'b R,
}

impl<'b, R: RelationRead + 'b> crate::prefetcher::PrefetcherSequenceFamily<'b, R>
    for MakePlainPrefetcher<'b, R>
{
    type P<S: crate::prefetcher::Sequence>
        = crate::prefetcher::PlainPrefetcher<'b, R, S>
    where
        S::Item: crate::fetch::Fetch<'b>;
    fn prefetch<S: crate::prefetcher::Sequence>(&mut self, seq: S) -> Self::P<S>
    where
        S::Item: crate::fetch::Fetch<'b>,
    {
        crate::prefetcher::PlainPrefetcher::new(self.index, seq)
    }
    fn is_not_plain(&self) -> bool {
        false
    }
}

impl<'b, R: RelationRead + 'b> crate::prefetcher::PrefetcherHeapFamily<'b, R>
    for MakePlainPrefetcher<'b, R>
{
    type P<T>
        = crate::prefetcher::PlainPrefetcher<'b, R, std::collections::BinaryHeap<T>>
    where
        T: Ord + crate::fetch::Fetch<'b>;
    fn prefetch<T>(&mut self, seq: Vec<T>) -> Self::P<T>
    where
        T: Ord + crate::fetch::Fetch<'b>,
    {
        crate::prefetcher::PlainPrefetcher::new(self.index, seq.into())
    }
    fn is_not_plain(&self) -> bool {
        false
    }
}
