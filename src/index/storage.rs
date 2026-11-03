// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

pub mod buffered;

use index::fetch::Fetch;
use index::relation::{
    Hints, Opaque, Page, PageGuard, ReadStream, Relation, RelationPrefetch, RelationRead,
    RelationReadStream, RelationReadStreamTypes, RelationReadTypes, RelationWrite,
    RelationWriteTypes,
};
use std::collections::VecDeque;
use std::iter::{Chain, Flatten};
use std::marker::PhantomData;
use std::mem::{MaybeUninit, offset_of};
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

#[repr(C, align(8))]
#[derive(Debug)]
pub struct PostgresPage<O> {
    header: pgrx::pg_sys::PageHeaderData,
    content: [u8; pgrx::pg_sys::BLCKSZ as usize - size_of::<pgrx::pg_sys::PageHeaderData>()],
    _opaque: PhantomData<fn(O) -> O>,
}

// It is a non-guaranteed detection.
// If `PageHeaderData` contains padding bytes, const-eval probably fails.
const _: () = {
    use pgrx::pg_sys::PageHeaderData as T;
    use std::mem::{transmute, zeroed};
    const _ZERO: &[u8; size_of::<T>()] = unsafe { transmute(&zeroed::<T>()) };
};

// Layout checks of header.
const _: () = {
    use pgrx::pg_sys::{MAXIMUM_ALIGNOF, PageHeaderData as T};
    assert!(size_of::<T>() == offset_of!(T, pd_linp));
    assert!(size_of::<T>() % MAXIMUM_ALIGNOF as usize == 0);
};

const _: () = assert!(align_of::<PostgresPage<()>>() == pgrx::pg_sys::MAXIMUM_ALIGNOF as usize);
const _: () = assert!(size_of::<PostgresPage<()>>() == pgrx::pg_sys::BLCKSZ as usize);

impl<O: Opaque> PostgresPage<O> {
    pub fn clone_into_boxed(&self) -> Box<Self> {
        let mut result = Box::new_uninit();
        unsafe {
            std::ptr::copy(self as *const Self, result.as_mut_ptr(), 1);
            result.assume_init()
        }
    }
}

impl<O: Opaque> Page for PostgresPage<O> {
    type Opaque = O;
    fn get_opaque(&self) -> &O {
        assert!(self.header.pd_special as usize + size_of::<O>() == size_of::<Self>());
        unsafe { &*((self as *const _ as *const O).byte_add(self.header.pd_special as _)) }
    }
    fn get_opaque_mut(&mut self) -> &mut O {
        assert!(self.header.pd_special as usize + size_of::<O>() == size_of::<Self>());
        unsafe { &mut *((self as *mut _ as *mut O).byte_add(self.header.pd_special as _)) }
    }
    fn len(&self) -> u16 {
        use pgrx::pg_sys::{ItemIdData, PageHeaderData};
        assert!(self.header.pd_lower as usize <= size_of::<Self>());
        assert!(self.header.pd_upper as usize <= size_of::<Self>());
        let lower = self.header.pd_lower as usize;
        let upper = self.header.pd_upper as usize;
        assert!(offset_of!(PageHeaderData, pd_linp) <= lower && lower <= upper);
        ((lower - offset_of!(PageHeaderData, pd_linp)) / size_of::<ItemIdData>()) as u16
    }
    fn get(&self, i: u16) -> Option<&[u8]> {
        use pgrx::pg_sys::{ItemIdData, PageHeaderData};
        if i == 0 {
            return None;
        }
        assert!(self.header.pd_lower as usize <= size_of::<Self>());
        let lower = self.header.pd_lower as usize;
        assert!(offset_of!(PageHeaderData, pd_linp) <= lower);
        let n = ((lower - offset_of!(PageHeaderData, pd_linp)) / size_of::<ItemIdData>()) as u16;
        if i > n {
            return None;
        }
        let iid = unsafe { self.header.pd_linp.as_ptr().add((i - 1) as _).read() };
        let lp_off = iid.lp_off() as usize;
        let lp_len = iid.lp_len() as usize;
        match lp_flags(iid) {
            pgrx::pg_sys::LP_UNUSED => return None,
            pgrx::pg_sys::LP_NORMAL => (),
            pgrx::pg_sys::LP_REDIRECT => unimplemented!(),
            pgrx::pg_sys::LP_DEAD => unimplemented!(),
            _ => unreachable!(),
        }
        assert!(offset_of!(PageHeaderData, pd_linp) <= lp_off);
        assert!(lp_off <= size_of::<Self>());
        assert!(lp_len <= size_of::<Self>());
        assert!(lp_off + lp_len <= size_of::<Self>());
        unsafe {
            let ptr = (self as *const Self).cast::<u8>().add(lp_off as _);
            Some(std::slice::from_raw_parts(ptr, lp_len as _))
        }
    }
    fn get_mut(&mut self, i: u16) -> Option<&mut [u8]> {
        use pgrx::pg_sys::{ItemIdData, PageHeaderData};
        if i == 0 {
            return None;
        }
        assert!(self.header.pd_lower as usize <= size_of::<Self>());
        let lower = self.header.pd_lower as usize;
        assert!(offset_of!(PageHeaderData, pd_linp) <= lower);
        let n = ((lower - offset_of!(PageHeaderData, pd_linp)) / size_of::<ItemIdData>()) as u16;
        if i > n {
            return None;
        }
        let iid = unsafe { self.header.pd_linp.as_ptr().add((i - 1) as _).read() };
        let lp_off = iid.lp_off() as usize;
        let lp_len = iid.lp_len() as usize;
        match lp_flags(iid) {
            pgrx::pg_sys::LP_UNUSED => return None,
            pgrx::pg_sys::LP_NORMAL => (),
            pgrx::pg_sys::LP_REDIRECT => unimplemented!(),
            pgrx::pg_sys::LP_DEAD => unimplemented!(),
            _ => unreachable!(),
        }
        assert!(offset_of!(PageHeaderData, pd_linp) <= lp_off);
        assert!(lp_off <= size_of::<Self>());
        assert!(lp_len <= size_of::<Self>());
        assert!(lp_off + lp_len <= size_of::<Self>());
        unsafe {
            let ptr = (self as *mut Self).cast::<u8>().add(lp_off as _);
            Some(std::slice::from_raw_parts_mut(ptr, lp_len as _))
        }
    }
    fn alloc(&mut self, data: &[u8]) -> Option<u16> {
        unsafe {
            let i = pgrx::pg_sys::PageAddItemExtended(
                (self as *const Self).cast_mut().cast(),
                data.as_ptr().cast_mut().cast(),
                data.len(),
                0,
                0,
            );
            if i == 0 { None } else { Some(i) }
        }
    }
    fn free(&mut self, i: u16) {
        unsafe {
            pgrx::pg_sys::PageIndexTupleDeleteNoCompact((self as *mut Self).cast(), i);
        }
    }
    fn freespace(&self) -> u16 {
        unsafe { pgrx::pg_sys::PageGetFreeSpace((self as *const Self).cast_mut().cast()) as u16 }
    }
    fn clear(&mut self, opaque: O) {
        unsafe {
            page_init(self, opaque);
        }
    }
}

unsafe fn page_init<O: Opaque>(this: *mut PostgresPage<O>, opaque: O) {
    unsafe {
        use pgrx::pg_sys::{BLCKSZ, PageHeaderData, PageInit};
        PageInit(this.cast(), BLCKSZ as usize, size_of::<O>());
        assert_eq!(
            (*this.cast::<PageHeaderData>()).pd_special as usize + size_of::<O>(),
            size_of::<PostgresPage<O>>()
        );
        this.cast::<O>()
            .byte_add(size_of::<PostgresPage<O>>() - size_of::<O>())
            .write(opaque);
    }
}

pub struct PostgresBufferReadGuard<Opaque> {
    buf: i32,
    page: NonNull<PostgresPage<Opaque>>,
    id: u32,
}

impl<Opaque> PageGuard for PostgresBufferReadGuard<Opaque> {
    fn id(&self) -> u32 {
        self.id
    }
}

impl<Opaque> Deref for PostgresBufferReadGuard<Opaque> {
    type Target = PostgresPage<Opaque>;

    fn deref(&self) -> &PostgresPage<Opaque> {
        unsafe { self.page.as_ref() }
    }
}

impl<Opaque> Drop for PostgresBufferReadGuard<Opaque> {
    fn drop(&mut self) {
        if crate::index::cleanup::pg_cleanup_allowed() {
            unsafe {
                pgrx::pg_sys::UnlockReleaseBuffer(self.buf);
            }
        }
    }
}

pub struct PostgresBufferWriteGuard<O: Opaque> {
    raw: pgrx::pg_sys::Relation,
    buf: i32,
    page: NonNull<PostgresPage<O>>,
    state: *mut pgrx::pg_sys::GenericXLogState,
    id: u32,
    tracking_freespace: bool,
}

impl<O: Opaque> PageGuard for PostgresBufferWriteGuard<O> {
    fn id(&self) -> u32 {
        self.id
    }
}

impl<O: Opaque> Deref for PostgresBufferWriteGuard<O> {
    type Target = PostgresPage<O>;

    fn deref(&self) -> &PostgresPage<O> {
        unsafe { self.page.as_ref() }
    }
}

impl<O: Opaque> DerefMut for PostgresBufferWriteGuard<O> {
    fn deref_mut(&mut self) -> &mut PostgresPage<O> {
        unsafe { self.page.as_mut() }
    }
}

impl<O: Opaque> Drop for PostgresBufferWriteGuard<O> {
    fn drop(&mut self) {
        unsafe {
            if std::thread::panicking() {
                pgrx::pg_sys::GenericXLogAbort(self.state);
            } else {
                if self.tracking_freespace {
                    pgrx::pg_sys::RecordPageWithFreeSpace(self.raw, self.id, self.freespace() as _);
                    pgrx::pg_sys::FreeSpaceMapVacuumRange(self.raw, self.id, self.id + 1);
                }
                pgrx::pg_sys::GenericXLogFinish(self.state);
            }
            pgrx::pg_sys::UnlockReleaseBuffer(self.buf);
        }
    }
}

#[derive(Debug, Clone)]
pub struct PostgresRelation<Opaque> {
    raw: pgrx::pg_sys::Relation,
    _phantom: PhantomData<fn(Opaque) -> Opaque>,
}

impl<Opaque> PostgresRelation<Opaque> {
    pub unsafe fn new(raw: pgrx::pg_sys::Relation) -> Self {
        Self {
            raw,
            _phantom: PhantomData,
        }
    }
}

impl<O: Opaque> Relation for PostgresRelation<O> {
    type Page = PostgresPage<O>;
}

impl<O: Opaque> RelationReadTypes for PostgresRelation<O> {
    type ReadGuard<'a> = PostgresBufferReadGuard<O>;
}

impl<O: Opaque> RelationRead for PostgresRelation<O> {
    fn read(&self, id: u32) -> Self::ReadGuard<'_> {
        assert!(id != u32::MAX, "no such page");
        unsafe {
            use pgrx::pg_sys::{
                BUFFER_LOCK_SHARE, BufferGetPage, ForkNumber, LockBuffer, ReadBufferExtended,
                ReadBufferMode,
            };
            let buf = ReadBufferExtended(
                self.raw,
                ForkNumber::MAIN_FORKNUM,
                id,
                ReadBufferMode::RBM_NORMAL,
                std::ptr::null_mut(),
            );
            LockBuffer(buf, BUFFER_LOCK_SHARE as _);
            let page = NonNull::new(BufferGetPage(buf).cast()).expect("failed to get page");
            PostgresBufferReadGuard { buf, page, id }
        }
    }
}

impl<O: Opaque> RelationWriteTypes for PostgresRelation<O> {
    type WriteGuard<'a> = PostgresBufferWriteGuard<O>;
}

impl<O: Opaque> PostgresRelation<O> {
    /// Pins block `id` and locks it exclusively.
    unsafe fn lock_exclusive(&self, id: u32) -> pgrx::pg_sys::Buffer {
        assert!(id != u32::MAX, "no such page");
        unsafe {
            use pgrx::pg_sys::{
                BUFFER_LOCK_EXCLUSIVE, ForkNumber, LockBuffer, ReadBufferExtended, ReadBufferMode,
            };
            let buf = ReadBufferExtended(
                self.raw,
                ForkNumber::MAIN_FORKNUM,
                id,
                ReadBufferMode::RBM_NORMAL,
                std::ptr::null_mut(),
            );
            LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE as _);
            buf
        }
    }

    /// Starts a WAL-logged change of `buf`, which must be pinned and locked
    /// exclusively by `lock_exclusive(id)`.
    unsafe fn start_write(
        &self,
        buf: pgrx::pg_sys::Buffer,
        id: u32,
        tracking_freespace: bool,
    ) -> PostgresBufferWriteGuard<O> {
        unsafe {
            use pgrx::pg_sys::{GenericXLogRegisterBuffer, GenericXLogStart};
            let state = GenericXLogStart(self.raw);
            let page = NonNull::new(
                GenericXLogRegisterBuffer(state, buf, 0).cast::<MaybeUninit<PostgresPage<O>>>(),
            )
            .expect("failed to get page");
            PostgresBufferWriteGuard {
                raw: self.raw,
                buf,
                page: page.cast(),
                state,
                id,
                tracking_freespace,
            }
        }
    }
}

impl<O: Opaque> RelationWrite for PostgresRelation<O> {
    fn write(&self, id: u32, tracking_freespace: bool) -> PostgresBufferWriteGuard<O> {
        unsafe {
            let buf = self.lock_exclusive(id);
            self.start_write(buf, id, tracking_freespace)
        }
    }
    fn extend(
        &self,
        opaque: <Self::Page as Page>::Opaque,
        tracking_freespace: bool,
    ) -> PostgresBufferWriteGuard<O> {
        unsafe {
            use pgrx::pg_sys::{
                GENERIC_XLOG_FULL_IMAGE, GenericXLogRegisterBuffer, GenericXLogStart,
            };
            let buf;
            #[cfg(any(feature = "pg14", feature = "pg15"))]
            {
                use pgrx::pg_sys::{
                    BUFFER_LOCK_EXCLUSIVE, ExclusiveLock, ForkNumber, LockBuffer,
                    LockRelationForExtension, ReadBufferExtended, ReadBufferMode,
                    UnlockRelationForExtension,
                };
                LockRelationForExtension(self.raw, ExclusiveLock as _);
                buf = ReadBufferExtended(
                    self.raw,
                    ForkNumber::MAIN_FORKNUM,
                    u32::MAX,
                    ReadBufferMode::RBM_NORMAL,
                    std::ptr::null_mut(),
                );
                UnlockRelationForExtension(self.raw, ExclusiveLock as _);
                LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE as _);
            }
            #[cfg(any(feature = "pg16", feature = "pg17", feature = "pg18", feature = "pg19"))]
            {
                use pgrx::pg_sys::{
                    BufferManagerRelation, ExtendBufferedFlags, ExtendBufferedRel, ForkNumber,
                };
                let bmr = BufferManagerRelation {
                    rel: self.raw,
                    smgr: std::ptr::null_mut(),
                    relpersistence: 0,
                };
                buf = ExtendBufferedRel(
                    bmr,
                    ForkNumber::MAIN_FORKNUM,
                    std::ptr::null_mut(),
                    ExtendBufferedFlags::EB_LOCK_FIRST as _,
                );
            }
            let state = GenericXLogStart(self.raw);
            let mut page = NonNull::new(
                GenericXLogRegisterBuffer(state, buf, GENERIC_XLOG_FULL_IMAGE as _)
                    .cast::<MaybeUninit<PostgresPage<O>>>(),
            )
            .expect("failed to get page");
            page_init(page.as_mut().as_mut_ptr(), opaque);
            PostgresBufferWriteGuard {
                raw: self.raw,
                buf,
                page: page.cast(),
                state,
                id: pgrx::pg_sys::BufferGetBlockNumber(buf),
                tracking_freespace,
            }
        }
    }
    fn search(&self, freespace: usize) -> Option<PostgresBufferWriteGuard<O>> {
        unsafe {
            use pgrx::pg_sys::{
                BufferGetPage, FreeSpaceMapVacuumRange, GetPageWithFreeSpace,
                RecordPageWithFreeSpace, UnlockReleaseBuffer,
            };
            loop {
                let id = GetPageWithFreeSpace(self.raw, freespace);
                if id == u32::MAX {
                    return None;
                }
                let buf = self.lock_exclusive(id);
                let actual = (*BufferGetPage(buf).cast::<PostgresPage<O>>()).freespace();
                if (actual as usize) < freespace {
                    // The free space is recorded incorrectly. Correct the FSM
                    // only: the page is unchanged, so it needs no WAL record.
                    UnlockReleaseBuffer(buf);
                    RecordPageWithFreeSpace(self.raw, id, actual as _);
                    FreeSpaceMapVacuumRange(self.raw, id, id + 1);
                    continue;
                }
                return Some(self.start_write(buf, id, true));
            }
        }
    }
}

impl<O: Opaque> RelationPrefetch for PostgresRelation<O> {
    fn prefetch(&self, id: u32) {
        assert!(id != u32::MAX, "no such page");
        unsafe {
            use pgrx::pg_sys::PrefetchBuffer;
            PrefetchBuffer(self.raw, 0, id);
        }
    }
}

pub struct Cache<'b, I: Iterator> {
    window: VecDeque<I::Item>,
    tail: VecDeque<u32>,
    iter: Option<I>,
    _phantom: PhantomData<&'b mut ()>,
}

impl<'b, I: Iterator> Default for Cache<'b, I> {
    fn default() -> Self {
        Self {
            window: Default::default(),
            tail: Default::default(),
            iter: Default::default(),
            _phantom: PhantomData,
        }
    }
}

impl<'b, I: Iterator> Cache<'b, I>
where
    I::Item: Fetch<'b>,
{
    #[allow(dead_code)]
    pub fn pop_id(&mut self) -> Option<u32> {
        while self.tail.is_empty()
            && let Some(iter) = self.iter.as_mut()
            && let Some(e) = iter.next()
        {
            for id in e.fetch() {
                self.tail.push_back(id);
            }
            self.window.push_back(e);
        }
        self.tail.pop_front()
    }
    #[allow(dead_code)]
    pub fn pop_item_if(&mut self, predicate: impl FnOnce(&I::Item) -> bool) -> Option<I::Item> {
        while self.window.is_empty()
            && let Some(iter) = self.iter.as_mut()
            && let Some(e) = iter.next()
        {
            for id in e.fetch() {
                self.tail.push_back(id);
            }
            self.window.push_back(e);
        }
        self.window.pop_front_if(move |x| predicate(x))
    }
    #[allow(dead_code)]
    pub fn pop_item(&mut self) -> Option<I::Item> {
        while self.window.is_empty()
            && let Some(iter) = self.iter.as_mut()
            && let Some(e) = iter.next()
        {
            for id in e.fetch() {
                self.tail.push_back(id);
            }
            self.window.push_back(e);
        }
        self.window.pop_front()
    }
}

pub struct PostgresReadStream<'b, O: Opaque, I: Iterator> {
    #[cfg(any(feature = "pg17", feature = "pg18", feature = "pg19"))]
    raw: *mut pgrx::pg_sys::ReadStream,
    // Because of `Box`'s special alias rules, `Box` cannot be used here.
    cache: RawBox<Cache<'b, I>>,
    _phantom: PhantomData<fn(O) -> O>,
}

/// The buffers of one item of a `PostgresReadStream`.
///
/// `read_stream_next_buffer` returns buffers strictly in callback order, which
/// is the order of the items and of the blocks within each item. A consumer
/// may stop early, so `Drop` releases the item's remaining buffers and the
/// next item starts at its own first buffer.
///
/// `raw` is borrowed from the stream without a lifetime, because the guards
/// type of `ReadStream` cannot borrow the stream. Callers must drop the guards
/// before the stream, and before taking the next item from it. All current
/// consumers do so: they pass the guards by value to a reader and drop them
/// before the next `next`/`next_if`.
pub struct PostgresReadStreamGuards<O, I, L: Iterator<Item = u32>> {
    #[cfg(any(feature = "pg17", feature = "pg18", feature = "pg19"))]
    raw: *mut pgrx::pg_sys::ReadStream,
    list: L,
    _phantom: PhantomData<fn(O, I) -> (O, I)>,
}

/// Returns the next buffer of the stream, which must hold block `expected`.
#[cfg(any(feature = "pg17", feature = "pg18", feature = "pg19"))]
unsafe fn read_stream_next_expected(
    raw: *mut pgrx::pg_sys::ReadStream,
    expected: u32,
) -> pgrx::pg_sys::Buffer {
    unsafe {
        use pgrx::pg_sys::{
            Buffer, BufferGetBlockNumber, InvalidBuffer, ReleaseBuffer, read_stream_next_buffer,
        };
        let buf = read_stream_next_buffer(raw, core::ptr::null_mut());
        if buf == InvalidBuffer as Buffer {
            panic!("internal error: read stream ended before block {expected}");
        }
        let actual = BufferGetBlockNumber(buf);
        if actual != expected {
            ReleaseBuffer(buf);
            panic!(
                "internal error: read stream returned block {actual}, expected block {expected}"
            );
        }
        buf
    }
}

impl<O: Opaque, I: Iterator, L> Iterator for PostgresReadStreamGuards<O, I, L>
where
    L: Iterator<Item = u32>,
{
    type Item = PostgresBufferReadGuard<O>;

    #[cfg(any(feature = "pg14", feature = "pg15", feature = "pg16"))]
    fn next(&mut self) -> Option<Self::Item> {
        unimplemented!()
    }

    #[cfg(any(feature = "pg17", feature = "pg18", feature = "pg19"))]
    fn next(&mut self) -> Option<Self::Item> {
        let id = self.list.next()?;
        unsafe {
            use pgrx::pg_sys::{BUFFER_LOCK_SHARE, BufferGetPage, LockBuffer};
            let buf = read_stream_next_expected(self.raw, id);
            LockBuffer(buf, BUFFER_LOCK_SHARE as _);
            let page = NonNull::new(BufferGetPage(buf).cast()).expect("failed to get page");
            Some(PostgresBufferReadGuard { buf, page, id })
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.list.size_hint()
    }
}

#[cfg(any(feature = "pg17", feature = "pg18", feature = "pg19"))]
impl<O, I, L: Iterator<Item = u32>> Drop for PostgresReadStreamGuards<O, I, L> {
    fn drop(&mut self) {
        // After an error, PostgreSQL releases the pins and the stream.
        if crate::index::cleanup::pg_cleanup_allowed() {
            for id in self.list.by_ref() {
                unsafe {
                    let buf = read_stream_next_expected(self.raw, id);
                    pgrx::pg_sys::ReleaseBuffer(buf);
                }
            }
        }
    }
}

impl<O: Opaque, I: Iterator, L> ExactSizeIterator for PostgresReadStreamGuards<O, I, L> where
    L: ExactSizeIterator<Item = u32>
{
}

#[cfg(any(feature = "pg17", feature = "pg18", feature = "pg19"))]
impl<'b, O: Opaque, I: Iterator> PostgresReadStream<'b, O, I>
where
    I::Item: Fetch<'b>,
{
    fn read<L: Iterator<Item = u32>>(&mut self, fetch: L) -> PostgresReadStreamGuards<O, I, L> {
        PostgresReadStreamGuards {
            raw: self.raw,
            list: fetch,
            _phantom: PhantomData,
        }
    }
}

impl<'b, O: Opaque, I: Iterator> ReadStream<'b> for PostgresReadStream<'b, O, I>
where
    I::Item: Fetch<'b>,
{
    type Relation = PostgresRelation<O>;

    type Guards = PostgresReadStreamGuards<O, I, <I::Item as Fetch<'b>>::Iter>;

    type Item = I::Item;

    type Inner =
        Chain<std::collections::vec_deque::IntoIter<I::Item>, Flatten<std::option::IntoIter<I>>>;

    #[cfg(any(feature = "pg14", feature = "pg15", feature = "pg16"))]
    fn next(&mut self) -> Option<(I::Item, Self::Guards)> {
        panic!("read_stream is not supported on PostgreSQL versions earlier than 17.");
    }

    #[cfg(any(feature = "pg14", feature = "pg15", feature = "pg16"))]
    fn next_if<P: FnOnce(&I::Item) -> bool>(
        &mut self,
        _predicate: P,
    ) -> Option<(I::Item, Self::Guards)> {
        panic!("read_stream is not supported on PostgreSQL versions earlier than 17.");
    }

    #[cfg(any(feature = "pg17", feature = "pg18", feature = "pg19"))]
    fn next(&mut self) -> Option<(I::Item, Self::Guards)> {
        if let Some(e) = unsafe { self.cache.as_mut().pop_item() } {
            let list = self.read(e.fetch());
            Some((e, list))
        } else {
            None
        }
    }

    #[cfg(any(feature = "pg17", feature = "pg18", feature = "pg19"))]
    fn next_if<P: FnOnce(&I::Item) -> bool>(
        &mut self,
        predicate: P,
    ) -> Option<(I::Item, Self::Guards)> {
        if let Some(e) = unsafe { self.cache.as_mut().pop_item_if(predicate) } {
            let list = self.read(e.fetch());
            Some((e, list))
        } else {
            None
        }
    }

    fn into_inner(mut self) -> Self::Inner {
        let cache = unsafe { std::mem::take(self.cache.as_mut()) };
        cache
            .window
            .into_iter()
            .chain(cache.iter.into_iter().flatten())
    }
}

impl<'b, O: Opaque, I: Iterator> Drop for PostgresReadStream<'b, O, I> {
    fn drop(&mut self) {
        unsafe {
            // Leave valid, empty callback data until read_stream_end finishes.
            // The RawBox field is freed even if this cleanup raises an error.
            let _ = std::mem::take(self.cache.as_mut());
            #[cfg(any(feature = "pg17", feature = "pg18", feature = "pg19"))]
            if crate::index::cleanup::pg_cleanup_allowed() {
                pgrx::pg_sys::read_stream_end(self.raw);
            }
        }
    }
}

impl<O: Opaque> RelationReadStreamTypes for PostgresRelation<O> {
    type ReadStream<'b, I: Iterator>
        = PostgresReadStream<'b, O, I>
    where
        I::Item: Fetch<'b>;
}

impl<O: Opaque> RelationReadStream for PostgresRelation<O> {
    #[cfg(any(feature = "pg14", feature = "pg15", feature = "pg16"))]
    fn read_stream<'b, I: Iterator>(&'b self, _iter: I, _hints: Hints) -> Self::ReadStream<'b, I>
    where
        I::Item: Fetch<'b>,
    {
        panic!("read_stream is not supported on PostgreSQL versions earlier than 17.");
    }

    #[cfg(any(feature = "pg17", feature = "pg18", feature = "pg19"))]
    fn read_stream<'b, I: Iterator>(&'b self, iter: I, hints: Hints) -> Self::ReadStream<'b, I>
    where
        I::Item: Fetch<'b>,
    {
        #[pgrx::pg_guard]
        unsafe extern "C-unwind" fn callback<'b, I: Iterator>(
            _stream: *mut pgrx::pg_sys::ReadStream,
            callback_private_data: *mut core::ffi::c_void,
            _per_buffer_data: *mut core::ffi::c_void,
        ) -> pgrx::pg_sys::BlockNumber
        where
            I::Item: Fetch<'b>,
        {
            unsafe {
                use pgrx::pg_sys::InvalidBlockNumber;
                let inner = callback_private_data.cast::<Cache<I>>();
                (*inner).pop_id().unwrap_or(InvalidBlockNumber)
            }
        }
        let cache = RawBox::new(Cache {
            window: VecDeque::new(),
            tail: VecDeque::new(),
            iter: Some(iter),
            _phantom: PhantomData,
        });
        let raw = unsafe {
            let mut flags = pgrx::pg_sys::READ_STREAM_DEFAULT;
            if hints.full {
                flags |= pgrx::pg_sys::READ_STREAM_FULL;
            }
            // In batch mode the callback must not block (see
            // `pgaio_enter_batchmode`). The callback pulls `iter`, which may
            // fetch heap tuples and evaluate quals (`vchordrq.prefilter`), so
            // only pure in-memory sequences may set `batch`.
            #[cfg(any(feature = "pg18", feature = "pg19"))]
            if hints.batch {
                flags |= pgrx::pg_sys::READ_STREAM_USE_BATCHING;
            }
            pgrx::pg_sys::read_stream_begin_relation(
                flags as i32,
                core::ptr::null_mut(),
                self.raw,
                pgrx::pg_sys::ForkNumber::MAIN_FORKNUM,
                Some(callback::<I>),
                cache.as_ptr().cast(),
                0,
            )
        };
        PostgresReadStream {
            raw,
            cache,
            _phantom: PhantomData,
        }
    }
}

#[inline(always)]
fn lp_flags(x: pgrx::pg_sys::ItemIdData) -> u32 {
    let x: u32 = unsafe { std::mem::transmute(x) };
    (x >> 15) & 0b11
}

// Own callback data without Box's aliasing requirements when the owner moves.
// Keeping an owner on the stack also frees it if read_stream_begin_relation fails.
struct RawBox<T>(NonNull<T>);

impl<T> RawBox<T> {
    #[cfg_attr(
        any(feature = "pg14", feature = "pg15", feature = "pg16"),
        expect(dead_code)
    )]
    fn new(value: T) -> Self {
        let ptr = Box::into_raw(Box::new(value));
        Self(NonNull::new(ptr).expect("Box::into_raw returned a null pointer"))
    }

    #[cfg_attr(
        any(feature = "pg14", feature = "pg15", feature = "pg16"),
        expect(dead_code)
    )]
    fn as_ptr(&self) -> *mut T {
        self.0.as_ptr()
    }

    // The caller must exclude access through the PostgreSQL callback pointer.
    unsafe fn as_mut(&mut self) -> &mut T {
        unsafe { self.0.as_mut() }
    }
}

impl<T> Drop for RawBox<T> {
    fn drop(&mut self) {
        unsafe { drop(Box::from_raw(self.0.as_ptr())) }
    }
}
