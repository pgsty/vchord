// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use crate::index::fetcher::Fetcher;
use crate::recorder::Recorder;
use index::bump::Bump;
use index::relation::{Page, RelationPrefetch, RelationRead, RelationReadStream};
use pgrx::pg_sys::Datum;

#[derive(Debug, Clone, Copy)]
pub enum Io {
    Plain,
    Simple,
    #[cfg_attr(any(feature = "pg15", feature = "pg16"), expect(dead_code))]
    Stream,
}

/// How a vchordrq search reads list pages: as an `Io`, or `Mixed`, a read stream on dense spans and `Io::Simple` elsewhere.
#[derive(Debug, Clone, Copy)]
pub enum SearchIo {
    Io(Io),
    #[cfg_attr(any(feature = "pg15", feature = "pg16"), expect(dead_code))]
    Mixed,
}

pub trait SearchBuilder: 'static {
    type Options;

    type Opfamily;

    type Opaque: Copy;

    fn new(opfamily: Self::Opfamily) -> Self;

    unsafe fn add(&mut self, strategy: u16, datum: Option<Datum>);

    fn build<'b, R>(
        self,
        relation: &'b R,
        options: Self::Options,
        fetcher: impl Fetcher + 'b,
        bump: &'b impl Bump,
        recorder: impl Recorder,
    ) -> Box<dyn Iterator<Item = (f32, [u16; 3], bool)> + 'b>
    where
        R: RelationRead + RelationPrefetch + RelationReadStream,
        R::Page: Page<Opaque = Self::Opaque>;
}

pub(super) unsafe fn null_order(scan: pgrx::pg_sys::IndexScanDesc) -> bool {
    unsafe {
        (*scan).numberOfOrderBys == 1
            && (*(*scan).orderByData).sk_flags & pgrx::pg_sys::SK_ISNULL as i32 != 0
    }
}

pub(super) struct NullOrderRows {
    context: pgrx::pg_sys::MemoryContext,
    store: *mut pgrx::pg_sys::Tuplestorestate,
    slot: *mut pgrx::pg_sys::TupleTableSlot,
}

impl NullOrderRows {
    pub(super) unsafe fn new(scan: pgrx::pg_sys::IndexScanDesc) -> Self {
        use crate::index::traverse::{HeapTraverser, Traverser};
        use pgrx::pg_sys;
        use std::ptr::{null, null_mut};
        unsafe {
            let context = pg_sys::AllocSetContextCreateInternal(
                pg_sys::GetMemoryChunkContext(scan.cast()),
                c"vchord NULL ordering".as_ptr(),
                pg_sys::ALLOCSET_DEFAULT_MINSIZE as _,
                pg_sys::ALLOCSET_DEFAULT_INITSIZE as _,
                pg_sys::ALLOCSET_DEFAULT_MAXSIZE as _,
            );
            let _switch = crate::index::MemoryContextSwitch::new(context);
            let desc = pg_sys::CreateTemplateTupleDesc(1);
            pg_sys::TupleDescInitEntry(desc, 1, null(), pg_sys::TIDOID, -1, 0);
            let store = pg_sys::tuplestore_begin_heap(false, false, pg_sys::work_mem);
            let slot =
                pg_sys::MakeSingleTupleTableSlot(desc, &raw const pg_sys::TTSOpsMinimalTuple);
            let index = (*scan).indexRelation;
            let heap = (*scan).heapRelation;
            let info = pg_sys::BuildIndexInfo(index);
            (*info).ii_Concurrent = true;
            (*info).ii_NumIndexAttrs = 0;
            (*info).ii_NumIndexKeyAttrs = 0;
            (*info).ii_Expressions = null_mut();
            let table_scan =
                pg_sys::table_beginscan_strat(heap, (*scan).xs_snapshot, 0, null_mut(), true, true);
            HeapTraverser::new(heap, index, info, table_scan).traverse(
                false,
                |row: &mut dyn crate::index::traverse::Tuple| {
                    let mut tid = row.id();
                    let mut value = Datum::from(&raw mut tid);
                    pg_sys::tuplestore_putvalues(store, desc, &raw mut value, &mut false);
                },
            );
            Self {
                context,
                store,
                slot,
            }
        }
    }
}

impl Iterator for NullOrderRows {
    type Item = (f32, [u16; 3], bool);

    fn next(&mut self) -> Option<Self::Item> {
        use pgrx::pg_sys;
        unsafe {
            if !pg_sys::tuplestore_gettupleslot(self.store, true, false, self.slot) {
                return None;
            }
            let value = pg_sys::slot_getattr(self.slot, 1, &mut false);
            let tid = *value.cast_mut_ptr::<pg_sys::ItemPointerData>();
            Some((0.0, crate::index::fetcher::ctid_to_key(tid), true))
        }
    }
}

impl Drop for NullOrderRows {
    fn drop(&mut self) {
        if crate::index::cleanup::pg_cleanup_allowed() {
            unsafe {
                pgrx::pg_sys::ExecDropSingleTupleTableSlot(self.slot);
                pgrx::pg_sys::tuplestore_end(self.store);
                pgrx::pg_sys::MemoryContextDelete(self.context);
            }
        }
    }
}
