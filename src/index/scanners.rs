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
    heap: pgrx::pg_sys::Relation,
    table_scan: *mut pgrx::pg_sys::TableScanDescData,
    slot: *mut pgrx::pg_sys::TupleTableSlot,
    estate: *mut pgrx::pg_sys::EState,
    econtext: *mut pgrx::pg_sys::ExprContext,
    predicate: *mut pgrx::pg_sys::ExprState,
    remaining: Option<u32>,
    root_block: Option<pgrx::pg_sys::BlockNumber>,
    root_offsets: Vec<pgrx::pg_sys::OffsetNumber>,
}

impl NullOrderRows {
    pub(super) unsafe fn new(
        scan: pgrx::pg_sys::IndexScanDesc,
        max_scan_tuples: Option<u32>,
    ) -> Self {
        use pgrx::pg_sys;
        use std::ptr::null_mut;
        unsafe {
            let context = pg_sys::AllocSetContextCreateInternal(
                pg_sys::GetMemoryChunkContext(scan.cast()),
                c"vchord NULL ordering".as_ptr(),
                pg_sys::ALLOCSET_DEFAULT_MINSIZE as _,
                pg_sys::ALLOCSET_DEFAULT_INITSIZE as _,
                pg_sys::ALLOCSET_DEFAULT_MAXSIZE as _,
            );
            let _switch = crate::index::MemoryContextSwitch::new(context);
            let index = (*scan).indexRelation;
            let heap = (*scan).heapRelation;
            let table_scan =
                pg_sys::table_beginscan_strat(heap, (*scan).xs_snapshot, 0, null_mut(), true, true);
            let slot = pg_sys::table_slot_create(heap, null_mut());
            let pred = pg_sys::RelationGetIndexPredicate(index);
            let (estate, econtext, predicate) = if pred.is_null() {
                (null_mut(), null_mut(), null_mut())
            } else {
                let estate = pg_sys::CreateExecutorState();
                let predicate = {
                    let _switch = crate::index::MemoryContextSwitch::new((*estate).es_query_cxt);
                    pg_sys::ExecPrepareQual(pred, estate)
                };
                let econtext = pg_sys::MakePerTupleExprContext(estate);
                (estate, econtext, predicate)
            };
            Self {
                context,
                heap,
                table_scan,
                slot,
                estate,
                econtext,
                predicate,
                remaining: max_scan_tuples,
                root_block: None,
                root_offsets: Vec::new(),
            }
        }
    }
}

impl Iterator for NullOrderRows {
    type Item = (f32, [u16; 3], bool);

    fn next(&mut self) -> Option<Self::Item> {
        use pgrx::datum::FromDatum;
        use pgrx::pg_sys;
        use pgrx::pg_sys::ffi::pg_guard_ffi_boundary;
        if self.remaining == Some(0) {
            return None;
        }
        unsafe {
            let table_am = (*self.heap).rd_tableam;
            if table_am.is_null() {
                panic!("unknown heap access method");
            }
            let scan_getnextslot = (*table_am)
                .scan_getnextslot
                .expect("unsupported heap access method");
            loop {
                pgrx::check_for_interrupts();
                #[allow(ffi_unwind_calls, reason = "protected by pg_guard_ffi_boundary")]
                let found = pg_guard_ffi_boundary(|| {
                    scan_getnextslot(
                        self.table_scan,
                        pg_sys::ScanDirection::ForwardScanDirection,
                        self.slot,
                    )
                });
                if !found {
                    return None;
                }
                if !self.predicate.is_null() {
                    let evalfunc = (*self.predicate).evalfunc.expect("no evalfunc for qual");
                    (*self.econtext).ecxt_scantuple = self.slot;
                    pg_sys::MemoryContextReset((*self.econtext).ecxt_per_tuple_memory);
                    let result = {
                        let _switch = crate::index::MemoryContextSwitch::new(
                            (*self.econtext).ecxt_per_tuple_memory,
                        );
                        let mut is_null = true;
                        #[allow(ffi_unwind_calls, reason = "protected by pg_guard_ffi_boundary")]
                        let datum = pg_guard_ffi_boundary(|| {
                            evalfunc(self.predicate, self.econtext, &mut is_null)
                        });
                        bool::from_datum(datum, is_null)
                    };
                    if result != Some(true) {
                        continue;
                    }
                }
                if let Some(remaining) = &mut self.remaining {
                    *remaining -= 1;
                }
                let mut tid = (*self.slot).tts_tid;
                // `index_getnext_slot` rejects heap-only TIDs; map them to their HOT root.
                if std::ptr::eq(
                    (*self.slot).tts_ops,
                    &raw const pg_sys::TTSOpsBufferHeapTuple,
                ) {
                    let bslot = self.slot.cast::<pg_sys::BufferHeapTupleTableSlot>();
                    let tuple = (*bslot).base.tuple;
                    if !tuple.is_null()
                        && !(*tuple).t_data.is_null()
                        && ((*(*tuple).t_data).t_infomask2 & pg_sys::HEAP_ONLY_TUPLE as u16) != 0
                    {
                        let blkno = pg_sys::ItemPointerGetBlockNumber(&tid);
                        let offnum = pg_sys::ItemPointerGetOffsetNumber(&tid);
                        if self.root_block != Some(blkno) {
                            if self.root_offsets.is_empty() {
                                self.root_offsets.resize(
                                    (pg_sys::BLCKSZ as usize) / size_of::<pg_sys::ItemIdData>(),
                                    0,
                                );
                            }
                            let buf = (*bslot).buffer;
                            pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_SHARE as _);
                            pg_sys::heap_get_root_tuples(
                                pgrx::BufferGetPage!(buf).get(),
                                self.root_offsets.as_mut_ptr(),
                            );
                            pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_UNLOCK as _);
                            self.root_block = Some(blkno);
                        }
                        let root_off = self.root_offsets[(offnum - 1) as usize];
                        if root_off != 0 {
                            pg_sys::ItemPointerSetOffsetNumber(&mut tid, root_off);
                        }
                    }
                }
                return Some((0.0, crate::index::fetcher::ctid_to_key(tid), true));
            }
        }
    }
}

impl Drop for NullOrderRows {
    fn drop(&mut self) {
        if crate::index::cleanup::pg_cleanup_allowed() {
            unsafe {
                use pgrx::pg_sys::ffi::pg_guard_ffi_boundary;
                pgrx::pg_sys::ExecDropSingleTupleTableSlot(self.slot);
                let table_am = (*self.heap).rd_tableam;
                if !table_am.is_null()
                    && let Some(scan_end) = (*table_am).scan_end
                {
                    #[allow(ffi_unwind_calls, reason = "protected by pg_guard_ffi_boundary")]
                    pg_guard_ffi_boundary(|| scan_end(self.table_scan));
                }
                if !self.estate.is_null() {
                    pgrx::pg_sys::FreeExecutorState(self.estate);
                }
                pgrx::pg_sys::MemoryContextDelete(self.context);
            }
        }
    }
}
