// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

mod am_build;

use crate::index::fetcher::*;
use crate::index::gucs;
use crate::index::scanners::SearchBuilder;
use crate::index::storage::PostgresRelation;
use crate::index::vchordg::opclass::opfamily;
use crate::index::vchordg::scanners::*;
use crate::recorder::DefaultRecorder;
use pgrx::datum::Internal;
use pgrx::pg_sys::Datum;
use std::cell::LazyCell;
use std::ffi::CStr;
use std::num::NonZero;
use std::ops::DerefMut;
use std::ptr::NonNull;
use std::sync::OnceLock;

#[repr(C)]
pub struct Reloption {
    vl_len_: i32,
    options: i32,
    ef_search: i32,
}

impl Reloption {
    unsafe fn options<'a>(this: *const Self, default: &'static CStr) -> &'a CStr {
        unsafe {
            if this.is_null() {
                return default;
            }
            let count = (&raw const (*this).options).read();
            if count == 0 {
                return default;
            }
            let ptr = this.cast::<u8>().add(count as _);
            CStr::from_ptr(ptr.cast())
        }
    }
    pub unsafe fn ef_search(this: *const Self, default: i32) -> i32 {
        unsafe {
            if this.is_null() {
                return default;
            }
            (*this).ef_search
        }
    }
}

// The isset_offset field exists only in PG18; PG19 removed it.
const TABLE: &[pgrx::pg_sys::relopt_parse_elt] = &[
    pgrx::pg_sys::relopt_parse_elt {
        optname: c"options".as_ptr(),
        opttype: pgrx::pg_sys::relopt_type::RELOPT_TYPE_STRING,
        offset: std::mem::offset_of!(Reloption, options) as i32,
        #[cfg(feature = "pg18")]
        isset_offset: 0,
    },
    pgrx::pg_sys::relopt_parse_elt {
        optname: c"ef_search".as_ptr(),
        opttype: pgrx::pg_sys::relopt_type::RELOPT_TYPE_INT,
        offset: std::mem::offset_of!(Reloption, ef_search) as i32,
        #[cfg(feature = "pg18")]
        isset_offset: 0,
    },
];

static RELOPT_KIND: OnceLock<pgrx::pg_sys::relopt_kind::Type> = OnceLock::new();

/// Validates the `options` reloption at `CREATE INDEX ... WITH` and
/// `ALTER INDEX ... SET`, so a malformed value is rejected immediately instead
/// of failing at the next REINDEX (#35). Checks that need the indexed column
/// (dimensions, vector type) still happen when the index is built.
#[pgrx::pg_guard]
unsafe extern "C-unwind" fn validate_options(value: *const core::ffi::c_char) {
    if value.is_null() {
        return;
    }
    let s = unsafe { CStr::from_ptr(value) }.to_string_lossy();
    match toml::from_str::<crate::index::vchordg::types::VchordgIndexingOptions>(&s) {
        Ok(options) => {
            if let Err(errors) = validator::Validate::validate(&options) {
                pgrx::error!("error while validating options: {}", errors);
            }
        }
        Err(e) => pgrx::error!("failed to parse options: {}", e),
    }
}

pub fn init() {
    RELOPT_KIND.get_or_init(|| {
        let kind;
        unsafe {
            kind = pgrx::pg_sys::add_reloption_kind();
            pgrx::pg_sys::add_string_reloption(
                kind as _,
                c"options".as_ptr(),
                c"Vector index options, represented as a TOML string.".as_ptr(),
                c"".as_ptr(),
                Some(validate_options),
                pgrx::pg_sys::AccessExclusiveLock as pgrx::pg_sys::LOCKMODE,
            );
            pgrx::pg_sys::add_int_reloption(
                kind as _,
                c"ef_search".as_ptr(),
                c"Search parameter `vchordg.ef_search`".as_ptr(),
                64,
                1,
                65535,
                pgrx::pg_sys::AccessExclusiveLock as pgrx::pg_sys::LOCKMODE,
            );
        }
        kind
    });
}

#[pgrx::pg_extern(sql = "")]
fn _vchordg_amhandler(_fcinfo: pgrx::pg_sys::FunctionCallInfo) -> Internal {
    type T = pgrx::pg_sys::IndexAmRoutine;
    unsafe {
        let index_am_routine = pgrx::pg_sys::palloc0(size_of::<T>()) as *mut T;
        index_am_routine.write(AM_HANDLER);
        Internal::from(Some(Datum::from(index_am_routine)))
    }
}

const AM_HANDLER: pgrx::pg_sys::IndexAmRoutine = const {
    let mut am_routine = unsafe { std::mem::zeroed::<pgrx::pg_sys::IndexAmRoutine>() };

    am_routine.type_ = pgrx::pg_sys::NodeTag::T_IndexAmRoutine;

    am_routine.amsupport = 1;
    am_routine.amcanorderbyop = true;

    #[cfg(any(feature = "pg17", feature = "pg18", feature = "pg19"))]
    {
        am_routine.amcanbuildparallel = true;
    }

    // Index access methods that set `amoptionalkey` to `false`
    // must index all tuples, even if the first column is `NULL`.
    // However, PostgreSQL does not generate a path if there is no
    // index clauses, even if there is a `ORDER BY` clause.
    // So we have to set it to `true` and set costs of every path
    // for vector index scans without `ORDER BY` clauses a large number
    // and throw errors if someone really wants such a path.
    am_routine.amoptionalkey = true;

    am_routine.amvalidate = Some(amvalidate);
    am_routine.amoptions = Some(amoptions);
    am_routine.amcostestimate = Some(amcostestimate);

    am_routine.ambuildphasename = Some(am_build::ambuildphasename);
    am_routine.ambuild = Some(am_build::ambuild);
    am_routine.ambuildempty = Some(am_build::ambuildempty);
    am_routine.aminsert = Some(aminsert);
    am_routine.ambulkdelete = Some(ambulkdelete);
    am_routine.amvacuumcleanup = Some(amvacuumcleanup);

    am_routine.ambeginscan = Some(ambeginscan);
    am_routine.amrescan = Some(amrescan);
    am_routine.amgettuple = Some(amgettuple);
    am_routine.amendscan = Some(amendscan);

    am_routine.amparallelvacuumoptions = pgrx::pg_sys::VACUUM_OPTION_PARALLEL_BULKDEL as u8
        | pgrx::pg_sys::VACUUM_OPTION_PARALLEL_CLEANUP as u8;

    am_routine
};

#[pgrx::pg_guard]
pub unsafe extern "C-unwind" fn amvalidate(_opclass_oid: pgrx::pg_sys::Oid) -> bool {
    true
}

#[pgrx::pg_guard]
pub unsafe extern "C-unwind" fn amoptions(
    reloptions: Datum,
    validate: bool,
) -> *mut pgrx::pg_sys::bytea {
    let relopt_kind = RELOPT_KIND.get().copied().expect("init is not called");
    let rdopts = unsafe {
        pgrx::pg_sys::build_reloptions(
            reloptions,
            validate,
            relopt_kind,
            size_of::<Reloption>(),
            TABLE.as_ptr(),
            TABLE.len() as _,
        )
    };
    rdopts as *mut pgrx::pg_sys::bytea
}

#[pgrx::pg_guard]
pub unsafe extern "C-unwind" fn amcostestimate(
    _root: *mut pgrx::pg_sys::PlannerInfo,
    path: *mut pgrx::pg_sys::IndexPath,
    _loop_count: f64,
    index_startup_cost: *mut pgrx::pg_sys::Cost,
    index_total_cost: *mut pgrx::pg_sys::Cost,
    index_selectivity: *mut pgrx::pg_sys::Selectivity,
    index_correlation: *mut f64,
    index_pages: *mut f64,
) {
    unsafe {
        use pgrx::pg_sys::disable_cost;
        // Without ORDER BY or vector clauses, PostgreSQL still builds paths for
        // a partial index whose predicate the query implies, and index-only
        // paths when the query needs no columns. `amrescan` rejects the former,
        // and this index cannot return tuple data for the latter.
        let unusable = ((*path).indexorderbys.is_null() && (*path).indexclauses.is_null())
            || (*path).path.pathtype == pgrx::pg_sys::NodeTag::T_IndexOnlyScan;
        if unusable {
            // Since PostgreSQL 18, `disable_cost` alone does not lose against
            // paths disabled by `enable_*` settings, so count the path as
            // disabled as well. Before PostgreSQL 18, `disable_cost` is
            // additive: a seq scan disabled by `enable_seqscan = off` costs
            // more than this path at `disable_cost`, so give it a cost that
            // dominates realistic plans (#32). It stays finite on purpose:
            // planner arithmetic such as `inf - inf` yields NaN.
            #[cfg(any(feature = "pg18", feature = "pg19"))]
            {
                (*path).path.disabled_nodes += 1;
            }
            *index_startup_cost = crate::index::UNUSABLE_PATH_COST;
            *index_total_cost = crate::index::UNUSABLE_PATH_COST;
            *index_selectivity = 0.0;
            *index_correlation = 0.0;
            *index_pages = 1.0;
            return;
        }
        if !gucs::vchordg_enable_scan() {
            // `vchordg.enable_scan = off` discourages the path like the
            // built-in `enable_*` settings do.
            #[cfg(any(feature = "pg18", feature = "pg19"))]
            {
                (*path).path.disabled_nodes += 1;
            }
            *index_startup_cost = disable_cost;
            *index_total_cost = disable_cost;
            *index_selectivity = 0.0;
            *index_correlation = 0.0;
            *index_pages = 1.0;
            return;
        }
        *index_startup_cost = 0.0;
        *index_total_cost = 0.0;
        *index_selectivity = 1.0;
        *index_correlation = 0.0;
        *index_pages = 1.0;
    }
}

#[pgrx::pg_guard]
pub unsafe extern "C-unwind" fn aminsert(
    index_relation: pgrx::pg_sys::Relation,
    values: *mut Datum,
    is_null: *mut bool,
    heap_tid: pgrx::pg_sys::ItemPointer,
    _heap_relation: pgrx::pg_sys::Relation,
    _check_unique: pgrx::pg_sys::IndexUniqueCheck::Type,
    _index_unchanged: bool,
    _index_info: *mut pgrx::pg_sys::IndexInfo,
) -> bool {
    let opfamily = unsafe { opfamily(index_relation) };
    let index = unsafe { PostgresRelation::new(index_relation) };
    let datum = unsafe { (!is_null.add(0).read()).then_some(values.add(0).read()) };
    let ctid = unsafe { heap_tid.read() };
    if let Some(store) = unsafe { datum.and_then(|x| opfamily.store(x)) } {
        for (vector, extra) in store {
            let key = ctid_to_key(ctid);
            let payload = kv_to_pointer((key, extra));
            crate::index::vchordg::dispatch::insert(opfamily, &index, payload, vector);
        }
    }
    false
}

#[pgrx::pg_guard]
pub unsafe extern "C-unwind" fn ambulkdelete(
    info: *mut pgrx::pg_sys::IndexVacuumInfo,
    stats: *mut pgrx::pg_sys::IndexBulkDeleteResult,
    callback: pgrx::pg_sys::IndexBulkDeleteCallback,
    callback_state: *mut std::os::raw::c_void,
) -> *mut pgrx::pg_sys::IndexBulkDeleteResult {
    use pgrx::pg_sys::ffi::pg_guard_ffi_boundary;
    let mut stats = stats;
    if stats.is_null() {
        stats = unsafe {
            pgrx::pg_sys::palloc0(size_of::<pgrx::pg_sys::IndexBulkDeleteResult>()).cast()
        };
    }
    let opfamily = unsafe { opfamily((*info).index) };
    let index = unsafe { PostgresRelation::new((*info).index) };
    let check = || unsafe {
        #[cfg(any(feature = "pg14", feature = "pg15", feature = "pg16", feature = "pg17"))]
        pgrx::pg_sys::vacuum_delay_point();
        #[cfg(any(feature = "pg18", feature = "pg19"))]
        pgrx::pg_sys::vacuum_delay_point(false);
    };
    let callback = callback.expect("null function pointer");
    let callback = |pointer: NonZero<u64>| {
        let (key, _) = pointer_to_kv(pointer);
        let mut ctid = key_to_ctid(key);
        #[allow(ffi_unwind_calls, reason = "protected by pg_guard_ffi_boundary")]
        unsafe {
            pg_guard_ffi_boundary(|| callback(&mut ctid, callback_state))
        }
    };
    let removed = crate::index::vchordg::dispatch::bulkdelete(opfamily, &index, check, callback);
    unsafe {
        (*stats).num_pages = relation_pages((*info).index);
        (*stats).tuples_removed += removed as f64;
    }
    stats
}

#[pgrx::pg_guard]
pub unsafe extern "C-unwind" fn amvacuumcleanup(
    info: *mut pgrx::pg_sys::IndexVacuumInfo,
    stats: *mut pgrx::pg_sys::IndexBulkDeleteResult,
) -> *mut pgrx::pg_sys::IndexBulkDeleteResult {
    // ANALYZE calls this without VACUUM. Nothing was deleted, so leave the
    // index (and its statistics) untouched: `maintain` would rewrite and
    // WAL-log every page.
    if unsafe { (*info).analyze_only } {
        return stats;
    }
    // Without a bulk delete in this VACUUM, rewrite nothing unless an earlier
    // VACUUM left deleted vertices or unreferenced vectors behind. Returning
    // NULL tells PostgreSQL the index did not change (#24).
    if stats.is_null() {
        let index = unsafe { PostgresRelation::new((*info).index) };
        if !vchordg::pending(&index) {
            return stats;
        }
    }
    let mut stats = stats;
    if stats.is_null() {
        stats = unsafe {
            pgrx::pg_sys::palloc0(size_of::<pgrx::pg_sys::IndexBulkDeleteResult>()).cast()
        };
    }
    let opfamily = unsafe { opfamily((*info).index) };
    let index = unsafe { PostgresRelation::new((*info).index) };
    let check = || unsafe {
        #[cfg(any(feature = "pg14", feature = "pg15", feature = "pg16", feature = "pg17"))]
        pgrx::pg_sys::vacuum_delay_point();
        #[cfg(any(feature = "pg18", feature = "pg19"))]
        pgrx::pg_sys::vacuum_delay_point(false);
    };
    // maintenance_work_mem is in kilobytes
    let memory_limit =
        usize::try_from(unsafe { pgrx::pg_sys::maintenance_work_mem }).unwrap_or(0) * 1024;
    let maintained =
        crate::index::vchordg::dispatch::maintain(opfamily, &index, check, memory_limit);
    let name = unsafe {
        CStr::from_ptr((*(*(*info).index).rd_rel).relname.data.as_ptr())
            .to_string_lossy()
            .into_owned()
    };
    match maintained.reachability {
        vchordg::Reachability::Checked {
            disconnected,
            unreachable_before,
            live,
        } => {
            if disconnected > 0 {
                let message = format!(
                    "vchordg index \"{name}\": {disconnected} of {live} live vertices may have become unreachable from the entry point during VACUUM (concurrent inserts are not accounted for); REINDEX to restore recall"
                );
                // Edge repair can cut off a few vertices after ordinary
                // deletes; warn the session only when it is a noticeable
                // share of the index, and keep smaller counts in the log.
                if disconnected.saturating_mul(100) >= live {
                    pgrx::warning!("{message}");
                } else {
                    pgrx::log!("{message}");
                }
            }
            pgrx::debug1!(
                "vchordg index \"{name}\": reachability check: {disconnected} live vertices disconnected by VACUUM, {unreachable_before} unreachable before it, {live} live"
            );
        }
        vchordg::Reachability::Skipped(reason) => {
            pgrx::debug1!("vchordg index \"{name}\": reachability check skipped: {reason}");
        }
    }
    unsafe {
        (*stats).num_pages = relation_pages((*info).index);
        (*stats).num_index_tuples = maintained.live as f64;
        (*stats).estimated_count = false;
    }
    stats
}

unsafe fn relation_pages(index_relation: pgrx::pg_sys::Relation) -> pgrx::pg_sys::BlockNumber {
    unsafe {
        pgrx::pg_sys::RelationGetNumberOfBlocksInFork(
            index_relation,
            pgrx::pg_sys::ForkNumber::MAIN_FORKNUM,
        )
    }
}

#[pgrx::pg_guard]
pub unsafe extern "C-unwind" fn ambeginscan(
    index_relation: pgrx::pg_sys::Relation,
    n_keys: std::os::raw::c_int,
    n_orderbys: std::os::raw::c_int,
) -> pgrx::pg_sys::IndexScanDesc {
    use pgrx::memcxt::PgMemoryContexts::CurrentMemoryContext;

    let scan = unsafe { pgrx::pg_sys::RelationGetIndexScan(index_relation, n_keys, n_orderbys) };
    let scanner: Scanner = Scanner {
        hack: None,
        scanning: None,
        bump: Box::new(bumpalo::Bump::new()),
    };
    unsafe {
        (*scan).opaque = CurrentMemoryContext.leak_and_drop_on_delete(scanner).cast();
    }
    scan
}

#[pgrx::pg_guard]
pub unsafe extern "C-unwind" fn amrescan(
    scan: pgrx::pg_sys::IndexScanDesc,
    keys: pgrx::pg_sys::ScanKey,
    _n_keys: std::os::raw::c_int,
    orderbys: pgrx::pg_sys::ScanKey,
    _n_orderbys: std::os::raw::c_int,
) {
    unsafe {
        use crate::index::vchordg::opclass::Opfamily;
        if !keys.is_null() && (*scan).numberOfKeys > 0 {
            std::ptr::copy(keys, (*scan).keyData, (*scan).numberOfKeys as _);
        }
        if !orderbys.is_null() && (*scan).numberOfOrderBys > 0 {
            std::ptr::copy(orderbys, (*scan).orderByData, (*scan).numberOfOrderBys as _);
        }
        if (*scan).numberOfOrderBys == 0 && (*scan).numberOfKeys == 0 {
            pgrx::error!(
                "vector search with no WHERE clause and no ORDER BY clause is not supported"
            );
        }
        let scanner = &mut *(*scan).opaque.cast::<Scanner>();
        drop(scanner.scanning.take());
        scanner.bump.reset();
        let opfamily = opfamily((*scan).indexRelation);
        let index = PostgresRelation::new((*scan).indexRelation);
        let options = SearchOptions {
            ef_search: gucs::vchordg_ef_search((*scan).indexRelation),
            beam_search: gucs::vchordg_beam_search(),
            max_scan_tuples: gucs::vchordg_max_scan_tuples(),
            io_search: gucs::vchordg_io_search(),
            io_rerank: gucs::vchordg_io_rerank(),
        };
        let fetcher = {
            let hack = scanner.hack;
            LazyCell::new(move || {
                HeapFetcher::new(
                    (*scan).indexRelation,
                    (*scan).heapRelation,
                    (*scan).xs_snapshot,
                    (*scan).xs_heapfetch,
                    if let Some(hack) = hack {
                        hack.as_ptr()
                    } else {
                        std::ptr::null_mut()
                    },
                )
            })
        };
        // Query recorde is disable for vchordg indexes for now.
        let recorder = DefaultRecorder {
            enable: false,
            rate: None,
            max_records: 0,
            index: (*(*scan).indexRelation).rd_id.to_u32(),
        };
        // PAY ATTENTATION: `scanning` references `bump`, so `scanning` must be dropped before `bump`.
        let bump = scanner.bump.as_ref();
        scanner.scanning = Some(match opfamily {
            Opfamily::VectorL2
            | Opfamily::VectorCosine
            | Opfamily::VectorIp
            | Opfamily::HalfvecL2
            | Opfamily::HalfvecCosine
            | Opfamily::HalfvecIp
            | Opfamily::Rabitq8L2
            | Opfamily::Rabitq8Cosine
            | Opfamily::Rabitq8Ip
            | Opfamily::Rabitq4L2
            | Opfamily::Rabitq4Cosine
            | Opfamily::Rabitq4Ip => {
                let mut builder = DefaultBuilder::new(opfamily);
                for i in 0..(*scan).numberOfOrderBys {
                    let data = (*scan).orderByData.add(i as usize);
                    let value = (*data).sk_argument;
                    let is_null = ((*data).sk_flags & pgrx::pg_sys::SK_ISNULL as i32) != 0;
                    builder.add((*data).sk_strategy, (!is_null).then_some(value));
                }
                for i in 0..(*scan).numberOfKeys {
                    let data = (*scan).keyData.add(i as usize);
                    let value = (*data).sk_argument;
                    let is_null = ((*data).sk_flags & pgrx::pg_sys::SK_ISNULL as i32) != 0;
                    builder.add((*data).sk_strategy, (!is_null).then_some(value));
                }
                LazyCell::new(Box::new(move || {
                    // only do this since `PostgresRelation` has no destructor
                    let index = bump.alloc(index.clone());
                    builder.build(index, options, fetcher, bump, recorder)
                }))
            }
        });
    }
}

#[pgrx::pg_guard]
pub unsafe extern "C-unwind" fn amgettuple(
    scan: pgrx::pg_sys::IndexScanDesc,
    direction: pgrx::pg_sys::ScanDirection::Type,
) -> bool {
    if direction != pgrx::pg_sys::ScanDirection::ForwardScanDirection {
        pgrx::error!("vector search without a forward scan direction is not supported");
    }
    // `index_getnext_slot` calls back without checking interrupts while heap
    // tuples are invisible; no buffer lock is held here (#28).
    pgrx::pg_sys::check_for_interrupts!();
    // https://www.postgresql.org/docs/current/index-locking.html
    // If heap entries referenced physical pointers are deleted before
    // they are consumed by PostgreSQL, PostgreSQL will received wrong
    // physical pointers: no rows or irreverent rows are referenced.
    if unsafe { (*(*scan).xs_snapshot).snapshot_type } != pgrx::pg_sys::SnapshotType::SNAPSHOT_MVCC
    {
        pgrx::error!("scanning with a non-MVCC-compliant snapshot is not supported");
    }
    let scanner = unsafe { (*scan).opaque.cast::<Scanner>().as_mut().unwrap_unchecked() };
    let Some(scanning) = scanner.scanning.as_mut() else {
        return false;
    };
    if let Some((_, key, recheck)) = scanning.deref_mut().next() {
        unsafe {
            (*scan).xs_heaptid = key_to_ctid(key);
            (*scan).xs_recheck = recheck;
            (*scan).xs_recheckorderby = false;
        }
        true
    } else {
        false
    }
}

#[pgrx::pg_guard]
pub unsafe extern "C-unwind" fn amendscan(scan: pgrx::pg_sys::IndexScanDesc) {
    let scanner = unsafe { &mut *(*scan).opaque.cast::<Scanner>() };
    drop(scanner.scanning.take());
    scanner.bump.reset();
}

type Iter = Box<dyn Iterator<Item = (f32, [u16; 3], bool)>>;

pub struct Scanner {
    pub hack: Option<NonNull<pgrx::pg_sys::IndexScanState>>,
    scanning: Option<LazyCell<Iter, Box<dyn FnOnce() -> Iter>>>,
    bump: Box<bumpalo::Bump>,
}

impl Drop for Scanner {
    fn drop(&mut self) {
        // The MemoryContext callback is the sole owner of this Scanner. Normal
        // amendscan already emptied it; after an error PostgreSQL may have
        // released its resources before this callback (even after ROLLBACK TO).
        // Drop the iterator before the bump it borrows, freeing Rust memory only.
        crate::index::cleanup::abandon(self.scanning.take());
    }
}
