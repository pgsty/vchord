// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

mod cleanup;
mod fetcher;
mod functions;
mod gucs;
mod hook;
mod opclass;
mod sample;
mod scanners;
mod storage;
mod traverse;
mod vchordg;
mod vchordrq;

pub fn init() {
    gucs::init();
    hook::init();
    vchordrq::am::init();
    vchordg::am::init();
    ::index::interrupt::set(check_for_interrupts);
}

/// Interrupt check for the algorithm crates' long loops (#28). Only the
/// backend's main thread may call into PostgreSQL; build worker threads skip
/// it. pgrx turns a raised error into a Rust panic that unwinds normally.
fn check_for_interrupts() {
    if crate::is_main() {
        pgrx::check_for_interrupts();
    }
}

/// Cost given to index paths the access methods cannot execute (no ORDER BY
/// vector and no index clause, or index-only scans). Finite on purpose:
/// `inf - inf` in planner arithmetic yields NaN. This is a mitigation, not a
/// structural exclusion (#32).
pub const UNUSABLE_PATH_COST: f64 = 1.0e30;

/// Vector builds size parallel scans by heap blocks and share the keyed sort budget.
pub(crate) unsafe fn compute_parallel_workers(
    heap_relation: pgrx::pg_sys::Relation,
    index_relation: pgrx::pg_sys::Relation,
) -> i32 {
    use pgrx::pg_sys::*;
    unsafe {
        if !IsUnderPostmaster
            || max_parallel_maintenance_workers == 0
            || (*(*heap_relation).rd_rel).relpersistence == RELPERSISTENCE_TEMP as std::ffi::c_char
        {
            return 0;
        }
        let options = (*heap_relation).rd_options.cast::<StdRdOptions>().as_ref();
        let requested = options.map_or(-1, |o| o.parallel_workers);
        let workers = if requested < 0 {
            max_parallel_maintenance_workers
        } else {
            requested.min(max_parallel_maintenance_workers)
        };
        let blocks = RelationGetNumberOfBlocksInFork(heap_relation, ForkNumber::MAIN_FORKNUM);
        let workers = workers.min(blocks.saturating_sub(1).min(i32::MAX as u32) as i32);
        if workers == 0 {
            return 0;
        }
        let expressions = RelationGetIndexExpressions(index_relation);
        let predicate = RelationGetIndexPredicate(index_relation);
        if !expressions.is_null() || !predicate.is_null() {
            // An index build has no query-level InitPlans or known-safe execution parameters.
            let mut glob = PlannerGlobal {
                type_: NodeTag::T_PlannerGlobal,
                maxParallelHazard: PROPARALLEL_UNSAFE as _,
                ..Default::default()
            };
            let mut root = PlannerInfo {
                type_: NodeTag::T_PlannerInfo,
                glob: &raw mut glob,
                ..Default::default()
            };
            if !is_parallel_safe(&raw mut root, expressions.cast())
                || !is_parallel_safe(&raw mut root, predicate.cast())
            {
                return 0;
            }
        }
        workers
    }
}

/// Mirror PostgreSQL's `pgstat_count_index_scan(rel)` macro so
/// `pg_stat_user_indexes.idx_scan` tracks index searches across PG 15–19.
#[inline]
pub(crate) unsafe fn count_index_scan(rel: pgrx::pg_sys::Relation) {
    unsafe {
        if (*rel).pgstat_info.is_null() {
            if !(*rel).pgstat_enabled {
                return;
            }
            pgrx::pg_sys::pgstat_assoc_relation(rel);
        }
        #[cfg(feature = "pg15")]
        {
            (*(*rel).pgstat_info).t_counts.t_numscans += 1;
        }
        #[cfg(any(feature = "pg16", feature = "pg17", feature = "pg18", feature = "pg19"))]
        {
            (*(*rel).pgstat_info).counts.numscans += 1;
        }
    }
}

/// A handler's C entry points without pgrx's guard: the planner calls it per operator class member.
macro_rules! am_handler {
    ($name:ident, $routine:expr) => {
        paste::paste! {
            #[unsafe(no_mangle)]
            #[doc(hidden)]
            pub extern "C" fn [<pg_finfo_ $name>]() -> &'static pgrx::pg_sys::Pg_finfo_record {
                &pgrx::pg_sys::Pg_finfo_record { api_version: 1 }
            }
            #[unsafe(no_mangle)]
            #[doc(hidden)]
            pub extern "C" fn $name(_: pgrx::pg_sys::FunctionCallInfo) -> pgrx::pg_sys::Datum {
                unsafe { crate::index::am_routine(&$routine) }
            }
        }
    };
}
pub(crate) use am_handler;

/// A handler's result: the routine itself where PostgreSQL takes it as static, else a copy it frees.
#[inline(always)]
pub unsafe fn am_routine(routine: &'static pgrx::pg_sys::IndexAmRoutine) -> pgrx::pg_sys::Datum {
    #[cfg(feature = "pg19")]
    let routine = std::ptr::from_ref(routine).cast_mut();
    #[cfg(not(feature = "pg19"))]
    let routine = unsafe {
        unsafe extern "C" {
            fn palloc(size: usize) -> *mut std::ffi::c_void;
        }
        let copy = palloc(size_of::<pgrx::pg_sys::IndexAmRoutine>())
            .cast::<pgrx::pg_sys::IndexAmRoutine>();
        copy.copy_from_nonoverlapping(routine, 1);
        copy
    };
    pgrx::pg_sys::Datum::from(routine)
}

/// Switches `CurrentMemoryContext` and restores it when dropped, including
/// during unwinding.
pub struct MemoryContextSwitch(pgrx::pg_sys::MemoryContext);

impl MemoryContextSwitch {
    pub unsafe fn new(context: pgrx::pg_sys::MemoryContext) -> Self {
        Self(unsafe { pgrx::pg_sys::MemoryContextSwitchTo(context) })
    }
}

impl Drop for MemoryContextSwitch {
    fn drop(&mut self) {
        unsafe {
            pgrx::pg_sys::MemoryContextSwitchTo(self.0);
        }
    }
}
