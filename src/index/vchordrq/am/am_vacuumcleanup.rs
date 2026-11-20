// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use crate::index::vchordrq::am::PostgresRelation;
use crate::index::vchordrq::dispatch;
use crate::index::vchordrq::opclass::opfamily;
use std::cell::Cell;
use vchordrq::MaintainChooser;

#[pgrx::pg_guard]
pub unsafe extern "C-unwind" fn amvacuumcleanup(
    info: *mut pgrx::pg_sys::IndexVacuumInfo,
    stats: *mut pgrx::pg_sys::IndexBulkDeleteResult,
) -> *mut pgrx::pg_sys::IndexBulkDeleteResult {
    // ANALYZE calls this without a preceding bulk delete; compacting the whole
    // index there only writes WAL. Like nbtree, do nothing and report nothing.
    if unsafe { (*info).analyze_only } {
        return stats;
    }
    let mut stats = stats;
    if stats.is_null() {
        stats = unsafe {
            pgrx::pg_sys::palloc0(size_of::<pgrx::pg_sys::IndexBulkDeleteResult>()).cast()
        };
    }
    let index_relation = unsafe { (*info).index };
    let tuples = unsafe { sequential_vacuumcleanup(index_relation, || (), || ()) };
    unsafe {
        // Compacted and skipped lists are both counted, each when it is
        // visited; tuples inserted into a list after that are not, as in
        // 1.2.0 (which counted during compaction) and in btree.
        (*stats).num_index_tuples = tuples as f64;
        (*stats).estimated_count = false;
        (*stats).num_pages = pgrx::pg_sys::RelationGetNumberOfBlocksInFork(
            index_relation,
            pgrx::pg_sys::ForkNumber::MAIN_FORKNUM,
        );
    }
    stats
}

unsafe fn sequential_vacuumcleanup(
    index_relation: pgrx::pg_sys::Relation,
    sync_0: impl FnOnce(),
    sync_1: impl FnOnce(),
) -> u64 {
    struct ChooseAll;
    impl MaintainChooser for ChooseAll {
        fn choose(&mut self, _: usize) -> bool {
            true
        }
        // Lists with nothing to compact are left alone: VACUUM used to
        // rewrite and WAL-log every list, even with nothing deleted (#24).
        fn skip_unchanged(&self) -> bool {
            true
        }
    }

    let opfamily = unsafe { opfamily(index_relation) };
    let index = unsafe { PostgresRelation::new(index_relation) };
    let pending = vchordrq::Pending::default();
    let held = unsafe { pgrx::pg_sys::InterruptHoldoffCount };
    let checking = Cell::new(false);
    // interrupts are served only here, where maintain locks no page: one more held is a buffer lock
    let check = || unsafe {
        debug_assert_eq!({ pgrx::pg_sys::InterruptHoldoffCount }, held + 1);
        checking.set(true);
        pgrx::pg_sys::InterruptHoldoffCount -= 1;
        #[cfg(any(feature = "pg15", feature = "pg16", feature = "pg17"))]
        pgrx::pg_sys::vacuum_delay_point();
        #[cfg(any(feature = "pg18", feature = "pg19"))]
        pgrx::pg_sys::vacuum_delay_point(false);
        pgrx::pg_sys::InterruptHoldoffCount += 1;
        checking.set(false);
    };
    let give_back = || {
        if checking.get() {
            pending.give_back(&index);
        }
    };

    sync_0();

    let mut chooser = ChooseAll;
    let tuples = unsafe {
        pgrx::pg_sys::InterruptHoldoffCount += 1;
        let maintain = || dispatch::maintain(opfamily, &index, &mut chooser, &pending, check);
        let tuples = ensure_cleanup(&give_back, maintain);
        pgrx::pg_sys::InterruptHoldoffCount -= 1;
        tuples
    };

    sync_1();

    tuples
}

/// Runs `f`; if it raises an error or the backend exits in it, runs `cleanup` first, with interrupts held.
unsafe fn ensure_cleanup<T>(cleanup: &dyn Fn(), f: impl FnOnce() -> T) -> T {
    #[pgrx::pg_guard]
    unsafe extern "C-unwind" fn exit(_code: std::ffi::c_int, arg: pgrx::pg_sys::Datum) {
        unsafe { (*arg.cast_mut_ptr::<&dyn Fn()>())() };
    }
    let arg = pgrx::pg_sys::Datum::from(&raw const cleanup);
    unsafe { pgrx::pg_sys::before_shmem_exit(Some(exit), arg) };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    unsafe { pgrx::pg_sys::cancel_before_shmem_exit(Some(exit), arg) };
    result.unwrap_or_else(|error| unsafe {
        pgrx::pg_sys::InterruptHoldoffCount += 1;
        cleanup();
        pgrx::pg_sys::InterruptHoldoffCount -= 1;
        std::panic::resume_unwind(error)
    })
}
