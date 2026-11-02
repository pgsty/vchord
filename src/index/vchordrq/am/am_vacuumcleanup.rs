// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use crate::index::vchordrq::am::PostgresRelation;
use crate::index::vchordrq::opclass::opfamily;
use vchordrq::MaintainChooser;

#[pgrx::pg_guard]
pub unsafe extern "C-unwind" fn amvacuumcleanup(
    info: *mut pgrx::pg_sys::IndexVacuumInfo,
    stats: *mut pgrx::pg_sys::IndexBulkDeleteResult,
) -> *mut pgrx::pg_sys::IndexBulkDeleteResult {
    let mut stats = stats;
    if stats.is_null() {
        stats = unsafe {
            pgrx::pg_sys::palloc0(size_of::<pgrx::pg_sys::IndexBulkDeleteResult>()).cast()
        };
    }
    let index_relation = unsafe { (*info).index };
    unsafe {
        sequential_vacuumcleanup(index_relation, || (), || ());
    }
    stats
}

unsafe fn sequential_vacuumcleanup(
    index_relation: pgrx::pg_sys::Relation,
    sync_0: impl FnOnce(),
    sync_1: impl FnOnce(),
) {
    struct ChooseAll;
    impl MaintainChooser for ChooseAll {
        fn choose(&mut self, _: usize) -> bool {
            true
        }
    }

    let opfamily = unsafe { opfamily(index_relation) };
    let index = unsafe { PostgresRelation::new(index_relation) };
    let check = || unsafe {
        #[cfg(any(feature = "pg14", feature = "pg15", feature = "pg16", feature = "pg17"))]
        pgrx::pg_sys::vacuum_delay_point();
        #[cfg(any(feature = "pg18", feature = "pg19"))]
        pgrx::pg_sys::vacuum_delay_point(false);
    };

    sync_0();

    let mut chooser = ChooseAll;
    crate::index::vchordrq::dispatch::maintain(opfamily, &index, &mut chooser, check);

    sync_1();
}
