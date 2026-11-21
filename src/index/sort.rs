// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use pgrx::pg_sys::{Datum, SortCoordinateData, Tuplesortstate, varlena};

/// Rows sorted by their bytes, through PostgreSQL's tuplesort of `bytea`.
pub struct Sort {
    state: *mut Tuplesortstate,
    /// The row last taken, which this owns before PostgreSQL 16.
    #[cfg(feature = "pg15")]
    last: *mut varlena,
}

impl Sort {
    /// `memory` in kilobytes; `coordinate`, if it is a participant or the leader of a parallel sort.
    pub unsafe fn new(memory: i32, coordinate: Option<SortCoordinateData>) -> Self {
        use pgrx::pg_sys::{
            BYTEAOID, InvalidOid, TUPLESORT_NONE, TYPECACHE_LT_OPR, lookup_type_cache,
        };
        let mut coordinate = coordinate;
        let coordinate = coordinate
            .as_mut()
            .map_or(std::ptr::null_mut(), |x| x as *mut _);
        let state = unsafe {
            let lt_opr = (*lookup_type_cache(BYTEAOID, TYPECACHE_LT_OPR as _)).lt_opr;
            pgrx::pg_sys::tuplesort_begin_datum(
                BYTEAOID,
                lt_opr,
                InvalidOid,
                false,
                memory.max(64),
                coordinate,
                TUPLESORT_NONE as _,
            )
        };
        Self {
            state,
            #[cfg(feature = "pg15")]
            last: std::ptr::null_mut(),
        }
    }
    pub fn put(&mut self, bytes: &[u8]) {
        let datum = pgrx::rust_byte_slice_to_bytea(bytes).into_pg();
        unsafe {
            pgrx::pg_sys::tuplesort_putdatum(self.state, Datum::from(datum), false);
            pgrx::pg_sys::pfree(datum.cast());
        }
    }
    pub fn perform(&mut self) {
        unsafe {
            pgrx::pg_sys::tuplesort_performsort(self.state);
        }
    }
}

impl vchordrq::Sorted for Sort {
    fn next(&mut self) -> Option<&[u8]> {
        let mut datum = Datum::null();
        let mut is_null = false;
        #[cfg(feature = "pg15")]
        let found = unsafe {
            if !self.last.is_null() {
                pgrx::pg_sys::pfree(self.last.cast());
                self.last = std::ptr::null_mut();
            }
            let found = pgrx::pg_sys::tuplesort_getdatum(
                self.state,
                true,
                &mut datum,
                &mut is_null,
                std::ptr::null_mut(),
            );
            self.last = datum.cast_mut_ptr();
            found
        };
        #[cfg(not(feature = "pg15"))]
        let found = unsafe {
            pgrx::pg_sys::tuplesort_getdatum(
                self.state,
                true,
                false,
                &mut datum,
                &mut is_null,
                std::ptr::null_mut(),
            )
        };
        found.then(|| unsafe {
            pgrx::varlena::varlena_to_byte_slice(datum.cast_mut_ptr::<varlena>())
        })
    }
}

impl Drop for Sort {
    fn drop(&mut self) {
        // after an error, PostgreSQL ends the sort with its memory context and resource owner
        if crate::index::cleanup::pg_cleanup_allowed() {
            unsafe {
                pgrx::pg_sys::tuplesort_end(self.state);
            }
        }
    }
}
