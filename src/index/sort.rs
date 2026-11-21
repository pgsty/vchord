// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use pgrx::pg_sys::{BufFile, Datum, SortCoordinateData, Tuplesortstate, varlena};

/// Rows sorted by their bytes, through PostgreSQL's tuplesort of `bytea`.
pub struct Sort {
    state: *mut Tuplesortstate,
    /// The row being put, after its header; the sort copies it.
    row: *mut varlena,
    capacity: usize,
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
            row: std::ptr::null_mut(),
            capacity: 0,
            #[cfg(feature = "pg15")]
            last: std::ptr::null_mut(),
        }
    }
    pub fn put(&mut self, bytes: &[u8]) {
        let size = size_of::<u32>() + bytes.len();
        unsafe {
            if self.capacity < size {
                self.row = match self.row.is_null() {
                    true => pgrx::pg_sys::palloc(size),
                    false => pgrx::pg_sys::repalloc(self.row.cast(), size),
                }
                .cast();
                self.capacity = size;
            }
            // the header of an uncompressed value, as SET_VARSIZE_4B writes it
            let header = match cfg!(target_endian = "little") {
                true => (size as u32) << 2,
                false => size as u32 & 0x3fff_ffff,
            };
            self.row.cast::<u32>().write(header);
            let data = self.row.cast::<u8>().add(size_of::<u32>());
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), data, bytes.len());
            pgrx::pg_sys::tuplesort_putdatum(self.state, Datum::from(self.row), false);
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

/// Pieces in a temporary file, read back in the order written.
pub struct Spill {
    file: *mut BufFile,
    pieces: usize,
    piece: Vec<u8>,
}

impl Default for Spill {
    fn default() -> Self {
        Self {
            file: std::ptr::null_mut(),
            pieces: 0,
            piece: Vec::new(),
        }
    }
}

impl vchordrq::Spill for Spill {
    fn put(&mut self, bytes: &[u8]) {
        use pgrx::pg_sys::{BufFileCreateTemp, BufFileWrite};
        unsafe {
            if self.file.is_null() {
                self.file = BufFileCreateTemp(false);
            }
            let len = bytes.len().to_ne_bytes();
            BufFileWrite(self.file, len.as_ptr().cast_mut().cast(), len.len());
            BufFileWrite(self.file, bytes.as_ptr().cast_mut().cast(), bytes.len());
        }
        self.pieces += 1;
    }
    fn take(&mut self, mut f: impl FnMut(&[u8])) {
        use pgrx::pg_sys::{BufFileClose, BufFileRead, BufFileSeekBlock};
        if self.pieces == 0 {
            return;
        }
        let read = |file, out: &mut [u8]| {
            let size = unsafe { BufFileRead(file, out.as_mut_ptr().cast(), out.len()) };
            assert_eq!(size, out.len(), "could not read a temporary file");
        };
        unsafe {
            assert_eq!(
                BufFileSeekBlock(self.file, 0),
                0,
                "could not seek in a temporary file"
            );
        }
        for _ in 0..std::mem::take(&mut self.pieces) {
            let mut len = [0; size_of::<usize>()];
            read(self.file, &mut len);
            self.piece.resize(usize::from_ne_bytes(len), 0);
            read(self.file, &mut self.piece);
            f(&self.piece);
        }
        self.piece = Vec::new();
        unsafe {
            BufFileClose(std::mem::replace(&mut self.file, std::ptr::null_mut()));
        }
    }
}

impl Drop for Spill {
    fn drop(&mut self) {
        // after an error, PostgreSQL closes the file with its resource owner
        if !self.file.is_null() && crate::index::cleanup::pg_cleanup_allowed() {
            unsafe {
                pgrx::pg_sys::BufFileClose(self.file);
            }
        }
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
