// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use crate::datatype::memory_rabitq8::{Rabitq8Input, Rabitq8Output};
use crate::datatype::rabitq_io;
use pgrx::datum::Internal;
use pgrx::pg_sys::Oid;
use vector::VectorBorrowed;
use vector::rabitq8::Rabitq8Borrowed;

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchord_rabitq8_send(vector: Rabitq8Input<'_>) -> Vec<u8> {
    let vector = vector.as_borrowed();
    let mut stream = Vec::<u8>::new();
    stream.extend(vector.dim().to_be_bytes());
    stream.extend(vector.sum_of_x2().to_be_bytes());
    stream.extend(vector.norm_of_lattice().to_be_bytes());
    stream.extend(vector.sum_of_code().to_be_bytes());
    stream.extend(vector.sum_of_abs_x().to_be_bytes());
    for &c in vector.packed_code() {
        stream.extend(c.to_be_bytes());
    }
    stream
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchord_rabitq8_recv(mut internal: Internal, oid: Oid, typmod: i32) -> Rabitq8Output {
    let _ = oid;
    let buf = unsafe { internal.get_mut::<pgrx::pg_sys::StringInfoData>().unwrap() };
    let mut reader = rabitq_io::Reader::new(buf);

    let dim = reader.u32();
    rabitq_io::check_dim("rabitq8", dim);
    rabitq_io::check_typmod("rabitq8", typmod, dim);
    let metadata = reader.metadata();
    let packed_code = reader.bytes(dim.div_ceil(1) as usize);

    let Some(x) = Rabitq8Borrowed::new_checked(
        dim,
        metadata.sum_of_x2,
        metadata.norm_of_lattice,
        metadata.sum_of_code,
        metadata.sum_of_abs_x,
        packed_code,
    ) else {
        pgrx::ereport!(
            ERROR,
            pgrx::PgSqlErrorCode::ERRCODE_INVALID_BINARY_REPRESENTATION,
            "invalid binary data for type rabitq8"
        );
    };
    Rabitq8Output::new(x)
}
