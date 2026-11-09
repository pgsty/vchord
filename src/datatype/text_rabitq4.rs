// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use crate::datatype::memory_rabitq4::{Rabitq4Input, Rabitq4Output};
use crate::datatype::rabitq_io;
use pgrx::pg_sys::Oid;
use std::ffi::{CStr, CString};
use vector::rabitq4::Rabitq4Borrowed;

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchord_rabitq4_in(input: &CStr, oid: Oid, typmod: i32) -> Rabitq4Output {
    // Typmod stays unenforced until #31: INSERT passes -1, so checking only
    // COPY/receive would make dumps of accepted rows unrestorable.
    let _ = (oid, typmod);
    let (metadata, unpacked_code) = rabitq_io::parse_text(input, "rabitq4", 15);
    let packed_code = rabitq::halfbyte::pack_code(&unpacked_code);
    let Some(x) = Rabitq4Borrowed::new_checked(
        unpacked_code.len() as _,
        metadata.sum_of_x2,
        metadata.norm_of_lattice,
        metadata.sum_of_code,
        metadata.sum_of_abs_x,
        &packed_code,
    ) else {
        pgrx::error!("invalid input syntax for type rabitq4");
    };
    Rabitq4Output::new(x)
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchord_rabitq4_out(vector: Rabitq4Input<'_>) -> CString {
    let vector = vector.as_borrowed();
    let mut buffer = String::new();
    buffer.push('(');
    buffer.push_str(format!("{}", vector.sum_of_x2()).as_str());
    buffer.push_str(format!(", {}", vector.norm_of_lattice()).as_str());
    buffer.push_str(format!(", {}", vector.sum_of_code()).as_str());
    buffer.push_str(format!(", {}", vector.sum_of_abs_x()).as_str());
    buffer.push(')');
    buffer.push('[');
    let mut unpacked_code = vector.unpacked_code();
    if let Some(x) = unpacked_code.next() {
        buffer.push_str(format!("{x}").as_str());
    }
    for x in unpacked_code {
        buffer.push_str(format!(", {x}").as_str());
    }
    buffer.push(']');
    CString::new(buffer).unwrap()
}
