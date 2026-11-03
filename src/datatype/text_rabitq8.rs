// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use crate::datatype::memory_rabitq8::{Rabitq8Input, Rabitq8Output};
use crate::datatype::rabitq_io;
use pgrx::pg_sys::Oid;
use std::ffi::{CStr, CString};
use vector::rabitq8::Rabitq8Borrowed;

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchord_rabitq8_in(input: &CStr, oid: Oid, typmod: i32) -> Rabitq8Output {
    let _ = (oid, typmod);
    let (metadata, unpacked_code) = rabitq_io::parse_text(input, "rabitq8", 255);
    let packed_code = rabitq::byte::pack_code(&unpacked_code);
    let Some(x) = Rabitq8Borrowed::new_checked(
        unpacked_code.len() as _,
        metadata.sum_of_x2,
        metadata.norm_of_lattice,
        metadata.sum_of_code,
        metadata.sum_of_abs_x,
        &packed_code,
    ) else {
        pgrx::error!("invalid input syntax for type rabitq8");
    };
    Rabitq8Output::new(x)
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchord_rabitq8_out(vector: Rabitq8Input<'_>) -> CString {
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
