// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use simd::f16;
use vector::rabitq4::Rabitq4Borrowed;
use vector::rabitq8::Rabitq8Borrowed;
use vector::vect::VectBorrowed;

pub fn vector_out(vector: VectBorrowed<'_, f32>) -> String {
    let mut result = String::from("[");
    for x in vector.slice() {
        if !result.ends_with('[') {
            result.push(',');
        }
        result.push_str(&x.to_string());
    }
    result.push(']');
    result
}

pub fn halfvec_out(vector: VectBorrowed<'_, f16>) -> String {
    let mut result = String::from("[");
    for x in vector.slice() {
        if !result.ends_with('[') {
            result.push(',');
        }
        result.push_str(&x.to_string());
    }
    result.push(']');
    result
}

pub fn rabitq8_out(vector: Rabitq8Borrowed<'_>) -> String {
    let mut result = String::new();
    result.push('(');
    result.push_str(&vector.sum_of_x2().to_string());
    result.push(',');
    result.push_str(&vector.norm_of_lattice().to_string());
    result.push(',');
    result.push_str(&vector.sum_of_code().to_string());
    result.push(',');
    result.push_str(&vector.sum_of_abs_x().to_string());
    result.push(')');
    result.push('[');
    for x in vector.unpacked_code() {
        if !result.ends_with('[') {
            result.push(',');
        }
        result.push_str(&x.to_string());
    }
    result.push(']');
    result
}

pub fn rabitq4_out(vector: Rabitq4Borrowed<'_>) -> String {
    let mut result = String::new();
    result.push('(');
    result.push_str(&vector.sum_of_x2().to_string());
    result.push(',');
    result.push_str(&vector.norm_of_lattice().to_string());
    result.push(',');
    result.push_str(&vector.sum_of_code().to_string());
    result.push(',');
    result.push_str(&vector.sum_of_abs_x().to_string());
    result.push(')');
    result.push('[');
    for x in vector.unpacked_code() {
        if !result.ends_with('[') {
            result.push(',');
        }
        result.push_str(&x.to_string());
    }
    result.push(']');
    result
}
