// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use crate::datatype::memory_halfvec::{HalfvecInput, HalfvecOutput};
use crate::datatype::memory_rabitq4::{Rabitq4Input, Rabitq4Output};
use crate::datatype::memory_vector::{VectorInput, VectorOutput};
use crate::datatype::rabitq_io;
use simd::{Floating, f16};
use vector::VectorBorrowed;
use vector::rabitq4::Rabitq4Borrowed;
use vector::vect::VectBorrowed;

fn quantized(
    dim: u32,
    metadata: rabitq::halfbyte::CodeMetadata,
    sum_of_abs_x: f32,
    elements: &[u8],
) -> Rabitq4Output {
    let metadata = rabitq_io::Metadata {
        sum_of_x2: metadata.dis_u_2,
        norm_of_lattice: metadata.norm_of_lattice,
        sum_of_code: metadata.sum_of_code,
        sum_of_abs_x,
    };
    // Non-finite metadata makes every distance NaN. Input functions still
    // accept it so that values stored by earlier versions can be restored.
    rabitq_io::check_quantized("rabitq4", metadata);
    Rabitq4Output::new(Rabitq4Borrowed::new(
        dim,
        metadata.sum_of_x2,
        metadata.norm_of_lattice,
        metadata.sum_of_code,
        metadata.sum_of_abs_x,
        elements,
    ))
}

#[pgrx::pg_extern(sql = "")]
fn _vchord_vector_quantize_to_rabitq4(vector: VectorInput) -> Rabitq4Output {
    let vector = vector.as_borrowed();
    let dim = vector.dim();
    let mut vector = vector.slice().to_vec();
    rabitq::rotate::rotate_inplace(&mut vector);
    let (metadata, elements) = rabitq::halfbyte::ugly_code(&vector);
    let elements = rabitq::halfbyte::pack_code(&elements);
    quantized(dim, metadata, f32::reduce_sum_of_abs_x(&vector), &elements)
}

#[pgrx::pg_extern(sql = "")]
fn _vchord_halfvec_quantize_to_rabitq4(vector: HalfvecInput) -> Rabitq4Output {
    let vector = vector.as_borrowed();
    let dim = vector.dim();
    let mut vector = f16::vector_to_f32(vector.slice());
    rabitq::rotate::rotate_inplace(&mut vector);
    let (metadata, elements) = rabitq::halfbyte::ugly_code(&vector);
    let elements = rabitq::halfbyte::pack_code(&elements);
    quantized(dim, metadata, f32::reduce_sum_of_abs_x(&vector), &elements)
}

#[pgrx::pg_extern(sql = "")]
fn _vchord_rabitq4_dequantize_to_vector(vector: Rabitq4Input) -> VectorOutput {
    let vector = vector.as_borrowed();
    let scale = vector.sum_of_x2().sqrt() / vector.norm_of_lattice();
    let mut result = Vec::with_capacity(vector.dim() as _);
    for c in vector.unpacked_code() {
        let base = -0.5 * ((1 << 4) - 1) as f32;
        result.push((base + c as f32) * scale);
    }
    rabitq::rotate::rotate_reversed_inplace(&mut result);
    VectorOutput::new(VectBorrowed::new(&result))
}

#[pgrx::pg_extern(sql = "")]
fn _vchord_rabitq4_dequantize_to_halfvec(vector: Rabitq4Input) -> HalfvecOutput {
    let vector = vector.as_borrowed();
    let scale = vector.sum_of_x2().sqrt() / vector.norm_of_lattice();
    let mut result = Vec::with_capacity(vector.dim() as _);
    for c in vector.unpacked_code() {
        let base = -0.5 * ((1 << 4) - 1) as f32;
        result.push((base + c as f32) * scale);
    }
    rabitq::rotate::rotate_reversed_inplace(&mut result);
    let result = f16::vector_from_f32(&result);
    HalfvecOutput::new(VectBorrowed::new(&result))
}
