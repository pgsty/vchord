// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

pub use crate::extended::{Code, CodeMetadata};

#[deprecated]
pub fn code(vector: &[f32]) -> Code {
    crate::extended::code::<4>(vector)
}

pub fn ugly_code(vector: &[f32]) -> Code {
    crate::extended::ugly_code::<4>(vector)
}

pub fn pack_code(input: &[u8]) -> Vec<u8> {
    let f = |t: &[u8; 2]| t[0] | t[1] << 4;
    let (arrays, remainder) = input.as_chunks::<2>();
    let mut buffer = [0u8; 2];
    let tailing = if !remainder.is_empty() {
        buffer[..remainder.len()].copy_from_slice(remainder);
        Some(&buffer)
    } else {
        None
    };
    arrays.iter().chain(tailing).map(f).collect()
}

pub mod binary {
    use crate::extended::CodeMetadata;

    const BITS: usize = 4;

    pub type BinaryLutMetadata = CodeMetadata;
    pub type BinaryLut = (BinaryLutMetadata, Vec<u8>);
    pub type BinaryCode<'a> = ((f32, f32, f32, f32), &'a [u8]);

    #[deprecated]
    pub fn preprocess(vector: &[f32]) -> BinaryLut {
        let (metadata, elements) = crate::extended::code::<BITS>(vector);
        (metadata, super::pack_code(&elements))
    }

    pub fn ugly_preprocess(vector: &[f32]) -> BinaryLut {
        let (metadata, elements) = crate::extended::ugly_code::<BITS>(vector);
        (metadata, super::pack_code(&elements))
    }

    pub fn accumulate(x: &[u8], y: &[u8]) -> u32 {
        simd::halfbyte::reduce_sum_of_xy(x, y)
    }

    pub fn half_process_dot(
        dim: u32,
        sum: u32,
        code: CodeMetadata,
        lut: BinaryLutMetadata,
    ) -> (f32,) {
        let rough = crate::extended::half_process_dot::<4, BITS>(dim, sum, code, lut);
        (rough,)
    }

    pub fn half_process_l2s(
        dim: u32,
        sum: u32,
        code: CodeMetadata,
        lut: BinaryLutMetadata,
    ) -> (f32,) {
        let rough = crate::extended::half_process_l2s::<4, BITS>(dim, sum, code, lut);
        (rough,)
    }

    pub fn half_process_cos(
        dim: u32,
        sum: u32,
        code: CodeMetadata,
        lut: BinaryLutMetadata,
    ) -> (f32,) {
        let rough = crate::extended::half_process_cos::<4, BITS>(dim, sum, code, lut);
        (rough,)
    }
}
