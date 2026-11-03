// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use crate::{VectorBorrowed, VectorOwned};
use distance::Distance;

#[derive(Debug, Clone)]
pub struct Rabitq4Owned {
    dim: u32,
    sum_of_x2: f32,
    norm_of_lattice: f32,
    sum_of_code: f32,
    sum_of_abs_x: f32,
    packed_code: Vec<u8>,
}

impl Rabitq4Owned {
    #[inline(always)]
    pub fn new(
        dim: u32,
        sum_of_x2: f32,
        norm_of_lattice: f32,
        sum_of_code: f32,
        sum_of_abs_x: f32,
        packed_code: Vec<u8>,
    ) -> Self {
        Self::new_checked(
            dim,
            sum_of_x2,
            norm_of_lattice,
            sum_of_code,
            sum_of_abs_x,
            packed_code,
        )
        .expect("invalid data")
    }

    #[inline(always)]
    pub fn new_checked(
        dim: u32,
        sum_of_x2: f32,
        norm_of_lattice: f32,
        sum_of_code: f32,
        sum_of_abs_x: f32,
        packed_code: Vec<u8>,
    ) -> Option<Self> {
        if !(1..=65535).contains(&dim) {
            return None;
        }
        if dim.div_ceil(2) as usize != packed_code.len() {
            return None;
        }
        if dim % 2 == 1 && packed_code.last().copied().unwrap_or_default() >> 4 != 0 {
            return None;
        }
        #[allow(unsafe_code)]
        Some(unsafe {
            Self::new_unchecked(
                dim,
                sum_of_x2,
                norm_of_lattice,
                sum_of_code,
                sum_of_abs_x,
                packed_code,
            )
        })
    }

    /// # Safety
    ///
    /// * `dim` must not be zero.
    /// * `dim` must be less than 65536.
    /// * `dim` must be equal to 1/2 of `code.len()`, rounding to infinity.
    /// * `packed_code` is filled with zero bits correctly.
    #[allow(unsafe_code)]
    #[inline(always)]
    pub unsafe fn new_unchecked(
        dim: u32,
        sum_of_x2: f32,
        norm_of_lattice: f32,
        sum_of_code: f32,
        sum_of_abs_x: f32,
        packed_code: Vec<u8>,
    ) -> Self {
        Self {
            dim,
            sum_of_x2,
            norm_of_lattice,
            sum_of_code,
            sum_of_abs_x,
            packed_code,
        }
    }
}

impl VectorOwned for Rabitq4Owned {
    type Borrowed<'a> = Rabitq4Borrowed<'a>;

    #[inline(always)]
    fn as_borrowed(&self) -> Rabitq4Borrowed<'_> {
        Rabitq4Borrowed {
            dim: self.dim,
            sum_of_x2: self.sum_of_x2,
            norm_of_lattice: self.norm_of_lattice,
            sum_of_code: self.sum_of_code,
            sum_of_abs_x: self.sum_of_abs_x,
            packed_code: self.packed_code.as_slice(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Rabitq4Borrowed<'a> {
    dim: u32,
    sum_of_x2: f32,
    norm_of_lattice: f32,
    sum_of_code: f32,
    sum_of_abs_x: f32,
    packed_code: &'a [u8],
}

impl<'a> Rabitq4Borrowed<'a> {
    #[inline(always)]
    pub fn new(
        dim: u32,
        sum_of_x2: f32,
        norm_of_lattice: f32,
        sum_of_code: f32,
        sum_of_abs_x: f32,
        packed_code: &'a [u8],
    ) -> Self {
        Self::new_checked(
            dim,
            sum_of_x2,
            norm_of_lattice,
            sum_of_code,
            sum_of_abs_x,
            packed_code,
        )
        .expect("invalid data")
    }

    #[inline(always)]
    pub fn new_checked(
        dim: u32,
        sum_of_x2: f32,
        norm_of_lattice: f32,
        sum_of_code: f32,
        sum_of_abs_x: f32,
        packed_code: &'a [u8],
    ) -> Option<Self> {
        // The on-disk header stores `dim` as `u16`, so bound the dimension
        // itself rather than the packed length (which is about half of it).
        if !(1..=65535).contains(&dim) {
            return None;
        }
        if dim.div_ceil(2) as usize != packed_code.len() {
            return None;
        }
        if dim % 2 == 1 && packed_code.last().copied().unwrap_or_default() >> 4 != 0 {
            return None;
        }
        #[allow(unsafe_code)]
        Some(unsafe {
            Self::new_unchecked(
                dim,
                sum_of_x2,
                norm_of_lattice,
                sum_of_code,
                sum_of_abs_x,
                packed_code,
            )
        })
    }

    /// # Safety
    ///
    /// * `dim` must not be zero.
    /// * `dim` must be less than 65536.
    /// * `dim` must be equal to 1/2 of `code.len()`, rounding to infinity.
    /// * `packed_code` is filled with zero bits correctly.
    #[allow(unsafe_code)]
    #[inline(always)]
    pub unsafe fn new_unchecked(
        dim: u32,
        sum_of_x2: f32,
        norm_of_lattice: f32,
        sum_of_code: f32,
        sum_of_abs_x: f32,
        packed_code: &'a [u8],
    ) -> Self {
        Self {
            dim,
            sum_of_x2,
            norm_of_lattice,
            sum_of_code,
            sum_of_abs_x,
            packed_code,
        }
    }

    #[inline(always)]
    pub fn sum_of_x2(&self) -> f32 {
        self.sum_of_x2
    }

    #[inline(always)]
    pub fn norm_of_lattice(&self) -> f32 {
        self.norm_of_lattice
    }

    #[inline(always)]
    pub fn sum_of_code(&self) -> f32 {
        self.sum_of_code
    }

    #[inline(always)]
    pub fn sum_of_abs_x(&self) -> f32 {
        self.sum_of_abs_x
    }

    #[inline(always)]
    pub fn packed_code(&self) -> &'a [u8] {
        self.packed_code
    }

    #[inline(always)]
    pub fn unpacked_code(&self) -> impl Iterator<Item = u8> {
        self.packed_code
            .iter()
            .flat_map(|x| [x & 0xf, x >> 4])
            .take(self.dim as _)
    }
}

impl VectorBorrowed for Rabitq4Borrowed<'_> {
    type Owned = Rabitq4Owned;

    #[inline(always)]
    fn dim(&self) -> u32 {
        self.dim
    }

    #[inline(always)]
    fn own(&self) -> Rabitq4Owned {
        Rabitq4Owned {
            dim: self.dim,
            sum_of_x2: self.sum_of_x2,
            norm_of_lattice: self.norm_of_lattice,
            sum_of_code: self.sum_of_code,
            sum_of_abs_x: self.sum_of_abs_x,
            packed_code: self.packed_code.to_owned(),
        }
    }

    #[inline(always)]
    fn norm(&self) -> f32 {
        self.sum_of_x2.sqrt()
    }

    #[inline(always)]
    fn operator_dot(self, rhs: Self) -> Distance {
        let dim = self.dim();
        let sum = rabitq::halfbyte::binary::accumulate(self.packed_code, rhs.packed_code);
        Distance::from_f32(
            rabitq::halfbyte::binary::half_process_dot(
                dim,
                sum,
                rabitq::halfbyte::CodeMetadata {
                    dis_u_2: self.sum_of_x2,
                    norm_of_lattice: self.norm_of_lattice,
                    sum_of_code: self.sum_of_code,
                },
                rabitq::halfbyte::CodeMetadata {
                    dis_u_2: rhs.sum_of_x2,
                    norm_of_lattice: rhs.norm_of_lattice,
                    sum_of_code: rhs.sum_of_code,
                },
            )
            .0,
        )
    }

    #[inline(always)]
    fn operator_l2s(self, rhs: Self) -> Distance {
        let dim = self.dim();
        let sum = rabitq::halfbyte::binary::accumulate(self.packed_code, rhs.packed_code);
        Distance::from_f32(
            rabitq::halfbyte::binary::half_process_l2s(
                dim,
                sum,
                rabitq::halfbyte::CodeMetadata {
                    dis_u_2: self.sum_of_x2,
                    norm_of_lattice: self.norm_of_lattice,
                    sum_of_code: self.sum_of_code,
                },
                rabitq::halfbyte::CodeMetadata {
                    dis_u_2: rhs.sum_of_x2,
                    norm_of_lattice: rhs.norm_of_lattice,
                    sum_of_code: rhs.sum_of_code,
                },
            )
            .0,
        )
    }

    #[inline(always)]
    fn operator_cos(self, rhs: Self) -> Distance {
        let dim = self.dim();
        let sum = rabitq::halfbyte::binary::accumulate(self.packed_code, rhs.packed_code);
        Distance::from_f32(
            rabitq::halfbyte::binary::half_process_cos(
                dim,
                sum,
                rabitq::halfbyte::CodeMetadata {
                    dis_u_2: self.sum_of_x2,
                    norm_of_lattice: self.norm_of_lattice,
                    sum_of_code: self.sum_of_code,
                },
                rabitq::halfbyte::CodeMetadata {
                    dis_u_2: rhs.sum_of_x2,
                    norm_of_lattice: rhs.norm_of_lattice,
                    sum_of_code: rhs.sum_of_code,
                },
            )
            .0,
        )
    }

    #[inline(always)]
    fn operator_hamming(self, _: Self) -> Distance {
        unimplemented!()
    }

    #[inline(always)]
    fn operator_jaccard(self, _: Self) -> Distance {
        unimplemented!()
    }

    #[inline(always)]
    fn function_normalize(&self) -> Rabitq4Owned {
        Rabitq4Owned {
            dim: self.dim,
            sum_of_x2: 1.0,
            norm_of_lattice: self.norm_of_lattice,
            sum_of_code: self.sum_of_code,
            sum_of_abs_x: self.sum_of_abs_x / self.sum_of_x2.sqrt(),
            packed_code: self.packed_code.to_owned(),
        }
    }

    fn operator_add(&self, _: Self) -> Self::Owned {
        unimplemented!()
    }

    fn operator_sub(&self, _: Self) -> Self::Owned {
        unimplemented!()
    }

    fn operator_mul(&self, _: Self) -> Self::Owned {
        unimplemented!()
    }

    fn operator_and(&self, _: Self) -> Self::Owned {
        unimplemented!()
    }

    fn operator_or(&self, _: Self) -> Self::Owned {
        unimplemented!()
    }

    fn operator_xor(&self, _: Self) -> Self::Owned {
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_checked_bounds_dim() {
        // 65536 dimensions pack into 32768 bytes, but `dim` must fit in u16.
        let code = vec![0u8; 32768];
        assert!(Rabitq4Borrowed::new_checked(65536, 1.0, 1.0, 0.0, 1.0, &code).is_none());
        assert!(Rabitq4Borrowed::new_checked(65535, 1.0, 1.0, 0.0, 1.0, &code).is_some());
        assert!(Rabitq4Borrowed::new_checked(0, 1.0, 1.0, 0.0, 1.0, &[]).is_none());
        assert!(Rabitq4Owned::new_checked(65536, 1.0, 1.0, 0.0, 1.0, code).is_none());
    }
}
