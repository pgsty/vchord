// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

pub trait PackedRefMut {
    type T;
    fn get(&self) -> &Self::T;
    fn get_mut(&mut self) -> &mut Self::T;
}

impl<T> PackedRefMut for &mut T {
    type T = T;
    #[inline(always)]
    fn get(&self) -> &T {
        self
    }
    #[inline(always)]
    fn get_mut(&mut self) -> &mut T {
        self
    }
}

#[repr(Rust, packed(4))]
pub struct PackedRefMut4<'b, T>(pub &'b mut T);

impl<'b, T> PackedRefMut for PackedRefMut4<'b, T> {
    type T = T;
    #[inline(always)]
    fn get(&self) -> &T {
        self.0
    }
    #[inline(always)]
    fn get_mut(&mut self) -> &mut T {
        self.0
    }
}

#[repr(Rust, packed(8))]
pub struct PackedRefMut8<'b, T>(pub &'b mut T);

impl<'a, T> PackedRefMut for PackedRefMut8<'a, T> {
    type T = T;
    #[inline(always)]
    fn get(&self) -> &T {
        self.0
    }
    #[inline(always)]
    fn get_mut(&mut self) -> &mut T {
        self.0
    }
}
