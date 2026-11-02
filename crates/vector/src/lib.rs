// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.

pub mod bvect;
pub mod rabitq4;
pub mod rabitq8;
pub mod svect;
pub mod vect;

pub trait VectorOwned: Clone + 'static {
    type Borrowed<'a>: VectorBorrowed<Owned = Self>;

    fn as_borrowed(&self) -> Self::Borrowed<'_>;
}

pub trait VectorBorrowed: Copy {
    type Owned: VectorOwned;

    fn own(&self) -> Self::Owned;

    fn dim(&self) -> u32;

    fn norm(&self) -> f32;

    fn operator_dot(self, rhs: Self) -> distance::Distance;

    fn operator_l2s(self, rhs: Self) -> distance::Distance;

    fn operator_cos(self, rhs: Self) -> distance::Distance;

    fn operator_hamming(self, rhs: Self) -> distance::Distance;

    fn operator_jaccard(self, rhs: Self) -> distance::Distance;

    fn function_normalize(&self) -> Self::Owned;

    fn operator_add(&self, rhs: Self) -> Self::Owned;

    fn operator_sub(&self, rhs: Self) -> Self::Owned;

    fn operator_mul(&self, rhs: Self) -> Self::Owned;

    fn operator_and(&self, rhs: Self) -> Self::Owned;

    fn operator_or(&self, rhs: Self) -> Self::Owned;

    fn operator_xor(&self, rhs: Self) -> Self::Owned;
}
