// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

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

/// Cosine distance `1 - xy / (|x| |y|)` from f32 reductions.
///
/// The norms are taken separately (`sqrt(x2 * y2)` underflowed to 0 for tiny
/// vectors and overflowed for huge ones). If a reduction itself overflowed,
/// the sums are recomputed in f64 by `wide`. A zero vector still gives NaN,
/// like pgvector (#35).
#[inline(always)]
pub fn cosine_distance(xy: f32, x2: f32, y2: f32, wide: impl FnOnce() -> (f64, f64, f64)) -> f32 {
    if xy.is_finite() && x2.is_finite() && y2.is_finite() {
        1.0 - xy / (x2.sqrt() * y2.sqrt())
    } else {
        let (xy, x2, y2) = wide();
        (1.0 - xy / (x2.sqrt() * y2.sqrt())) as f32
    }
}

#[test]
fn cosine_distance_extremes() {
    use simd::Floating;
    let cos = |x: &[f32], y: &[f32]| {
        let xy = f32::reduce_sum_of_xy(x, y);
        let x2 = f32::reduce_sum_of_x2(x);
        let y2 = f32::reduce_sum_of_x2(y);
        cosine_distance(xy, x2, y2, || {
            let mut s = (0.0f64, 0.0f64, 0.0f64);
            for (&a, &b) in x.iter().zip(y.iter()) {
                let (a, b) = (f64::from(a), f64::from(b));
                s = (s.0 + a * b, s.1 + a * a, s.2 + b * b);
            }
            s
        })
    };
    let tiny = [1.0e-13f32; 3];
    assert!(cos(&tiny, &tiny).abs() < 1.0e-6);
    let huge = [1.0e19f32, 1.0e19];
    assert!(cos(&huge, &huge).abs() < 1.0e-6);
    let huge_opposite = [-1.0e30f32, -1.0e30];
    let huge_positive = [1.0e30f32, 1.0e30];
    assert!((cos(&huge_positive, &huge_opposite) - 2.0).abs() < 1.0e-6);
    assert!(cos(&[0.0, 0.0], &[1.0, 0.0]).is_nan());
}
