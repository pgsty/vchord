// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use crate::{VectorBorrowed, VectorOwned};
use distance::Distance;
use simd::Floating;
use std::cmp::Ordering;

#[derive(Debug, Clone, PartialEq)]
#[repr(transparent)]
pub struct VectOwned<S>(Vec<S>);

impl<S: Floating> VectOwned<S> {
    #[inline(always)]
    pub fn new(slice: Vec<S>) -> Self {
        Self::new_checked(slice).expect("invalid data")
    }

    #[inline(always)]
    pub fn new_checked(slice: Vec<S>) -> Option<Self> {
        if !(1..=65535).contains(&slice.len()) {
            return None;
        }
        #[allow(unsafe_code)]
        Some(unsafe { Self::new_unchecked(slice) })
    }

    /// # Safety
    ///
    /// * `slice.len()` must not be zero.
    /// * `slice.len()` must be less than 65536.
    #[allow(unsafe_code)]
    #[inline(always)]
    pub unsafe fn new_unchecked(slice: Vec<S>) -> Self {
        Self(slice)
    }

    #[inline(always)]
    pub fn slice(&self) -> &[S] {
        self.0.as_slice()
    }

    #[inline(always)]
    pub fn slice_mut(&mut self) -> &mut [S] {
        self.0.as_mut_slice()
    }

    #[inline(always)]
    pub fn into_vec(self) -> Vec<S> {
        self.0
    }
}

impl<S: Floating> VectorOwned for VectOwned<S> {
    type Borrowed<'a> = VectBorrowed<'a, S>;

    #[inline(always)]
    fn as_borrowed(&self) -> VectBorrowed<'_, S> {
        VectBorrowed(self.0.as_slice())
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct VectBorrowed<'a, S>(&'a [S]);

impl<'a, S: Floating> VectBorrowed<'a, S> {
    #[inline(always)]
    pub fn new(slice: &'a [S]) -> Self {
        Self::new_checked(slice).expect("invalid data")
    }

    #[inline(always)]
    pub fn new_checked(slice: &'a [S]) -> Option<Self> {
        if !(1..=65535).contains(&slice.len()) {
            return None;
        }
        #[allow(unsafe_code)]
        Some(unsafe { Self::new_unchecked(slice) })
    }

    /// # Safety
    ///
    /// * `slice.len()` must not be zero.
    /// * `slice.len()` must be less than 65536.
    #[allow(unsafe_code)]
    #[inline(always)]
    pub unsafe fn new_unchecked(slice: &'a [S]) -> Self {
        Self(slice)
    }

    #[inline(always)]
    pub fn slice(&self) -> &'a [S] {
        self.0
    }

    /// Non-zero; the cosine opclasses and `function_normalize` share this test.
    #[inline(always)]
    pub fn has_direction(&self) -> bool {
        has_direction(self.0)
    }
}

impl<S: Floating> VectorBorrowed for VectBorrowed<'_, S> {
    type Owned = VectOwned<S>;

    #[inline(always)]
    fn dim(&self) -> u32 {
        self.0.len() as u32
    }

    #[inline(always)]
    fn own(&self) -> VectOwned<S> {
        VectOwned(self.0.to_vec())
    }

    #[inline(always)]
    fn norm(&self) -> f32 {
        S::reduce_sum_of_x2(self.0).sqrt()
    }

    #[inline(always)]
    fn operator_dot(self, rhs: Self) -> Distance {
        Distance::from(-S::reduce_sum_of_xy(self.slice(), rhs.slice()))
    }

    #[inline(always)]
    fn operator_l2s(self, rhs: Self) -> Distance {
        Distance::from(S::reduce_sum_of_d2(self.slice(), rhs.slice()))
    }

    #[inline(always)]
    fn operator_cos(self, rhs: Self) -> Distance {
        Distance::from(cosine_distance(self.0, rhs.0))
    }

    /// Scales to unit length; a vector without a direction is returned unchanged.
    #[inline(always)]
    fn function_normalize(&self) -> VectOwned<S> {
        VectOwned(normalized(self.0).unwrap_or_else(|| self.0.to_vec()))
    }
}

impl<S: Floating> PartialEq for VectBorrowed<'_, S> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<S: Floating> PartialOrd for VectBorrowed<'_, S> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.0.len() != other.0.len() {
            return None;
        }
        self.0.partial_cmp(other.0)
    }
}

/// Cosine distance, rescaling out-of-range norms exactly; a zero vector gives NaN as in pgvector.
#[inline(always)]
fn cosine_distance<S: Floating>(x: &[S], y: &[S]) -> f32 {
    let (x2, y2) = (S::reduce_sum_of_x2(x), S::reduce_sum_of_x2(y));
    let p = x2 * y2;
    if x2.min(y2) >= MIN_FAST_X2 && p.is_normal() {
        1.0 - S::reduce_sum_of_xy(x, y) / p.sqrt()
    } else {
        cosine_distance_rescaled(x, y)
    }
}

#[cold]
fn cosine_distance_rescaled<S: Floating>(x: &[S], y: &[S]) -> f32 {
    let (Some(x), Some(y)) = (rescaled(x), rescaled(y)) else {
        return f32::NAN;
    };
    let xy = f32::reduce_sum_of_xy(&x, &y);
    1.0 - xy / (f32::reduce_sum_of_x2(&x) * f32::reduce_sum_of_x2(&y)).sqrt()
}

/// Below this, subnormal squares could bias `x2`, so the vector is rescaled.
pub const MIN_FAST_X2: f32 = 1e-30;

/// The power of two that brings the largest magnitude near 1, or `None` for a zero vector.
fn scale<S: Floating>(values: &[S]) -> Option<f32> {
    let (min, max) = S::reduce_min_max_of_x(values);
    let m = max.max(-min);
    (m > 0.0).then(|| {
        let e = ((m.to_bits() >> 23) as i32 - 127).min(126);
        f32::from_bits(((127 - e) as u32) << 23)
    })
}

fn rescaled<S: Floating>(values: &[S]) -> Option<Vec<f32>> {
    let s = scale(values)?;
    Some(f32::vector_mul_scalar(
        S::vector_to_f32_borrowed(values).as_ref(),
        s,
    ))
}

/// Whether `values` is non-zero, so that `normalized` returns a vector.
fn has_direction<S: Floating>(values: &[S]) -> bool {
    let x2 = S::reduce_sum_of_x2(values);
    (MIN_FAST_X2..f32::INFINITY).contains(&x2) || scale(values).is_some()
}

/// `values` at unit length, or `None` without a direction.
fn normalized<S: Floating>(values: &[S]) -> Option<Vec<S>> {
    let x2 = S::reduce_sum_of_x2(values);
    if (MIN_FAST_X2..f32::INFINITY).contains(&x2) {
        return Some(S::vector_mul_scalar(values, 1.0 / x2.sqrt()));
    }
    let mut x = rescaled(values)?;
    let l = f32::reduce_sum_of_x2(&x).sqrt();
    f32::vector_mul_scalar_inplace(&mut x, 1.0 / l);
    Some(S::vector_from_f32(&x))
}

#[cfg(test)]
mod tests {
    use super::*;
    use simd::{F16, f16};

    fn lcg(seed: u64) -> impl FnMut() -> f32 {
        let mut state = seed;
        move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }
    }

    fn f16s(x: &[f32]) -> Vec<f16> {
        x.iter().copied().map(f16::_from_f32).collect()
    }

    fn norm<S: Floating>(x: &[S]) -> f64 {
        S::vector_to_f32(x)
            .iter()
            .map(|&x| x as f64 * x as f64)
            .sum::<f64>()
            .sqrt()
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn normalize_zero_vector() {
        for n in [1, 2, 3, 64, 784] {
            let zero = vec![0.0f32; n];
            let r = VectBorrowed::new(&zero).function_normalize();
            assert_eq!(r.slice(), &zero[..]);
            let zero = f16s(&zero);
            let r = VectBorrowed::new(&zero).function_normalize();
            assert!(r.slice().iter().all(|x| x._to_f32() == 0.0));
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn normalize_unit_length() {
        let mut rng = lcg(5);
        for n in [1, 2, 3, 7, 64, 783, 784, 785] {
            for scale in [1e-3f32, 1.0, 255.0, 60000.0] {
                let x = (0..n).map(|_| rng() * scale).collect::<Vec<_>>();
                let r = VectBorrowed::new(&x).function_normalize();
                assert!((norm(r.slice()) - 1.0).abs() < 1e-5, "f32: n = {n}");
                // Elements in `0..=255` (pixels) and up to the f16 maximum: the
                // squared norm overflows binary16, and `1 / norm` underflows it.
                let x = f16s(&x);
                let r = VectBorrowed::new(&x).function_normalize();
                let l = norm(r.slice());
                assert!(
                    (l - 1.0).abs() < 2e-3,
                    "f16: n = {n}, scale = {scale}, l = {l}"
                );
            }
            let x = (0..n)
                .map(|_| (rng().abs() * 255.0).round())
                .collect::<Vec<_>>();
            if x.iter().any(|&x| x != 0.0) {
                let r = VectBorrowed::new(&f16s(&x)).function_normalize();
                let l = norm(r.slice());
                assert!((l - 1.0).abs() < 2e-3, "f16 pixels: n = {n}, l = {l}");
            }
        }
    }

    #[test]
    fn cosine_against_zero_vector() {
        let x = vec![0.6f32, 0.8, 0.0];
        let zero = vec![0.0f32; 3];
        let (x, zero) = (VectBorrowed::new(&x), VectBorrowed::new(&zero));
        // The exact cosine distance is undefined; it is NaN, which orders last.
        let d = x.operator_cos(zero);
        assert_eq!(d, Distance::NAN);
        assert!(d > x.operator_cos(VectBorrowed::new(&[-0.6, -0.8, 0.0])));
        // After normalization (as the cosine opclasses do), it is finite.
        let (xn, zn) = (x.function_normalize(), zero.function_normalize());
        let d = xn.as_borrowed().operator_dot(zn.as_borrowed()).to_f32();
        assert_eq!(d, 0.0);
        let d = xn.as_borrowed().operator_l2s(zn.as_borrowed()).to_f32();
        assert!((d - 1.0).abs() < 1e-6);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn f16_large_magnitude_distances() {
        let mut rng = lcg(9);
        for n in [1, 3, 784, 785] {
            let x = (0..n)
                .map(|_| (rng().abs() * 255.0).round())
                .collect::<Vec<_>>();
            let y = (0..n)
                .map(|_| (rng().abs() * 255.0).round())
                .collect::<Vec<_>>();
            let exact_l2s = std::iter::zip(&x, &y)
                .map(|(&x, &y)| (x as f64 - y as f64).powi(2))
                .sum::<f64>();
            let exact_dot = std::iter::zip(&x, &y)
                .map(|(&x, &y)| x as f64 * y as f64)
                .sum::<f64>();
            let (xh, yh) = (f16s(&x), f16s(&y));
            let (xh, yh) = (VectBorrowed::new(&xh), VectBorrowed::new(&yh));
            let l2s = xh.operator_l2s(yh).to_f32() as f64;
            let dot = -xh.operator_dot(yh).to_f32() as f64;
            assert!(
                (l2s - exact_l2s).abs() <= 1e-5 * exact_l2s + 1e-3,
                "n = {n}"
            );
            assert!(
                (dot - exact_dot).abs() <= 1e-5 * exact_dot + 1e-3,
                "n = {n}"
            );
            assert_eq!(xh.operator_l2s(xh), Distance::ZERO);
            assert!((xh.norm() as f64 - norm(xh.slice())).abs() <= 1e-5 * norm(xh.slice()));
        }
    }

    #[test]
    fn cosine_distance_extremes() {
        let cos = |x: &[f32], y: &[f32]| cosine_distance(x, y);
        let tiny = [1.0e-13f32; 3];
        assert!(cos(&tiny, &tiny).abs() < 1.0e-6);
        let tinier = [1.0e-23f32, 2.0e-23, 0.0];
        assert!(cos(&tinier, &tinier).abs() < 1.0e-6);
        assert!((cos(&tinier, &[2.0, 4.0, 0.0])).abs() < 1.0e-6);
        assert!((cos(&tinier, &[0.0, 0.0, 1.0]) - 1.0).abs() < 1.0e-6);
        let huge = [1.0e19f32, 1.0e19];
        assert!(cos(&huge, &huge).abs() < 1.0e-6);
        let huge_opposite = [-1.0e30f32, -1.0e30];
        let huge_positive = [1.0e30f32, 1.0e30];
        assert!((cos(&huge_positive, &huge_opposite) - 2.0).abs() < 1.0e-6);
        let max = [f32::MAX, f32::MAX, 0.0];
        assert!((cos(&max, &tinier) - (1.0 - 3.0 / 10f32.sqrt())).abs() < 1.0e-6);
        let sub = [1.0e-40f32, -1.0e-40];
        assert!((cos(&sub, &[1.0, -1.0])).abs() < 1.0e-6);
        assert!(cos(&[0.0, 0.0], &[1.0, 0.0]).is_nan());
        assert!(cos(&[0.0, 0.0], &[0.0, 0.0]).is_nan());
        // Subnormal squares: rescaled exactly like the fast path on the scaled-up vector.
        let up = |x: &[f32]| x.iter().map(|&x| x * 2f32.powi(70)).collect::<Vec<_>>();
        let sub = vec![1.1e-21f32; 4096];
        let mut ones = vec![1.0f32; 4096];
        assert_eq!(cos(&sub, &ones).to_bits(), cos(&up(&sub), &ones).to_bits());
        assert!(cos(&sub, &ones).abs() < 1e-4);
        ones[0] = -4094.0;
        assert_eq!(cos(&sub, &ones).to_bits(), cos(&up(&sub), &ones).to_bits());
        assert!((cos(&sub, &ones) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn normalized_subnormal_squares() {
        let norm = |x: &[f32]| x.iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt();
        let up = |x: &[f32]| x.iter().map(|&x| x * 2f32.powi(70)).collect::<Vec<_>>();
        let mut mixed = vec![1e-21f32; 4096];
        mixed[0] = 1.1e-21;
        for x in [vec![1.1e-21f32; 4096], mixed] {
            assert!(has_direction(&x));
            let r = normalized(&x).unwrap();
            assert_eq!(r, normalized(&up(&x)).unwrap());
            assert!((norm(&r) - 1.0).abs() < 1e-4, "{}", norm(&r));
        }
    }
}
