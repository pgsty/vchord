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

#[derive(Debug, Clone)]
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

    /// The norm, if the vector has a direction: positive and finite.
    ///
    /// Cosine is undefined without one. `function_normalize` and the cosine
    /// operator classes use this same test, so a vector a cosine index
    /// stores is always one that normalization can scale to unit length.
    #[inline(always)]
    pub fn direction_norm(&self) -> Option<f32> {
        let l = S::reduce_sum_of_x2(self.0).sqrt();
        (l > 0.0 && l.is_finite()).then_some(l)
    }

    #[inline(always)]
    pub fn has_direction(&self) -> bool {
        self.direction_norm().is_some()
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
        let xy = S::reduce_sum_of_xy(self.slice(), rhs.slice());
        let x2 = S::reduce_sum_of_x2(self.0);
        let y2 = S::reduce_sum_of_x2(rhs.0);
        Distance::from(crate::cosine_distance(xy, x2, y2, || {
            let x = S::vector_to_f32_borrowed(self.0);
            let y = S::vector_to_f32_borrowed(rhs.0);
            let (x, y) = (x.as_ref(), y.as_ref());
            let mut sums = (0.0f64, 0.0f64, 0.0f64);
            for (&a, &b) in x.iter().zip(y.iter()) {
                let (a, b) = (f64::from(a), f64::from(b));
                sums = (sums.0 + a * b, sums.1 + a * a, sums.2 + b * b);
            }
            sums
        }))
    }

    #[inline(always)]
    fn operator_hamming(self, _: Self) -> Distance {
        unimplemented!()
    }

    #[inline(always)]
    fn operator_jaccard(self, _: Self) -> Distance {
        unimplemented!()
    }

    /// Scales the vector to unit length.
    ///
    /// A vector whose norm is zero (or not finite) has no direction, and
    /// scaling it by `1 / l` would fill it with NaN, so it is returned
    /// unchanged. A zero vector therefore stays zero: its dot product with any
    /// unit vector is 0, i.e. it is treated as orthogonal to everything.
    #[inline(always)]
    fn function_normalize(&self) -> VectOwned<S> {
        let mut data = self.0.to_vec();
        if let Some(l) = self.direction_norm() {
            S::vector_mul_scalar_inplace(&mut data, 1.0 / l);
        }
        VectOwned(data)
    }

    fn operator_add(&self, rhs: Self) -> Self::Owned {
        VectOwned::new(S::vector_add(self.slice(), rhs.slice()))
    }

    fn operator_sub(&self, rhs: Self) -> Self::Owned {
        VectOwned::new(S::vector_sub(self.slice(), rhs.slice()))
    }

    fn operator_mul(&self, rhs: Self) -> Self::Owned {
        VectOwned::new(S::vector_mul(self.slice(), rhs.slice()))
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

impl<S: Floating> PartialEq for VectBorrowed<'_, S> {
    fn eq(&self, other: &Self) -> bool {
        if self.0.len() != other.0.len() {
            return false;
        }
        let n = self.0.len();
        for i in 0..n {
            if self.0[i] != other.0[i] {
                return false;
            }
        }
        true
    }
}

impl<S: Floating> PartialOrd for VectBorrowed<'_, S> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.0.len() != other.0.len() {
            return None;
        }
        let n = self.0.len();
        for i in 0..n {
            match PartialOrd::partial_cmp(&self.0[i], &other.0[i])? {
                Ordering::Less => return Some(Ordering::Less),
                Ordering::Equal => continue,
                Ordering::Greater => return Some(Ordering::Greater),
            }
        }
        Some(Ordering::Equal)
    }
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
    fn normalize_non_finite_norm() {
        // `1e30^2` overflows f32, so the norm is infinite and `1 / l` is zero.
        let x = vec![1e30f32, -1e30, 3.0];
        let r = VectBorrowed::new(&x).function_normalize();
        assert_eq!(r.slice(), &x[..]);
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
}
