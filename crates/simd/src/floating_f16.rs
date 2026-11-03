// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use crate::{F16, Floating, f16};

impl Floating for f16 {
    #[inline(always)]
    fn zero() -> Self {
        f16::_ZERO
    }

    #[inline(always)]
    fn infinity() -> Self {
        f16::INFINITY
    }

    #[inline(always)]
    fn mask(self, m: bool) -> Self {
        f16::from_bits(self.to_bits() & (m as u16).wrapping_neg())
    }

    #[inline(always)]
    fn scalar_neg(this: Self) -> Self {
        -this
    }

    #[inline(always)]
    fn scalar_add(lhs: Self, rhs: Self) -> Self {
        lhs + rhs
    }

    #[inline(always)]
    fn scalar_sub(lhs: Self, rhs: Self) -> Self {
        lhs - rhs
    }

    #[inline(always)]
    fn scalar_mul(lhs: Self, rhs: Self) -> Self {
        lhs * rhs
    }

    #[inline(always)]
    fn reduce_or_of_is_zero_x(this: &[f16]) -> bool {
        reduce_or_of_is_zero_x::reduce_or_of_is_zero_x(this)
    }

    #[inline(always)]
    fn reduce_sum_of_x(this: &[f16]) -> f32 {
        reduce_sum_of_x::reduce_sum_of_x(this)
    }

    #[inline(always)]
    fn reduce_sum_of_abs_x(this: &[f16]) -> f32 {
        reduce_sum_of_abs_x::reduce_sum_of_abs_x(this)
    }

    #[inline(always)]
    fn reduce_sum_of_x2(this: &[f16]) -> f32 {
        reduce_sum_of_x2::reduce_sum_of_x2(this)
    }

    #[inline(always)]
    fn reduce_min_max_of_x(this: &[f16]) -> (f32, f32) {
        reduce_min_max_of_x::reduce_min_max_of_x(this)
    }

    #[inline(always)]
    fn reduce_sum_of_xy(lhs: &[Self], rhs: &[Self]) -> f32 {
        reduce_sum_of_xy::reduce_sum_of_xy(lhs, rhs)
    }

    #[inline(always)]
    fn reduce_sum_of_d2(lhs: &[f16], rhs: &[f16]) -> f32 {
        reduce_sum_of_d2::reduce_sum_of_d2(lhs, rhs)
    }

    #[inline(always)]
    fn reduce_sum_of_xy_sparse(lidx: &[u32], lval: &[f16], ridx: &[u32], rval: &[f16]) -> f32 {
        reduce_sum_of_xy_sparse::reduce_sum_of_xy_sparse(lidx, lval, ridx, rval)
    }

    #[inline(always)]
    fn reduce_sum_of_d2_sparse(lidx: &[u32], lval: &[f16], ridx: &[u32], rval: &[f16]) -> f32 {
        reduce_sum_of_d2_sparse::reduce_sum_of_d2_sparse(lidx, lval, ridx, rval)
    }

    #[inline(always)]
    fn vector_from_f32(this: &[f32]) -> Vec<Self> {
        vector_from_f32::vector_from_f32(this)
    }

    #[inline(always)]
    fn vector_to_f32(this: &[Self]) -> Vec<f32> {
        vector_to_f32::vector_to_f32(this)
    }

    #[inline(always)]
    fn vector_add(lhs: &[Self], rhs: &[Self]) -> Vec<Self> {
        vector_add::vector_add(lhs, rhs)
    }

    #[inline(always)]
    fn vector_add_inplace(lhs: &mut [Self], rhs: &[Self]) {
        vector_add_inplace::vector_add_inplace(lhs, rhs)
    }

    #[inline(always)]
    fn vector_sub(lhs: &[Self], rhs: &[Self]) -> Vec<Self> {
        vector_sub::vector_sub(lhs, rhs)
    }

    #[inline(always)]
    fn vector_mul(lhs: &[Self], rhs: &[Self]) -> Vec<Self> {
        vector_mul::vector_mul(lhs, rhs)
    }

    #[inline(always)]
    fn vector_mul_scalar(lhs: &[Self], rhs: f32) -> Vec<Self> {
        vector_mul_scalar::vector_mul_scalar(lhs, rhs)
    }

    #[inline(always)]
    fn vector_mul_scalar_inplace(lhs: &mut [Self], rhs: f32) {
        vector_mul_scalar_inplace::vector_mul_scalar_inplace(lhs, rhs)
    }

    #[inline(always)]
    fn vector_to_f32_borrowed(this: &[Self]) -> impl AsRef<[f32]> {
        Self::vector_to_f32(this)
    }

    #[inline(always)]
    fn vector_abs_inplace(this: &mut [Self]) {
        vector_abs_inplace::vector_abs_inplace(this);
    }
}

mod reduce_or_of_is_zero_x {
    use crate::{F16, f16};

    #[crate::multiversion(
        "v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn reduce_or_of_is_zero_x(this: &[f16]) -> bool {
        for &x in this {
            if x == f16::_ZERO {
                return true;
            }
        }
        false
    }
}

// All reductions below accumulate in f32 on every code path.
//
// Accumulating in binary16 (as the former `v4:avx512fp16`, `a2:fp16` and SVE
// kernels did) overflows at 65504 and loses precision after a few thousand
// additions, which made halfvec distances meaningless for large-magnitude data.
// On aarch64 with FEAT_FHM, `FMLAL`/`FMLSL` widen each f16 product to f32
// exactly and accumulate in f32, so they keep most of the speed of fp16
// arithmetic without its range. Without FEAT_FHM, and on x86_64, every element
// is converted to f32 before it is accumulated.

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use core::arch::aarch64::*;

    /// Sums all lanes of all accumulators.
    #[inline]
    #[crate::target_cpu(enable = "a2")]
    pub(super) fn reduce_add<const N: usize>(mut sum: [float32x4_t; N]) -> f32 {
        let mut n = N;
        while n > 1 {
            let h = n / 2;
            for i in 0..h {
                sum[i] = vaddq_f32(sum[i], sum[n - h + i]);
            }
            n -= h;
        }
        vaddvq_f32(sum[0])
    }
}

#[cfg(test)]
mod test_util {
    use crate::{F16, f16};
    use rand::RngExt;

    /// Odd, short, around-the-unroll-boundary, and realistic lengths.
    pub(super) fn lengths() -> impl Iterator<Item = usize> {
        (1..=70).chain([127, 128, 129, 255, 783, 784, 785, 1023, 4095, 4097])
    }

    pub(super) fn random(n: usize, lo: f32, hi: f32) -> Vec<f16> {
        let mut rng = rand::rng();
        (0..n)
            .map(|_| f16::_from_f32(rng.random_range(lo..=hi)))
            .collect()
    }

    /// Integers in `[lo, hi]`; e.g. pixel intensities `0..=255`.
    pub(super) fn integers(n: usize, lo: i32, hi: i32) -> Vec<f16> {
        let mut rng = rand::rng();
        (0..n)
            .map(|_| f16::_from_f32(rng.random_range(lo..=hi) as f32))
            .collect()
    }

    /// Asserts `got` is within the worst-case error of an f32 summation of `n`
    /// terms, `(n + 1) * 2^-24 * sum_of_abs`, from the exact `(sum, sum_of_abs)`.
    ///
    /// Accumulating in f16 instead has an error of about `2^-11` per addition.
    #[track_caller]
    pub(super) fn assert_close(name: &str, n: usize, got: f32, (sum, sum_of_abs): (f64, f64)) {
        let tolerance = (n + 1) as f64 * 2.0f64.powi(-24) * sum_of_abs + 1e-6;
        assert!(
            ((got as f64) - sum).abs() <= tolerance,
            "{name}: n = {n}, got = {got}, expected = {sum}, tolerance = {tolerance}"
        );
    }
}

mod reduce_sum_of_x {
    use crate::{F16, f16};

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    fn reduce_sum_of_x_v4(this: &[f16]) -> f32 {
        use core::arch::x86_64::*;
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut sum = _mm512_setzero_ps();
        while n >= 16 {
            let x = unsafe { _mm512_cvtph_ps(_mm256_loadu_epi16(a.cast())) };
            sum = _mm512_add_ps(sum, x);
            (n, a) = unsafe { (n - 16, a.add(16)) };
        }
        if n > 0 {
            let mask = _bzhi_u32(0xffff, n as u32) as u16;
            let x = unsafe { _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, a.cast())) };
            sum = _mm512_add_ps(sum, x);
        }
        _mm512_reduce_add_ps(sum)
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_sum_of_x_v4_test() {
        use rand::RngExt;
        const EPSILON: f32 = 2.0;
        if !crate::is_cpu_detected!("v4") {
            println!("test {} ... skipped (v4)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let this = (0..n)
                .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            let specialized = unsafe { reduce_sum_of_x_v4(&this) };
            let fallback = fallback(&this);
            assert!(
                (specialized - fallback).abs() < EPSILON,
                "specialized = {specialized}, fallback = {fallback}."
            );
        }
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v3")]
    fn reduce_sum_of_x_v3(this: &[f16]) -> f32 {
        use crate::emulate::{emulate_mm256_reduce_add_ps, partial_load};
        use core::arch::x86_64::*;
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut sum = _mm256_setzero_ps();
        while n >= 8 {
            let x = unsafe { _mm256_cvtph_ps(_mm_loadu_si128(a.cast())) };
            sum = _mm256_add_ps(sum, x);
            (n, a) = unsafe { (n - 8, a.add(8)) };
        }
        if n > 0 {
            let (_a,) = unsafe { partial_load!(8, n, a) };
            (a,) = (_a.as_ptr(),);
            let x = unsafe { _mm256_cvtph_ps(_mm_loadu_si128(a.cast())) };
            sum = _mm256_add_ps(sum, x);
        }
        emulate_mm256_reduce_add_ps(sum)
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_sum_of_x_v3_test() {
        use rand::RngExt;
        const EPSILON: f32 = 2.0;
        if !crate::is_cpu_detected!("v3") {
            println!("test {} ... skipped (v3)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let this = (0..n)
                .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            for z in 3984..4016 {
                let this = &this[..z];
                let specialized = unsafe { reduce_sum_of_x_v3(this) };
                let fallback = fallback(this);
                assert!(
                    (specialized - fallback).abs() < EPSILON,
                    "specialized = {specialized}, fallback = {fallback}."
                );
            }
        }
    }

    #[inline]
    #[cfg(target_arch = "aarch64")]
    #[crate::target_cpu(enable = "a2")]
    #[target_feature(enable = "fhm")]
    fn reduce_sum_of_x_a2_fhm(this: &[f16]) -> f32 {
        use crate::emulate::partial_load;
        use core::arch::aarch64::*;
        // `x * 1.0` is exact, so `FMLAL` widens every element to f32.
        let ones = vreinterpretq_f16_u16(vdupq_n_u16(0x3c00));
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut sum = [vdupq_n_f32(0.0); 16];
        while n >= 64 {
            for i in 0..8 {
                let x = vreinterpretq_f16_u16(unsafe { vld1q_u16(a.add(8 * i).cast()) });
                sum[2 * i] = vfmlalq_low_f16(sum[2 * i], x, ones);
                sum[2 * i + 1] = vfmlalq_high_f16(sum[2 * i + 1], x, ones);
            }
            (n, a) = unsafe { (n - 64, a.add(64)) };
        }
        while n >= 8 {
            let x = vreinterpretq_f16_u16(unsafe { vld1q_u16(a.cast()) });
            sum[0] = vfmlalq_low_f16(sum[0], x, ones);
            sum[1] = vfmlalq_high_f16(sum[1], x, ones);
            (n, a) = unsafe { (n - 8, a.add(8)) };
        }
        if n > 0 {
            let (_a,) = unsafe { partial_load!(8, n, a) };
            (a,) = (_a.as_ptr(),);
            let x = vreinterpretq_f16_u16(unsafe { vld1q_u16(a.cast()) });
            sum[2] = vfmlalq_low_f16(sum[2], x, ones);
            sum[3] = vfmlalq_high_f16(sum[3], x, ones);
        }
        super::aarch64::reduce_add(sum)
    }

    #[inline]
    #[cfg(target_arch = "aarch64")]
    #[crate::target_cpu(enable = "a2")]
    fn reduce_sum_of_x_a2(this: &[f16]) -> f32 {
        use crate::emulate::{emulate_vreinterpretq_f16_u16, partial_load};
        use core::arch::aarch64::*;
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut sum = [vdupq_n_f32(0.0); 8];
        while n >= 32 {
            for i in 0..4 {
                let x = emulate_vreinterpretq_f16_u16(unsafe { vld1q_u16(a.add(8 * i).cast()) });
                sum[2 * i] = vaddq_f32(sum[2 * i], vcvt_f32_f16(vget_low_f16(x)));
                sum[2 * i + 1] = vaddq_f32(sum[2 * i + 1], vcvt_high_f32_f16(x));
            }
            (n, a) = unsafe { (n - 32, a.add(32)) };
        }
        while n >= 8 {
            let x = emulate_vreinterpretq_f16_u16(unsafe { vld1q_u16(a.cast()) });
            sum[0] = vaddq_f32(sum[0], vcvt_f32_f16(vget_low_f16(x)));
            sum[1] = vaddq_f32(sum[1], vcvt_high_f32_f16(x));
            (n, a) = unsafe { (n - 8, a.add(8)) };
        }
        if n > 0 {
            let (_a,) = unsafe { partial_load!(8, n, a) };
            (a,) = (_a.as_ptr(),);
            let x = emulate_vreinterpretq_f16_u16(unsafe { vld1q_u16(a.cast()) });
            sum[2] = vaddq_f32(sum[2], vcvt_f32_f16(vget_low_f16(x)));
            sum[3] = vaddq_f32(sum[3], vcvt_high_f32_f16(x));
        }
        super::aarch64::reduce_add(sum)
    }

    #[crate::multiversion(
        @"v4", @"v3", "v2", @"a2:fhm", @"a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn reduce_sum_of_x(this: &[f16]) -> f32 {
        let n = this.len();
        let mut x = 0.0f32;
        for i in 0..n {
            x += this[i]._to_f32();
        }
        x
    }

    #[cfg(test)]
    #[allow(clippy::type_complexity)]
    fn implementations() -> Vec<(&'static str, unsafe fn(&[f16]) -> f32)> {
        let mut r: Vec<(&'static str, unsafe fn(&[f16]) -> f32)> =
            vec![("fallback", fallback), ("dispatch", |x| reduce_sum_of_x(x))];
        #[cfg(target_arch = "x86_64")]
        if crate::is_cpu_detected!("v4") {
            r.push(("v4", reduce_sum_of_x_v4));
        }
        #[cfg(target_arch = "x86_64")]
        if crate::is_cpu_detected!("v3") {
            r.push(("v3", reduce_sum_of_x_v3));
        }
        #[cfg(target_arch = "aarch64")]
        if crate::is_cpu_detected!("a2") && crate::is_feature_detected!("fhm") {
            r.push(("a2:fhm", reduce_sum_of_x_a2_fhm));
        }
        #[cfg(target_arch = "aarch64")]
        if crate::is_cpu_detected!("a2") {
            r.push(("a2", reduce_sum_of_x_a2));
        }
        r
    }

    #[cfg(test)]
    fn exact(this: &[f16]) -> (f64, f64) {
        let sum = this.iter().map(|x| x._to_f32() as f64).sum();
        let sum_of_abs = this.iter().map(|x| x._to_f32().abs() as f64).sum();
        (sum, sum_of_abs)
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_sum_of_x_accuracy_test() {
        use super::test_util::*;
        for (lo, hi) in [(0, 255), (-255, 255), (60000, 65504), (-65504, 65504)] {
            for n in lengths() {
                let x = integers(n, lo, hi);
                for (name, f) in implementations() {
                    assert_close(name, n, unsafe { f(&x) }, exact(&x));
                }
            }
        }
        for n in lengths() {
            for (lo, hi) in [(-1.0, 1.0), (-1e-5, 1e-5)] {
                let x = random(n, lo, hi);
                for (name, f) in implementations() {
                    assert_close(name, n, unsafe { f(&x) }, exact(&x));
                }
            }
        }
    }
}

mod reduce_sum_of_abs_x {
    use crate::{F16, f16};

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    fn reduce_sum_of_abs_x_v4(this: &[f16]) -> f32 {
        use core::arch::x86_64::*;
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut sum = _mm512_setzero_ps();
        while n >= 16 {
            let x = unsafe { _mm512_cvtph_ps(_mm256_loadu_epi16(a.cast())) };
            sum = _mm512_add_ps(sum, _mm512_abs_ps(x));
            (n, a) = unsafe { (n - 16, a.add(16)) };
        }
        if n > 0 {
            let mask = _bzhi_u32(0xffff, n as u32) as u16;
            let x = unsafe { _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, a.cast())) };
            sum = _mm512_add_ps(sum, _mm512_abs_ps(x));
        }
        _mm512_reduce_add_ps(sum)
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_sum_of_abs_x_v4_test() {
        use rand::RngExt;
        const EPSILON: f32 = 2.0;
        if !crate::is_cpu_detected!("v4") {
            println!("test {} ... skipped (v4)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let this = (0..n)
                .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            let specialized = unsafe { reduce_sum_of_abs_x_v4(&this) };
            let fallback = fallback(&this);
            assert!(
                (specialized - fallback).abs() < EPSILON,
                "specialized = {specialized}, fallback = {fallback}."
            );
        }
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v3")]
    fn reduce_sum_of_abs_x_v3(this: &[f16]) -> f32 {
        use crate::emulate::{emulate_mm256_reduce_add_ps, partial_load};
        use core::arch::x86_64::*;
        let abs = _mm256_castsi256_ps(_mm256_srli_epi32(_mm256_set1_epi32(-1), 1));
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut sum = _mm256_setzero_ps();
        while n >= 8 {
            let x = unsafe { _mm256_cvtph_ps(_mm_loadu_si128(a.cast())) };
            sum = _mm256_add_ps(sum, _mm256_and_ps(abs, x));
            (n, a) = unsafe { (n - 8, a.add(8)) };
        }
        if n > 0 {
            let (_a,) = unsafe { partial_load!(8, n, a) };
            (a,) = (_a.as_ptr(),);
            let x = unsafe { _mm256_cvtph_ps(_mm_loadu_si128(a.cast())) };
            sum = _mm256_add_ps(sum, _mm256_and_ps(abs, x));
        }
        emulate_mm256_reduce_add_ps(sum)
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_sum_of_abs_x_v3_test() {
        use rand::RngExt;
        const EPSILON: f32 = 2.0;
        if !crate::is_cpu_detected!("v3") {
            println!("test {} ... skipped (v3)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let this = (0..n)
                .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            for z in 3984..4016 {
                let this = &this[..z];
                let specialized = unsafe { reduce_sum_of_abs_x_v3(this) };
                let fallback = fallback(this);
                assert!(
                    (specialized - fallback).abs() < EPSILON,
                    "specialized = {specialized}, fallback = {fallback}."
                );
            }
        }
    }

    #[inline]
    #[cfg(target_arch = "aarch64")]
    #[crate::target_cpu(enable = "a2")]
    #[target_feature(enable = "fhm")]
    fn reduce_sum_of_abs_x_a2_fhm(this: &[f16]) -> f32 {
        use crate::emulate::partial_load;
        use core::arch::aarch64::*;
        // `|x| * 1.0` is exact, so `FMLAL` widens every element to f32.
        let ones = vreinterpretq_f16_u16(vdupq_n_u16(0x3c00));
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut sum = [vdupq_n_f32(0.0); 16];
        while n >= 64 {
            for i in 0..8 {
                let x = vreinterpretq_f16_u16(unsafe { vld1q_u16(a.add(8 * i).cast()) });
                let x = vabsq_f16(x);
                sum[2 * i] = vfmlalq_low_f16(sum[2 * i], x, ones);
                sum[2 * i + 1] = vfmlalq_high_f16(sum[2 * i + 1], x, ones);
            }
            (n, a) = unsafe { (n - 64, a.add(64)) };
        }
        while n >= 8 {
            let x = vabsq_f16(vreinterpretq_f16_u16(unsafe { vld1q_u16(a.cast()) }));
            sum[0] = vfmlalq_low_f16(sum[0], x, ones);
            sum[1] = vfmlalq_high_f16(sum[1], x, ones);
            (n, a) = unsafe { (n - 8, a.add(8)) };
        }
        if n > 0 {
            let (_a,) = unsafe { partial_load!(8, n, a) };
            (a,) = (_a.as_ptr(),);
            let x = vabsq_f16(vreinterpretq_f16_u16(unsafe { vld1q_u16(a.cast()) }));
            sum[2] = vfmlalq_low_f16(sum[2], x, ones);
            sum[3] = vfmlalq_high_f16(sum[3], x, ones);
        }
        super::aarch64::reduce_add(sum)
    }

    #[inline]
    #[cfg(target_arch = "aarch64")]
    #[crate::target_cpu(enable = "a2")]
    fn reduce_sum_of_abs_x_a2(this: &[f16]) -> f32 {
        use crate::emulate::{emulate_vreinterpretq_f16_u16, partial_load};
        use core::arch::aarch64::*;
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut sum = [vdupq_n_f32(0.0); 8];
        while n >= 32 {
            for i in 0..4 {
                let x = emulate_vreinterpretq_f16_u16(unsafe { vld1q_u16(a.add(8 * i).cast()) });
                let x_0 = vabsq_f32(vcvt_f32_f16(vget_low_f16(x)));
                let x_1 = vabsq_f32(vcvt_high_f32_f16(x));
                sum[2 * i] = vaddq_f32(sum[2 * i], x_0);
                sum[2 * i + 1] = vaddq_f32(sum[2 * i + 1], x_1);
            }
            (n, a) = unsafe { (n - 32, a.add(32)) };
        }
        while n >= 8 {
            let x = emulate_vreinterpretq_f16_u16(unsafe { vld1q_u16(a.cast()) });
            sum[0] = vaddq_f32(sum[0], vabsq_f32(vcvt_f32_f16(vget_low_f16(x))));
            sum[1] = vaddq_f32(sum[1], vabsq_f32(vcvt_high_f32_f16(x)));
            (n, a) = unsafe { (n - 8, a.add(8)) };
        }
        if n > 0 {
            let (_a,) = unsafe { partial_load!(8, n, a) };
            (a,) = (_a.as_ptr(),);
            let x = emulate_vreinterpretq_f16_u16(unsafe { vld1q_u16(a.cast()) });
            sum[2] = vaddq_f32(sum[2], vabsq_f32(vcvt_f32_f16(vget_low_f16(x))));
            sum[3] = vaddq_f32(sum[3], vabsq_f32(vcvt_high_f32_f16(x)));
        }
        super::aarch64::reduce_add(sum)
    }

    #[crate::multiversion(
        @"v4", @"v3", "v2", @"a2:fhm", @"a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn reduce_sum_of_abs_x(this: &[f16]) -> f32 {
        let n = this.len();
        let mut x = 0.0f32;
        for i in 0..n {
            x += this[i]._to_f32().abs();
        }
        x
    }

    #[cfg(test)]
    #[allow(clippy::type_complexity)]
    fn implementations() -> Vec<(&'static str, unsafe fn(&[f16]) -> f32)> {
        let mut r: Vec<(&'static str, unsafe fn(&[f16]) -> f32)> = vec![
            ("fallback", fallback),
            ("dispatch", |x| reduce_sum_of_abs_x(x)),
        ];
        #[cfg(target_arch = "x86_64")]
        if crate::is_cpu_detected!("v4") {
            r.push(("v4", reduce_sum_of_abs_x_v4));
        }
        #[cfg(target_arch = "x86_64")]
        if crate::is_cpu_detected!("v3") {
            r.push(("v3", reduce_sum_of_abs_x_v3));
        }
        #[cfg(target_arch = "aarch64")]
        if crate::is_cpu_detected!("a2") && crate::is_feature_detected!("fhm") {
            r.push(("a2:fhm", reduce_sum_of_abs_x_a2_fhm));
        }
        #[cfg(target_arch = "aarch64")]
        if crate::is_cpu_detected!("a2") {
            r.push(("a2", reduce_sum_of_abs_x_a2));
        }
        r
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_sum_of_abs_x_accuracy_test() {
        use super::test_util::*;
        let exact = |x: &[f16]| {
            let s = x.iter().map(|x| x._to_f32().abs() as f64).sum::<f64>();
            (s, s)
        };
        for (lo, hi) in [(0, 255), (-255, 255), (60000, 65504), (-65504, 65504)] {
            for n in lengths() {
                let x = integers(n, lo, hi);
                for (name, f) in implementations() {
                    assert_close(name, n, unsafe { f(&x) }, exact(&x));
                }
            }
        }
        for n in lengths() {
            for (lo, hi) in [(-1.0, 1.0), (-1e-5, 1e-5)] {
                let x = random(n, lo, hi);
                for (name, f) in implementations() {
                    assert_close(name, n, unsafe { f(&x) }, exact(&x));
                }
            }
        }
    }
}

mod reduce_sum_of_x2 {
    use crate::{F16, f16};

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    fn reduce_sum_of_x2_v4(this: &[f16]) -> f32 {
        use core::arch::x86_64::*;
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut sum = _mm512_setzero_ps();
        while n >= 16 {
            let x = unsafe { _mm512_cvtph_ps(_mm256_loadu_epi16(a.cast())) };
            sum = _mm512_fmadd_ps(x, x, sum);
            (n, a) = unsafe { (n - 16, a.add(16)) };
        }
        if n > 0 {
            let mask = _bzhi_u32(0xffff, n as u32) as u16;
            let x = unsafe { _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, a.cast())) };
            sum = _mm512_fmadd_ps(x, x, sum);
        }
        _mm512_reduce_add_ps(sum)
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_sum_of_x2_v4_test() {
        use rand::RngExt;
        const EPSILON: f32 = 2.0;
        if !crate::is_cpu_detected!("v4") {
            println!("test {} ... skipped (v4)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let this = (0..n)
                .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            let specialized = unsafe { reduce_sum_of_x2_v4(&this) };
            let fallback = fallback(&this);
            assert!(
                (specialized - fallback).abs() < EPSILON,
                "specialized = {specialized}, fallback = {fallback}."
            );
        }
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v3")]
    fn reduce_sum_of_x2_v3(this: &[f16]) -> f32 {
        use crate::emulate::{emulate_mm256_reduce_add_ps, partial_load};
        use core::arch::x86_64::*;
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut sum = _mm256_setzero_ps();
        while n >= 8 {
            let x = unsafe { _mm256_cvtph_ps(_mm_loadu_si128(a.cast())) };
            sum = _mm256_fmadd_ps(x, x, sum);
            (n, a) = unsafe { (n - 8, a.add(8)) };
        }
        if n > 0 {
            let (_a,) = unsafe { partial_load!(8, n, a) };
            (a,) = (_a.as_ptr(),);
            let x = unsafe { _mm256_cvtph_ps(_mm_loadu_si128(a.cast())) };
            sum = _mm256_fmadd_ps(x, x, sum);
        }
        emulate_mm256_reduce_add_ps(sum)
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_sum_of_x2_v3_test() {
        use rand::RngExt;
        const EPSILON: f32 = 2.0;
        if !crate::is_cpu_detected!("v3") {
            println!("test {} ... skipped (v3)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let this = (0..n)
                .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            for z in 3984..4016 {
                let this = &this[..z];
                let specialized = unsafe { reduce_sum_of_x2_v3(this) };
                let fallback = fallback(this);
                assert!(
                    (specialized - fallback).abs() < EPSILON,
                    "specialized = {specialized}, fallback = {fallback}."
                );
            }
        }
    }

    #[inline]
    #[cfg(target_arch = "aarch64")]
    #[crate::target_cpu(enable = "a2")]
    #[target_feature(enable = "fhm")]
    fn reduce_sum_of_x2_a2_fhm(this: &[f16]) -> f32 {
        use crate::emulate::partial_load;
        use core::arch::aarch64::*;
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut sum = [vdupq_n_f32(0.0); 16];
        while n >= 64 {
            for i in 0..8 {
                let x = vreinterpretq_f16_u16(unsafe { vld1q_u16(a.add(8 * i).cast()) });
                sum[2 * i] = vfmlalq_low_f16(sum[2 * i], x, x);
                sum[2 * i + 1] = vfmlalq_high_f16(sum[2 * i + 1], x, x);
            }
            (n, a) = unsafe { (n - 64, a.add(64)) };
        }
        while n >= 8 {
            let x = vreinterpretq_f16_u16(unsafe { vld1q_u16(a.cast()) });
            sum[0] = vfmlalq_low_f16(sum[0], x, x);
            sum[1] = vfmlalq_high_f16(sum[1], x, x);
            (n, a) = unsafe { (n - 8, a.add(8)) };
        }
        if n > 0 {
            let (_a,) = unsafe { partial_load!(8, n, a) };
            (a,) = (_a.as_ptr(),);
            let x = vreinterpretq_f16_u16(unsafe { vld1q_u16(a.cast()) });
            sum[2] = vfmlalq_low_f16(sum[2], x, x);
            sum[3] = vfmlalq_high_f16(sum[3], x, x);
        }
        super::aarch64::reduce_add(sum)
    }

    #[inline]
    #[cfg(target_arch = "aarch64")]
    #[crate::target_cpu(enable = "a2")]
    fn reduce_sum_of_x2_a2(this: &[f16]) -> f32 {
        use crate::emulate::{emulate_vreinterpretq_f16_u16, partial_load};
        use core::arch::aarch64::*;
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut sum = [vdupq_n_f32(0.0); 8];
        while n >= 32 {
            for i in 0..4 {
                let x = emulate_vreinterpretq_f16_u16(unsafe { vld1q_u16(a.add(8 * i).cast()) });
                let x_0 = vcvt_f32_f16(vget_low_f16(x));
                let x_1 = vcvt_high_f32_f16(x);
                sum[2 * i] = vfmaq_f32(sum[2 * i], x_0, x_0);
                sum[2 * i + 1] = vfmaq_f32(sum[2 * i + 1], x_1, x_1);
            }
            (n, a) = unsafe { (n - 32, a.add(32)) };
        }
        while n >= 8 {
            let x = emulate_vreinterpretq_f16_u16(unsafe { vld1q_u16(a.cast()) });
            let x_0 = vcvt_f32_f16(vget_low_f16(x));
            let x_1 = vcvt_high_f32_f16(x);
            sum[0] = vfmaq_f32(sum[0], x_0, x_0);
            sum[1] = vfmaq_f32(sum[1], x_1, x_1);
            (n, a) = unsafe { (n - 8, a.add(8)) };
        }
        if n > 0 {
            let (_a,) = unsafe { partial_load!(8, n, a) };
            (a,) = (_a.as_ptr(),);
            let x = emulate_vreinterpretq_f16_u16(unsafe { vld1q_u16(a.cast()) });
            let x_0 = vcvt_f32_f16(vget_low_f16(x));
            let x_1 = vcvt_high_f32_f16(x);
            sum[2] = vfmaq_f32(sum[2], x_0, x_0);
            sum[3] = vfmaq_f32(sum[3], x_1, x_1);
        }
        super::aarch64::reduce_add(sum)
    }

    #[crate::multiversion(
        @"v4", @"v3", "v2", @"a2:fhm", @"a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn reduce_sum_of_x2(this: &[f16]) -> f32 {
        let n = this.len();
        let mut x2 = 0.0f32;
        for i in 0..n {
            x2 += this[i]._to_f32() * this[i]._to_f32();
        }
        x2
    }

    #[cfg(test)]
    #[allow(clippy::type_complexity)]
    fn implementations() -> Vec<(&'static str, unsafe fn(&[f16]) -> f32)> {
        let mut r: Vec<(&'static str, unsafe fn(&[f16]) -> f32)> = vec![
            ("fallback", fallback),
            ("dispatch", |x| reduce_sum_of_x2(x)),
        ];
        #[cfg(target_arch = "x86_64")]
        if crate::is_cpu_detected!("v4") {
            r.push(("v4", reduce_sum_of_x2_v4));
        }
        #[cfg(target_arch = "x86_64")]
        if crate::is_cpu_detected!("v3") {
            r.push(("v3", reduce_sum_of_x2_v3));
        }
        #[cfg(target_arch = "aarch64")]
        if crate::is_cpu_detected!("a2") && crate::is_feature_detected!("fhm") {
            r.push(("a2:fhm", reduce_sum_of_x2_a2_fhm));
        }
        #[cfg(target_arch = "aarch64")]
        if crate::is_cpu_detected!("a2") {
            r.push(("a2", reduce_sum_of_x2_a2));
        }
        r
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_sum_of_x2_accuracy_test() {
        use super::test_util::*;
        let exact = |x: &[f16]| {
            let s = x
                .iter()
                .map(|x| (x._to_f32() as f64) * (x._to_f32() as f64))
                .sum::<f64>();
            (s, s)
        };
        // `255 * 255 * 784` overflows binary16 by a factor of more than 700.
        for (lo, hi) in [(0, 255), (-255, 255), (-4096, 4096)] {
            for n in lengths() {
                let x = integers(n, lo, hi);
                for (name, f) in implementations() {
                    assert_close(name, n, unsafe { f(&x) }, exact(&x));
                }
            }
        }
        for n in lengths() {
            for (lo, hi) in [(-1.0, 1.0), (-1e-3, 1e-3)] {
                let x = random(n, lo, hi);
                for (name, f) in implementations() {
                    assert_close(name, n, unsafe { f(&x) }, exact(&x));
                }
            }
        }
    }
}

mod reduce_min_max_of_x {
    use crate::{F16, f16};

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    #[target_feature(enable = "avx512fp16")]
    fn reduce_min_max_of_x_v4_avx512fp16(this: &[f16]) -> (f32, f32) {
        use core::arch::x86_64::*;
        let mut n = this.len();
        let mut a = this.as_ptr();
        // Bit casts: `_mm512_cvtepi16_ph` would convert the bit patterns as
        // integers (to 31744.0 and -1024.0) instead of producing +/-infinity.
        let mut min = _mm512_castsi512_ph(_mm512_set1_epi16(f16::INFINITY.to_bits() as _));
        let mut max = _mm512_castsi512_ph(_mm512_set1_epi16(f16::NEG_INFINITY.to_bits() as _));
        while n >= 32 {
            let x = unsafe { _mm512_castsi512_ph(_mm512_loadu_epi16(a.cast())) };
            min = _mm512_min_ph(x, min);
            max = _mm512_max_ph(x, max);
            (n, a) = unsafe { (n - 32, a.add(32)) };
        }
        if n > 0 {
            let mask = _bzhi_u32(0xffffffff, n as u32);
            let x = unsafe { _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, a.cast())) };
            min = _mm512_mask_min_ph(min, mask, x, min);
            max = _mm512_mask_max_ph(max, mask, x, max);
        }
        let min = {
            let s_0 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(_mm512_castph_si512(min), 0));
            let s_1 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(_mm512_castph_si512(min), 1));
            _mm512_reduce_min_ps(_mm512_min_ps(s_0, s_1))
        };
        let max = {
            let s_0 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(_mm512_castph_si512(max), 0));
            let s_1 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(_mm512_castph_si512(max), 1));
            _mm512_reduce_max_ps(_mm512_max_ps(s_0, s_1))
        };
        (min, max)
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_min_max_of_x_v4_avx512fp16_test() {
        use rand::RngExt;
        if !crate::is_cpu_detected!("v4") || !crate::is_feature_detected!("avx512fp16") {
            println!("test {} ... skipped (v4:avx512fp16)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 200;
            let mut x = (0..n)
                .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            (x[0], x[1]) = (f16::NAN, -f16::NAN);
            for z in 50..200 {
                let x = &x[..z];
                let specialized = unsafe { reduce_min_max_of_x_v4_avx512fp16(x) };
                let fallback = fallback(x);
                assert_eq!(specialized.0, fallback.0);
                assert_eq!(specialized.1, fallback.1);
            }
        }
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    fn reduce_min_max_of_x_v4(this: &[f16]) -> (f32, f32) {
        use core::arch::x86_64::*;
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut min = _mm512_set1_ps(f32::INFINITY);
        let mut max = _mm512_set1_ps(f32::NEG_INFINITY);
        while n >= 16 {
            let x = unsafe { _mm512_cvtph_ps(_mm256_loadu_epi16(a.cast())) };
            min = _mm512_min_ps(x, min);
            max = _mm512_max_ps(x, max);
            (n, a) = unsafe { (n - 16, a.add(16)) };
        }
        if n > 0 {
            let mask = _bzhi_u32(0xffff, n as u32) as u16;
            let x = unsafe { _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, a.cast())) };
            min = _mm512_mask_min_ps(min, mask, x, min);
            max = _mm512_mask_max_ps(max, mask, x, max);
        }
        let min = _mm512_reduce_min_ps(min);
        let max = _mm512_reduce_max_ps(max);
        (min, max)
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_min_max_of_x_v4_test() {
        use rand::RngExt;
        if !crate::is_cpu_detected!("v4") {
            println!("test {} ... skipped (v4)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 200;
            let mut x = (0..n)
                .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            (x[0], x[1]) = (f16::NAN, -f16::NAN);
            for z in 50..200 {
                let x = &x[..z];
                let specialized = unsafe { reduce_min_max_of_x_v4(x) };
                let fallback = fallback(x);
                assert_eq!(specialized.0, fallback.0);
                assert_eq!(specialized.1, fallback.1);
            }
        }
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v3")]
    fn reduce_min_max_of_x_v3(this: &[f16]) -> (f32, f32) {
        use crate::emulate::{
            emulate_mm256_reduce_max_ps, emulate_mm256_reduce_min_ps, partial_load,
        };
        use core::arch::x86_64::*;
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut min = _mm256_set1_ps(f32::INFINITY);
        let mut max = _mm256_set1_ps(f32::NEG_INFINITY);
        while n >= 8 {
            let x = unsafe { _mm256_cvtph_ps(_mm_loadu_si128(a.cast())) };
            min = _mm256_min_ps(x, min);
            max = _mm256_max_ps(x, max);
            (n, a) = unsafe { (n - 8, a.add(8)) };
        }
        if n > 0 {
            let (_a,) = unsafe { partial_load!(8, n, a = f16::NAN) };
            (a,) = (_a.as_ptr(),);
            let x = unsafe { _mm256_cvtph_ps(_mm_loadu_si128(a.cast())) };
            min = _mm256_min_ps(x, min);
            max = _mm256_max_ps(x, max);
        }
        (
            emulate_mm256_reduce_min_ps(min),
            emulate_mm256_reduce_max_ps(max),
        )
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    fn reduce_min_max_of_x_v3_test() {
        use rand::RngExt;
        if !crate::is_cpu_detected!("v3") {
            println!("test {} ... skipped (v3)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 200;
            let mut x = (0..n)
                .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            (x[0], x[1]) = (f16::NAN, -f16::NAN);
            for z in 50..200 {
                let x = &x[..z];
                let specialized = unsafe { reduce_min_max_of_x_v3(x) };
                let fallback = fallback(x);
                assert_eq!(specialized.0, fallback.0,);
                assert_eq!(specialized.1, fallback.1,);
            }
        }
    }

    #[inline]
    #[cfg(target_arch = "aarch64")]
    #[crate::target_cpu(enable = "a2")]
    #[target_feature(enable = "fp16")]
    fn reduce_min_max_of_x_a2_fp16(this: &[f16]) -> (f32, f32) {
        use crate::emulate::{emulate_vreinterpretq_f16_u16, partial_load};
        use core::arch::aarch64::*;
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut min = emulate_vreinterpretq_f16_u16(vdupq_n_u16(0x7C00u16));
        let mut max = emulate_vreinterpretq_f16_u16(vdupq_n_u16(0xFC00u16));
        while n >= 8 {
            let x = emulate_vreinterpretq_f16_u16(unsafe { vld1q_u16(a.cast()) });
            min = vminnmq_f16(x, min);
            max = vmaxnmq_f16(x, max);
            (n, a) = unsafe { (n - 8, a.add(8)) };
        }
        if n > 0 {
            let (_a,) = unsafe { partial_load!(8, n, a = f16::NAN) };
            (a,) = (_a.as_ptr(),);
            let x = emulate_vreinterpretq_f16_u16(unsafe { vld1q_u16(a.cast()) });
            min = vminnmq_f16(x, min);
            max = vmaxnmq_f16(x, max);
        }
        (
            vminnmvq_f32(vcvt_f32_f16(vminnm_f16(
                vget_low_f16(min),
                vget_high_f16(min),
            ))),
            vmaxnmvq_f32(vcvt_f32_f16(vmaxnm_f16(
                vget_low_f16(max),
                vget_high_f16(max),
            ))),
        )
    }

    #[cfg(all(target_arch = "aarch64", test))]
    #[test]
    fn reduce_min_max_of_x_a2_fp16_test() {
        use rand::RngExt;
        if !crate::is_cpu_detected!("a2") || !crate::is_feature_detected!("fp16") {
            println!("test {} ... skipped (a2:fp16)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 200;
            let mut x = (0..n)
                .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            (x[0], x[1]) = (f16::NAN, -f16::NAN);
            for z in 50..200 {
                let x = &x[..z];
                let specialized = unsafe { reduce_min_max_of_x_a2_fp16(x) };
                let fallback = fallback(x);
                assert_eq!(specialized.0, fallback.0,);
                assert_eq!(specialized.1, fallback.1,);
            }
        }
    }

    #[inline]
    #[cfg(target_arch = "aarch64")]
    #[crate::target_cpu(enable = "a2")]
    fn reduce_min_max_of_x_a2(this: &[f16]) -> (f32, f32) {
        use crate::emulate::{emulate_vreinterpret_f16_u16, partial_load};
        use core::arch::aarch64::*;
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut min = vdupq_n_f32(f32::INFINITY);
        let mut max = vdupq_n_f32(f32::NEG_INFINITY);
        while n >= 4 {
            let x = emulate_vreinterpret_f16_u16(unsafe { vld1_u16(a.cast()) });
            let x = vcvt_f32_f16(x);
            min = vminnmq_f32(x, min);
            max = vmaxnmq_f32(x, max);
            (n, a) = unsafe { (n - 4, a.add(4)) };
        }
        if n > 0 {
            let (_a,) = unsafe { partial_load!(4, n, a = f16::NAN) };
            (a,) = (_a.as_ptr(),);
            let x = emulate_vreinterpret_f16_u16(unsafe { vld1_u16(a.cast()) });
            let x = vcvt_f32_f16(x);
            min = vminnmq_f32(x, min);
            max = vmaxnmq_f32(x, max);
        }
        (vminnmvq_f32(min), vmaxnmvq_f32(max))
    }

    #[cfg(all(target_arch = "aarch64", test))]
    #[test]
    fn reduce_min_max_of_x_a2_test() {
        use rand::RngExt;
        if !crate::is_cpu_detected!("a2") {
            println!("test {} ... skipped (a2)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 200;
            let mut x = (0..n)
                .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            (x[0], x[1]) = (f16::NAN, -f16::NAN);
            for z in 50..200 {
                let x = &x[..z];
                let specialized = unsafe { reduce_min_max_of_x_a2(x) };
                let fallback = fallback(x);
                assert_eq!(specialized.0, fallback.0,);
                assert_eq!(specialized.1, fallback.1,);
            }
        }
    }

    #[crate::multiversion(
        @"v4:avx512fp16", @"v4", @"v3", "v2", @"a2:fp16", @"a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn reduce_min_max_of_x(this: &[f16]) -> (f32, f32) {
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let n = this.len();
        for i in 0..n {
            min = min.min(this[i]._to_f32());
            max = max.max(this[i]._to_f32());
        }
        (min, max)
    }

    /// Every element lies outside `[-1024, 31744]`, the range the avx512fp16
    /// kernel used to start from when it converted the bit patterns of
    /// +/-infinity as integers.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_min_max_of_x_out_of_range_test() {
        #[allow(clippy::type_complexity)]
        let mut implementations: Vec<(&str, unsafe fn(&[f16]) -> (f32, f32))> = vec![
            ("fallback", fallback),
            ("dispatch", |x| reduce_min_max_of_x(x)),
        ];
        #[cfg(target_arch = "x86_64")]
        if crate::is_cpu_detected!("v4") && crate::is_feature_detected!("avx512fp16") {
            implementations.push(("v4:avx512fp16", reduce_min_max_of_x_v4_avx512fp16));
        }
        #[cfg(target_arch = "x86_64")]
        if crate::is_cpu_detected!("v4") {
            implementations.push(("v4", reduce_min_max_of_x_v4));
        }
        #[cfg(target_arch = "x86_64")]
        if crate::is_cpu_detected!("v3") {
            implementations.push(("v3", reduce_min_max_of_x_v3));
        }
        #[cfg(target_arch = "aarch64")]
        if crate::is_cpu_detected!("a2") && crate::is_feature_detected!("fp16") {
            implementations.push(("a2:fp16", reduce_min_max_of_x_a2_fp16));
        }
        #[cfg(target_arch = "aarch64")]
        if crate::is_cpu_detected!("a2") {
            implementations.push(("a2", reduce_min_max_of_x_a2));
        }
        for (lo, hi) in [(40000, 65504), (-65504, -2000), (32000, 32760)] {
            for n in super::test_util::lengths() {
                let x = super::test_util::integers(n, lo, hi);
                let min = x.iter().map(|x| x._to_f32()).fold(f32::INFINITY, f32::min);
                let max = x
                    .iter()
                    .map(|x| x._to_f32())
                    .fold(f32::NEG_INFINITY, f32::max);
                for (name, f) in implementations.iter() {
                    let (got_min, got_max) = unsafe { f(&x) };
                    assert_eq!((got_min, got_max), (min, max), "{name}: n = {n}");
                }
            }
        }
    }
}

#[cfg_attr(feature = "internal", simd_macros::public)]
mod reduce_sum_of_xy {
    use crate::{F16, f16};

    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    fn reduce_sum_of_xy_v4(lhs: &[f16], rhs: &[f16]) -> f32 {
        assert!(lhs.len() == rhs.len());
        use core::arch::x86_64::*;
        let mut n = lhs.len();
        let mut a = lhs.as_ptr();
        let mut b = rhs.as_ptr();
        let mut sum = _mm512_setzero_ps();
        while n >= 16 {
            let x = unsafe { _mm512_cvtph_ps(_mm256_loadu_epi16(a.cast())) };
            let y = unsafe { _mm512_cvtph_ps(_mm256_loadu_epi16(b.cast())) };
            sum = _mm512_fmadd_ps(x, y, sum);
            (n, a, b) = unsafe { (n - 16, a.add(16), b.add(16)) };
        }
        if n > 0 {
            let mask = _bzhi_u32(0xffff, n as u32) as u16;
            let x = unsafe { _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, a.cast())) };
            let y = unsafe { _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, b.cast())) };
            sum = _mm512_fmadd_ps(x, y, sum);
        }
        _mm512_reduce_add_ps(sum)
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_sum_of_xy_v4_test() {
        use rand::RngExt;
        const EPSILON: f32 = 2.0;
        if !crate::is_cpu_detected!("v4") {
            println!("test {} ... skipped (v4)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let lhs = (0..n)
                .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            let rhs = (0..n)
                .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            let specialized = unsafe { reduce_sum_of_xy_v4(&lhs, &rhs) };
            let fallback = fallback(&lhs, &rhs);
            assert!(
                (specialized - fallback).abs() < EPSILON,
                "specialized = {specialized}, fallback = {fallback}."
            );
        }
    }

    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v3")]
    fn reduce_sum_of_xy_v3(lhs: &[f16], rhs: &[f16]) -> f32 {
        use crate::emulate::{emulate_mm256_reduce_add_ps, partial_load};
        use core::arch::x86_64::*;
        assert!(lhs.len() == rhs.len());
        let mut n = lhs.len();
        let mut a = lhs.as_ptr();
        let mut b = rhs.as_ptr();
        let mut sum = _mm256_setzero_ps();
        while n >= 8 {
            let x = unsafe { _mm256_cvtph_ps(_mm_loadu_si128(a.cast())) };
            let y = unsafe { _mm256_cvtph_ps(_mm_loadu_si128(b.cast())) };
            sum = _mm256_fmadd_ps(x, y, sum);
            (n, a, b) = unsafe { (n - 8, a.add(8), b.add(8)) };
        }
        if n > 0 {
            let (_a, _b) = unsafe { partial_load!(8, n, a, b) };
            (a, b) = (_a.as_ptr(), _b.as_ptr());
            let x = unsafe { _mm256_cvtph_ps(_mm_loadu_si128(a.cast())) };
            let y = unsafe { _mm256_cvtph_ps(_mm_loadu_si128(b.cast())) };
            sum = _mm256_fmadd_ps(x, y, sum);
        }
        emulate_mm256_reduce_add_ps(sum)
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_sum_of_xy_v3_test() {
        use rand::RngExt;
        const EPSILON: f32 = 2.0;
        if !crate::is_cpu_detected!("v3") {
            println!("test {} ... skipped (v3)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let lhs = (0..n)
                .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            let rhs = (0..n)
                .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            for z in 3984..4016 {
                let lhs = &lhs[..z];
                let rhs = &rhs[..z];
                let specialized = unsafe { reduce_sum_of_xy_v3(lhs, rhs) };
                let fallback = fallback(lhs, rhs);
                assert!(
                    (specialized - fallback).abs() < EPSILON,
                    "specialized = {specialized}, fallback = {fallback}."
                );
            }
        }
    }

    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    // Not dispatched until validated on SVE-512 hardware; kept for its test.
    #[cfg_attr(not(test), allow(dead_code))]
    #[crate::target_cpu(enable = "a3.512")]
    fn reduce_sum_of_xy_a3_512(lhs: &[f16], rhs: &[f16]) -> f32 {
        unsafe extern "C" {
            #[link_name = "fp16_reduce_sum_of_xy_a3_512"]
            unsafe fn f(n: usize, a: *const f16, b: *const f16) -> f32;
        }
        assert!(lhs.len() == rhs.len());
        let n = lhs.len();
        let a = lhs.as_ptr();
        let b = rhs.as_ptr();
        unsafe { f(n, a, b) }
    }

    #[cfg(all(target_arch = "aarch64", target_endian = "little", test))]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_sum_of_xy_a3_512_test() {
        use rand::RngExt;
        const EPSILON: f32 = 2.0;
        if !crate::is_cpu_detected!("a3.512") {
            println!("test {} ... skipped (a3.512)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let lhs = (0..n)
                .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            let rhs = (0..n)
                .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            for z in 3984..4016 {
                let lhs = &lhs[..z];
                let rhs = &rhs[..z];
                let specialized = unsafe { reduce_sum_of_xy_a3_512(lhs, rhs) };
                let fallback = fallback(lhs, rhs);
                assert!(
                    (specialized - fallback).abs() < EPSILON,
                    "specialized = {specialized}, fallback = {fallback}."
                );
            }
        }
    }

    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "aarch64")]
    #[crate::target_cpu(enable = "a2")]
    #[target_feature(enable = "fhm")]
    fn reduce_sum_of_xy_a2_fhm(lhs: &[f16], rhs: &[f16]) -> f32 {
        use crate::emulate::partial_load;
        use core::arch::aarch64::*;
        assert!(lhs.len() == rhs.len());
        let mut n = lhs.len();
        let mut a = lhs.as_ptr();
        let mut b = rhs.as_ptr();
        let mut sum = [vdupq_n_f32(0.0); 16];
        while n >= 64 {
            for i in 0..8 {
                let x = vreinterpretq_f16_u16(unsafe { vld1q_u16(a.add(8 * i).cast()) });
                let y = vreinterpretq_f16_u16(unsafe { vld1q_u16(b.add(8 * i).cast()) });
                sum[2 * i] = vfmlalq_low_f16(sum[2 * i], x, y);
                sum[2 * i + 1] = vfmlalq_high_f16(sum[2 * i + 1], x, y);
            }
            (n, a, b) = unsafe { (n - 64, a.add(64), b.add(64)) };
        }
        while n >= 8 {
            let x = vreinterpretq_f16_u16(unsafe { vld1q_u16(a.cast()) });
            let y = vreinterpretq_f16_u16(unsafe { vld1q_u16(b.cast()) });
            sum[0] = vfmlalq_low_f16(sum[0], x, y);
            sum[1] = vfmlalq_high_f16(sum[1], x, y);
            (n, a, b) = unsafe { (n - 8, a.add(8), b.add(8)) };
        }
        if n > 0 {
            let (_a, _b) = unsafe { partial_load!(8, n, a, b) };
            (a, b) = (_a.as_ptr(), _b.as_ptr());
            let x = vreinterpretq_f16_u16(unsafe { vld1q_u16(a.cast()) });
            let y = vreinterpretq_f16_u16(unsafe { vld1q_u16(b.cast()) });
            sum[2] = vfmlalq_low_f16(sum[2], x, y);
            sum[3] = vfmlalq_high_f16(sum[3], x, y);
        }
        super::aarch64::reduce_add(sum)
    }

    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "aarch64")]
    #[crate::target_cpu(enable = "a2")]
    fn reduce_sum_of_xy_a2(lhs: &[f16], rhs: &[f16]) -> f32 {
        use crate::emulate::{emulate_vreinterpretq_f16_u16, partial_load};
        use core::arch::aarch64::*;
        assert!(lhs.len() == rhs.len());
        let mut n = lhs.len();
        let mut a = lhs.as_ptr();
        let mut b = rhs.as_ptr();
        let mut sum = [vdupq_n_f32(0.0); 8];
        while n >= 32 {
            for i in 0..4 {
                let x = emulate_vreinterpretq_f16_u16(unsafe { vld1q_u16(a.add(8 * i).cast()) });
                let y = emulate_vreinterpretq_f16_u16(unsafe { vld1q_u16(b.add(8 * i).cast()) });
                let (x_0, x_1) = (vcvt_f32_f16(vget_low_f16(x)), vcvt_high_f32_f16(x));
                let (y_0, y_1) = (vcvt_f32_f16(vget_low_f16(y)), vcvt_high_f32_f16(y));
                sum[2 * i] = vfmaq_f32(sum[2 * i], x_0, y_0);
                sum[2 * i + 1] = vfmaq_f32(sum[2 * i + 1], x_1, y_1);
            }
            (n, a, b) = unsafe { (n - 32, a.add(32), b.add(32)) };
        }
        while n >= 8 {
            let x = emulate_vreinterpretq_f16_u16(unsafe { vld1q_u16(a.cast()) });
            let y = emulate_vreinterpretq_f16_u16(unsafe { vld1q_u16(b.cast()) });
            let (x_0, x_1) = (vcvt_f32_f16(vget_low_f16(x)), vcvt_high_f32_f16(x));
            let (y_0, y_1) = (vcvt_f32_f16(vget_low_f16(y)), vcvt_high_f32_f16(y));
            sum[0] = vfmaq_f32(sum[0], x_0, y_0);
            sum[1] = vfmaq_f32(sum[1], x_1, y_1);
            (n, a, b) = unsafe { (n - 8, a.add(8), b.add(8)) };
        }
        if n > 0 {
            let (_a, _b) = unsafe { partial_load!(8, n, a, b) };
            (a, b) = (_a.as_ptr(), _b.as_ptr());
            let x = emulate_vreinterpretq_f16_u16(unsafe { vld1q_u16(a.cast()) });
            let y = emulate_vreinterpretq_f16_u16(unsafe { vld1q_u16(b.cast()) });
            let (x_0, x_1) = (vcvt_f32_f16(vget_low_f16(x)), vcvt_high_f32_f16(x));
            let (y_0, y_1) = (vcvt_f32_f16(vget_low_f16(y)), vcvt_high_f32_f16(y));
            sum[2] = vfmaq_f32(sum[2], x_0, y_0);
            sum[3] = vfmaq_f32(sum[3], x_1, y_1);
        }
        super::aarch64::reduce_add(sum)
    }

    #[crate::multiversion(@"v4", @"v3", @"a2:fhm", @"a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1")]
    pub fn reduce_sum_of_xy(lhs: &[f16], rhs: &[f16]) -> f32 {
        assert!(lhs.len() == rhs.len());
        let n = lhs.len();
        let mut sum = 0.0f32;
        for i in 0..n {
            sum += lhs[i]._to_f32() * rhs[i]._to_f32();
        }
        sum
    }

    #[cfg(test)]
    #[allow(clippy::type_complexity)]
    fn implementations() -> Vec<(&'static str, unsafe fn(&[f16], &[f16]) -> f32)> {
        let mut r: Vec<(&'static str, unsafe fn(&[f16], &[f16]) -> f32)> = vec![
            ("fallback", fallback),
            ("dispatch", |x, y| reduce_sum_of_xy(x, y)),
        ];
        #[cfg(target_arch = "x86_64")]
        if crate::is_cpu_detected!("v4") {
            r.push(("v4", reduce_sum_of_xy_v4));
        }
        #[cfg(target_arch = "x86_64")]
        if crate::is_cpu_detected!("v3") {
            r.push(("v3", reduce_sum_of_xy_v3));
        }
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        if crate::is_cpu_detected!("a3.512") {
            r.push(("a3.512", reduce_sum_of_xy_a3_512));
        }
        #[cfg(target_arch = "aarch64")]
        if crate::is_cpu_detected!("a2") && crate::is_feature_detected!("fhm") {
            r.push(("a2:fhm", reduce_sum_of_xy_a2_fhm));
        }
        #[cfg(target_arch = "aarch64")]
        if crate::is_cpu_detected!("a2") {
            r.push(("a2", reduce_sum_of_xy_a2));
        }
        r
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_sum_of_xy_accuracy_test() {
        use super::test_util::*;
        let exact = |x: &[f16], y: &[f16]| {
            let t =
                || std::iter::zip(x, y).map(|(x, y)| (x._to_f32() as f64) * (y._to_f32() as f64));
            (t().sum::<f64>(), t().map(f64::abs).sum::<f64>())
        };
        for (lo, hi) in [(0, 255), (-255, 255), (-4096, 4096)] {
            for n in lengths() {
                let (x, y) = (integers(n, lo, hi), integers(n, lo, hi));
                for (name, f) in implementations() {
                    assert_close(name, n, unsafe { f(&x, &y) }, exact(&x, &y));
                    // A vector against itself: the norm of a large-magnitude vector.
                    assert_close(name, n, unsafe { f(&x, &x) }, exact(&x, &x));
                }
            }
        }
        for n in lengths() {
            for (lo, hi) in [(-1.0, 1.0), (-1e-3, 1e-3)] {
                let (x, y) = (random(n, lo, hi), random(n, lo, hi));
                for (name, f) in implementations() {
                    assert_close(name, n, unsafe { f(&x, &y) }, exact(&x, &y));
                }
            }
        }
    }
}

#[cfg_attr(feature = "internal", simd_macros::public)]
mod reduce_sum_of_d2 {
    use crate::{F16, f16};

    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    fn reduce_sum_of_d2_v4(lhs: &[f16], rhs: &[f16]) -> f32 {
        assert!(lhs.len() == rhs.len());
        use core::arch::x86_64::*;
        let mut n = lhs.len();
        let mut a = lhs.as_ptr();
        let mut b = rhs.as_ptr();
        let mut sum = _mm512_setzero_ps();
        while n >= 16 {
            let x = unsafe { _mm512_cvtph_ps(_mm256_loadu_epi16(a.cast())) };
            let y = unsafe { _mm512_cvtph_ps(_mm256_loadu_epi16(b.cast())) };
            let d = _mm512_sub_ps(x, y);
            sum = _mm512_fmadd_ps(d, d, sum);
            (n, a, b) = unsafe { (n - 16, a.add(16), b.add(16)) };
        }
        if n > 0 {
            let mask = _bzhi_u32(0xffff, n as u32) as u16;
            let x = unsafe { _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, a.cast())) };
            let y = unsafe { _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, b.cast())) };
            let d = _mm512_sub_ps(x, y);
            sum = _mm512_fmadd_ps(d, d, sum);
        }
        _mm512_reduce_add_ps(sum)
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_sum_of_d2_v4_test() {
        use rand::RngExt;
        const EPSILON: f32 = 2.0;
        if !crate::is_cpu_detected!("v4") {
            println!("test {} ... skipped (v4)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let lhs = (0..n)
                .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            let rhs = (0..n)
                .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            for z in 3984..4016 {
                let lhs = &lhs[..z];
                let rhs = &rhs[..z];
                let specialized = unsafe { reduce_sum_of_d2_v4(lhs, rhs) };
                let fallback = fallback(lhs, rhs);
                assert!(
                    (specialized - fallback).abs() < EPSILON,
                    "specialized = {specialized}, fallback = {fallback}."
                );
            }
        }
    }

    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v3")]
    fn reduce_sum_of_d2_v3(lhs: &[f16], rhs: &[f16]) -> f32 {
        use crate::emulate::{emulate_mm256_reduce_add_ps, partial_load};
        assert!(lhs.len() == rhs.len());
        use core::arch::x86_64::*;
        let mut n = lhs.len();
        let mut a = lhs.as_ptr();
        let mut b = rhs.as_ptr();
        let mut sum = _mm256_setzero_ps();
        while n >= 8 {
            let x = unsafe { _mm256_cvtph_ps(_mm_loadu_si128(a.cast())) };
            let y = unsafe { _mm256_cvtph_ps(_mm_loadu_si128(b.cast())) };
            let d = _mm256_sub_ps(x, y);
            sum = _mm256_fmadd_ps(d, d, sum);
            (n, a, b) = unsafe { (n - 8, a.add(8), b.add(8)) };
        }
        if n > 0 {
            let (_a, _b) = unsafe { partial_load!(8, n, a, b) };
            (a, b) = (_a.as_ptr(), _b.as_ptr());

            let x = unsafe { _mm256_cvtph_ps(_mm_loadu_si128(a.cast())) };
            let y = unsafe { _mm256_cvtph_ps(_mm_loadu_si128(b.cast())) };
            let d = _mm256_sub_ps(x, y);
            sum = _mm256_fmadd_ps(d, d, sum);
        }
        emulate_mm256_reduce_add_ps(sum)
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_sum_of_d2_v3_test() {
        use rand::RngExt;
        const EPSILON: f32 = 2.0;
        if !crate::is_cpu_detected!("v3") {
            println!("test {} ... skipped (v3)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let lhs = (0..n)
                .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            let rhs = (0..n)
                .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            for z in 3984..4016 {
                let lhs = &lhs[..z];
                let rhs = &rhs[..z];
                let specialized = unsafe { reduce_sum_of_d2_v3(lhs, rhs) };
                let fallback = fallback(lhs, rhs);
                assert!(
                    (specialized - fallback).abs() < EPSILON,
                    "specialized = {specialized}, fallback = {fallback}."
                );
            }
        }
    }

    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    // Not dispatched until validated on SVE-512 hardware; kept for its test.
    #[cfg_attr(not(test), allow(dead_code))]
    #[crate::target_cpu(enable = "a3.512")]
    fn reduce_sum_of_d2_a3_512(lhs: &[f16], rhs: &[f16]) -> f32 {
        unsafe extern "C" {
            #[link_name = "fp16_reduce_sum_of_d2_a3_512"]
            unsafe fn f(n: usize, a: *const f16, b: *const f16) -> f32;
        }
        assert!(lhs.len() == rhs.len());
        let n = lhs.len();
        let a = lhs.as_ptr();
        let b = rhs.as_ptr();
        unsafe { f(n, a, b) }
    }

    #[cfg(all(target_arch = "aarch64", target_endian = "little", test))]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_sum_of_d2_a3_512_test() {
        use rand::RngExt;
        const EPSILON: f32 = 2.0;
        if !crate::is_cpu_detected!("a3.512") {
            println!("test {} ... skipped (a3.512)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let lhs = (0..n)
                .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            let rhs = (0..n)
                .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            for z in 3984..4016 {
                let lhs = &lhs[..z];
                let rhs = &rhs[..z];
                let specialized = unsafe { reduce_sum_of_d2_a3_512(lhs, rhs) };
                let fallback = fallback(lhs, rhs);
                assert!(
                    (specialized - fallback).abs() < EPSILON,
                    "specialized = {specialized}, fallback = {fallback}."
                );
            }
        }
    }

    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "aarch64")]
    #[crate::target_cpu(enable = "a2")]
    #[target_feature(enable = "fhm")]
    fn reduce_sum_of_d2_a2_fhm(lhs: &[f16], rhs: &[f16]) -> f32 {
        use crate::emulate::partial_load;
        use core::arch::aarch64::*;
        // `x - y` is computed in f32 (`FMLSL` with `y * 1.0`, which is exact),
        // never in f16: an f16 difference would be rounded, and it overflows
        // once `|x - y| > 65504`.
        let ones = vreinterpretq_f16_u16(vdupq_n_u16(0x3c00));
        assert!(lhs.len() == rhs.len());
        let mut n = lhs.len();
        let mut a = lhs.as_ptr();
        let mut b = rhs.as_ptr();
        let mut sum = [vdupq_n_f32(0.0); 16];
        while n >= 64 {
            for i in 0..8 {
                let x = vreinterpretq_f16_u16(unsafe { vld1q_u16(a.add(8 * i).cast()) });
                let y = vreinterpretq_f16_u16(unsafe { vld1q_u16(b.add(8 * i).cast()) });
                let d_0 = vfmlslq_low_f16(vcvt_f32_f16(vget_low_f16(x)), y, ones);
                let d_1 = vfmlslq_high_f16(vcvt_high_f32_f16(x), y, ones);
                sum[2 * i] = vfmaq_f32(sum[2 * i], d_0, d_0);
                sum[2 * i + 1] = vfmaq_f32(sum[2 * i + 1], d_1, d_1);
            }
            (n, a, b) = unsafe { (n - 64, a.add(64), b.add(64)) };
        }
        while n >= 8 {
            let x = vreinterpretq_f16_u16(unsafe { vld1q_u16(a.cast()) });
            let y = vreinterpretq_f16_u16(unsafe { vld1q_u16(b.cast()) });
            let d_0 = vfmlslq_low_f16(vcvt_f32_f16(vget_low_f16(x)), y, ones);
            let d_1 = vfmlslq_high_f16(vcvt_high_f32_f16(x), y, ones);
            sum[0] = vfmaq_f32(sum[0], d_0, d_0);
            sum[1] = vfmaq_f32(sum[1], d_1, d_1);
            (n, a, b) = unsafe { (n - 8, a.add(8), b.add(8)) };
        }
        if n > 0 {
            let (_a, _b) = unsafe { partial_load!(8, n, a, b) };
            (a, b) = (_a.as_ptr(), _b.as_ptr());
            let x = vreinterpretq_f16_u16(unsafe { vld1q_u16(a.cast()) });
            let y = vreinterpretq_f16_u16(unsafe { vld1q_u16(b.cast()) });
            let d_0 = vfmlslq_low_f16(vcvt_f32_f16(vget_low_f16(x)), y, ones);
            let d_1 = vfmlslq_high_f16(vcvt_high_f32_f16(x), y, ones);
            sum[2] = vfmaq_f32(sum[2], d_0, d_0);
            sum[3] = vfmaq_f32(sum[3], d_1, d_1);
        }
        super::aarch64::reduce_add(sum)
    }

    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "aarch64")]
    #[crate::target_cpu(enable = "a2")]
    fn reduce_sum_of_d2_a2(lhs: &[f16], rhs: &[f16]) -> f32 {
        use crate::emulate::{emulate_vreinterpretq_f16_u16, partial_load};
        use core::arch::aarch64::*;
        assert!(lhs.len() == rhs.len());
        let mut n = lhs.len();
        let mut a = lhs.as_ptr();
        let mut b = rhs.as_ptr();
        let mut sum = [vdupq_n_f32(0.0); 8];
        while n >= 32 {
            for i in 0..4 {
                let x = emulate_vreinterpretq_f16_u16(unsafe { vld1q_u16(a.add(8 * i).cast()) });
                let y = emulate_vreinterpretq_f16_u16(unsafe { vld1q_u16(b.add(8 * i).cast()) });
                let d_0 = vsubq_f32(vcvt_f32_f16(vget_low_f16(x)), vcvt_f32_f16(vget_low_f16(y)));
                let d_1 = vsubq_f32(vcvt_high_f32_f16(x), vcvt_high_f32_f16(y));
                sum[2 * i] = vfmaq_f32(sum[2 * i], d_0, d_0);
                sum[2 * i + 1] = vfmaq_f32(sum[2 * i + 1], d_1, d_1);
            }
            (n, a, b) = unsafe { (n - 32, a.add(32), b.add(32)) };
        }
        while n >= 8 {
            let x = emulate_vreinterpretq_f16_u16(unsafe { vld1q_u16(a.cast()) });
            let y = emulate_vreinterpretq_f16_u16(unsafe { vld1q_u16(b.cast()) });
            let d_0 = vsubq_f32(vcvt_f32_f16(vget_low_f16(x)), vcvt_f32_f16(vget_low_f16(y)));
            let d_1 = vsubq_f32(vcvt_high_f32_f16(x), vcvt_high_f32_f16(y));
            sum[0] = vfmaq_f32(sum[0], d_0, d_0);
            sum[1] = vfmaq_f32(sum[1], d_1, d_1);
            (n, a, b) = unsafe { (n - 8, a.add(8), b.add(8)) };
        }
        if n > 0 {
            let (_a, _b) = unsafe { partial_load!(8, n, a, b) };
            (a, b) = (_a.as_ptr(), _b.as_ptr());
            let x = emulate_vreinterpretq_f16_u16(unsafe { vld1q_u16(a.cast()) });
            let y = emulate_vreinterpretq_f16_u16(unsafe { vld1q_u16(b.cast()) });
            let d_0 = vsubq_f32(vcvt_f32_f16(vget_low_f16(x)), vcvt_f32_f16(vget_low_f16(y)));
            let d_1 = vsubq_f32(vcvt_high_f32_f16(x), vcvt_high_f32_f16(y));
            sum[2] = vfmaq_f32(sum[2], d_0, d_0);
            sum[3] = vfmaq_f32(sum[3], d_1, d_1);
        }
        super::aarch64::reduce_add(sum)
    }

    #[crate::multiversion(@"v4", @"v3", @"a2:fhm", @"a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1")]
    pub fn reduce_sum_of_d2(lhs: &[f16], rhs: &[f16]) -> f32 {
        assert!(lhs.len() == rhs.len());
        let n = lhs.len();
        let mut sum = 0.0_f32;
        for i in 0..n {
            let d = lhs[i]._to_f32() - rhs[i]._to_f32();
            sum += d * d;
        }
        sum
    }

    #[cfg(test)]
    #[allow(clippy::type_complexity)]
    fn implementations() -> Vec<(&'static str, unsafe fn(&[f16], &[f16]) -> f32)> {
        let mut r: Vec<(&'static str, unsafe fn(&[f16], &[f16]) -> f32)> = vec![
            ("fallback", fallback),
            ("dispatch", |x, y| reduce_sum_of_d2(x, y)),
        ];
        #[cfg(target_arch = "x86_64")]
        if crate::is_cpu_detected!("v4") {
            r.push(("v4", reduce_sum_of_d2_v4));
        }
        #[cfg(target_arch = "x86_64")]
        if crate::is_cpu_detected!("v3") {
            r.push(("v3", reduce_sum_of_d2_v3));
        }
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        if crate::is_cpu_detected!("a3.512") {
            r.push(("a3.512", reduce_sum_of_d2_a3_512));
        }
        #[cfg(target_arch = "aarch64")]
        if crate::is_cpu_detected!("a2") && crate::is_feature_detected!("fhm") {
            r.push(("a2:fhm", reduce_sum_of_d2_a2_fhm));
        }
        #[cfg(target_arch = "aarch64")]
        if crate::is_cpu_detected!("a2") {
            r.push(("a2", reduce_sum_of_d2_a2));
        }
        r
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_sum_of_d2_accuracy_test() {
        use super::test_util::*;
        let exact = |x: &[f16], y: &[f16]| {
            let s = std::iter::zip(x, y)
                .map(|(x, y)| (x._to_f32() as f64 - y._to_f32() as f64).powi(2))
                .sum::<f64>();
            (s, s)
        };
        // `x - y` reaches 131008 for the last range, beyond the f16 range.
        for (lo, hi) in [(0, 255), (-255, 255), (-4096, 4096), (-65504, 65504)] {
            for n in lengths() {
                let (x, y) = (integers(n, lo, hi), integers(n, lo, hi));
                let zero = vec![f16::_ZERO; n];
                for (name, f) in implementations() {
                    assert_close(name, n, unsafe { f(&x, &y) }, exact(&x, &y));
                    assert_close(name, n, unsafe { f(&x, &zero) }, exact(&x, &zero));
                    assert_eq!(unsafe { f(&x, &x) }, 0.0, "{name}: n = {n}");
                }
            }
        }
        for n in lengths() {
            // The second range covers subnormal halves.
            for (lo, hi) in [(-1.0, 1.0), (-1e-4, 1e-4)] {
                let (x, y) = (random(n, lo, hi), random(n, lo, hi));
                for (name, f) in implementations() {
                    assert_close(name, n, unsafe { f(&x, &y) }, exact(&x, &y));
                }
            }
        }
    }
}

mod reduce_sum_of_xy_sparse {
    // There is no manually-implemented SIMD version.
    // Add it if `svecf16` is supported.

    use crate::{F16, f16};

    #[crate::multiversion(
        "v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn reduce_sum_of_xy_sparse(lidx: &[u32], lval: &[f16], ridx: &[u32], rval: &[f16]) -> f32 {
        use std::cmp::Ordering;
        assert_eq!(lidx.len(), lval.len());
        assert_eq!(ridx.len(), rval.len());
        let (mut lp, ln) = (0, lidx.len());
        let (mut rp, rn) = (0, ridx.len());
        let mut sum = 0.0f32;
        while lp < ln && rp < rn {
            match Ord::cmp(&lidx[lp], &ridx[rp]) {
                Ordering::Equal => {
                    sum += lval[lp]._to_f32() * rval[rp]._to_f32();
                    lp += 1;
                    rp += 1;
                }
                Ordering::Less => {
                    lp += 1;
                }
                Ordering::Greater => {
                    rp += 1;
                }
            }
        }
        sum
    }
}

mod reduce_sum_of_d2_sparse {
    // There is no manually-implemented SIMD version.
    // Add it if `svecf16` is supported.

    use crate::{F16, f16};

    #[crate::multiversion(
        "v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn reduce_sum_of_d2_sparse(lidx: &[u32], lval: &[f16], ridx: &[u32], rval: &[f16]) -> f32 {
        use std::cmp::Ordering;
        assert_eq!(lidx.len(), lval.len());
        assert_eq!(ridx.len(), rval.len());
        let (mut lp, ln) = (0, lidx.len());
        let (mut rp, rn) = (0, ridx.len());
        let mut sum = 0.0f32;
        while lp < ln && rp < rn {
            match Ord::cmp(&lidx[lp], &ridx[rp]) {
                Ordering::Equal => {
                    let d = lval[lp]._to_f32() - rval[rp]._to_f32();
                    sum += d * d;
                    lp += 1;
                    rp += 1;
                }
                Ordering::Less => {
                    sum += lval[lp]._to_f32() * lval[lp]._to_f32();
                    lp += 1;
                }
                Ordering::Greater => {
                    sum += rval[rp]._to_f32() * rval[rp]._to_f32();
                    rp += 1;
                }
            }
        }
        for i in lp..ln {
            sum += lval[i]._to_f32() * lval[i]._to_f32();
        }
        for i in rp..rn {
            sum += rval[i]._to_f32() * rval[i]._to_f32();
        }
        sum
    }
}

mod vector_add {
    use crate::f16;

    #[crate::multiversion(
        "v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn vector_add(lhs: &[f16], rhs: &[f16]) -> Vec<f16> {
        assert_eq!(lhs.len(), rhs.len());
        let n = lhs.len();
        let mut r = Vec::<f16>::with_capacity(n);
        for i in 0..n {
            unsafe {
                r.as_mut_ptr().add(i).write(lhs[i] + rhs[i]);
            }
        }
        unsafe {
            r.set_len(n);
        }
        r
    }
}

mod vector_add_inplace {
    use crate::f16;

    #[crate::multiversion(
        "v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn vector_add_inplace(lhs: &mut [f16], rhs: &[f16]) {
        assert_eq!(lhs.len(), rhs.len());
        let n = lhs.len();
        for i in 0..n {
            lhs[i] += rhs[i];
        }
    }
}

mod vector_sub {
    use crate::f16;

    #[crate::multiversion(
        "v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn vector_sub(lhs: &[f16], rhs: &[f16]) -> Vec<f16> {
        assert_eq!(lhs.len(), rhs.len());
        let n = lhs.len();
        let mut r = Vec::<f16>::with_capacity(n);
        for i in 0..n {
            unsafe {
                r.as_mut_ptr().add(i).write(lhs[i] - rhs[i]);
            }
        }
        unsafe {
            r.set_len(n);
        }
        r
    }
}

mod vector_mul {
    use crate::f16;

    #[crate::multiversion(
        "v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn vector_mul(lhs: &[f16], rhs: &[f16]) -> Vec<f16> {
        assert_eq!(lhs.len(), rhs.len());
        let n = lhs.len();
        let mut r = Vec::<f16>::with_capacity(n);
        for i in 0..n {
            unsafe {
                r.as_mut_ptr().add(i).write(lhs[i] * rhs[i]);
            }
        }
        unsafe {
            r.set_len(n);
        }
        r
    }
}

mod vector_mul_scalar {
    use crate::{F16, f16};

    // The scale factor stays in f32: rounding it to f16 first would lose
    // precision, and would flush factors below 2^-24 (e.g. `1 / norm` for a
    // large-magnitude vector) to zero.
    #[crate::multiversion(
        "v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn vector_mul_scalar(lhs: &[f16], rhs: f32) -> Vec<f16> {
        let n = lhs.len();
        let mut r = Vec::<f16>::with_capacity(n);
        for i in 0..n {
            unsafe {
                r.as_mut_ptr()
                    .add(i)
                    .write(f16::_from_f32(lhs[i]._to_f32() * rhs));
            }
        }
        unsafe {
            r.set_len(n);
        }
        r
    }

    #[test]
    fn vector_mul_scalar_test() {
        let x = [1.0f32, -3.0, 1000.0, 60000.0, 0.0]
            .into_iter()
            .map(f16::_from_f32)
            .collect::<Vec<_>>();
        // `1 / 3` is 0.33325195 in f16; in f32, `3 * (1 / 3)` rounds back to 1.
        let r = vector_mul_scalar(&x, 1.0 / 3.0);
        let expected = x.iter().map(|x| f16::_from_f32(x._to_f32() / 3.0));
        assert!(std::iter::zip(&r, expected).all(|(a, b)| a.to_bits() == b.to_bits()));
        assert_eq!(r[1]._to_f32(), -1.0);
        // `1e-8` is zero in f16.
        let r = vector_mul_scalar(&x, 1e-8);
        assert_eq!(r[3]._to_f32(), f16::_from_f32(6e-4)._to_f32());
    }
}

mod vector_mul_scalar_inplace {
    use crate::{F16, f16};

    // See `vector_mul_scalar`.
    #[crate::multiversion(
        "v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn vector_mul_scalar_inplace(lhs: &mut [f16], rhs: f32) {
        let n = lhs.len();
        for i in 0..n {
            lhs[i] = f16::_from_f32(lhs[i]._to_f32() * rhs);
        }
    }

    #[test]
    fn vector_mul_scalar_inplace_test() {
        let mut x = [3.0f32, 255.0, 60000.0]
            .into_iter()
            .map(f16::_from_f32)
            .collect::<Vec<_>>();
        vector_mul_scalar_inplace(&mut x, 1.0 / 3.0);
        assert_eq!(x[0]._to_f32(), 1.0);
        assert_eq!(x[1]._to_f32(), 85.0);
        assert_eq!(x[2]._to_f32(), 20000.0);
        let mut x = vec![f16::_from_f32(60000.0)];
        vector_mul_scalar_inplace(&mut x, 1e-8);
        assert_eq!(x[0]._to_f32(), f16::_from_f32(6e-4)._to_f32());
    }
}

mod vector_abs_inplace {
    use crate::{F16, f16};

    #[crate::multiversion(
        "v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn vector_abs_inplace(this: &mut [f16]) {
        let n = this.len();
        for i in 0..n {
            this[i] = f16::_from_f32(this[i]._to_f32().abs());
        }
    }
}

mod vector_from_f32 {
    use crate::{F16, f16};

    #[crate::multiversion(
        "v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn vector_from_f32(this: &[f32]) -> Vec<f16> {
        let n = this.len();
        let mut r = Vec::<f16>::with_capacity(n);
        for i in 0..n {
            unsafe {
                r.as_mut_ptr().add(i).write(f16::_from_f32(this[i]));
            }
        }
        unsafe {
            r.set_len(n);
        }
        r
    }
}

mod vector_to_f32 {
    use crate::{F16, f16};

    #[crate::multiversion(
        "v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn vector_to_f32(this: &[f16]) -> Vec<f32> {
        let n = this.len();
        let mut r = Vec::<f32>::with_capacity(n);
        for i in 0..n {
            unsafe {
                r.as_mut_ptr().add(i).write(this[i]._to_f32());
            }
        }
        unsafe {
            r.set_len(n);
        }
        r
    }
}
