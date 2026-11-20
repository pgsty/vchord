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
    fn reduce_sum_of_abs_x(this: &[f16]) -> f32 {
        this.iter().map(|x| x._to_f32().abs()).sum()
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
    fn reduce_sum_of_d2_approx(lhs: &[f16], rhs: &[f16]) -> f32 {
        if reduce_sum_of_d2_approx::is_fast_path_detected() {
            return reduce_sum_of_d2_approx::reduce_sum_of_d2_approx(lhs, rhs);
        }
        // No additional runtime dispatch on CPUs without a fast kernel.
        Self::reduce_sum_of_d2(lhs, rhs)
    }

    #[inline(always)]
    fn reduce_sum_of_d2_approx_is_exact() -> bool {
        !reduce_sum_of_d2_approx::is_fast_path_detected()
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
    fn vector_add_inplace(lhs: &mut [Self], rhs: &[Self]) {
        vector_add_inplace::vector_add_inplace(lhs, rhs)
    }

    #[inline(always)]
    fn vector_sub(lhs: &[Self], rhs: &[Self]) -> Vec<Self> {
        vector_sub::vector_sub(lhs, rhs)
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

// Every kernel accumulates in f32: binary16 overflows at 65504. FHM's `FMLAL`/`FMLSL` widen exactly.
// No AVX-512 x², xy or d² kernels: their 512-bit conversions are slower on some CPUs.

#[cfg_attr(feature = "internal", simd_macros::public)]
mod reduce_sum_of_x2 {
    use crate::{F16, f16};

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v3")]
    fn reduce_sum_of_x2_v3(this: &[f16]) -> f32 {
        use core::arch::x86_64::*;
        crate::reduce::v3::sum::<8, 1, _>([this], |s, [x]| _mm256_fmadd_ps(x, x, s))
    }

    #[inline]
    #[cfg(target_arch = "aarch64")]
    #[crate::target_cpu(enable = "a2")]
    #[target_feature(enable = "fhm")]
    fn reduce_sum_of_x2_a2_fhm(this: &[f16]) -> f32 {
        use core::arch::aarch64::*;
        crate::reduce::a2::sum_f16::<8, 1>([this], |s, [x]| {
            [vfmlalq_low_f16(s[0], x, x), vfmlalq_high_f16(s[1], x, x)]
        })
    }

    #[inline]
    #[cfg(target_arch = "aarch64")]
    #[crate::target_cpu(enable = "a2")]
    fn reduce_sum_of_x2_a2(this: &[f16]) -> f32 {
        use core::arch::aarch64::*;
        crate::reduce::a2::sum::<4, 1, _>([this], |s, [x]| {
            [vfmaq_f32(s[0], x[0], x[0]), vfmaq_f32(s[1], x[1], x[1])]
        })
    }

    #[crate::multiversion(
        @"v3", "v2", @"a2:fhm", @"a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn reduce_sum_of_x2(this: &[f16]) -> f32 {
        let n = this.len();
        let mut x2 = 0.0f32;
        for i in 0..n {
            x2 += this[i]._to_f32() * this[i]._to_f32();
        }
        x2
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_sum_of_x2_test() {
        crate::test_util::check_unary(implementations(), |x| x * x);
    }
}

mod reduce_min_max_of_x {
    use crate::{F16, f16};

    #[crate::multiversion(
        "v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
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
}

#[cfg_attr(feature = "internal", simd_macros::public)]
mod reduce_sum_of_xy {
    use crate::{F16, f16};

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v3")]
    fn reduce_sum_of_xy_v3(lhs: &[f16], rhs: &[f16]) -> f32 {
        use core::arch::x86_64::*;
        crate::reduce::v3::sum::<4, 2, _>([lhs, rhs], |s, [x, y]| _mm256_fmadd_ps(x, y, s))
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

    #[inline]
    #[cfg(target_arch = "aarch64")]
    #[crate::target_cpu(enable = "a2")]
    #[target_feature(enable = "fhm")]
    fn reduce_sum_of_xy_a2_fhm(lhs: &[f16], rhs: &[f16]) -> f32 {
        use core::arch::aarch64::*;
        crate::reduce::a2::sum_f16::<8, 2>([lhs, rhs], |s, [x, y]| {
            [vfmlalq_low_f16(s[0], x, y), vfmlalq_high_f16(s[1], x, y)]
        })
    }

    #[inline]
    #[cfg(target_arch = "aarch64")]
    #[crate::target_cpu(enable = "a2")]
    fn reduce_sum_of_xy_a2(lhs: &[f16], rhs: &[f16]) -> f32 {
        use core::arch::aarch64::*;
        crate::reduce::a2::sum::<4, 2, _>([lhs, rhs], |s, [x, y]| {
            [vfmaq_f32(s[0], x[0], y[0]), vfmaq_f32(s[1], x[1], y[1])]
        })
    }

    #[crate::multiversion(@"v3", @"a2:fhm", @"a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1")]
    pub fn reduce_sum_of_xy(lhs: &[f16], rhs: &[f16]) -> f32 {
        assert!(lhs.len() == rhs.len());
        let n = lhs.len();
        let mut sum = 0.0f32;
        for i in 0..n {
            sum += lhs[i]._to_f32() * rhs[i]._to_f32();
        }
        sum
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_sum_of_xy_test() {
        crate::test_util::check_binary(implementations(), |x, y| x * y);
    }
}

#[cfg_attr(feature = "internal", simd_macros::public)]
mod reduce_sum_of_d2 {
    use crate::{F16, f16};

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v3")]
    fn reduce_sum_of_d2_v3(lhs: &[f16], rhs: &[f16]) -> f32 {
        use core::arch::x86_64::*;
        crate::reduce::v3::sum::<4, 2, _>([lhs, rhs], |s, [x, y]| {
            let d = _mm256_sub_ps(x, y);
            _mm256_fmadd_ps(d, d, s)
        })
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

    #[inline]
    #[cfg(target_arch = "aarch64")]
    #[crate::target_cpu(enable = "a2")]
    #[target_feature(enable = "fhm")]
    fn reduce_sum_of_d2_a2_fhm(lhs: &[f16], rhs: &[f16]) -> f32 {
        use core::arch::aarch64::*;
        // `x - y` is computed in f32 (`FMLSL` with `y * 1.0`, which is exact),
        // never in f16: an f16 difference would be rounded, and it overflows
        // once `|x - y| > 65504`.
        let ones = vreinterpretq_f16_u16(vdupq_n_u16(0x3c00));
        crate::reduce::a2::sum_f16::<8, 2>([lhs, rhs], |s, [x, y]| {
            let d_0 = vfmlslq_low_f16(vcvt_f32_f16(vget_low_f16(x)), y, ones);
            let d_1 = vfmlslq_high_f16(vcvt_high_f32_f16(x), y, ones);
            [vfmaq_f32(s[0], d_0, d_0), vfmaq_f32(s[1], d_1, d_1)]
        })
    }

    #[inline]
    #[cfg(target_arch = "aarch64")]
    #[crate::target_cpu(enable = "a2")]
    fn reduce_sum_of_d2_a2(lhs: &[f16], rhs: &[f16]) -> f32 {
        use core::arch::aarch64::*;
        crate::reduce::a2::sum::<4, 2, _>([lhs, rhs], |s, [x, y]| {
            let d = [vsubq_f32(x[0], y[0]), vsubq_f32(x[1], y[1])];
            [vfmaq_f32(s[0], d[0], d[0]), vfmaq_f32(s[1], d[1], d[1])]
        })
    }

    #[crate::multiversion(@"v3", @"a2:fhm", @"a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1")]
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

    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_sum_of_d2_test() {
        crate::test_util::check_binary(implementations(), |x, y| (x - y).powi(2));
    }
}

#[cfg_attr(feature = "internal", simd_macros::public)]
mod reduce_sum_of_d2_approx {
    use crate::{Floating, f16};

    /// Whether `reduce_sum_of_d2_approx` has a fast kernel; CPUs with this feature have the rest.
    #[inline]
    pub fn is_fast_path_detected() -> bool {
        #[cfg(target_arch = "aarch64")]
        return crate::is_feature_detected!("fp16");
        #[cfg(target_arch = "x86_64")]
        return crate::is_feature_detected!("avx512fp16");
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        false
    }

    /// Retries in f32 when the difference of two finite halves overflows f16.
    #[inline(always)]
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    fn finite_or_exact(result: f32, lhs: &[f16], rhs: &[f16]) -> f32 {
        if result.is_finite() {
            result
        } else {
            f16::reduce_sum_of_d2(lhs, rhs)
        }
    }

    // Only the difference is rounded (to f16), so each term is within a relative (1 + 2^-11)^2 - 1 of exact.

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    #[target_feature(enable = "avx512fp16")]
    fn reduce_sum_of_d2_approx_v4_avx512fp16(lhs: &[f16], rhs: &[f16]) -> f32 {
        use core::arch::x86_64::*;
        let load = |p: *const f16, len: usize| unsafe {
            if len == 32 {
                _mm512_castsi512_ph(_mm512_loadu_epi16(p.cast()))
            } else {
                let mask = _bzhi_u32(0xffffffff, len as u32);
                _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, p.cast()))
            }
        };
        let add =
            |x: [__m512; 2], y: [__m512; 2]| [_mm512_add_ps(x[0], y[0]), _mm512_add_ps(x[1], y[1])];
        let sum = crate::reduce::fold::<_, _, 2, 32, 2, 2>(
            [lhs, rhs],
            [_mm512_setzero_ps(); 2],
            add,
            |s, [x, y], len| {
                let d = _mm512_sub_ph(load(x, len), load(y, len));
                let d_0 = _mm512_cvtxph_ps(_mm512_castph512_ph256(d));
                let d_1 = _mm512_castph_si512(d);
                let d_1 =
                    _mm512_cvtxph_ps(_mm256_castsi256_ph(_mm512_extracti64x4_epi64::<1>(d_1)));
                [
                    _mm512_fmadd_ps(d_0, d_0, s[0]),
                    _mm512_fmadd_ps(d_1, d_1, s[1]),
                ]
            },
        );
        finite_or_exact(
            _mm512_reduce_add_ps(_mm512_add_ps(sum[0], sum[1])),
            lhs,
            rhs,
        )
    }

    #[inline]
    #[cfg(target_arch = "aarch64")]
    #[crate::target_cpu(enable = "a2")]
    #[target_feature(enable = "fp16,fhm")]
    fn reduce_sum_of_d2_approx_a2_fp16_fhm(lhs: &[f16], rhs: &[f16]) -> f32 {
        use core::arch::aarch64::*;
        let sum = crate::reduce::a2::sum_f16::<8, 2>([lhs, rhs], |s, [x, y]| {
            let d = vsubq_f16(x, y);
            [vfmlalq_low_f16(s[0], d, d), vfmlalq_high_f16(s[1], d, d)]
        });
        finite_or_exact(sum, lhs, rhs)
    }

    #[inline]
    #[cfg(target_arch = "aarch64")]
    #[crate::target_cpu(enable = "a2")]
    #[target_feature(enable = "fp16")]
    fn reduce_sum_of_d2_approx_a2_fp16(lhs: &[f16], rhs: &[f16]) -> f32 {
        use core::arch::aarch64::*;
        let sum = crate::reduce::a2::sum_f16::<8, 2>([lhs, rhs], |s, [x, y]| {
            let d = vsubq_f16(x, y);
            let (d_0, d_1) = (vcvt_f32_f16(vget_low_f16(d)), vcvt_high_f32_f16(d));
            [vfmaq_f32(s[0], d_0, d_0), vfmaq_f32(s[1], d_1, d_1)]
        });
        finite_or_exact(sum, lhs, rhs)
    }

    #[crate::multiversion(@"v4:avx512fp16", @"a2:fp16:fhm", @"a2:fp16")]
    pub fn reduce_sum_of_d2_approx(lhs: &[f16], rhs: &[f16]) -> f32 {
        f16::reduce_sum_of_d2(lhs, rhs)
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn bounded_error_and_overflow_fallback() {
        use crate::test_util::{lengths, random};
        let gamma = |k: f64| k * 2f64.powi(-24) / (1.0 - k * 2f64.powi(-24));
        for n in lengths() {
            let bound = (1.0 + 2f64.powi(-11)).powi(2) * (1.0 + gamma(n as f64 + 2.0)) - 1.0;
            let adjacent = (1..=n as u16)
                .map(|a| (f16::from_bits(a), f16::from_bits(a + 1)))
                .unzip();
            for (x, y) in [
                (random(n, -1.0, 1.0), random(n, -1.0, 1.0)),
                (random(n, -60000.0, 60000.0), random(n, -1e-3, 1e-3)),
                adjacent,
                (vec![f16::MAX; n], vec![f16::MIN; n]),
            ] {
                let exact: f64 = std::iter::zip(&x, &y)
                    .map(|(a, b)| (a.to_f64() - b.to_f64()).powi(2))
                    .sum();
                for (name, f) in implementations() {
                    let error = (f(&x, &y) as f64 - exact).abs();
                    assert!(error <= bound * exact, "{name}: n = {n}");
                }
            }
        }
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
