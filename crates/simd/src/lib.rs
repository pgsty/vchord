// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

#![allow(unsafe_code)]
#![cfg_attr(feature = "nightly_f16", feature(f16))]
// Newer toolchains stabilized the s390x target features this crate enables, so
// the gate is unused there (`unused_features`); older ones still require it.
#![cfg_attr(target_arch = "s390x", allow(unknown_lints, unused_features))]
#![cfg_attr(target_arch = "s390x", feature(s390x_target_feature))]
#![cfg_attr(target_arch = "s390x", feature(stdarch_s390x))]
#![cfg_attr(target_arch = "powerpc64", feature(stdarch_powerpc_feature_detection))]
#![cfg_attr(target_arch = "powerpc64", feature(powerpc_target_feature))]
#![cfg_attr(target_arch = "powerpc64", feature(stdarch_powerpc))]
#![cfg_attr(target_arch = "riscv64", feature(stdarch_riscv_feature_detection))]
#![cfg_attr(target_arch = "riscv64", feature(riscv_target_feature))]

mod aligned;
mod emulate;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
mod reduce;

#[cfg(not(feature = "internal"))]
mod floating_f16;

#[cfg(feature = "internal")]
pub mod floating_f16;

#[cfg(not(feature = "internal"))]
mod floating_f32;

#[cfg(feature = "internal")]
pub mod floating_f32;

#[cfg(test)]
mod test_util {
    use crate::Floating;
    use rand::RngExt;

    /// Lengths covering every tail and unroll boundary, then common widths.
    pub(crate) fn lengths() -> impl Iterator<Item = usize> {
        (0..=300).chain([
            511, 512, 513, 767, 768, 769, 783, 784, 785, 1023, 1024, 1025, 1535, 1536, 1537, 4095,
            4096, 4097,
        ])
    }

    pub(crate) fn random<S: Floating>(n: usize, lo: f32, hi: f32) -> Vec<S> {
        let mut rng = rand::rng();
        S::vector_from_f32(
            &(0..n)
                .map(|_| rng.random_range(lo..=hi))
                .collect::<Vec<_>>(),
        )
    }

    /// Non-zero integers in `[lo, hi]`; small ones sum exactly in any order, so a lost one shows.
    pub(crate) fn integers<S: Floating>(n: usize, lo: i32, hi: i32) -> Vec<S> {
        let mut rng = rand::rng();
        S::vector_from_f32(
            &(0..n)
                .map(|_| match rng.random_range(lo..hi) {
                    v if v >= 0 => v + 1,
                    v => v,
                } as f32)
                .collect::<Vec<_>>(),
        )
    }

    /// Checks every implementation against the f64 sum of `term` over non-zero integers.
    #[track_caller]
    pub(crate) fn check_unary<S: Floating>(
        implementations: Vec<(&str, fn(&[S]) -> f32)>,
        term: impl Fn(f64) -> f64,
    ) {
        for n in lengths() {
            for hi in [8, 65504] {
                let x = integers::<S>(n, -hi, hi);
                let terms = S::vector_to_f32(&x).into_iter().map(|x| term(x as f64));
                for (name, f) in &implementations {
                    assert_sum(name, n, f(&x), terms.clone());
                }
            }
        }
    }

    /// Checks every implementation against the f64 sum of `term` over non-zero integers.
    #[track_caller]
    pub(crate) fn check_binary<S: Floating>(
        implementations: Vec<(&str, fn(&[S], &[S]) -> f32)>,
        term: impl Fn(f64, f64) -> f64,
    ) {
        for n in lengths() {
            for hi in [8, 65504] {
                let (x, y) = (integers::<S>(n, -hi, hi), integers::<S>(n, -hi, hi));
                let terms = std::iter::zip(S::vector_to_f32(&x), S::vector_to_f32(&y))
                    .map(|(x, y)| term(x as f64, y as f64));
                for (name, f) in &implementations {
                    assert_sum(name, n, f(&x, &y), terms.clone());
                }
            }
        }
    }

    /// Asserts `got` is the sum of the integer `terms`: exactly below 2^24, where every partial sum
    /// is exact, else within the worst-case f32 summation error (f16 accumulation exceeds it).
    #[track_caller]
    fn assert_sum(name: &str, n: usize, got: f32, terms: impl Iterator<Item = f64> + Clone) {
        let (sum, sum_of_abs) = (terms.clone().sum::<f64>(), terms.map(f64::abs).sum::<f64>());
        let tolerance = if sum_of_abs < 2f64.powi(24) {
            0.0
        } else {
            (n + 1) as f64 * 2f64.powi(-24) * sum_of_abs
        };
        assert!(
            ((got as f64) - sum).abs() <= tolerance,
            "{name}: n = {n}, got = {got}, expected = {sum}, tolerance = {tolerance}"
        );
    }
}

pub mod bit;
pub mod byte;
pub mod column;
pub mod fast_scan;
pub mod fht;
pub mod halfbyte;
pub mod quantize;
pub mod rotate;

#[cfg(not(feature = "nightly_f16"))]
pub use half::f16;

#[cfg(feature = "nightly_f16")]
pub use f16;

pub trait F16: Sized {
    const _ZERO: Self;

    fn _from_f32(x: f32) -> Self;

    fn _to_f32(self) -> f32;
}

#[cfg(not(feature = "nightly_f16"))]
impl F16 for f16 {
    const _ZERO: Self = f16::ZERO;

    #[inline(always)]
    fn _from_f32(x: f32) -> Self {
        #[cfg(not(miri))]
        {
            f16::from_f32(x)
        }
        #[cfg(miri)]
        {
            f16::from_f32_const(x)
        }
    }

    #[inline(always)]
    fn _to_f32(self) -> f32 {
        #[cfg(not(miri))]
        {
            self.to_f32()
        }
        #[cfg(miri)]
        {
            self.to_f32_const()
        }
    }
}

#[cfg(feature = "nightly_f16")]
impl F16 for f16 {
    const _ZERO: Self = 0.0;

    #[inline(always)]
    fn _from_f32(x: f32) -> Self {
        x as _
    }

    #[inline(always)]
    fn _to_f32(self) -> f32 {
        self as _
    }
}

pub trait Floating:
    Copy + Send + Sync + std::fmt::Debug + Default + 'static + PartialEq + PartialOrd
{
    fn reduce_sum_of_abs_x(this: &[Self]) -> f32;
    fn reduce_sum_of_x2(this: &[Self]) -> f32;
    fn reduce_min_max_of_x(this: &[Self]) -> (f32, f32);
    fn reduce_sum_of_xy(lhs: &[Self], rhs: &[Self]) -> f32;
    fn reduce_sum_of_d2(lhs: &[Self], rhs: &[Self]) -> f32;
    /// Distance for ANN routing and graph construction; not for final reranking,
    /// SQL predicates or any bound that the reranker trusts. Implementations
    /// may round intermediate differences.
    #[inline(always)]
    fn reduce_sum_of_d2_approx(lhs: &[Self], rhs: &[Self]) -> f32 {
        Self::reduce_sum_of_d2(lhs, rhs)
    }
    /// Whether `reduce_sum_of_d2_approx` returns exactly `reduce_sum_of_d2` on
    /// this CPU, so callers can skip recomputing an exact distance.
    #[inline(always)]
    fn reduce_sum_of_d2_approx_is_exact() -> bool {
        true
    }

    fn vector_from_f32(this: &[f32]) -> Vec<Self>;
    fn vector_to_f32(this: &[Self]) -> Vec<f32>;
    fn vector_to_f32_borrowed(this: &[Self]) -> impl AsRef<[f32]>;
    fn vector_add_inplace(lhs: &mut [Self], rhs: &[Self]);
    fn vector_sub(lhs: &[Self], rhs: &[Self]) -> Vec<Self>;
    fn vector_mul_scalar(lhs: &[Self], rhs: f32) -> Vec<Self>;
    fn vector_mul_scalar_inplace(lhs: &mut [Self], rhs: f32);
    fn vector_abs_inplace(this: &mut [Self]);
}

#[doc(hidden)]
pub mod internal {
    #[cfg(target_arch = "x86_64")]
    simd_macros::define_is_cpu_detected!("x86_64");

    #[cfg(target_arch = "aarch64")]
    simd_macros::define_is_cpu_detected!("aarch64");

    #[cfg(target_arch = "s390x")]
    simd_macros::define_is_cpu_detected!("s390x");

    #[cfg(target_arch = "powerpc64")]
    simd_macros::define_is_cpu_detected!("powerpc64");

    #[cfg(target_arch = "riscv64")]
    simd_macros::define_is_cpu_detected!("riscv64");

    #[cfg(target_arch = "x86_64")]
    #[allow(unused_imports)]
    pub use is_x86_64_cpu_detected;

    #[cfg(target_arch = "aarch64")]
    #[allow(unused_imports)]
    pub use is_aarch64_cpu_detected;

    #[cfg(target_arch = "s390x")]
    #[allow(unused_imports)]
    pub use is_s390x_cpu_detected;

    #[cfg(target_arch = "powerpc64")]
    #[allow(unused_imports)]
    pub use is_powerpc64_cpu_detected;

    #[cfg(target_arch = "riscv64")]
    #[allow(unused_imports)]
    pub use is_riscv64_cpu_detected;

    #[cfg(target_arch = "x86_64")]
    pub fn is_v4_detected() -> bool {
        std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512cd")
            && std::arch::is_x86_feature_detected!("avx512dq")
            && std::arch::is_x86_feature_detected!("avx512vl")
            && std::arch::is_x86_feature_detected!("bmi1")
            && std::arch::is_x86_feature_detected!("bmi2")
            && std::arch::is_x86_feature_detected!("lzcnt")
            && std::arch::is_x86_feature_detected!("movbe")
            && std::arch::is_x86_feature_detected!("popcnt")
    }

    #[cfg(target_arch = "x86_64")]
    pub fn is_v3_detected() -> bool {
        std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("f16c")
            && std::arch::is_x86_feature_detected!("fma")
            && std::arch::is_x86_feature_detected!("bmi1")
            && std::arch::is_x86_feature_detected!("bmi2")
            && std::arch::is_x86_feature_detected!("lzcnt")
            && std::arch::is_x86_feature_detected!("movbe")
            && std::arch::is_x86_feature_detected!("popcnt")
    }

    #[cfg(target_arch = "x86_64")]
    pub fn is_v2_detected() -> bool {
        std::arch::is_x86_feature_detected!("sse4.2")
            && std::arch::is_x86_feature_detected!("popcnt")
    }

    #[cfg(target_arch = "aarch64")]
    #[cfg_attr(target_endian = "big", expect(dead_code))]
    pub fn is_a3_512_detected() -> bool {
        #[target_feature(enable = "sve")]
        fn is_512_detected() -> bool {
            let vl: u64;
            unsafe {
                core::arch::asm!(
                    "rdvl {0}, #8",
                    out(reg) vl
                );
            }
            vl >= 512
        }
        std::arch::is_aarch64_feature_detected!("sve") && unsafe { is_512_detected() }
    }

    #[cfg(target_arch = "aarch64")]
    #[cfg_attr(target_endian = "big", expect(dead_code))]
    pub fn is_a3_256_detected() -> bool {
        #[target_feature(enable = "sve")]
        fn is_256_detected() -> bool {
            let vl: u64;
            unsafe {
                core::arch::asm!(
                    "rdvl {0}, #8",
                    out(reg) vl
                );
            }
            vl >= 256
        }
        std::arch::is_aarch64_feature_detected!("sve") && unsafe { is_256_detected() }
    }

    #[cfg(target_arch = "aarch64")]
    pub fn is_a3_128_detected() -> bool {
        std::arch::is_aarch64_feature_detected!("sve")
    }

    #[cfg(target_arch = "aarch64")]
    pub fn is_a2_detected() -> bool {
        std::arch::is_aarch64_feature_detected!("neon")
    }

    #[cfg(target_arch = "s390x")]
    pub fn is_z17_detected() -> bool {
        std::arch::is_s390x_feature_detected!("vector")
            && std::arch::is_s390x_feature_detected!("vector-enhancements-1")
            && std::arch::is_s390x_feature_detected!("vector-enhancements-2")
            && std::arch::is_s390x_feature_detected!("vector-enhancements-3")
            && std::arch::is_s390x_feature_detected!("vector-packed-decimal")
            && std::arch::is_s390x_feature_detected!("vector-packed-decimal-enhancement")
            && std::arch::is_s390x_feature_detected!("vector-packed-decimal-enhancement-2")
            && std::arch::is_s390x_feature_detected!("vector-packed-decimal-enhancement-3")
            && std::arch::is_s390x_feature_detected!("nnp-assist")
            && std::arch::is_s390x_feature_detected!("miscellaneous-extensions-2")
            && std::arch::is_s390x_feature_detected!("miscellaneous-extensions-3")
            && std::arch::is_s390x_feature_detected!("miscellaneous-extensions-4")
    }

    #[cfg(target_arch = "s390x")]
    pub fn is_z16_detected() -> bool {
        std::arch::is_s390x_feature_detected!("vector")
            && std::arch::is_s390x_feature_detected!("vector-enhancements-1")
            && std::arch::is_s390x_feature_detected!("vector-enhancements-2")
            && std::arch::is_s390x_feature_detected!("vector-packed-decimal")
            && std::arch::is_s390x_feature_detected!("vector-packed-decimal-enhancement")
            && std::arch::is_s390x_feature_detected!("vector-packed-decimal-enhancement-2")
            && std::arch::is_s390x_feature_detected!("nnp-assist")
            && std::arch::is_s390x_feature_detected!("miscellaneous-extensions-2")
            && std::arch::is_s390x_feature_detected!("miscellaneous-extensions-3")
    }

    #[cfg(target_arch = "s390x")]
    pub fn is_z15_detected() -> bool {
        std::arch::is_s390x_feature_detected!("vector")
            && std::arch::is_s390x_feature_detected!("vector-enhancements-1")
            && std::arch::is_s390x_feature_detected!("vector-enhancements-2")
            && std::arch::is_s390x_feature_detected!("vector-packed-decimal")
            && std::arch::is_s390x_feature_detected!("vector-packed-decimal-enhancement")
            && std::arch::is_s390x_feature_detected!("miscellaneous-extensions-2")
            && std::arch::is_s390x_feature_detected!("miscellaneous-extensions-3")
    }

    #[cfg(target_arch = "s390x")]
    pub fn is_z14_detected() -> bool {
        std::arch::is_s390x_feature_detected!("vector")
            && std::arch::is_s390x_feature_detected!("vector-enhancements-1")
            && std::arch::is_s390x_feature_detected!("vector-packed-decimal")
            && std::arch::is_s390x_feature_detected!("miscellaneous-extensions-2")
    }

    #[cfg(target_arch = "s390x")]
    pub fn is_z13_detected() -> bool {
        std::arch::is_s390x_feature_detected!("vector")
    }

    #[cfg(target_arch = "powerpc64")]
    pub fn is_p9_detected() -> bool {
        std::arch::is_powerpc64_feature_detected!("altivec")
            && std::arch::is_powerpc64_feature_detected!("vsx")
            && std::arch::is_powerpc64_feature_detected!("power8-altivec")
            && std::arch::is_powerpc64_feature_detected!("power8-crypto")
            && std::arch::is_powerpc64_feature_detected!("power8-vector")
            && std::arch::is_powerpc64_feature_detected!("power9-altivec")
            && std::arch::is_powerpc64_feature_detected!("power9-vector")
    }

    #[cfg(target_arch = "powerpc64")]
    pub fn is_p8_detected() -> bool {
        std::arch::is_powerpc64_feature_detected!("altivec")
            && std::arch::is_powerpc64_feature_detected!("vsx")
            && std::arch::is_powerpc64_feature_detected!("power8-altivec")
            && std::arch::is_powerpc64_feature_detected!("power8-crypto")
            && std::arch::is_powerpc64_feature_detected!("power8-vector")
    }

    #[cfg(target_arch = "powerpc64")]
    pub fn is_p7_detected() -> bool {
        std::arch::is_powerpc64_feature_detected!("altivec")
            && std::arch::is_powerpc64_feature_detected!("vsx")
    }

    #[cfg(target_arch = "riscv64")]
    pub fn is_r1_detected() -> bool {
        std::arch::is_riscv_feature_detected!("v")
    }
}

pub use simd_macros::{multiversion, target_cpu};

#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
pub use std::arch::is_x86_feature_detected as is_feature_detected;

#[cfg(target_arch = "aarch64")]
#[allow(unused_imports)]
pub use std::arch::is_aarch64_feature_detected as is_feature_detected;

#[cfg(target_arch = "s390x")]
#[allow(unused_imports)]
pub use std::arch::is_s390x_feature_detected as is_feature_detected;

#[cfg(target_arch = "powerpc64")]
#[allow(unused_imports)]
pub use std::arch::is_powerpc64_feature_detected as is_feature_detected;

#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
pub use internal::is_x86_64_cpu_detected as is_cpu_detected;

#[cfg(target_arch = "aarch64")]
#[allow(unused_imports)]
pub use internal::is_aarch64_cpu_detected as is_cpu_detected;

#[cfg(target_arch = "s390x")]
#[allow(unused_imports)]
pub use internal::is_s390x_cpu_detected as is_cpu_detected;

#[cfg(target_arch = "powerpc64")]
#[allow(unused_imports)]
pub use internal::is_powerpc64_cpu_detected as is_cpu_detected;

#[cfg(target_arch = "riscv64")]
#[allow(unused_imports)]
pub use internal::is_riscv64_cpu_detected as is_cpu_detected;
