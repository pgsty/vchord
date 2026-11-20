// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

/// `len < N` elements from `p`, then `fill`.
#[inline(always)]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub unsafe fn partial<T: Copy, const N: usize>(p: *const T, len: usize, fill: T) -> [T; N] {
    unsafe { core::hint::assert_unchecked(len < N) };
    // Element by element for short arrays, where a copy loop becomes a `memcpy` call, and in a
    // loop for wider ones, where unrolling every element costs more.
    if N <= 8 {
        return core::array::from_fn(|i| {
            if i < len {
                unsafe { p.add(i).read() }
            } else {
                fill
            }
        });
    }
    let mut r = [fill; N];
    for (i, r) in r.iter_mut().enumerate().take(len) {
        *r = unsafe { p.add(i).read() };
    }
    r
}

#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
pub fn emulate_mm256_reduce_add_ps(mut x: core::arch::x86_64::__m256) -> f32 {
    use core::arch::x86_64::*;
    x = _mm256_add_ps(x, _mm256_permute2f128_ps(x, x, 1));
    x = _mm256_hadd_ps(x, x);
    x = _mm256_hadd_ps(x, x);
    _mm256_cvtss_f32(x)
}

#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
pub fn emulate_mm_reduce_add_ps(mut x: core::arch::x86_64::__m128) -> f32 {
    use core::arch::x86_64::*;
    x = _mm_hadd_ps(x, x);
    x = _mm_hadd_ps(x, x);
    _mm_cvtss_f32(x)
}

#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub fn emulate_mm256_reduce_add_epi32(mut x: core::arch::x86_64::__m256i) -> i32 {
    use core::arch::x86_64::*;
    x = _mm256_add_epi32(x, _mm256_permute2f128_si256(x, x, 1));
    x = _mm256_hadd_epi32(x, x);
    x = _mm256_hadd_epi32(x, x);
    _mm256_cvtsi256_si32(x)
}

#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
pub fn emulate_mm_reduce_add_epi32(mut x: core::arch::x86_64::__m128i) -> i32 {
    use core::arch::x86_64::*;
    x = _mm_hadd_epi32(x, x);
    x = _mm_hadd_epi32(x, x);
    _mm_cvtsi128_si32(x)
}

#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
pub fn emulate_mm256_reduce_min_ps(x: core::arch::x86_64::__m256) -> f32 {
    use core::arch::x86_64::*;
    let (lo, hi) = (_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
    emulate_mm_reduce_min_ps(_mm_min_ps(lo, hi))
}

#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse")]
pub fn emulate_mm_reduce_min_ps(x: core::arch::x86_64::__m128) -> f32 {
    use crate::aligned::Aligned16;
    use core::arch::x86_64::*;
    let min = x;
    let mut x = Aligned16([0.0f32; 4]);
    unsafe {
        _mm_store_ps(x.0.as_mut_ptr(), min);
    }
    f32::min(f32::min(x.0[0], x.0[1]), f32::min(x.0[2], x.0[3]))
}

#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
pub fn emulate_mm256_reduce_max_ps(x: core::arch::x86_64::__m256) -> f32 {
    use core::arch::x86_64::*;
    let (lo, hi) = (_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
    emulate_mm_reduce_max_ps(_mm_max_ps(lo, hi))
}

#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse")]
pub fn emulate_mm_reduce_max_ps(x: core::arch::x86_64::__m128) -> f32 {
    use crate::aligned::Aligned16;
    use core::arch::x86_64::*;
    let max = x;
    let mut x = Aligned16([0.0f32; 4]);
    unsafe {
        _mm_store_ps(x.0.as_mut_ptr(), max);
    }
    f32::max(f32::max(x.0[0], x.0[1]), f32::max(x.0[2], x.0[3]))
}

#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub fn emulate_mm256_reduce_add_epi64(mut x: core::arch::x86_64::__m256i) -> i64 {
    use core::arch::x86_64::*;
    x = _mm256_add_epi64(x, _mm256_permute2f128_si256(x, x, 1));
    _mm256_extract_epi64(x, 0) + _mm256_extract_epi64(x, 1)
}

#[inline]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub fn emulate_vreinterpretq_f16_u16(
    x: core::arch::aarch64::uint16x8_t,
) -> core::arch::aarch64::float16x8_t {
    unsafe { core::mem::transmute(x) }
}
