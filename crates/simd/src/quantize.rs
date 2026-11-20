// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

mod sub_mul_round {
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    fn sub_mul_round_v4(this: &[f32], b: f32, k: f32) -> Vec<u8> {
        let mut r = Vec::<u8>::with_capacity(this.len());
        use core::arch::x86_64::*;
        let lb = _mm512_set1_ps(b);
        let lk = _mm512_set1_ps(k);
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut p = r.as_mut_ptr();
        while n >= 16 {
            let x = unsafe { _mm512_loadu_ps(a) };
            let v = _mm512_mul_ps(_mm512_sub_ps(x, lb), lk);
            let v = _mm512_cvtps_epi32(v);
            let vfl = _mm512_cvtepi32_epi8(v);
            unsafe { _mm_storeu_si128(p.cast(), vfl) };
            (n, a, p) = unsafe { (n - 16, a.add(16), p.add(16)) };
        }
        if n > 0 {
            let mask = _bzhi_u32(0xffff, n as u32) as u16;
            let x = unsafe { _mm512_maskz_loadu_ps(mask, a) };
            let v = _mm512_mul_ps(_mm512_sub_ps(x, lb), lk);
            let v = _mm512_cvtps_epi32(v);
            let vfl = _mm512_cvtepi32_epi8(v);
            unsafe { _mm_mask_storeu_epi8(p.cast(), mask, vfl) };
        }
        unsafe {
            r.set_len(this.len());
        }
        r
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v3")]
    fn sub_mul_round_v3(this: &[f32], b: f32, k: f32) -> Vec<u8> {
        let mut r = Vec::<u8>::with_capacity(this.len().next_multiple_of(8));
        use crate::reduce::Lanes;
        use core::arch::x86_64::*;
        let cons = _mm256_setr_epi8(
            0, 4, 8, 12, -1, -1, -1, -1, // 0..8
            -1, -1, -1, -1, -1, -1, -1, -1, // 8..15
            0, 4, 8, 12, -1, -1, -1, -1, // 16..24
            -1, -1, -1, -1, -1, -1, -1, -1, // 24..32
        );
        let lb = _mm256_set1_ps(b);
        let lk = _mm256_set1_ps(k);
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut p = r.as_mut_ptr();
        let step = |a: *const f32, p: *mut u8, len: usize| {
            let x = unsafe { f32::load_m256(a, len, 0.0) };
            let v = _mm256_mul_ps(_mm256_sub_ps(x, lb), lk);
            let v = _mm256_cvtps_epi32(_mm256_round_ps(v, 0x00));
            let vs = _mm256_shuffle_epi8(v, cons);
            let vlo = _mm256_extract_epi32::<0>(vs) as u32;
            let vhi = _mm256_extract_epi32::<4>(vs) as u32;
            let vfl = vlo as u64 | ((vhi as u64) << 32);
            unsafe { p.cast::<u64>().write_unaligned(vfl) };
        };
        while n >= 8 {
            step(a, p, 8);
            (n, a, p) = unsafe { (n - 8, a.add(8), p.add(8)) };
        }
        if n > 0 {
            step(a, p, n);
        }
        unsafe {
            r.set_len(this.len());
        }
        r
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v2")]
    fn sub_mul_round_v2(this: &[f32], b: f32, k: f32) -> Vec<u8> {
        let mut r = Vec::<u8>::with_capacity(this.len().next_multiple_of(4));
        use crate::reduce::v2;
        use core::arch::x86_64::*;
        let cons = _mm_setr_epi8(
            0, 4, 8, 12, -1, -1, -1, -1, // 0..8
            -1, -1, -1, -1, -1, -1, -1, -1, // 8..15
        );
        let lb = _mm_set1_ps(b);
        let lk = _mm_set1_ps(k);
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut p = r.as_mut_ptr();
        let step = |a: *const f32, p: *mut u8, len: usize| {
            let x = unsafe { v2::load(a, len, 0.0) };
            let v = _mm_mul_ps(_mm_sub_ps(x, lb), lk);
            let v = _mm_cvtps_epi32(_mm_round_ps(v, 0x00));
            let vs = _mm_shuffle_epi8(v, cons);
            let vfl = _mm_extract_epi32::<0>(vs) as u32;
            unsafe { p.cast::<u32>().write_unaligned(vfl) };
        };
        while n >= 4 {
            step(a, p, 4);
            (n, a, p) = unsafe { (n - 4, a.add(4), p.add(4)) };
        }
        if n > 0 {
            step(a, p, n);
        }
        unsafe {
            r.set_len(this.len());
        }
        r
    }

    #[inline]
    #[cfg(target_arch = "aarch64")]
    #[crate::target_cpu(enable = "a2")]
    fn sub_mul_round_a2(this: &[f32], b: f32, k: f32) -> Vec<u8> {
        let mut r = Vec::<u8>::with_capacity(this.len().next_multiple_of(4));
        use crate::emulate::partial;
        use core::arch::aarch64::*;
        #[cfg(target_endian = "little")]
        const CONS: [u8; 16] = [
            0, 4, 8, 12, 0xff, 0xff, 0xff, 0xff, // 0..8
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, // 8..15
        ];
        #[cfg(target_endian = "big")]
        const CONS: [u8; 16] = [
            12, 8, 4, 0, 0xff, 0xff, 0xff, 0xff, // 0..8
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, // 8..15
        ];
        let cons = unsafe { vld1q_u8(CONS.as_ptr()) };
        let lb = vdupq_n_f32(b);
        let lk = vdupq_n_f32(k);
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut p = r.as_mut_ptr();
        let step = |a: *const f32, p: *mut u8, len: usize| {
            let x = if len == 4 {
                unsafe { vld1q_f32(a) }
            } else {
                unsafe { vld1q_f32(partial::<_, 4>(a, len, 0.0).as_ptr()) }
            };
            let v = vmulq_f32(vsubq_f32(x, lb), lk);
            let v = vcvtnq_u32_f32(v);
            let vs = vqtbl1q_u8(vreinterpretq_u8_u32(v), cons);
            let vfl = vgetq_lane_u32::<0>(vreinterpretq_u32_u8(vs));
            unsafe { p.cast::<u32>().write_unaligned(vfl) };
        };
        while n >= 4 {
            step(a, p, 4);
            (n, a, p) = unsafe { (n - 4, a.add(4), p.add(4)) };
        }
        if n > 0 {
            step(a, p, n);
        }
        unsafe {
            r.set_len(this.len());
        }
        r
    }

    #[crate::multiversion(@"v4", @"v3", @"v2", @"a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1")]
    pub fn sub_mul_round(this: &[f32], b: f32, k: f32) -> Vec<u8> {
        let n = this.len();
        let mut r = Vec::<u8>::with_capacity(n);
        for i in 0..n {
            let x = this[i];
            let v = ((x - b) * k).round_ties_even() as u8;
            unsafe {
                r.as_mut_ptr().add(i).write(v);
            }
        }
        unsafe {
            r.set_len(n);
        }
        r
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn sub_mul_round_test() {
        let x = crate::test_util::random::<f32>(4010, -1.0, 1.0);
        for n in 3990..4010 {
            let expected = fallback(&x[..n], -1.0, 127.0);
            for (name, f) in implementations() {
                assert_eq!(f(&x[..n], -1.0, 127.0), expected, "{name}: n = {n}");
            }
        }
    }
}

/// Quantizes `lut` to codes in `0..=n` with `x ~= b + code * k`; returns `(k, b, codes)`.
/// Shifts by `b` before scaling: scaling first cancels and can exceed `n`; this way no clamp is needed.
#[inline(always)]
pub fn quantize(lut: &[f32], n: f32) -> (f32, f32, Vec<u8>) {
    use crate::Floating;
    let (min, max) = f32::reduce_min_max_of_x(lut);
    let span = max - min;
    let scale = n / span;
    if !(scale > 0.0 && scale.is_finite()) {
        return (0.0, min, vec![0; lut.len()]);
    }
    (span / n, min, sub_mul_round::sub_mul_round(lut, min, scale))
}

#[cfg(test)]
mod quantize_tests {
    use super::quantize;

    fn check(lut: &[f32], n: f32) {
        let (k, b, codes) = quantize(lut, n);
        assert!(k.is_finite() && k >= 0.0, "k = {k}");
        assert_eq!(codes.len(), lut.len());
        for (&x, &c) in lut.iter().zip(codes.iter()) {
            assert!(f32::from(c) <= n, "code {c} > {n}");
            let exact = (f64::from(x) - f64::from(b)) / f64::from(k);
            assert!(
                (f64::from(c) - exact).abs() <= 0.5 + 1e-4,
                "x = {x}, code {c}, exact {exact}"
            );
            let y = f64::from(b) + f64::from(c) * f64::from(k);
            // `x - b` rounds at the larger of the two magnitudes
            let rounding = (f64::from(x).abs() + f64::from(b).abs()) * 1e-6;
            assert!(
                (y - f64::from(x)).abs() <= f64::from(k) * 0.5 + rounding,
                "x = {x}, reconstructed {y}, k = {k}, b = {b}"
            );
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn random_offsets_and_spans() {
        use rand::RngExt;
        let mut rng = rand::rng();
        for len in [1, 2, 15, 16, 17, 63, 64, 65, 768, 3072, 3073] {
            for (offset, width) in [
                (0.0, 1.0),
                (1000.0, 1e-3),
                (-1e6, 1.0),
                (1e30, 1e25),
                (0.0, 1e-30),
                (0.0, 1e38),
            ] {
                let lut = (0..len)
                    .map(|_| offset + width * rng.random_range(-1.0..=1.0f32))
                    .collect::<Vec<_>>();
                for n in [15.0, 63.0, 255.0] {
                    if lut.iter().any(|&x| x != lut[0]) {
                        check(&lut, n);
                    }
                }
            }
        }
    }

    #[test]
    fn degenerate_spans() {
        for lut in [
            vec![5.0f32; 10],
            vec![0.0, f32::MIN_POSITIVE * 1.0e-3],
            vec![1.0e-40, 2.0e-40],
            vec![-3.0e38, 0.0, 3.0e38],
        ] {
            let (k, b, codes) = quantize(&lut, 63.0);
            assert_eq!(k, 0.0);
            assert_eq!(b, lut.iter().copied().fold(f32::INFINITY, f32::min));
            assert!(codes.iter().all(|&c| c == 0));
        }
    }
}
