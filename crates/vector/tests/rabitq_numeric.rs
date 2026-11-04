// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use simd::Floating;
use vector::rabitq4::Rabitq4Owned;
use vector::rabitq8::Rabitq8Owned;
use vector::{VectorBorrowed, VectorOwned};

fn lcg(seed: u64) -> impl FnMut() -> f32 {
    let mut state = seed;
    move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// What `quantize_to_rabitq8` does.
fn rabitq8(x: &[f32]) -> Rabitq8Owned {
    let (metadata, elements) = rabitq::byte::ugly_code(x);
    Rabitq8Owned::new(
        x.len() as u32,
        metadata.dis_u_2,
        metadata.norm_of_lattice,
        metadata.sum_of_code,
        f32::reduce_sum_of_abs_x(x),
        rabitq::byte::pack_code(&elements),
    )
}

/// What `quantize_to_rabitq4` does.
fn rabitq4(x: &[f32]) -> Rabitq4Owned {
    let (metadata, elements) = rabitq::halfbyte::ugly_code(x);
    Rabitq4Owned::new(
        x.len() as u32,
        metadata.dis_u_2,
        metadata.norm_of_lattice,
        metadata.sum_of_code,
        f32::reduce_sum_of_abs_x(x),
        rabitq::halfbyte::pack_code(&elements),
    )
}

/// `v <-> v` is `sqrt(operator_l2s(v, v))`; a negative estimate made it NaN.
#[test]
#[cfg_attr(miri, ignore)]
fn l2s_of_identical_vectors() {
    let mut rng = lcg(1);
    for n in [1, 2, 3, 64, 65, 768] {
        for _ in 0..1000 {
            let x = (0..n).map(|_| rng()).collect::<Vec<_>>();
            let (a, b) = (rabitq8(&x), rabitq4(&x));
            let d = a.as_borrowed().operator_l2s(a.as_borrowed()).to_f32();
            assert!(
                d >= 0.0 && d.sqrt().is_finite(),
                "rabitq8: n = {n}, d = {d}"
            );
            let d = b.as_borrowed().operator_l2s(b.as_borrowed()).to_f32();
            assert!(
                d >= 0.0 && d.sqrt().is_finite(),
                "rabitq4: n = {n}, d = {d}"
            );
        }
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn zero_vector() {
    let mut rng = lcg(2);
    for n in [1, 3, 64] {
        let zero = vec![0.0f32; n];
        let x = (0..n).map(|_| rng()).collect::<Vec<_>>();
        let x2 = f32::reduce_sum_of_x2(&x);
        let (z8, x8) = (rabitq8(&zero), rabitq8(&x));
        let (z4, x4) = (rabitq4(&zero), rabitq4(&x));
        // L2 to a zero vector is exact.
        let d = z8.as_borrowed().operator_l2s(x8.as_borrowed()).to_f32();
        assert!((d - x2).abs() <= 1e-5 * x2, "{d} {x2}");
        let d = z4.as_borrowed().operator_l2s(x4.as_borrowed()).to_f32();
        assert!((d - x2).abs() <= 1e-5 * x2, "{d} {x2}");
        assert_eq!(
            z8.as_borrowed().operator_l2s(z8.as_borrowed()).to_f32(),
            0.0
        );
        // Normalization keeps a zero vector zero (it used to claim a unit norm).
        let n8 = z8.as_borrowed().function_normalize();
        let n4 = z4.as_borrowed().function_normalize();
        assert_eq!(n8.as_borrowed().sum_of_x2(), 0.0);
        assert_eq!(n4.as_borrowed().sum_of_x2(), 0.0);
        assert_eq!(n8.as_borrowed().sum_of_abs_x(), 0.0);
        let u8 = x8.as_borrowed().function_normalize();
        let u4 = x4.as_borrowed().function_normalize();
        // The distances the cosine opclasses compute: dot (vchordrq), L2 (vchordg).
        let dot = n8.as_borrowed().operator_dot(u8.as_borrowed()).to_f32();
        assert_eq!(dot, 0.0);
        let l2s = n8.as_borrowed().operator_l2s(u8.as_borrowed()).to_f32();
        assert!((l2s - 1.0).abs() < 1e-5, "{l2s}");
        let dot = n4.as_borrowed().operator_dot(u4.as_borrowed()).to_f32();
        assert_eq!(dot, 0.0);
        let l2s = n4.as_borrowed().operator_l2s(u4.as_borrowed()).to_f32();
        assert!((l2s - 1.0).abs() < 1e-5, "{l2s}");
        // Non-zero vectors are still normalized to unit length.
        assert_eq!(u8.as_borrowed().sum_of_x2(), 1.0);
        assert_eq!(u4.as_borrowed().sum_of_x2(), 1.0);
    }
}
