// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use simd::Floating;

#[derive(Debug, Clone, Copy)]
pub struct CodeMetadata {
    pub dis_u_2: f32,
    pub norm_of_lattice: f32,
    pub sum_of_code: f32,
}

impl CodeMetadata {
    pub fn into_tuple(self) -> (f32, f32, f32) {
        (self.dis_u_2, self.norm_of_lattice, self.sum_of_code)
    }
    pub fn into_array(self) -> [f32; 3] {
        [self.dis_u_2, self.norm_of_lattice, self.sum_of_code]
    }
    pub fn from_tuple((dis_u_2, norm_of_lattice, sum_of_code): (f32, f32, f32)) -> Self {
        Self {
            dis_u_2,
            norm_of_lattice,
            sum_of_code,
        }
    }
    pub fn from_array([dis_u_2, norm_of_lattice, sum_of_code]: [f32; 3]) -> Self {
        Self {
            dis_u_2,
            norm_of_lattice,
            sum_of_code,
        }
    }
}

pub type Code = (CodeMetadata, Vec<u8>);

pub fn code<const BITS: usize>(vector: &[f32]) -> Code {
    assert!((1..=8).contains(&BITS));

    let n = vector.len();
    let dis_u_2 = f32::reduce_sum_of_x2(vector);
    let code = if dis_u_2 == 0.0 {
        zero_code::<BITS>(n)
    } else {
        let min = -(1 << (BITS - 1));
        let max = (1 << (BITS - 1)) - 1;
        let normalized_vector = {
            let mut vector = vector.to_vec();
            f32::vector_mul_scalar_inplace(&mut vector, 1.0 / dis_u_2.sqrt());
            vector
        };
        let scale = {
            let mut o = normalized_vector.clone();
            f32::vector_abs_inplace(&mut o);
            find_scale::<BITS>(&o)
        };
        let mut code = Vec::with_capacity(n as _);
        for i in 0..n {
            let v = scale * normalized_vector[i];
            let c = v.floor().clamp(min as f32, max as f32) as i32;
            code.push((c + (1 << (BITS - 1))) as _);
        }
        code
    };
    let norm_of_lattice = {
        let base = -0.5 * ((1 << BITS) - 1) as f32;
        let mut y = 0.0;
        for i in 0..n {
            let x = base + code[i] as f32;
            y += x * x;
        }
        y.sqrt()
    };
    let sum_of_code = {
        let mut y = 0;
        for i in 0..n {
            let x = code[i] as u32;
            y += x;
        }
        y as f32
    };
    (
        CodeMetadata {
            dis_u_2,
            norm_of_lattice,
            sum_of_code,
        },
        code,
    )
}

pub fn ugly_code<const BITS: usize>(vector: &[f32]) -> Code {
    assert!((1..=8).contains(&BITS));

    let n = vector.len();
    let dis_u_2 = f32::reduce_sum_of_x2(vector);
    let code = if dis_u_2 == 0.0 {
        zero_code::<BITS>(n)
    } else {
        let min = -(1 << (BITS - 1));
        let max = (1 << (BITS - 1)) - 1;
        let normalized_vector = {
            let mut vector = vector.to_vec();
            f32::vector_mul_scalar_inplace(&mut vector, 1.0 / dis_u_2.sqrt());
            vector
        };
        let scale = {
            let mut o = normalized_vector.clone();
            f32::vector_abs_inplace(&mut o);
            ugly_find_scale::<BITS>(&o)
        };
        let mut code = Vec::with_capacity(n as _);
        for i in 0..n {
            let v = scale * normalized_vector[i];
            let c = v.floor().clamp(min as f32, max as f32) as i32;
            code.push((c + (1 << (BITS - 1))) as _);
        }
        code
    };
    let norm_of_lattice = {
        let base = -0.5 * ((1 << BITS) - 1) as f32;
        let mut y = 0.0;
        for i in 0..n {
            let x = base + code[i] as f32;
            y += x * x;
        }
        y.sqrt()
    };
    let sum_of_code = {
        let mut y = 0;
        for i in 0..n {
            let x = code[i] as u32;
            y += x;
        }
        y as f32
    };
    (
        CodeMetadata {
            dis_u_2,
            norm_of_lattice,
            sum_of_code,
        },
        code,
    )
}

/// The code of a zero vector: every element is the middle of the lattice.
///
/// A zero vector has no direction to quantize. This is also the code the
/// general path produced for it, but through NaN arithmetic (`0 * (1 / 0)`).
fn zero_code<const BITS: usize>(n: usize) -> Vec<u8> {
    vec![(1_u32 << (BITS - 1)) as u8; n]
}

/// Estimates the squared Euclidean distance.
///
/// `|u|^2 + |v|^2 - 2 <u, v>` cancels catastrophically for (nearly) equal
/// vectors and can come out slightly negative, so the result is clamped to
/// zero; a NaN stays NaN.
pub fn half_process_l2s<const X: usize, const Y: usize>(
    dim: u32,
    sum: u32,
    lhs: CodeMetadata,
    rhs: CodeMetadata,
) -> f32 {
    assert!((1..=8).contains(&X));
    assert!((1..=8).contains(&Y));

    let c_x = ((1 << X) - 1) as f32 * 0.5;
    let c_y = ((1 << Y) - 1) as f32 * 0.5;

    let ip = sum as f32 - (c_y * lhs.sum_of_code + c_x * rhs.sum_of_code) + dim as f32 * c_x * c_y;
    let l2s = lhs.dis_u_2 + rhs.dis_u_2
        - 2.0
            * ip
            * (lhs.dis_u_2.sqrt() / lhs.norm_of_lattice)
            * (rhs.dis_u_2.sqrt() / rhs.norm_of_lattice);
    l2s.clamp(0.0, f32::INFINITY)
}

pub fn half_process_dot<const X: usize, const Y: usize>(
    dim: u32,
    sum: u32,
    lhs: CodeMetadata,
    rhs: CodeMetadata,
) -> f32 {
    assert!((1..=8).contains(&X));
    assert!((1..=8).contains(&Y));

    let c_x = ((1 << X) - 1) as f32 * 0.5;
    let c_y = ((1 << Y) - 1) as f32 * 0.5;

    let ip = sum as f32 - (c_y * lhs.sum_of_code + c_x * rhs.sum_of_code) + dim as f32 * c_x * c_y;
    -ip * (lhs.dis_u_2.sqrt() / lhs.norm_of_lattice) * (rhs.dis_u_2.sqrt() / rhs.norm_of_lattice)
}

pub fn half_process_cos<const X: usize, const Y: usize>(
    dim: u32,
    sum: u32,
    lhs: CodeMetadata,
    rhs: CodeMetadata,
) -> f32 {
    assert!((1..=8).contains(&X));
    assert!((1..=8).contains(&Y));

    let c_x = ((1 << X) - 1) as f32 * 0.5;
    let c_y = ((1 << Y) - 1) as f32 * 0.5;

    let ip = sum as f32 - (c_y * lhs.sum_of_code + c_x * rhs.sum_of_code) + dim as f32 * c_x * c_y;
    -ip / lhs.norm_of_lattice / rhs.norm_of_lattice
}

fn find_scale<const B: usize>(o: &[f32]) -> f32 {
    assert!((1..=8).contains(&B));

    let mask = (1_u32 << (B - 1)) - 1;
    let dim = o.len();

    let mut code = Vec::<u8>::with_capacity(dim);
    let mut numerator = 0.0f64;
    let mut sqr_denominator = 0.0f64;

    let (mut y_m, mut x_m);

    for i in 0..dim {
        code.push(0);
        let value = 0.5;
        numerator += value * o[i] as f64;
        sqr_denominator += value * value;
    }
    {
        let x = 0.0;
        let y = numerator / sqr_denominator.sqrt();
        (y_m, x_m) = (y, x);
    }

    let mut events = Vec::<(f64, usize)>::new();
    for i in 0..dim {
        for c in 1..=mask {
            let x = (c as f64) / o[i] as f64;
            events.push((x, i));
        }
    }
    events.sort_unstable_by(|(x, _), (y, _)| f64::total_cmp(x, y));
    for (x, i) in events.into_iter() {
        code[i] += 1;
        numerator += o[i] as f64;
        sqr_denominator += 2.0 * code[i] as f64;

        let y = numerator / sqr_denominator.sqrt();
        if y > y_m {
            (y_m, x_m) = (y, x);
        }
    }

    x_m as f32 + f32::EPSILON
}

fn ugly_find_scale<const B: usize>(o: &[f32]) -> f32 {
    assert!((1..=8).contains(&B));

    (1 << (B - 1)) as f32 / f32::reduce_min_max_of_x(o).1
}

pub fn pack_code<const BITS: usize>(input: &[u8]) -> [Vec<u64>; BITS] {
    #[inline(always)]
    fn f(array: &[u8; 64], bit: usize) -> u64 {
        let mut result = 0_u64;
        for i in 0..64 {
            result |= ((array[i] as u64 >> bit) & 1) << i;
        }
        result
    }
    let (arrays, remainder) = input.as_chunks::<64>();
    let mut buffer = [0_u8; 64];
    let tailing = if !remainder.is_empty() {
        buffer[..remainder.len()].copy_from_slice(remainder);
        Some(&buffer)
    } else {
        None
    };
    std::array::from_fn(|bit| arrays.iter().chain(tailing).map(|t| f(t, bit)).collect())
}

pub fn accumulate<const X: usize, const Y: usize>(lhs: &[u64], rhs: &[Vec<u64>; Y]) -> u32 {
    assert!(lhs.len().is_multiple_of(X));
    let d = lhs.len() / X;
    let mut result = 0_u32;
    for i in 0..X {
        for j in 0..Y {
            result += simd::bit::reduce_sum_of_and(&lhs[i * d..][..d], &rhs[j]) << (i + j);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: u64) -> impl FnMut() -> f32 {
        let mut state = seed;
        move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }
    }

    fn check_zero<const BITS: usize>(n: usize, (metadata, code): Code) {
        assert_eq!(code, vec![1_u8 << (BITS - 1); n]);
        let half = (1u32 << (BITS - 1)) as f32;
        assert_eq!(metadata.dis_u_2, 0.0);
        assert_eq!(metadata.sum_of_code, n as f32 * half);
        let expected = (n as f32 * 0.25).sqrt();
        assert!((metadata.norm_of_lattice - expected).abs() <= 1e-6 * expected);
    }

    #[test]
    fn zero_vector() {
        for n in [1, 2, 3, 64, 65, 784] {
            for zero in [vec![0.0f32; n], vec![-0.0f32; n]] {
                check_zero::<1>(n, code::<1>(&zero));
                check_zero::<2>(n, code::<2>(&zero));
                check_zero::<4>(n, code::<4>(&zero));
                check_zero::<8>(n, code::<8>(&zero));
                check_zero::<1>(n, ugly_code::<1>(&zero));
                check_zero::<2>(n, ugly_code::<2>(&zero));
                check_zero::<4>(n, ugly_code::<4>(&zero));
                check_zero::<8>(n, ugly_code::<8>(&zero));
            }
        }
        // Distances to a zero vector are exact.
        let mut rng = lcg(7);
        for n in [1, 3, 64] {
            let v = (0..n).map(|_| rng() * 255.0).collect::<Vec<_>>();
            let v2 = f32::reduce_sum_of_x2(&v);
            let (z, zc) = ugly_code::<8>(&vec![0.0; n]);
            let (m, c) = ugly_code::<8>(&v);
            let sum = simd::byte::reduce_sum_of_xy(&zc, &c);
            let l2s = half_process_l2s::<8, 8>(n as u32, sum, z, m);
            assert!((l2s - v2).abs() <= 1e-6 * v2, "{l2s} {v2}");
            assert_eq!(half_process_dot::<8, 8>(n as u32, sum, z, m), 0.0);
            let sum = simd::byte::reduce_sum_of_xy(&zc, &zc);
            assert_eq!(half_process_l2s::<8, 8>(n as u32, sum, z, z), 0.0);
        }
    }

    /// The estimate of `|v - v|^2` must never be negative (its square root, the
    /// `<->` operator of `rabitq8` and `rabitq4`, would be NaN).
    #[test]
    fn l2s_of_identical_vectors_is_not_negative() {
        let mut rng = lcg(42);
        for n in [1, 2, 3, 7, 64, 100, 768] {
            for _ in 0..1000 {
                let scale = [1e-3, 1.0, 255.0][(rng().abs() * 2.99) as usize];
                let v = (0..n).map(|_| rng() * scale).collect::<Vec<_>>();
                let (m8, c8) = ugly_code::<8>(&v);
                let sum = simd::byte::reduce_sum_of_xy(&c8, &c8);
                let l2s = half_process_l2s::<8, 8>(n as u32, sum, m8, m8);
                assert!(
                    l2s >= 0.0 && l2s <= 1e-3 * m8.dis_u_2,
                    "n = {n}, l2s = {l2s}"
                );
                let (m4, c4) = ugly_code::<4>(&v);
                let p4 = crate::halfbyte::pack_code(&c4);
                let sum = simd::halfbyte::reduce_sum_of_xy(&p4, &p4);
                let l2s = half_process_l2s::<4, 4>(n as u32, sum, m4, m4);
                assert!(
                    l2s >= 0.0 && l2s <= 1e-3 * m4.dis_u_2,
                    "n = {n}, l2s = {l2s}"
                );
            }
        }
        // NaN is not hidden by the clamp.
        let m = CodeMetadata {
            dis_u_2: f32::NAN,
            norm_of_lattice: 1.0,
            sum_of_code: 1.0,
        };
        assert!(half_process_l2s::<8, 8>(1, 1, m, m).is_nan());
    }
}
