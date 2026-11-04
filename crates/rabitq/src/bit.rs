// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use binary::BinaryLut;
use block::BlockLut;
use simd::Floating;

#[derive(Debug, Clone, Copy, Default)]
pub struct CodeMetadata {
    pub dis_u_2: f32,
    pub factor_cnt: f32,
    pub factor_ip: f32,
    pub factor_err: f32,
}

impl CodeMetadata {
    #[inline(always)]
    pub fn into_tuple(self) -> (f32, f32, f32, f32) {
        (
            self.dis_u_2,
            self.factor_cnt,
            self.factor_ip,
            self.factor_err,
        )
    }
    #[inline(always)]
    pub fn into_array(self) -> [f32; 4] {
        [
            self.dis_u_2,
            self.factor_cnt,
            self.factor_ip,
            self.factor_err,
        ]
    }
    #[inline(always)]
    pub fn from_tuple((dis_u_2, factor_cnt, factor_ip, factor_err): (f32, f32, f32, f32)) -> Self {
        Self {
            dis_u_2,
            factor_cnt,
            factor_ip,
            factor_err,
        }
    }
    #[inline(always)]
    pub fn from_array([dis_u_2, factor_cnt, factor_ip, factor_err]: [f32; 4]) -> Self {
        Self {
            dis_u_2,
            factor_cnt,
            factor_ip,
            factor_err,
        }
    }
}

pub type Code = (CodeMetadata, Vec<bool>);

/// Computes the RaBitQ factors of a vector.
///
/// For a non-degenerate vector this is the usual formula. Two inputs used to
/// produce NaN, whose sign differs between CPUs:
///
/// * a zero vector (`0 / 0`): it has no direction, so both factors are zero
///   and the estimates reduce to the exact distances (`|q|^2` and `0`);
/// * `n == 1`, where `1 / x_0^2 - 1` is zero or a rounding error of either
///   sign: the 1-bit code of a 1-dimensional vector is exact, so the error
///   factor is zero.
///
/// Rounding can also make `1 / x_0^2 - 1` slightly negative for vectors whose
/// elements all have the same magnitude; it is clamped to zero.
fn metadata(n: usize, sum_of_abs_x: f32, sum_of_x_2: f32, factor_cnt: f32) -> CodeMetadata {
    if sum_of_abs_x == 0.0 {
        return CodeMetadata {
            dis_u_2: sum_of_x_2,
            factor_cnt,
            factor_ip: 0.0,
            factor_err: 0.0,
        };
    }
    CodeMetadata {
        dis_u_2: sum_of_x_2,
        factor_cnt,
        factor_ip: sum_of_x_2 / sum_of_abs_x,
        factor_err: if n > 1 {
            let dis_u = sum_of_x_2.sqrt();
            let x_0 = sum_of_abs_x / dis_u / (n as f32).sqrt();
            dis_u * (1.0 / (x_0 * x_0) - 1.0).max(0.0).sqrt() / (n as f32 - 1.0).sqrt()
        } else {
            0.0
        },
    }
}

fn factor_cnt(vector: &[f32]) -> f32 {
    let cnt_pos = vector.iter().filter(|x| x.is_sign_positive()).count();
    let cnt_neg = vector.iter().filter(|x| x.is_sign_negative()).count();
    cnt_pos as f32 - cnt_neg as f32
}

pub fn code_metadata(vector: &[f32]) -> CodeMetadata {
    let n = vector.len();
    let sum_of_abs_x = f32::reduce_sum_of_abs_x(vector);
    let sum_of_x_2 = f32::reduce_sum_of_x2(vector);
    metadata(n, sum_of_abs_x, sum_of_x_2, factor_cnt(vector))
}

pub fn code_elements(vector: &[f32]) -> Vec<bool> {
    let n = vector.len();
    let mut signs = Vec::new();
    for i in 0..n {
        signs.push(vector[i].is_sign_positive());
    }
    signs
}

pub fn code(vector: &[f32]) -> Code {
    (code_metadata(vector), code_elements(vector))
}

pub fn preprocess(vector: &[f32]) -> (BlockLut, BinaryLut) {
    let dis_v_2 = f32::reduce_sum_of_x2(vector);
    (
        block::preprocess_with_distance(vector, dis_v_2),
        binary::preprocess_with_distance(vector, dis_v_2),
    )
}

pub mod binary {
    pub fn pack_code(input: &[bool]) -> Vec<u64> {
        let f = |t: &[bool; 64]| {
            let mut result = 0_u64;
            for i in 0..64 {
                result |= (t[i] as u64) << i;
            }
            result
        };
        let (arrays, remainder) = input.as_chunks::<64>();
        let mut buffer = [false; 64];
        let tailing = if !remainder.is_empty() {
            buffer[..remainder.len()].copy_from_slice(remainder);
            Some(&buffer)
        } else {
            None
        };
        arrays.iter().chain(tailing).map(f).collect()
    }

    use super::CodeMetadata;
    use simd::Floating;

    const BITS: usize = 6;

    #[derive(Debug, Clone, Copy)]
    pub struct BinaryLutMetadata {
        dis_v_2: f32,
        b: f32,
        k: f32,
        qvector_sum: f32,
    }

    pub type BinaryLut = (BinaryLutMetadata, [Vec<u64>; BITS]);
    pub type BinaryCode<'a> = ((f32, f32, f32, f32), &'a [u64]);

    #[inline(always)]
    pub fn preprocess(vector: &[f32]) -> BinaryLut {
        let dis_v_2 = f32::reduce_sum_of_x2(vector);
        preprocess_with_distance(vector, dis_v_2)
    }

    pub(crate) fn preprocess_with_distance(vector: &[f32], dis_v_2: f32) -> BinaryLut {
        let (k, b, qvector) = simd::quantize::quantize(vector, ((1 << BITS) - 1) as f32);
        let qvector_sum = simd::byte::reduce_sum_of_x(&qvector) as f32;
        (
            BinaryLutMetadata {
                dis_v_2,
                b,
                k,
                qvector_sum,
            },
            binarize(&qvector),
        )
    }

    #[inline(always)]
    pub fn accumulate(x: &[u64], y: &[impl AsRef<[u64]>; BITS]) -> u32 {
        let mut result = 0_u32;
        for i in 0..BITS {
            result += simd::bit::reduce_sum_of_and(x, y[i].as_ref()) << i;
        }
        result
    }

    #[inline(always)]
    pub fn half_process_l2s(
        sum: u32,
        CodeMetadata {
            dis_u_2,
            factor_cnt,
            factor_ip,
            factor_err,
        }: CodeMetadata,
        BinaryLutMetadata {
            dis_v_2,
            b,
            k,
            qvector_sum,
        }: BinaryLutMetadata,
    ) -> (f32, f32) {
        let e = k * ((2.0 * sum as f32) - qvector_sum) + b * factor_cnt;
        let rough = dis_u_2 + dis_v_2 - 2.0 * e * factor_ip;
        let err = 2.0 * factor_err * dis_v_2.sqrt();
        (rough, err)
    }

    #[inline(always)]
    pub fn half_process_l2s_residual(
        sum: u32,
        CodeMetadata {
            dis_u_2,
            factor_cnt,
            factor_ip,
            factor_err,
        }: CodeMetadata,
        BinaryLutMetadata {
            dis_v_2: _,
            b,
            k,
            qvector_sum,
        }: BinaryLutMetadata,
        dis_f: f32,
        delta: f32,
    ) -> (f32, f32) {
        let e = k * ((2.0 * sum as f32) - qvector_sum) + b * factor_cnt;
        let rough = dis_u_2 + dis_f - 2.0 * e * factor_ip + delta;
        let err = 2.0 * factor_err * dis_f.sqrt();
        (rough, err)
    }

    #[inline(always)]
    pub fn half_process_dot(
        sum: u32,
        CodeMetadata {
            dis_u_2: _,
            factor_cnt,
            factor_ip,
            factor_err,
        }: CodeMetadata,
        BinaryLutMetadata {
            dis_v_2,
            b,
            k,
            qvector_sum,
        }: BinaryLutMetadata,
    ) -> (f32, f32) {
        let e = k * ((2.0 * sum as f32) - qvector_sum) + b * factor_cnt;
        let rough = -e * factor_ip;
        let err = factor_err * dis_v_2.sqrt();
        (rough, err)
    }

    #[inline(always)]
    pub fn half_process_dot_residual(
        sum: u32,
        CodeMetadata {
            dis_u_2: _,
            factor_cnt,
            factor_ip,
            factor_err,
        }: CodeMetadata,
        BinaryLutMetadata {
            dis_v_2,
            b,
            k,
            qvector_sum,
        }: BinaryLutMetadata,
        dis_f: f32,
        delta: f32,
        norm: f32,
    ) -> (f32, f32) {
        let e = k * ((2.0 * sum as f32) - qvector_sum) + b * factor_cnt;
        let rough = -e * factor_ip + dis_f + delta;
        let err = factor_err * (dis_v_2 + norm * norm + 2.0 * dis_f).sqrt();
        (rough, err)
    }

    #[inline(always)]
    pub(crate) fn binarize(vector: &[u8]) -> [Vec<u64>; BITS] {
        let n = vector.len();
        let mut t: [_; BITS] = std::array::from_fn(|_| vec![0_u64; n.div_ceil(64)]);
        for i in 0..BITS {
            for j in 0..n {
                let bit = (vector[j] >> i) & 1;
                t[i][j / 64] |= (bit as u64) << (j % 64);
            }
        }
        t
    }
}

pub mod block {
    use super::CodeMetadata;
    use simd::Floating;

    const BITS: usize = 8;
    pub const STEP: usize = 65535_usize / ((1_usize << BITS) - 1);

    #[derive(Debug, Clone, Copy)]
    pub struct BlockLutMetadata {
        dis_v_2: f32,
        k: f32,
        c: f32,
    }

    pub type BlockLut = (BlockLutMetadata, Vec<[u8; 16]>);
    pub type BlockCode<'a> = (
        &'a [f32; 32],
        &'a [f32; 32],
        &'a [f32; 32],
        &'a [f32; 32],
        &'a [[u8; 16]],
    );

    #[inline(always)]
    pub fn preprocess(vector: &[f32]) -> BlockLut {
        let dis_v_2 = f32::reduce_sum_of_x2(vector);
        preprocess_with_distance(vector, dis_v_2)
    }

    pub(crate) fn preprocess_with_distance(vector: &[f32], dis_v_2: f32) -> BlockLut {
        let cvector = compress(vector);
        let (k, b, cqvector) =
            simd::quantize::quantize(cvector.as_flattened(), ((1 << BITS) - 1) as f32);
        (
            BlockLutMetadata {
                dis_v_2,
                k,
                c: b * vector.len().div_ceil(4) as f32,
            },
            cqvector.as_chunks::<16>().0.to_vec(),
        )
    }

    #[inline(always)]
    pub fn accumulate(t: &[[u8; 16]], s: &[[u8; 16]]) -> [u32; 32] {
        use std::iter::zip;
        let mut sum = [0_u32; 32];
        for (t, s) in zip(t.chunks(STEP), s.chunks(STEP)) {
            let delta = simd::fast_scan::scan(t, s);
            simd::fast_scan::accu(&mut sum, &delta);
        }
        sum
    }

    #[inline(always)]
    pub fn full_process_l2s(
        (dis_u_2, _, factor_ip, factor_err, t): BlockCode<'_>,
        lut: &BlockLut,
    ) -> [(f32, f32); 32] {
        use std::iter::zip;
        let &(BlockLutMetadata { dis_v_2, k, c }, ref s) = lut;
        let mut sum = [0_u32; 32];
        for (t, s) in zip(t.chunks(STEP), s.chunks(STEP)) {
            let delta = simd::fast_scan::scan(t, s);
            simd::fast_scan::accu(&mut sum, &delta);
        }
        std::array::from_fn(|i| {
            let e = k * (sum[i] as f32) + c;
            let rough = dis_u_2[i] + dis_v_2 - 2.0 * e * factor_ip[i];
            let err = 2.0 * factor_err[i] * dis_v_2.sqrt();
            (rough, err)
        })
    }

    #[inline(always)]
    pub fn full_process_dot(
        (_, _, factor_ip, factor_err, t): BlockCode<'_>,
        lut: &BlockLut,
    ) -> [(f32, f32); 32] {
        use std::iter::zip;
        let &(BlockLutMetadata { dis_v_2, k, c }, ref s) = lut;
        let mut sum = [0_u32; 32];
        for (t, s) in zip(t.chunks(STEP), s.chunks(STEP)) {
            let delta = simd::fast_scan::scan(t, s);
            simd::fast_scan::accu(&mut sum, &delta);
        }
        std::array::from_fn(|i| {
            let e = k * (sum[i] as f32) + c;
            let rough = -e * factor_ip[i];
            let err = factor_err[i] * dis_v_2.sqrt();
            (rough, err)
        })
    }

    #[inline(always)]
    pub fn half_process_l2s(
        sum: u32,
        CodeMetadata {
            dis_u_2,
            factor_cnt: _,
            factor_ip,
            factor_err,
        }: CodeMetadata,
        BlockLutMetadata { dis_v_2, k, c }: BlockLutMetadata,
    ) -> (f32, f32) {
        let e = k * (sum as f32) + c;
        let rough = dis_u_2 + dis_v_2 - 2.0 * e * factor_ip;
        let err = 2.0 * factor_err * dis_v_2.sqrt();
        (rough, err)
    }

    #[inline(always)]
    pub fn half_process_l2s_residual(
        sum: u32,
        CodeMetadata {
            dis_u_2,
            factor_cnt: _,
            factor_ip,
            factor_err,
        }: CodeMetadata,
        BlockLutMetadata { dis_v_2: _, k, c }: BlockLutMetadata,
        dis_f: f32,
        delta: f32,
    ) -> (f32, f32) {
        let e = k * (sum as f32) + c;
        let rough = dis_u_2 + dis_f - 2.0 * e * factor_ip + delta;
        let err = 2.0 * factor_err * dis_f.sqrt();
        (rough, err)
    }

    #[inline(always)]
    pub fn half_process_dot(
        sum: u32,
        CodeMetadata {
            dis_u_2: _,
            factor_cnt: _,
            factor_ip,
            factor_err,
        }: CodeMetadata,
        BlockLutMetadata { dis_v_2, k, c }: BlockLutMetadata,
    ) -> (f32, f32) {
        let e = k * (sum as f32) + c;
        let rough = -e * factor_ip;
        let err = factor_err * dis_v_2.sqrt();
        (rough, err)
    }

    #[inline(always)]
    pub fn half_process_dot_residual(
        sum: u32,
        CodeMetadata {
            dis_u_2: _,
            factor_cnt: _,
            factor_ip,
            factor_err,
        }: CodeMetadata,
        BlockLutMetadata { dis_v_2, k, c }: BlockLutMetadata,
        dis_f: f32,
        delta: f32,
        norm: f32,
    ) -> (f32, f32) {
        let e = k * (sum as f32) + c;
        let rough = -e * factor_ip + dis_f + delta;
        let err = factor_err * (dis_v_2 + norm * norm + 2.0 * dis_f).sqrt();
        (rough, err)
    }

    fn compress(vector: &[f32]) -> Vec<[f32; 16]> {
        let f = |&[t_0, t_1, t_2, t_3]: &[f32; 4]| {
            [
                0.0 - t_3 - t_2 - t_1 - t_0,
                0.0 - t_3 - t_2 - t_1 + t_0,
                0.0 - t_3 - t_2 + t_1 - t_0,
                0.0 - t_3 - t_2 + t_1 + t_0,
                0.0 - t_3 + t_2 - t_1 - t_0,
                0.0 - t_3 + t_2 - t_1 + t_0,
                0.0 - t_3 + t_2 + t_1 - t_0,
                0.0 - t_3 + t_2 + t_1 + t_0,
                0.0 + t_3 - t_2 - t_1 - t_0,
                0.0 + t_3 - t_2 - t_1 + t_0,
                0.0 + t_3 - t_2 + t_1 - t_0,
                0.0 + t_3 - t_2 + t_1 + t_0,
                0.0 + t_3 + t_2 - t_1 - t_0,
                0.0 + t_3 + t_2 - t_1 + t_0,
                0.0 + t_3 + t_2 + t_1 - t_0,
                0.0 + t_3 + t_2 + t_1 + t_0,
            ]
        };
        let (arrays, remainder) = vector.as_chunks::<4>();
        let mut buffer = [0.0f32; 4];
        let tailing = if !remainder.is_empty() {
            buffer[..remainder.len()].copy_from_slice(remainder);
            Some(&buffer)
        } else {
            None
        };
        arrays.iter().chain(tailing).map(f).collect()
    }
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

    fn assert_finite(m: CodeMetadata) {
        for x in m.into_array() {
            assert!(x.is_finite(), "{m:?}");
        }
    }

    /// The formula before zero vectors and `n == 1` were special-cased.
    fn reference(vector: &[f32]) -> CodeMetadata {
        let n = vector.len();
        let sum_of_abs_x = f32::reduce_sum_of_abs_x(vector);
        let sum_of_x_2 = f32::reduce_sum_of_x2(vector);
        CodeMetadata {
            dis_u_2: sum_of_x_2,
            factor_cnt: factor_cnt(vector),
            factor_ip: sum_of_x_2 / sum_of_abs_x,
            factor_err: {
                let dis_u = sum_of_x_2.sqrt();
                let x_0 = sum_of_abs_x / dis_u / (n as f32).sqrt();
                dis_u * (1.0 / (x_0 * x_0) - 1.0).sqrt() / (n as f32 - 1.0).sqrt()
            },
        }
    }

    fn estimate(vector: &[f32], query: &[f32], dot: bool) -> [(f32, f32); 2] {
        let (metadata, elements) = code(vector);
        let binary = {
            let lut = binary::preprocess(query);
            let sum = binary::accumulate(&binary::pack_code(&elements), &lut.1);
            if dot {
                binary::half_process_dot(sum, metadata, lut.0)
            } else {
                binary::half_process_l2s(sum, metadata, lut.0)
            }
        };
        let block = {
            let lut = block::preprocess(query);
            let packed = crate::packing::padding_pack([crate::packing::pack_to_u4(&elements)]);
            let sum = block::accumulate(&packed, &lut.1);
            if dot {
                block::half_process_dot(sum[0], metadata, lut.0)
            } else {
                block::half_process_l2s(sum[0], metadata, lut.0)
            }
        };
        [binary, block]
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn zero_vector() {
        let mut rng = lcg(1);
        for n in [1, 2, 3, 7, 64, 65, 784] {
            for zero in [vec![0.0f32; n], vec![-0.0f32; n]] {
                let m = code_metadata(&zero);
                assert_finite(m);
                assert_eq!((m.dis_u_2, m.factor_ip, m.factor_err), (0.0, 0.0, 0.0));
                let query = (0..n).map(|_| rng() * 255.0).collect::<Vec<_>>();
                let q2 = f32::reduce_sum_of_x2(&query);
                // The estimates are exact for a zero vector.
                for (rough, err) in estimate(&zero, &query, false) {
                    assert_eq!((rough, err), (q2, 0.0), "n = {n}");
                }
                for (rough, err) in estimate(&zero, &query, true) {
                    assert_eq!((rough, err), (0.0, 0.0), "n = {n}");
                }
                // A zero query against a zero vector.
                for (rough, err) in estimate(&zero, &vec![0.0; n], false) {
                    assert_eq!((rough, err), (0.0, 0.0), "n = {n}");
                }
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn one_dimension() {
        for x in [1.0f32, -1.0, 0.3, -255.0, 1e-20, 3.0e18] {
            let m = code_metadata(&[x]);
            assert_finite(m);
            assert_eq!(m.factor_err, 0.0);
            // `x * x / |x|`; `1e-20 * 1e-20` is subnormal.
            assert!((m.factor_ip - x.abs()).abs() <= 1e-4 * x.abs(), "{m:?}");
        }
        for x in [1.0f32, -1.0, 0.3, -255.0] {
            for q in [2.0f32, -2.0, 0.0, 255.0] {
                for (rough, err) in estimate(&[x], &[q], false) {
                    assert!(rough.is_finite() && err == 0.0, "x = {x}, q = {q}");
                    let exact = (x - q) * (x - q);
                    assert!(
                        (rough - exact).abs() <= 1e-3 * (x * x + q * q),
                        "x = {x}, q = {q}, rough = {rough}, exact = {exact}"
                    );
                }
            }
        }
    }

    #[test]
    fn equal_magnitudes() {
        // `x_0` is 1 up to rounding, so `1 / x_0^2 - 1` may round below zero.
        for n in 1..300 {
            for scale in [1.0f32, 0.1, 3.0, 255.0, 1e-3] {
                let v = (0..n)
                    .map(|i| if i % 3 == 0 { -scale } else { scale })
                    .collect::<Vec<_>>();
                let m = code_metadata(&v);
                assert_finite(m);
                // Mathematically zero; only rounding noise remains.
                assert!(m.factor_err >= 0.0 && m.factor_err <= 1e-2 * scale, "{m:?}");
            }
        }
    }

    #[test]
    fn non_degenerate_metadata_is_unchanged() {
        let mut rng = lcg(2);
        for n in [2, 3, 7, 64, 100, 768, 784, 1536] {
            for _ in 0..50 {
                let v = (0..n).map(|_| rng() * 10.0).collect::<Vec<_>>();
                let expected = reference(&v).into_array().map(f32::to_bits);
                assert_eq!(code_metadata(&v).into_array().map(f32::to_bits), expected);
                let (got, signs) = code(&v);
                assert_eq!(got.into_array().map(f32::to_bits), expected);
                assert_eq!(signs, code_elements(&v));
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn residual_equal_to_centroid() {
        // A centroid that is equal to a sample (quick k-means) leaves a zero residual.
        let mut rng = lcg(3);
        for n in [3, 64, 785] {
            let sample = (0..n).map(|_| rng()).collect::<Vec<_>>();
            let residual = f32::vector_sub(&sample, &sample);
            assert_finite(code_metadata(&residual));
            let query = (0..n).map(|_| rng()).collect::<Vec<_>>();
            let dis_f = f32::reduce_sum_of_d2(&query, &sample);
            let (metadata, elements) = code(&residual);
            let lut = binary::preprocess(&query);
            let sum = binary::accumulate(&binary::pack_code(&elements), &lut.1);
            let (rough, err) = binary::half_process_l2s_residual(sum, metadata, lut.0, dis_f, 0.0);
            assert_eq!((rough, err), (dis_f, 0.0));
        }
    }
}
