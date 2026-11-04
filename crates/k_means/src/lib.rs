// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

mod flat;
mod hierarchical;
mod index;
mod quick;
mod rabitq;

pub mod square;

use crate::square::{Square, SquareMut};
use rand::rngs::StdRng;

pub struct This<'a> {
    pool: &'a rayon::ThreadPool,
    rng: StdRng,
    d: usize,
    c: usize,
    is_spherical: bool,
}

pub trait KMeans {
    fn prefect_index(&self) -> Box<dyn Fn(&[f32]) -> (f32, usize) + Sync + '_>;
    fn index(&self) -> Box<dyn Fn(&[f32]) -> (f32, usize) + Sync + '_> {
        self.prefect_index()
    }
    fn assign(&mut self);
    fn update(&mut self);
    fn finish(self: Box<Self>) -> Square;
}

/// Scales a centroid to unit length for spherical k-means.
///
/// A centroid whose norm is zero (e.g. the mean of zero vectors, or of `x` and
/// `-x`) or not finite has no direction. Scaling it by `1 / l` would fill it
/// with NaN, which no sample is ever assigned to again and which would end up
/// in the index, so it is left unchanged.
fn normalize(centroid: &mut [f32]) {
    use simd::Floating;
    let l = f32::reduce_sum_of_x2(centroid).sqrt();
    if l > 0.0 && l.is_finite() {
        f32::vector_mul_scalar_inplace(centroid, 1.0 / l);
    }
}

/// Non-finite samples are skipped: as a centroid, or as part of one, they
/// would turn the centroid into NaN.
fn is_finite(sample: &[f32]) -> bool {
    sample.iter().all(|x| x.is_finite())
}

pub fn k_means_lookup(sample: &[f32], centroids: &Square) -> usize {
    use simd::Floating;
    let mut result = (f32::INFINITY, 0);
    for (i, centroid) in centroids.into_iter().enumerate() {
        let dis = f32::reduce_sum_of_d2(sample, centroid);
        if dis <= result.0 {
            result = (dis, i);
        }
    }
    result.1
}

pub fn lloyd_k_means<'a>(
    pool: &'a rayon::ThreadPool,
    d: usize,
    samples: SquareMut<'a>,
    c: usize,
    seed: [u8; 32],
    is_spherical: bool,
) -> Box<dyn KMeans + 'a> {
    assert!(d > 0 && c > 0);
    let n = samples.len();
    if n <= c {
        quick::new(pool, d, samples, c, seed, is_spherical)
    } else if n <= c * 2 {
        flat::new(pool, d, samples, c, seed, is_spherical)
    } else {
        rabitq::new(pool, d, samples, c, seed, is_spherical)
    }
}

pub fn hierarchical_k_means<'a>(
    pool: &'a rayon::ThreadPool,
    d: usize,
    samples: SquareMut<'a>,
    c: usize,
    seed: [u8; 32],
    is_spherical: bool,
) -> Box<dyn KMeans + 'a> {
    assert!(d > 0 && c > 0);
    hierarchical::new(pool, d, samples, c, seed, is_spherical)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};
    use simd::Floating;

    fn pool() -> rayon::ThreadPool {
        rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap()
    }

    /// Every `zero`-th sample is a zero vector and every `nan`-th is NaN.
    fn samples(n: usize, d: usize, offset: f32, zero: usize, nan: usize) -> Square {
        let mut rng = StdRng::seed_from_u64((n * 1000 + d) as u64);
        let mut s = Square::with_capacity(d, n);
        for i in 0..n {
            if zero != 0 && i % zero == 0 {
                s.push_iter(std::iter::repeat_n(0.0, d));
            } else if nan != 0 && i % nan == 0 {
                s.push_iter(std::iter::repeat_n(f32::NAN, d));
            } else {
                s.push_iter((0..d).map(|_| offset + rng.random_range(-1.0f32..1.0)));
            }
        }
        s
    }

    fn check(centroids: &Square, c: usize, is_spherical: bool, what: &str) {
        assert_eq!(centroids.len(), c, "{what}");
        for centroid in centroids {
            assert!(is_finite(centroid), "{what}: {centroid:?}");
            if is_spherical {
                let l = f32::reduce_sum_of_x2(centroid).sqrt();
                assert!(l == 0.0 || (l - 1.0).abs() < 1e-4, "{what}: norm = {l}");
            }
        }
    }

    fn run(
        hierarchical: bool,
        mut s: Square,
        c: usize,
        is_spherical: bool,
    ) -> (Square, Vec<usize>) {
        let pool = pool();
        let d = s.d();
        let probes = (0..s.len()).map(|i| s[i].to_vec()).collect::<Vec<_>>();
        let mut f = if hierarchical {
            hierarchical_k_means(&pool, d, s.as_mut_view(), c, [7; 32], is_spherical)
        } else {
            lloyd_k_means(&pool, d, s.as_mut_view(), c, [7; 32], is_spherical)
        };
        for _ in 0..10 {
            f.assign();
            f.update();
        }
        let ids = {
            let index = f.index();
            probes.iter().map(|x| index(x).1).collect::<Vec<_>>()
        };
        (f.finish(), ids)
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn zero_and_nan_samples() {
        // `n <= c` (quick), `c < n <= 2c` (flat) and `n > 2c` (rabitq).
        for (n, c) in [(5, 8), (8, 8), (12, 8), (100, 8), (1000, 16)] {
            for d in [1, 3, 64] {
                for is_spherical in [false, true] {
                    for (zero, nan) in [(3, 0), (0, 5), (2, 3), (1, 0)] {
                        let what = format!("n = {n}, c = {c}, d = {d}, zero = {zero}, nan = {nan}");
                        for hierarchical in [false, true] {
                            let s = samples(n, d, 0.0, zero, nan);
                            let (centroids, ids) = run(hierarchical, s, c, is_spherical);
                            check(&centroids, c, is_spherical, &what);
                            assert!(ids.iter().all(|&id| id < c), "{what}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn only_nan_samples() {
        for (n, c) in [(3, 8), (12, 8), (100, 8)] {
            for is_spherical in [false, true] {
                let s = samples(n, 4, 0.0, 0, 1);
                let (centroids, _) = run(false, s, c, is_spherical);
                check(&centroids, c, is_spherical, "only NaN");
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn hierarchical_empty_partition() {
        // With 2 samples far from the origin, 2 of the 4 coarse centroids are
        // random vectors that no sample is assigned to. Their seats used to
        // get a zero vector (NaN when spherical).
        for is_spherical in [false, true] {
            let s = samples(2, 8, 1000.0, 0, 0);
            let (centroids, _) = run(true, s, 16, is_spherical);
            check(&centroids, 16, is_spherical, "empty partition");
            for centroid in &centroids {
                assert!(centroid.iter().any(|&x| x != 0.0), "{centroid:?}");
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn centroid_equal_to_zero_sample() {
        // Quick k-means uses samples as centroids; a zero sample gives a zero
        // centroid, whose RaBitQ metadata used to be NaN so that the RaBitQ
        // index never chose it, even for the zero vector itself.
        let pool = pool();
        for d in [1, 3, 16, 100] {
            for zero in [0, 5, 31, 32, 39] {
                let mut centroids = samples(40, d, 0.0, 0, 0);
                centroids[zero].fill(0.0);
                let rabitq_index = index::rabitq_index(&pool, &centroids);
                let flat_index = index::flat_index(&centroids);
                let mut probes = vec![vec![0.0; d], vec![1e-3; d]];
                probes.extend((0..40).map(|i| centroids[i].to_vec()));
                for probe in probes {
                    assert_eq!(rabitq_index(&probe), flat_index(&probe), "d = {d}");
                }
                assert_eq!(rabitq_index(&vec![0.0; d]), (0.0, zero));
            }
        }
    }
}
