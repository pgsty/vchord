// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use crate::index::{flat_index as prefect_index, flat_index as index};
use crate::square::{Square, SquareMut};
use crate::{KMeans, This};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;

struct Flat<'a> {
    this: This<'a>,
    samples: SquareMut<'a>,
    centroids: Square,
    targets: Vec<usize>,
    finite_samples: Option<Vec<bool>>,
}

impl<'a> KMeans for Flat<'a> {
    fn prefect_index(&self) -> Box<dyn Fn(&[f32]) -> (f32, usize) + Sync + '_> {
        Box::new(prefect_index(&self.centroids))
    }

    fn assign(&mut self) {
        let this = &mut self.this;
        let samples = &mut self.samples;
        let centroids = &self.centroids;
        let targets = &mut self.targets;
        let index = index(centroids);
        this.pool.install(|| {
            targets
                .par_iter_mut()
                .zip(samples.par_iter_mut())
                .for_each(|(target, sample)| {
                    *target = index(sample).1;
                });
        });
    }

    fn update(&mut self) {
        // Samples stay fixed. Cache once, and skip the scan entirely when the
        // caller requests zero Lloyd iterations.
        let finite_samples = self.finite_samples.get_or_insert_with(|| {
            self.this.pool.install(|| {
                (0..self.samples.len())
                    .into_par_iter()
                    .map(|i| crate::is_finite(&self.samples[i]))
                    .collect()
            })
        });
        crate::index::update(
            &mut self.this,
            &self.samples,
            &self.targets,
            finite_samples,
            &mut self.centroids,
        );
    }

    fn finish(self: Box<Self>) -> Square {
        self.centroids
    }
}

pub fn new<'a>(
    pool: &'a rayon::ThreadPool,
    d: usize,
    samples: SquareMut<'a>,
    c: usize,
    seed: [u8; 32],
    is_spherical: bool,
) -> Box<dyn KMeans + 'a> {
    let mut rng = StdRng::from_seed(seed);

    let mut centroids = Square::with_capacity(d, c);

    for index in rand::seq::index::sample(&mut rng, samples.len(), c.min(samples.len())) {
        if crate::is_finite(&samples[index]) {
            centroids.push_slice(&samples[index]);
        }
    }

    if centroids.is_empty() && c == 1 {
        centroids.push_iter(std::iter::repeat_n(0.0, d as _));
    }

    while centroids.len() < c {
        centroids.push_iter((0..d).map(|_| rng.random_range(-1.0f32..1.0f32)));
    }

    pool.install(|| {
        if is_spherical {
            (&mut centroids).into_par_iter().for_each(crate::normalize);
        }
    });

    let targets = vec![0; samples.len()];

    Box::new(Flat {
        this: This {
            pool,
            rng,
            d,
            c,
            is_spherical,
        },
        samples,
        centroids,
        targets,
        finite_samples: None,
    })
}
