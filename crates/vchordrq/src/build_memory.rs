// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

//! Memory planning for the internal build (sampling and k-means).
//!
//! The options of the internal build only give upper bounds: `lists` and
//! `sampling_factor` say how many vectors sampling may collect, not how many
//! the table has. Allocations are planned from the data instead.

const F32: u128 = size_of::<f32>() as u128;

/// The most vectors that sampling may collect for `lists` (the number of lists
/// of the bottom level) and `sampling_factor`.
pub fn max_number_of_samples(lists: Option<u32>, sampling_factor: u32) -> u64 {
    lists.map_or(0, |lists| u64::from(lists) * u64::from(sampling_factor))
}

/// The number of vectors to reserve memory for before sampling: the upper
/// bound from the options, clamped by the estimated number of rows in the heap.
///
/// An unknown estimate (negative, NaN) counts as zero. A row can hold several
/// vectors (multivector), so the buffer must still be able to grow past this.
pub fn initial_sample_capacity(max_number_of_samples: u64, estimated_rows: f64) -> u64 {
    let estimated_rows = if estimated_rows.is_finite() && estimated_rows > 0.0 {
        // float-to-int `as` saturates
        estimated_rows.ceil() as u64
    } else {
        0
    };
    max_number_of_samples.min(estimated_rows)
}

/// Memory of clustering the bottom level, in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClusteringMemory {
    /// The number of sampled vectors.
    pub samples: u64,
    /// The number of lists of the bottom level.
    pub lists: u64,
    /// The sampled vectors.
    pub samples_bytes: u128,
    /// State that scales with the number of lists: centroids, partial sums of
    /// k-means and, with `kmeans_dimension`, the full-dimensional sums.
    pub centroids_bytes: u128,
}

impl ClusteringMemory {
    /// * `sample_dim`: the dimension that k-means runs in
    /// * `dim`: the dimension of the vectors
    /// * `build_threads`: the threads of k-means
    /// * `reduction_threads`: with `kmeans_dimension`, the threads that compute
    ///   the full-dimensional centroids
    pub fn new(
        samples: u64,
        lists: u64,
        sample_dim: u64,
        dim: u64,
        build_threads: u64,
        reduction_threads: Option<u64>,
    ) -> Self {
        let samples_bytes = u128::from(samples) * u128::from(sample_dim) * F32;
        // With no more samples than lists, k-means copies the samples, fills
        // the remaining centroids with random values and does not iterate.
        // Otherwise an iteration holds the centroids, the new centroids and one
        // partial sum per thread.
        let copies = if samples <= lists {
            1
        } else {
            2 + u128::from(build_threads)
        };
        let mut centroids_bytes = copies * u128::from(lists) * u128::from(sample_dim) * F32;
        if let Some(reduction_threads) = reduction_threads {
            // a sum and a count per list, for each thread and for the result
            centroids_bytes += (u128::from(reduction_threads) + 1)
                * u128::from(lists)
                * (u128::from(dim) + 1)
                * F32;
        }
        Self {
            samples,
            lists,
            samples_bytes,
            centroids_bytes,
        }
    }

    pub fn total_bytes(&self) -> u128 {
        self.samples_bytes + self.centroids_bytes
    }

    /// Checks the memory that is not backed by sampled data.
    ///
    /// With more lists than sampled vectors, most centroids are random and
    /// their memory only follows from the options (e.g. `lists = [16777216]`
    /// on a table of ten rows), so it must fit in `limit`. Otherwise the
    /// working set is proportional to the sampled data, which is already
    /// bounded by the table and `sampling_factor`, and is not limited here.
    ///
    /// Returns the required bytes on failure.
    pub fn check(&self, limit: u128) -> Result<(), u128> {
        if self.lists > self.samples && self.centroids_bytes > limit {
            Err(self.centroids_bytes)
        } else {
            Ok(())
        }
    }
}

#[test]
fn test_max_number_of_samples() {
    assert_eq!(max_number_of_samples(None, 256), 0);
    assert_eq!(max_number_of_samples(Some(1000), 256), 256_000);
    // `u32` arithmetic used to saturate here
    assert_eq!(
        max_number_of_samples(Some(1 << 24), 1024),
        (1_u64 << 24) * 1024
    );
}

#[test]
fn test_initial_sample_capacity() {
    let max = max_number_of_samples(Some(1 << 24), 1024);
    // a table of 10 rows reserves 10 vectors, not `lists * sampling_factor`
    assert_eq!(initial_sample_capacity(max, 10.0), 10);
    assert_eq!(initial_sample_capacity(max, 9.2), 10);
    // never more than the options allow
    assert_eq!(initial_sample_capacity(1000, 1e12), 1000);
    assert_eq!(initial_sample_capacity(1000, f64::MAX), 1000);
    // an unknown estimate (`reltuples = -1`) or an empty table reserves nothing
    assert_eq!(initial_sample_capacity(1000, -1.0), 0);
    assert_eq!(initial_sample_capacity(1000, 0.0), 0);
    assert_eq!(initial_sample_capacity(1000, f64::NAN), 0);
    assert_eq!(initial_sample_capacity(0, 1e6), 0);
}

#[test]
fn test_clustering_memory() {
    const MIB: u128 = 1 << 20;
    const GIB: u128 = 1 << 30;
    let maintenance_work_mem = 64 * MIB;
    // The reported case: lists = [16777216], sampling_factor = 1024,
    // vector(2000), ten rows. The samples are tiny now, but the random
    // centroids would need 125 GiB.
    let max = max_number_of_samples(Some(1 << 24), 1024);
    let samples = initial_sample_capacity(max, 10.0);
    let memory = ClusteringMemory::new(samples, 1 << 24, 2000, 2000, 1, None);
    assert_eq!(memory.samples_bytes, 10 * 2000 * 4);
    assert_eq!(memory.centroids_bytes, (1 << 24) * 2000 * 4);
    assert_eq!(
        memory.check(maintenance_work_mem),
        Err((1 << 24) * 2000 * 4)
    );
    // ... and also with `kmeans_dimension`.
    let memory = ClusteringMemory::new(10, 1 << 24, 100, 2000, 8, Some(1));
    assert!(memory.centroids_bytes > 250 * GIB);
    assert!(memory.check(maintenance_work_mem).is_err());
    // An index created on an empty table with a moderate number of lists.
    let memory = ClusteringMemory::new(0, 1000, 768, 768, 1, None);
    assert_eq!(memory.total_bytes(), 1000 * 768 * 4);
    assert_eq!(memory.check(maintenance_work_mem), Ok(()));
    // A low maintenance_work_mem rejects random centroids that do not fit.
    let memory = ClusteringMemory::new(10, 2048, 256, 256, 1, None);
    assert_eq!(memory.centroids_bytes, 2 * MIB);
    assert_eq!(memory.check(MIB), Err(2 * MIB));
    assert_eq!(memory.check(2 * MIB), Ok(()));
    // Builds backed by enough data are not limited by maintenance_work_mem:
    // 1000 lists of 768 dimensions with 256 samples per list need ~0.8 GiB.
    let memory = ClusteringMemory::new(256_000, 1000, 768, 768, 1, None);
    assert_eq!(memory.samples_bytes, 256_000 * 768 * 4);
    assert_eq!(memory.centroids_bytes, 3 * 1000 * 768 * 4);
    assert_eq!(memory.check(MIB), Ok(()));
    // Neither is a large build with `kmeans_dimension` and many threads.
    let memory = ClusteringMemory::new(5_120_000, 160_000, 100, 768, 32, Some(5));
    assert_eq!(
        memory.centroids_bytes,
        34 * 160_000 * 100 * 4 + 6 * 160_000 * 769 * 4
    );
    assert_eq!(memory.check(maintenance_work_mem), Ok(()));
}
