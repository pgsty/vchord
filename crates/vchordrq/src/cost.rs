// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use crate::tuples::{MetaTuple, WithReader};
use index::relation::{Page, RelationRead};

pub struct Cost {
    pub dim: u32,
    pub cells: Vec<u32>,
}

#[must_use]
pub fn cost<R: RelationRead>(index: &R) -> Cost {
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
    let dim = meta_tuple.dim();
    let cells = meta_tuple.cells().to_vec();

    drop(meta_guard);

    Cost { dim, cells }
}
