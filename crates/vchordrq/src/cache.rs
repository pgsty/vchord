// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use crate::list::walk;
use crate::tuples::{MetaTuple, WithReader};
use crate::{Opaque, tape};
use index::relation::{Page, PageGuard, RelationRead};

pub fn cache<R: RelationRead>(index: &R, level: i32) -> Vec<u32>
where
    R::Page: Page<Opaque = Opaque>,
{
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
    let centroids_first = meta_tuple.centroids_first();
    drop(meta_guard);

    let mut trace = Vec::new();

    if level >= 0 {
        // meta tuple
        trace.push(0);
    }

    if level >= 1 {
        // h1 tuples, then jump tuples
        let firsts = walk(index, || (), |id| trace.push(id));
        trace.extend(firsts);
    }

    if level >= 2 {
        // centroid tuples
        for guard in tape::by_next(index, centroids_first) {
            trace.push(guard.id());
        }
    }

    trace
}
