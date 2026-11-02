// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use crate::tuples::*;
use crate::{Opaque, freepages};
use index::relation::{Page, PageGuard, RelationRead, RelationWrite};

pub fn consume<R: RelationRead + RelationWrite>(index: &R)
where
    R::Page: Page<Opaque = Opaque>,
{
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
    let freepages_first = meta_tuple.freepages_first();

    drop(meta_guard);

    let id = index
        .extend(
            Opaque {
                next: u32::MAX,
                skip: u32::MAX,
            },
            false,
        )
        .id();

    freepages::free(index, freepages_first, id);
}
