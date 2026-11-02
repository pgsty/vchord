// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use crate::Opaque;
use crate::tuples::*;
use index::relation::{Page, RelationWrite};

pub fn alloc<R: RelationWrite>(index: &R, freepages_first: u32) -> Option<R::WriteGuard<'_>>
where
    R::Page: Page<Opaque = Opaque>,
{
    let mut freepages_guard = index.write(freepages_first, false);
    let freepages_bytes = freepages_guard.get_mut(1).expect("data corruption");
    let mut freepages_tuple = FreepagesTuple::deserialize_mut(freepages_bytes);
    let id = *freepages_tuple.first();
    if id != u32::MAX {
        let mut guard = index.write(id, false);
        *freepages_tuple.first() = guard.get_opaque_mut().next;
        drop(freepages_guard); // write log of freespaces_guard
        Some(guard)
    } else {
        None
    }
}

// the page must be inaccessible in the graph
pub fn free<R: RelationWrite>(index: &R, freepages_first: u32, id: u32)
where
    R::Page: Page<Opaque = Opaque>,
{
    let mut guard = index.write(id, false);
    let mut freepages_guard = index.write(freepages_first, false);
    let freepages_bytes = freepages_guard.get_mut(1).expect("data corruption");
    let mut freepages_tuple = FreepagesTuple::deserialize_mut(freepages_bytes);
    guard.get_opaque_mut().next = *freepages_tuple.first();
    drop(guard); // write log of guard
    *freepages_tuple.first() = id;
    drop(freepages_guard); // write log of freepages_guard
}
