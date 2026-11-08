// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

mod build;
mod bulkdelete;
mod candidates;
mod insert;
mod maintain;
mod prewarm;
mod prune;
mod results;
mod search;
mod tuples;
mod vectors;
mod visited;

pub mod operator;
pub mod types;

pub use build::build;
pub use bulkdelete::bulkdelete;
pub use insert::{InsertError, insert};
pub use maintain::{Maintained, Reachability, maintain, pending};
pub use prewarm::prewarm;
pub use search::search;

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

#[repr(C, align(8))]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, Immutable, KnownLayout)]
pub struct Opaque {
    pub next: u32,
    pub link: u32,
}

#[allow(unsafe_code)]
unsafe impl index::relation::Opaque for Opaque {}

pub type Id = (u32, u16);

impl index::fetch::Fetch1 for tuples::Pointer {
    fn fetch_1(&self) -> u32 {
        self.into_inner().0
    }
}

#[cfg(test)]
mod test_support;
