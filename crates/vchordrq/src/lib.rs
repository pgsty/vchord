// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

mod build;
mod bulkdelete;
mod cache;
mod centroids;
mod closure_lifetime_binder;
mod consume;
mod cost;
mod fast_heap;
mod freepages;
mod insert;
mod linked_vec;
mod maintain;
mod prewarm;
mod rerank;
mod search;
mod tape;
mod tape_writer;
mod tuples;
mod vectors;

pub mod operator;
pub mod types;

pub use build::build;
pub use bulkdelete::{bulkdelete, bulkdelete_vectors};
pub use cache::cache;
pub use consume::consume;
pub use cost::cost;
pub use fast_heap::FastHeap;
pub use insert::{InsertChooser, insert, insert_vector};
pub use maintain::{MaintainChooser, maintain};
pub use prewarm::prewarm;
pub use rerank::{how, rerank_heap, rerank_index};
pub use search::{default_search, maxsim_search};

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

#[repr(C, align(8))]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, Immutable, KnownLayout)]
pub struct Opaque {
    pub next: u32,
    pub skip: u32,
}

#[allow(unsafe_code)]
unsafe impl index::relation::Opaque for Opaque {}

pub(crate) struct Branch<T> {
    pub code: rabitq::bit::Code,
    pub delta: f32,
    pub prefetch: Vec<u32>,
    pub head: u16,
    pub norm: f32,
    pub extra: T,
}

#[derive(Debug, Clone, Copy)]
pub enum RerankMethod {
    Index,
    Heap,
}
