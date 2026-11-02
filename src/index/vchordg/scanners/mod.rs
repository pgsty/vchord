// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.

mod default;

pub use default::DefaultBuilder;

#[derive(Debug)]
pub struct SearchOptions {
    pub ef_search: u32,
    pub beam_search: u32,
    pub max_scan_tuples: Option<u32>,
    pub io_search: crate::index::scanners::Io,
    pub io_rerank: crate::index::scanners::Io,
}
