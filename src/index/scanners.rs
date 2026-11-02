// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.

use crate::index::fetcher::Fetcher;
use crate::recorder::Recorder;
use index::bump::Bump;
use index::relation::{Page, RelationPrefetch, RelationRead, RelationReadStream};
use pgrx::pg_sys::Datum;

#[derive(Debug, Clone, Copy)]
pub enum Io {
    Plain,
    Simple,
    #[cfg_attr(
        any(feature = "pg14", feature = "pg15", feature = "pg16"),
        expect(dead_code)
    )]
    Stream,
}

pub trait SearchBuilder: 'static {
    type Options;

    type Opfamily;

    type Opaque: Copy;

    fn new(opfamily: Self::Opfamily) -> Self;

    unsafe fn add(&mut self, strategy: u16, datum: Option<Datum>);

    fn build<'b, R>(
        self,
        relation: &'b R,
        options: Self::Options,
        fetcher: impl Fetcher + 'b,
        bump: &'b impl Bump,
        recorder: impl Recorder,
    ) -> Box<dyn Iterator<Item = (f32, [u16; 3], bool)> + 'b>
    where
        R: RelationRead + RelationPrefetch + RelationReadStream,
        R::Page: Page<Opaque = Self::Opaque>;
}
