// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use crate::Opaque;
use crate::operator::Operator;
use crate::tuples::{MetaTuple, VertexTuple, WithReader, WithWriter};
use index::relation::{Page, RelationRead, RelationWrite};
use std::num::NonZero;

/// Clears the payload of every vertex whose heap tuple is dead. Returns the
/// number of vertices that were cleared.
pub fn bulkdelete<R: RelationRead + RelationWrite, O: Operator>(
    index: &R,
    check: impl Fn(),
    callback: impl Fn(NonZero<u64>) -> bool,
) -> u64
where
    R::Page: Page<Opaque = Opaque>,
{
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
    let start = meta_tuple.start();
    let link = meta_guard.get_opaque().link;
    drop(meta_guard);
    let Some(_) = start.into_inner() else {
        return 0;
    };
    let mut removed = 0_u64;
    let mut current = link;
    while current != u32::MAX {
        check();
        let read = index.read(current);
        let flag = 'flag: {
            for i in 1..=read.len() {
                if let Some(bytes) = read.get(i) {
                    let tuple = VertexTuple::deserialize_ref(bytes);
                    let p = tuple.payload();
                    if Some(true) == p.map(&callback) {
                        break 'flag true;
                    }
                }
            }
            false
        };
        if flag {
            drop(read);
            let mut write = index.write(current, false);
            for i in 1..=write.len() {
                if let Some(bytes) = write.get_mut(i) {
                    let mut tuple = VertexTuple::deserialize_mut(bytes);
                    let p = tuple.payload();
                    if Some(true) == p.map(&callback) {
                        *p = None;
                        removed += 1;
                    }
                }
            }
            current = write.get_opaque().next;
        } else {
            current = read.get_opaque().next;
        }
    }
    removed
}
