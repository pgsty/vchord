// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.

use crate::Opaque;
use crate::operator::Operator;
use index::relation::{Page, RelationRead};

pub fn prewarm<R: RelationRead, O: Operator>(index: &R) -> String
where
    R::Page: Page<Opaque = Opaque>,
{
    use std::fmt::Write;
    let meta_guard = index.read(0);
    let link = meta_guard.get_opaque().link;
    drop(meta_guard);
    let mut number_of_vertex_pages = 0_u64;
    let mut number_of_vertices = 0_u64;
    {
        let mut current = link;
        while current != u32::MAX {
            let vertex_guard = index.read(current);
            number_of_vertex_pages += 1;
            for i in 1..=vertex_guard.len() {
                if vertex_guard.get(i).is_some() {
                    number_of_vertices += 1;
                }
            }
            current = vertex_guard.get_opaque().next;
        }
    }
    let mut message = String::new();
    writeln!(message, "number of vertex pages: {number_of_vertex_pages}").unwrap();
    writeln!(message, "number of vertices: {number_of_vertices}").unwrap();
    message
}
