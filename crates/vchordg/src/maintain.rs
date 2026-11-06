// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use crate::Opaque;
use crate::operator::{CloneAccessor, Operator};
use crate::tuples::{MetaTuple, Pointer, VertexTuple, WithReader};
use crate::types::DistanceKind;
use crate::vectors::{by_read, copy_all, copy_nothing, copy_outs, update};
use always_equal::AlwaysEqual;
use index::relation::{Page, PageGuard, RelationRead, RelationWrite};
use index_accessor::DefaultWithDimension;
use std::cmp::Reverse;
use vector::VectorOwned;

/// Repairs edges around deleted vertices, then frees deleted vertices and
/// unreachable vectors. Returns the number of live (non-deleted) vertices.
pub fn maintain<R: RelationRead + RelationWrite, O: Operator>(index: &R, check: impl Fn()) -> u64
where
    R::Page: Page<Opaque = Opaque>,
{
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
    let dim = meta_tuple.dim();
    let m = meta_tuple.m();
    let alpha = meta_tuple.alpha().to_vec();
    let start = meta_tuple.start();
    let link = meta_guard.get_opaque().link;
    drop(meta_guard);
    let Some(s) = start.into_inner() else {
        return 0;
    };
    // do it's best to remove broken edges
    {
        let mut current = link;
        while current != u32::MAX {
            check();
            let vertex_guard = index.read(current);
            let mut members = Vec::new();
            for i in 1..=vertex_guard.len() {
                if let Some(vertex_bytes) = vertex_guard.get(i) {
                    let vertex_tuple = VertexTuple::deserialize_ref(vertex_bytes);
                    // The start vertex is never freed, even after it is deleted,
                    // because every search enters the graph through it. Keep its
                    // edges pointing at live vertices so the graph stays reachable.
                    if vertex_tuple.payload().is_some() || (vertex_guard.id(), i) == s {
                        let pointers_u = vertex_tuple.pointers().to_vec();
                        members.push((pointers_u, (vertex_guard.id(), i)));
                    }
                }
            }
            let next = { vertex_guard }.get_opaque().next;
            for (pointers_u, u) in members {
                // A list whose neighbours are all live is left as it is:
                // pruning an unchanged list reproduces it, so reading the
                // neighbours' vectors and rewriting it is wasted work (#24).
                if neighbours_all_live::<R, O>(index, &pointers_u) {
                    continue;
                }
                'occ: loop {
                    let Ok((vector_u, neighbours_u, _, version)) = crate::vectors::read::<R, O, _, _>(
                        by_read::<R>(index, pointers_u.iter().copied()),
                        CloneAccessor::<O::Vector>::default_with_dimension(dim),
                        copy_all,
                    ) else {
                        // the link is broken
                        break 'occ;
                    };
                    let trace = {
                        let mut trace = Vec::new();
                        let mut extend = Vec::new();
                        for &(v, dis_v) in neighbours_u.iter() {
                            let vertex_guard = index.read(v.0);
                            let Some(vertex_bytes) = vertex_guard.get(v.1) else {
                                // the link is broken
                                continue;
                            };
                            let vertex_tuple = VertexTuple::deserialize_ref(vertex_bytes);
                            let pointers_v = vertex_tuple.pointers().to_vec();
                            let payload_v = vertex_tuple.payload();
                            drop(vertex_guard);
                            let Ok((vector_v, outs_v, _, _)) = crate::vectors::read::<R, O, _, _>(
                                by_read::<R>(index, pointers_v.iter().copied()),
                                CloneAccessor::<O::Vector>::default_with_dimension(dim),
                                copy_outs,
                            ) else {
                                // the link is broken
                                continue;
                            };
                            if payload_v.is_some() {
                                trace.push((
                                    (Reverse(dis_v), AlwaysEqual((pointers_v, v))),
                                    vector_v,
                                ));
                            } else {
                                extend.extend(outs_v.iter().copied());
                            }
                        }
                        for v in extend {
                            let vertex_guard = index.read(v.0);
                            let Some(vertex_bytes) = vertex_guard.get(v.1) else {
                                // the link is broken
                                continue;
                            };
                            let vertex_tuple = VertexTuple::deserialize_ref(vertex_bytes);
                            let pointers_v = vertex_tuple.pointers().to_vec();
                            let payload_v = vertex_tuple.payload();
                            drop(vertex_guard);
                            let Ok((vector_v, _, _, _)) = crate::vectors::read::<R, O, _, _>(
                                by_read::<R>(index, pointers_v.iter().copied()),
                                CloneAccessor::<O::Vector>::default_with_dimension(dim),
                                copy_nothing,
                            ) else {
                                // the link is broken
                                continue;
                            };
                            if payload_v.is_some() {
                                let dis_v =
                                    O::distance(vector_u.as_borrowed(), vector_v.as_borrowed());
                                trace.push((
                                    (Reverse(dis_v), AlwaysEqual((pointers_v, v))),
                                    vector_v,
                                ));
                            }
                        }
                        trace
                    };
                    let outs = crate::prune::prune(
                        |x, y| O::distance(x.as_borrowed(), y.as_borrowed()),
                        (pointers_u.to_vec(), u),
                        trace.into_iter(),
                        m,
                        &alpha,
                        |(_, u)| *u,
                        O::DISTANCE == DistanceKind::L2S,
                    );
                    if update::<R, O>(
                        (index, pointers_u.as_slice()),
                        (version, neighbours_u),
                        outs.iter()
                            .map(|&(Reverse(dis_u), AlwaysEqual((_, u)))| (u, dis_u)),
                    ) != Ok(false)
                    {
                        break 'occ;
                    }
                }
            }
            current = next;
        }
    }
    let mut live = 0_u64;
    // remove vertices and vectors
    {
        let mut current = link;
        while current != u32::MAX {
            check();
            // Decide under share locks whether this page has anything to free;
            // only then take the write locks, re-evaluating there (#24).
            let inspected = inspect_vertex_page(index, current, s);
            if !inspected.reclaim {
                live += inspected.live;
                current = inspected.next;
                continue;
            }
            let mut vertex_guard = index.write(current, true);
            let mut reachable_set = Vec::<(u32, u16)>::new();
            for i in 1..=vertex_guard.len() {
                if let Some(bytes) = vertex_guard.get(i) {
                    let tuple = VertexTuple::deserialize_ref(bytes);
                    let p = tuple.payload();
                    if p.is_none() && (current, i) != s {
                        vertex_guard.free(i);
                    } else {
                        live += u64::from(p.is_some());
                        let iter = tuple.pointers().iter().map(|pointer| pointer.into_inner());
                        reachable_set.extend(iter);
                    }
                }
            }
            reachable_set.sort_unstable();
            {
                let mut current = vertex_guard.get_opaque().link;
                while current != u32::MAX {
                    check();
                    let mut vector_guard = index.write(current, false);
                    for i in 1..=vector_guard.len() {
                        if vector_guard.get(i).is_some() {
                            if reachable_set.binary_search(&(current, i)).is_err() {
                                vector_guard.free(i);
                            }
                        }
                    }
                    current = vector_guard.get_opaque().next;
                }
            }
            current = vertex_guard.get_opaque().next;
        }
    }
    live
}

fn neighbours_all_live<R: RelationRead, O: Operator>(index: &R, pointers_u: &[Pointer]) -> bool
where
    R::Page: Page<Opaque = Opaque>,
{
    let Ok((neighbours, _, _)) =
        crate::vectors::read_without_accessor::<R, O, _>((index, pointers_u), copy_all)
    else {
        return false;
    };
    neighbours.iter().all(|&(v, _)| {
        let vertex_guard = index.read(v.0);
        let Some(vertex_bytes) = vertex_guard.get(v.1) else {
            return false;
        };
        VertexTuple::deserialize_ref(vertex_bytes)
            .payload()
            .is_some()
    })
}

pub(crate) struct InspectedVertexPage {
    /// The page holds a deleted vertex (other than the start vertex) or a
    /// vector tuple that none of its vertices reference.
    pub reclaim: bool,
    /// Live (non-deleted) vertices on the page.
    pub live: u64,
    pub next: u32,
}

/// Inspects a vertex page and its vector pages under share locks. The vertex
/// page stays share-locked while its vector pages are read: inserts write
/// their vector tuples while holding the vertex page exclusively, so none can
/// be observed half-inserted.
pub(crate) fn inspect_vertex_page<R: RelationRead>(
    index: &R,
    id: u32,
    s: (u32, u16),
) -> InspectedVertexPage
where
    R::Page: Page<Opaque = Opaque>,
{
    let vertex_guard = index.read(id);
    let mut reachable_set = Vec::<(u32, u16)>::new();
    let mut live = 0_u64;
    let mut reclaim = false;
    for i in 1..=vertex_guard.len() {
        if let Some(bytes) = vertex_guard.get(i) {
            let tuple = VertexTuple::deserialize_ref(bytes);
            let p = tuple.payload();
            if p.is_none() && (id, i) != s {
                reclaim = true;
            } else {
                live += u64::from(p.is_some());
                reachable_set.extend(tuple.pointers().iter().map(|pointer| pointer.into_inner()));
            }
        }
    }
    let opaque = *vertex_guard.get_opaque();
    if !reclaim {
        reachable_set.sort_unstable();
        let mut current = opaque.link;
        'vectors: while current != u32::MAX {
            let vector_guard = index.read(current);
            for i in 1..=vector_guard.len() {
                if vector_guard.get(i).is_some()
                    && reachable_set.binary_search(&(current, i)).is_err()
                {
                    reclaim = true;
                    break 'vectors;
                }
            }
            current = vector_guard.get_opaque().next;
        }
    }
    drop(vertex_guard);
    InspectedVertexPage {
        reclaim,
        live,
        next: opaque.next,
    }
}

/// Whether a VACUUM that deleted nothing still has work: deleted vertices
/// other than the start vertex, or vector tuples left unreferenced (for
/// example by an interrupted earlier VACUUM). Read-only (#24).
pub fn pending<R: RelationRead>(index: &R) -> bool
where
    R::Page: Page<Opaque = Opaque>,
{
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
    let start = meta_tuple.start();
    let link = meta_guard.get_opaque().link;
    drop(meta_guard);
    let Some(s) = start.into_inner() else {
        return false;
    };
    let mut current = link;
    while current != u32::MAX {
        let inspected = inspect_vertex_page(index, current, s);
        if inspected.reclaim {
            return true;
        }
        current = inspected.next;
    }
    false
}
