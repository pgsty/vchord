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
use std::cell::Cell;
use std::cmp::Reverse;
use std::collections::{HashSet, VecDeque};
use std::hash::Hash;
use vector::VectorOwned;

/// Deleted vertices an isolated non-start vertex may expand past 2 hops.
const DEEP_EXPANSION_CAP: usize = 64;

/// Deleted vertices an isolated start vertex may expand before falling back to
/// the live-vertex sample collected during the linear page scan.
const START_EXPANSION_CAP: usize = 512;

/// Visited-set cap for one local deleted-region expansion.
const MAX_VISITED: usize = 4096;

pub struct Maintained {
    /// Live (non-deleted) vertices.
    pub live: u64,
    /// Edges to freed vertex slots that repair dropped.
    pub stale_edges: u64,
    /// References to vertices whose tuples are damaged; repair skips them.
    pub broken: u64,
}

/// Repairs edges around deleted vertices, then frees deleted vertices and
/// unreachable vectors.
pub fn maintain<R: RelationRead + RelationWrite, O: Operator>(
    index: &R,
    check: impl Fn(),
) -> Maintained
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
        return Maintained {
            live: 0,
            stale_edges: 0,
            broken: 0,
        };
    };
    let stale_edges = Cell::new(0_u64);
    let broken = Cell::new(0_u64);
    let stale = || stale_edges.set(stale_edges.get() + 1);
    let damaged = || broken.set(broken.get() + 1);

    // Deterministic O(m) reservoir sample of live vertices seen during Phase 1,
    // used if the start vertex cannot reach a live vertex within its local cap.
    let mut live_fallback: Vec<((u32, u16), Vec<Pointer>)> = Vec::new();
    let mut seen_live = 0_u64;
    let mut rng = 0x9e37_79b9_7f4a_7c15_u64;
    let mut start_needs_fallback: Option<Vec<Pointer>> = None;

    // Phase 1: repair edges around deleted vertices.
    {
        let mut current = link;
        while current != u32::MAX {
            check();
            let vertex_guard = index.read(current);
            let mut members = Vec::new();
            for i in 1..=vertex_guard.len() {
                if let Some(vertex_bytes) = vertex_guard.get(i) {
                    let vertex_tuple = VertexTuple::deserialize_ref(vertex_bytes);
                    let live_u = vertex_tuple.payload().is_some();
                    let u = (vertex_guard.id(), i);
                    // The start vertex is never freed, even after deletion,
                    // because every search enters through it.
                    if live_u || u == s {
                        let pointers_u = vertex_tuple.pointers().to_vec();
                        if live_u && u != s {
                            seen_live += 1;
                            if live_fallback.len() < m as usize {
                                live_fallback.push((u, pointers_u.clone()));
                            } else if m > 0 {
                                rng = rng
                                    .wrapping_mul(6364136223846793005)
                                    .wrapping_add(1442695040888963407);
                                let slot = (rng >> 32) % seen_live;
                                if (slot as usize) < m as usize {
                                    live_fallback[slot as usize] = (u, pointers_u.clone());
                                }
                            }
                        }
                        members.push((pointers_u, u));
                    }
                }
            }
            let next = { vertex_guard }.get_opaque().next;
            for (pointers_u, u) in members {
                // Skip vertices whose neighbours are all live (#24).
                if all_neighbours_live::<R, O>(index, &pointers_u) {
                    continue;
                }
                'occ: loop {
                    check();
                    let Ok((vector_u, neighbours_u, _, version)) = crate::vectors::read::<R, O, _, _>(
                        by_read::<R>(index, pointers_u.iter().copied()),
                        CloneAccessor::<O::Vector>::default_with_dimension(dim),
                        copy_all,
                    ) else {
                        damaged();
                        break 'occ;
                    };
                    let trace = {
                        let mut trace = Vec::new();
                        let mut extend = Vec::new();
                        let mut visited = HashSet::from([u]);
                        for &(v, dis_v) in neighbours_u.iter() {
                            visited.insert(v);
                            let vertex_guard = index.read(v.0);
                            let Some(vertex_bytes) = vertex_guard.get(v.1) else {
                                stale();
                                continue;
                            };
                            let vertex_tuple = VertexTuple::deserialize_ref(vertex_bytes);
                            let pointers_v = vertex_tuple.pointers().to_vec();
                            let payload_v = vertex_tuple.payload();
                            drop(vertex_guard);
                            if payload_v.is_some() {
                                let Ok((vector_v, _, _, _)) = crate::vectors::read::<R, O, _, _>(
                                    by_read::<R>(index, pointers_v.iter().copied()),
                                    CloneAccessor::<O::Vector>::default_with_dimension(dim),
                                    copy_nothing,
                                ) else {
                                    damaged();
                                    continue;
                                };
                                trace.push((
                                    (Reverse(dis_v), AlwaysEqual((pointers_v, v))),
                                    vector_v,
                                ));
                            } else if let Ok((outs_v, _, _)) =
                                crate::vectors::read_without_accessor::<R, O, _>(
                                    (index, &pointers_v),
                                    copy_outs,
                                )
                            {
                                extend.extend(outs_v.iter().copied());
                            } else {
                                damaged();
                            }
                        }
                        // Second level of standard 2-hop ANN repair.
                        let mut deleted = Vec::new();
                        for v in extend {
                            if !visited.insert(v) {
                                continue;
                            }
                            let pointers_v = match probe(index, v) {
                                Probe::Missing => continue,
                                Probe::Deleted => {
                                    deleted.push(v);
                                    continue;
                                }
                                Probe::Live(pointers_v) => pointers_v,
                            };
                            let Ok((vector_v, _, _, _)) = crate::vectors::read::<R, O, _, _>(
                                by_read::<R>(index, pointers_v.iter().copied()),
                                CloneAccessor::<O::Vector>::default_with_dimension(dim),
                                copy_nothing,
                            ) else {
                                damaged();
                                continue;
                            };
                            let dis_v = O::distance(vector_u.as_borrowed(), vector_v.as_borrowed());
                            trace.push(((Reverse(dis_v), AlwaysEqual((pointers_v, v))), vector_v));
                        }
                        // Only expand past 2 hops when 2-hop repair found zero live
                        // neighbours, with a tight cap; never run unbounded BFS.
                        if trace.is_empty() && !deleted.is_empty() {
                            let cap = if u == s {
                                START_EXPANSION_CAP
                            } else {
                                DEEP_EXPANSION_CAP
                            };
                            let found = expand_deleted(
                                deleted,
                                &mut visited,
                                (cap, MAX_VISITED),
                                |v| probe(index, v),
                                |w| {
                                    check();
                                    let Some(pointers_w) = vertex_pointers(index, w) else {
                                        return Vec::new();
                                    };
                                    crate::vectors::read_without_accessor::<R, O, _>(
                                        (index, &pointers_w),
                                        copy_outs,
                                    )
                                    .map(|(outs, _, _)| Vec::from(outs))
                                    .unwrap_or_else(|_| {
                                        damaged();
                                        Vec::new()
                                    })
                                },
                            );
                            for (v, pointers_v) in found {
                                let Ok((vector_v, _, _, _)) = crate::vectors::read::<R, O, _, _>(
                                    by_read::<R>(index, pointers_v.iter().copied()),
                                    CloneAccessor::<O::Vector>::default_with_dimension(dim),
                                    copy_nothing,
                                ) else {
                                    damaged();
                                    continue;
                                };
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
                    if u == s && trace.is_empty() {
                        start_needs_fallback = Some(pointers_u.clone());
                        break 'occ;
                    }
                    let outs = crate::prune::prune(
                        |x, y| O::distance(x.as_borrowed(), y.as_borrowed()),
                        (pointers_u.to_vec(), u),
                        trace.into_iter(),
                        m,
                        &alpha,
                        |(_, u)| *u,
                        O::DISTANCE == DistanceKind::L2S,
                    );
                    match update::<R, O>(
                        (index, pointers_u.as_slice()),
                        (version, neighbours_u),
                        outs.iter()
                            .map(|&(Reverse(dis_u), AlwaysEqual((_, u)))| (u, dis_u)),
                    ) {
                        Ok(true) => break 'occ,
                        Ok(false) => {}
                        Err(()) => {
                            damaged();
                            break 'occ;
                        }
                    }
                }
            }
            current = next;
        }
    }

    // Reconnect an isolated start vertex directly to the O(m) live vertices
    // observed during Phase 1 so Phase 2 can always reclaim all dead space.
    if let Some(pointers_s) = start_needs_fallback {
        'occ: loop {
            check();
            let Ok((vector_s, neighbours_s, _, version)) = crate::vectors::read::<R, O, _, _>(
                by_read::<R>(index, pointers_s.iter().copied()),
                CloneAccessor::<O::Vector>::default_with_dimension(dim),
                copy_all,
            ) else {
                damaged();
                break 'occ;
            };
            let mut trace = Vec::with_capacity(live_fallback.len());
            for (v, pointers_v) in &live_fallback {
                let Ok((vector_v, _, _, _)) = crate::vectors::read::<R, O, _, _>(
                    by_read::<R>(index, pointers_v.iter().copied()),
                    CloneAccessor::<O::Vector>::default_with_dimension(dim),
                    copy_nothing,
                ) else {
                    continue;
                };
                let dis_v = O::distance(vector_s.as_borrowed(), vector_v.as_borrowed());
                trace.push((
                    (Reverse(dis_v), AlwaysEqual((pointers_v.clone(), *v))),
                    vector_v,
                ));
            }
            let outs = crate::prune::prune(
                |x, y| O::distance(x.as_borrowed(), y.as_borrowed()),
                (pointers_s.clone(), s),
                trace.into_iter(),
                m,
                &alpha,
                |(_, u)| *u,
                O::DISTANCE == DistanceKind::L2S,
            );
            match update::<R, O>(
                (index, pointers_s.as_slice()),
                (version, neighbours_s),
                outs.iter()
                    .map(|&(Reverse(dis_u), AlwaysEqual((_, u)))| (u, dis_u)),
            ) {
                Ok(true) => break 'occ,
                Ok(false) => {}
                Err(()) => {
                    damaged();
                    break 'occ;
                }
            }
        }
    }

    // Phase 2: remove deleted vertices and unreferenced vectors.
    let mut live = 0_u64;
    {
        let mut current = link;
        while current != u32::MAX {
            check();
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
                    let mut vector_guard = index.write(current, false);
                    for i in 1..=vector_guard.len() {
                        if vector_guard.get(i).is_some()
                            && reachable_set.binary_search(&(current, i)).is_err()
                        {
                            vector_guard.free(i);
                        }
                    }
                    current = vector_guard.get_opaque().next;
                }
            }
            current = vertex_guard.get_opaque().next;
        }
    }
    Maintained {
        live,
        stale_edges: stale_edges.get(),
        broken: broken.get(),
    }
}

/// Whether every neighbour in `pointers_u`'s list is live.
fn all_neighbours_live<R: RelationRead, O: Operator>(index: &R, pointers_u: &[Pointer]) -> bool
where
    R::Page: Page<Opaque = Opaque>,
{
    let Ok((neighbours, _, _)) =
        crate::vectors::read_without_accessor::<R, O, _>((index, pointers_u), copy_outs)
    else {
        return false;
    };
    neighbours.iter().all(|&v| {
        let vertex_guard = index.read(v.0);
        let Some(vertex_bytes) = vertex_guard.get(v.1) else {
            return false;
        };
        VertexTuple::deserialize_ref(vertex_bytes)
            .payload()
            .is_some()
    })
}

enum Probe<P> {
    Missing,
    Live(P),
    Deleted,
}

fn probe<R: RelationRead>(index: &R, v: (u32, u16)) -> Probe<Vec<Pointer>>
where
    R::Page: Page<Opaque = Opaque>,
{
    let vertex_guard = index.read(v.0);
    let Some(vertex_bytes) = vertex_guard.get(v.1) else {
        return Probe::Missing;
    };
    let vertex_tuple = VertexTuple::deserialize_ref(vertex_bytes);
    if vertex_tuple.payload().is_some() {
        Probe::Live(vertex_tuple.pointers().to_vec())
    } else {
        Probe::Deleted
    }
}

fn vertex_pointers<R: RelationRead>(index: &R, v: (u32, u16)) -> Option<Vec<Pointer>>
where
    R::Page: Page<Opaque = Opaque>,
{
    let vertex_guard = index.read(v.0);
    let vertex_bytes = vertex_guard.get(v.1)?;
    Some(
        VertexTuple::deserialize_ref(vertex_bytes)
            .pointers()
            .to_vec(),
    )
}

/// Bounded BFS through deleted vertices for live exits when 2-hop repair found none.
fn expand_deleted<V: Copy + Eq + Hash, P>(
    seeds: Vec<V>,
    visited: &mut HashSet<V>,
    (cap, max_visited): (usize, usize),
    mut probe: impl FnMut(V) -> Probe<P>,
    mut outs: impl FnMut(V) -> Vec<V>,
) -> Vec<(V, P)> {
    let mut queue = VecDeque::from(seeds);
    let mut found = Vec::new();
    let mut expanded = 0_usize;
    while let Some(w) = queue.pop_front() {
        if expanded >= cap {
            return found;
        }
        expanded += 1;
        for x in outs(w) {
            if visited.len() >= max_visited {
                return found;
            }
            if !visited.insert(x) {
                continue;
            }
            match probe(x) {
                Probe::Missing => {}
                Probe::Live(p) => found.push((x, p)),
                Probe::Deleted => queue.push_back(x),
            }
        }
        if !found.is_empty() {
            return found;
        }
    }
    found
}

pub(crate) struct InspectedVertexPage {
    pub reclaim: bool,
    pub live: u64,
    pub next: u32,
}

/// Inspects a vertex page and its vector pages under share locks.
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
/// other than the start vertex, or unreferenced vector tuples. Read-only (#24).
pub fn pending<R: RelationRead>(index: &R, check: impl Fn()) -> bool
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
        check();
        let inspected = inspect_vertex_page(index, current, s);
        if inspected.reclaim {
            return true;
        }
        current = inspected.next;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Graph {
        live: Vec<bool>,
        outs: Vec<Vec<usize>>,
    }

    impl Graph {
        fn new(n: usize) -> Self {
            Self {
                live: vec![false; n],
                outs: vec![Vec::new(); n],
            }
        }
        fn probe(&self, v: usize) -> Probe<usize> {
            match self.live.get(v) {
                None => Probe::Missing,
                Some(true) => Probe::Live(v),
                Some(false) => Probe::Deleted,
            }
        }
        fn expand(&self, u: usize, first: &[usize], cap: usize) -> Vec<usize> {
            self.expand_bounded(u, first, cap, usize::MAX)
        }
        fn expand_bounded(
            &self,
            u: usize,
            first: &[usize],
            cap: usize,
            max_visited: usize,
        ) -> Vec<usize> {
            let mut visited = HashSet::from([u]);
            visited.extend(first.iter().copied());
            let mut seeds = Vec::new();
            for &f in first {
                for &v in &self.outs[f] {
                    if visited.insert(v) && !self.live.get(v).copied().unwrap_or(true) {
                        seeds.push(v);
                    }
                }
            }
            let found = expand_deleted(
                seeds,
                &mut visited,
                (cap, max_visited),
                |v| self.probe(v),
                |w| self.outs.get(w).cloned().unwrap_or_default(),
            );
            let mut found = found.into_iter().map(|(v, _)| v).collect::<Vec<_>>();
            found.sort_unstable();
            found
        }
    }

    #[test]
    fn branching() {
        let mut g = Graph::new(7);
        g.live[5] = true;
        g.live[6] = true;
        g.outs[0] = vec![1];
        g.outs[1] = vec![2, 3];
        g.outs[2] = vec![4];
        g.outs[3] = vec![5];
        g.outs[4] = vec![6];
        assert_eq!(g.expand(0, &[1], 64), vec![5]);
    }

    #[test]
    fn long_chain_respects_cap() {
        let mut g = Graph::new(102);
        for v in 0..101 {
            g.outs[v] = vec![v + 1];
        }
        g.live[101] = true;
        assert_eq!(g.expand(0, &[1], 99), vec![101]);
        assert_eq!(g.expand(0, &[1], 98), Vec::<usize>::new());
    }

    #[test]
    fn pocket_without_exits() {
        let mut g = Graph::new(4);
        g.live[0] = true;
        g.outs[0] = vec![1];
        g.outs[1] = vec![2];
        g.outs[2] = vec![3, 9];
        g.outs[3] = vec![1, 0];
        assert_eq!(g.expand(0, &[1], 64), Vec::<usize>::new());
    }

    fn outs_of(r: &crate::test_support::TestRelation, n: u32) -> Vec<(u32, u16)> {
        let (outs, _, _) =
            crate::vectors::read_without_accessor::<_, crate::test_support::TestOp, _>(
                (r, &[Pointer::new((n + 2, 1))]),
                copy_outs,
            )
            .unwrap();
        Vec::from(outs)
    }

    #[test]
    fn start_vertex_reconnects_via_fallback_and_reclaims_all_deleted() {
        use crate::test_support::{TestOp, graph};
        // Start vertex 0 only points into a dead pocket 1 -> 2 -> 3 with no exits,
        // while live vertices 4 and 5 are unreachable from 0.
        let vertices: &[(bool, &[u16])] = &[
            (true, &[2]),
            (false, &[3]),
            (false, &[4]),
            (false, &[]),
            (true, &[]),
            (true, &[]),
        ];
        let r = graph(vertices, 4);
        let maintained = maintain::<_, TestOp>(&r, || {});
        assert_eq!(maintained.live, 3);
        let outs = outs_of(&r, 0);
        assert!(outs.contains(&(1, 5)));
        assert!(outs.contains(&(1, 6)));
        for i in 2..=4 {
            assert!(r.read(1).get(i).is_none(), "deleted vertex {i} was kept");
        }
        assert!(!pending(&r, || {}));
    }

    #[test]
    fn stale_edges_are_dropped_and_counted() {
        use crate::test_support::{TestOp, graph};
        let vertices: &[(bool, &[u16])] =
            &[(true, &[2, 3]), (false, &[3]), (true, &[]), (true, &[])];
        let r = graph(vertices, 4);
        r.write(1, false).free(3);
        let maintained = maintain::<_, TestOp>(&r, || {});
        assert_eq!(maintained.stale_edges, 1);
        assert_eq!(maintained.broken, 0);
        assert!(!outs_of(&r, 0).contains(&(1, 3)));
        assert!(r.read(1).get(2).is_none());
        assert!(!pending(&r, || {}));
    }
}
