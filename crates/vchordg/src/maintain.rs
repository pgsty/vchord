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
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::Hash;
use vector::VectorOwned;

/// Deleted vertices a non-start vertex may expand while looking for a live
/// neighbour (#21). The start vertex has no such cap; both are bounded by
/// `maintenance_work_mem` through `EXPANSION_BYTES_PER_VERTEX`.
const DEEP_EXPANSION_CAP: usize = 4096;

/// Estimated bytes per vertex an expansion visits: its entry in the visited
/// set (with the hash table's spare capacity) and in the queue.
const EXPANSION_BYTES_PER_VERTEX: usize = 48;

pub struct Maintained {
    /// Live (non-deleted) vertices.
    pub live: u64,
    pub reachability: Reachability,
    /// Set when deleted vertices were kept instead of freed, and why. A later
    /// VACUUM retries; `pending` reports the index as having work left.
    pub retained: Option<&'static str>,
    /// Edges to freed vertex slots that repair dropped. Such an edge leads
    /// nowhere, so dropping it cannot cut a path. Releases up to 1.2.0 could
    /// leave them behind when an insertion raced with VACUUM.
    pub stale_edges: u64,
    /// References to vertices whose tuples are damaged: a vertex whose vectors
    /// cannot be read, or a neighbour list that disappeared while it was being
    /// rewritten. Repair skips them; the index needs REINDEX.
    pub broken: u64,
}

/// Advisory reachability check (#21): compares what the start vertex reaches
/// over the neighbour lists before edge repair (through deleted vertices,
/// which searches traverse until they are freed) and after it (live vertices
/// only). Vertices inserted concurrently are not accounted for.
pub enum Reachability {
    Checked {
        /// Live vertices reachable before the repair but not after it.
        disconnected: u64,
        /// Live vertices not reachable before the repair. Graphs can have
        /// these without any deletion, so they are reported separately.
        unreachable_before: u64,
        live: u64,
    },
    /// The check was skipped for the given reason.
    Skipped(&'static str),
}

/// Repairs edges around deleted vertices, then frees deleted vertices and
/// unreachable vectors. `memory_limit` (bytes) bounds the neighbour lists
/// recorded for the reachability check.
pub fn maintain<R: RelationRead + RelationWrite, O: Operator>(
    index: &R,
    check: impl Fn(),
    memory_limit: usize,
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
            reachability: Reachability::Skipped("the index is empty"),
            retained: None,
            stale_edges: 0,
            broken: 0,
        };
    };
    let stale_edges = Cell::new(0_u64);
    let broken = Cell::new(0_u64);
    let stale = || stale_edges.set(stale_edges.get() + 1);
    let damaged = || broken.set(broken.get() + 1);
    let mut retained = None;
    // The recorded lists and one expansion share `maintenance_work_mem`,
    // half each. The expansion bound covers its visited set and queue, the
    // start vertex's included.
    let mut recorder = Recorder::new(memory_limit / 2);
    let max_visited = (memory_limit / 2 / EXPANSION_BYTES_PER_VERTEX).max(1);
    // do it's best to remove broken edges
    {
        let mut current = link;
        while current != u32::MAX {
            check();
            let vertex_guard = index.read(current);
            let mut members = Vec::new();
            let mut deleted_members = Vec::new();
            for i in 1..=vertex_guard.len() {
                if let Some(vertex_bytes) = vertex_guard.get(i) {
                    let vertex_tuple = VertexTuple::deserialize_ref(vertex_bytes);
                    let live_u = vertex_tuple.payload().is_some();
                    let pointers_u = vertex_tuple.pointers().to_vec();
                    // The start vertex is never freed, even after it is deleted,
                    // because every search enters the graph through it. Keep its
                    // edges pointing at live vertices so the graph stays reachable.
                    if live_u || (vertex_guard.id(), i) == s {
                        members.push((pointers_u, (vertex_guard.id(), i), live_u));
                    } else {
                        deleted_members.push((pointers_u, (vertex_guard.id(), i)));
                    }
                }
            }
            let next = { vertex_guard }.get_opaque().next;
            if recorder.active() {
                for (pointers_u, u) in deleted_members {
                    let outs_u = crate::vectors::read_without_accessor::<R, O, _>(
                        (index, &pointers_u),
                        copy_outs,
                    )
                    .map(|(outs, _, _)| Vec::from(outs))
                    .unwrap_or_default();
                    recorder.record_before(u, outs_u);
                }
            }
            for (pointers_u, u, live_u) in members {
                // A list whose neighbours are all live is left as it is:
                // pruning an unchanged list reproduces it, so reading the
                // neighbours' vectors and rewriting it is wasted work (#24).
                if let Some(outs_u) = live_neighbours::<R, O>(index, &pointers_u) {
                    recorder.record_before(u, outs_u.clone());
                    recorder.record_after(u, live_u, outs_u);
                    continue;
                }
                let mut before = true;
                'occ: loop {
                    // Retried while concurrent insertions change the list.
                    check();
                    let Ok((vector_u, neighbours_u, _, version)) = crate::vectors::read::<R, O, _, _>(
                        by_read::<R>(index, pointers_u.iter().copied()),
                        CloneAccessor::<O::Vector>::default_with_dimension(dim),
                        copy_all,
                    ) else {
                        // The vertex tuple exists but its vectors do not.
                        damaged();
                        recorder.skip("a broken link was found");
                        break 'occ;
                    };
                    if std::mem::take(&mut before) {
                        recorder.record_before(u, neighbours_u.iter().map(|&(v, _)| v).collect());
                    }
                    let trace = {
                        let mut trace = Vec::new();
                        let mut extend = Vec::new();
                        let mut visited = HashSet::from([u]);
                        for &(v, dis_v) in neighbours_u.iter() {
                            visited.insert(v);
                            let vertex_guard = index.read(v.0);
                            let Some(vertex_bytes) = vertex_guard.get(v.1) else {
                                // A stale edge to a freed slot: dropped.
                                stale();
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
                                damaged();
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
                        // Second level. Vertices seen already (`u` itself, the
                        // first level, repeats) would only be deduplicated or
                        // removed by `prune`, so they are skipped.
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
                        // Fill spare degree through deleted paths even when a
                        // nearby live neighbour was found: it need not lead to
                        // the live vertices beyond a deleted bridge (#21).
                        // Keep the bounded, first-exit search and usual pruning;
                        // a full candidate list needs no extra expansion. This
                        // is local ANN repair, not a connectivity guarantee.
                        let mut cut_off = false;
                        if trace.len() < m as usize && !deleted.is_empty() {
                            let cap = if u == s {
                                None
                            } else {
                                Some(DEEP_EXPANSION_CAP)
                            };
                            let (found, exhausted) = expand_deleted(
                                deleted,
                                &mut visited,
                                (cap, max_visited),
                                |v| probe(index, v),
                                |w| {
                                    check();
                                    let Some(pointers_w) = vertex_pointers(index, w) else {
                                        // the link is broken
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
                            // Every search enters through the start vertex. If
                            // the memory bound stopped its expansion before a
                            // live vertex turned up, an empty list would cut
                            // off the whole graph; keep its list and every
                            // deleted vertex instead, for a VACUUM with more
                            // memory. Other vertices keep the capped
                            // behaviour: their in-edges, not their own list,
                            // keep them reachable.
                            cut_off = u == s && trace.is_empty() && found.is_empty() && !exhausted;
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
                        (trace, cut_off)
                    };
                    let (trace, cut_off) = trace;
                    if cut_off {
                        retained = Some(
                            "the start vertex could not be reconnected to a live vertex within maintenance_work_mem",
                        );
                        recorder.skip("deleted vertices were kept");
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
                    let outs_u = outs
                        .iter()
                        .map(|&(_, AlwaysEqual((_, v)))| v)
                        .collect::<Vec<_>>();
                    match update::<R, O>(
                        (index, pointers_u.as_slice()),
                        (version, neighbours_u),
                        outs.iter()
                            .map(|&(Reverse(dis_u), AlwaysEqual((_, u)))| (u, dis_u)),
                    ) {
                        Ok(true) => {
                            recorder.record_after(u, live_u, outs_u);
                            break 'occ;
                        }
                        Ok(false) => {}
                        Err(()) => {
                            // The vectors of `u` disappeared while it was
                            // being rewritten.
                            damaged();
                            recorder.skip("a broken link was found");
                            break 'occ;
                        }
                    }
                }
            }
            current = next;
        }
    }
    let reachability = recorder.finish(s);
    let mut live = 0_u64;
    if retained.is_some() {
        // Count only; nothing is freed until a VACUUM can reconnect the start
        // vertex. Vectors left unreferenced also wait for that pass.
        let mut current = link;
        while current != u32::MAX {
            check();
            let inspected = inspect_vertex_page(index, current, s);
            live += inspected.live;
            current = inspected.next;
        }
        return Maintained {
            live,
            reachability,
            retained,
            stale_edges: stale_edges.get(),
            broken: broken.get(),
        };
    }
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
                // No `check()` here: the vertex page is locked exclusively, and
                // a cost-based sleep or an interrupt must not happen under it.
                let mut current = vertex_guard.get_opaque().link;
                while current != u32::MAX {
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
    Maintained {
        live,
        reachability,
        retained,
        stale_edges: stale_edges.get(),
        broken: broken.get(),
    }
}

/// The neighbour list of a vertex, if every neighbour is live.
fn live_neighbours<R: RelationRead, O: Operator>(
    index: &R,
    pointers_u: &[Pointer],
) -> Option<Vec<(u32, u16)>>
where
    R::Page: Page<Opaque = Opaque>,
{
    let (neighbours, _, _) =
        crate::vectors::read_without_accessor::<R, O, _>((index, pointers_u), copy_all).ok()?;
    let all_live = neighbours.iter().all(|&(v, _)| {
        let vertex_guard = index.read(v.0);
        let Some(vertex_bytes) = vertex_guard.get(v.1) else {
            return false;
        };
        VertexTuple::deserialize_ref(vertex_bytes)
            .payload()
            .is_some()
    });
    all_live.then(|| neighbours.into_iter().map(|(v, _)| v).collect())
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
        // the link is broken
        return Probe::Missing;
    };
    let vertex_tuple = VertexTuple::deserialize_ref(vertex_bytes);
    let pointers = vertex_tuple.pointers().to_vec();
    if vertex_tuple.payload().is_some() {
        Probe::Live(pointers)
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

/// Breadth-first search through deleted vertices, starting from `seeds`
/// (deleted vertices), for live vertices not in `visited`. Stops after the
/// first expanded vertex that leads to a live vertex, when the reachable
/// deleted region is exhausted, after expanding `cap` deleted vertices, or
/// once `visited` holds `max_visited` vertices; returns the live vertices
/// found (#21), and whether the region was exhausted (so that an empty
/// result means no live vertex is reachable, rather than that a bound
/// stopped the search). The queue holds vertex ids only; `outs` reads a
/// vertex's neighbours when it is expanded.
fn expand_deleted<V: Copy + Eq + Hash, P>(
    seeds: Vec<V>,
    visited: &mut HashSet<V>,
    (cap, max_visited): (Option<usize>, usize),
    mut probe: impl FnMut(V) -> Probe<P>,
    mut outs: impl FnMut(V) -> Vec<V>,
) -> (Vec<(V, P)>, bool) {
    let mut queue = VecDeque::from(seeds);
    let mut found = Vec::new();
    let mut expanded = 0_usize;
    while let Some(w) = queue.pop_front() {
        if cap.is_some_and(|cap| expanded >= cap) {
            return (found, false);
        }
        expanded += 1;
        for x in outs(w) {
            if visited.len() >= max_visited {
                return (found, false);
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
            return (found, false);
        }
    }
    (found, true)
}

/// Neighbour lists seen by edge repair, within a memory limit: before the
/// repair for every vertex (deleted ones included), after it for the
/// vertices the repair keeps (live vertices and the start vertex).
struct Recorder {
    before: HashMap<(u32, u16), Vec<(u32, u16)>>,
    after: HashMap<(u32, u16), (bool, Vec<(u32, u16)>)>,
    bytes: usize,
    limit: usize,
    skipped: Option<&'static str>,
}

impl Recorder {
    /// Estimated bytes per recorded list besides its elements: the map slot
    /// with the hash table's spare capacity (up to about 2.3 slots per entry
    /// after growth), the list's allocation overhead, and the vertex's share
    /// of the visited sets and queues that the check allocates.
    const ENTRY: usize = 192;

    fn new(limit: usize) -> Self {
        Self {
            before: HashMap::new(),
            after: HashMap::new(),
            bytes: 0,
            limit,
            skipped: None,
        }
    }
    fn active(&self) -> bool {
        self.skipped.is_none()
    }
    fn charge(&mut self, outs: &[(u32, u16)]) -> bool {
        if self.skipped.is_some() {
            return false;
        }
        self.bytes = self
            .bytes
            .saturating_add(Self::ENTRY + size_of_val(outs).next_multiple_of(16));
        if self.bytes > self.limit {
            self.skip("maintenance_work_mem is too small to hold the neighbour lists");
            return false;
        }
        true
    }
    fn record_before(&mut self, u: (u32, u16), mut outs: Vec<(u32, u16)>) {
        if self.charge(&outs) {
            outs.shrink_to_fit();
            self.before.insert(u, outs);
        }
    }
    fn record_after(&mut self, u: (u32, u16), live: bool, mut outs: Vec<(u32, u16)>) {
        if self.charge(&outs) {
            outs.shrink_to_fit();
            self.after.insert(u, (live, outs));
        }
    }
    fn skip(&mut self, reason: &'static str) {
        if self.skipped.is_none() {
            self.skipped = Some(reason);
            self.before = HashMap::new();
            self.after = HashMap::new();
        }
    }
    fn finish(self, s: (u32, u16)) -> Reachability {
        if let Some(reason) = self.skipped {
            return Reachability::Skipped(reason);
        }
        if !self.before.contains_key(&s) || !self.after.contains_key(&s) {
            return Reachability::Skipped("the start vertex was not found");
        }
        compare_reachability(s, &self.before, &self.after)
    }
}

/// Vertices reachable from `start`, following only vertices `outs` knows.
fn reachable<'a, V: Copy + Eq + Hash + 'a>(
    start: V,
    outs: impl Fn(V) -> Option<&'a [V]>,
) -> HashSet<V> {
    let mut visited = HashSet::from([start]);
    let mut queue = VecDeque::from([start]);
    while let Some(v) = queue.pop_front() {
        for &x in outs(v).unwrap_or_default() {
            if outs(x).is_some() && visited.insert(x) {
                queue.push_back(x);
            }
        }
    }
    visited
}

fn compare_reachability<V: Copy + Eq + Hash>(
    start: V,
    before: &HashMap<V, Vec<V>>,
    after: &HashMap<V, (bool, Vec<V>)>,
) -> Reachability {
    let reached_before = reachable(start, |v| before.get(&v).map(Vec::as_slice));
    let reached_after = reachable(start, |v| after.get(&v).map(|(_, outs)| outs.as_slice()));
    let (mut disconnected, mut unreachable_before, mut live) = (0, 0, 0);
    for (v, (live_v, _)) in after {
        if !live_v {
            continue;
        }
        live += 1;
        match (reached_before.contains(v), reached_after.contains(v)) {
            (false, _) => unreachable_before += 1,
            (true, false) => disconnected += 1,
            (true, true) => {}
        }
    }
    Reachability::Checked {
        disconnected,
        unreachable_before,
        live,
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A test graph: `live[v]` and `outs[v]`; vertices beyond `live` are
    /// missing.
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
        fn expand(&self, u: usize, first: &[usize], cap: Option<usize>) -> Vec<usize> {
            self.expand_bounded(u, first, cap, usize::MAX)
        }
        /// Expands from `u`, whose neighbours `first` were all deleted and
        /// were expanded once already (two-level repair).
        fn expand_bounded(
            &self,
            u: usize,
            first: &[usize],
            cap: Option<usize>,
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
            self.expand_full(u, first, cap, max_visited).0
        }
        /// Like `expand_bounded`, also reporting whether the deleted region
        /// was exhausted.
        fn expand_full(
            &self,
            u: usize,
            first: &[usize],
            cap: Option<usize>,
            max_visited: usize,
        ) -> (Vec<usize>, bool) {
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
            let (found, exhausted) = expand_deleted(
                seeds,
                &mut visited,
                (cap, max_visited),
                |v| self.probe(v),
                |w| self.outs.get(w).cloned().unwrap_or_default(),
            );
            let mut found = found.into_iter().map(|(v, _)| v).collect::<Vec<_>>();
            found.sort_unstable();
            (found, exhausted)
        }
    }

    #[test]
    fn branching() {
        // u=0 -> 1 (deleted) -> {2, 3} (deleted); 2 -> 4 -> 6 (live), 3 -> 5 (live)
        let mut g = Graph::new(7);
        g.live[5] = true;
        g.live[6] = true;
        g.outs[0] = vec![1];
        g.outs[1] = vec![2, 3];
        g.outs[2] = vec![4];
        g.outs[3] = vec![5];
        g.outs[4] = vec![6];
        // 2 and 3 are the seeds; expanding 2 finds nothing, 3 finds 5 and
        // the search stops before reaching 6.
        assert_eq!(g.expand(0, &[1], None), vec![5]);
    }

    #[test]
    fn long_chain() {
        // 0 -> 1 -> 2 -> ... -> 100 (deleted) -> 101 (live)
        let mut g = Graph::new(102);
        for v in 0..101 {
            g.outs[v] = vec![v + 1];
        }
        g.live[101] = true;
        assert_eq!(g.expand(0, &[1], None), vec![101]);
        assert_eq!(g.expand(0, &[1], Some(4096)), vec![101]);
        // The cap counts expanded deleted vertices; 2..=100 are 99.
        assert_eq!(g.expand(0, &[1], Some(99)), vec![101]);
        assert_eq!(g.expand(0, &[1], Some(98)), Vec::<usize>::new());
        // An empty result is distinguishable from an exhausted region.
        assert_eq!(
            g.expand_full(0, &[1], Some(98), usize::MAX),
            (vec![], false)
        );
    }

    #[test]
    fn wide_frontier_is_bounded() {
        // 0 -> 1 -> {2..=1001} (deleted), each -> v + 1000 (deleted), and only
        // 3000 is live, behind the second layer.
        let mut g = Graph::new(3001);
        g.live[3000] = true;
        g.outs[0] = vec![1];
        g.outs[1] = (2..=1001).collect();
        for v in 2..=1001 {
            g.outs[v] = vec![v + 1000];
        }
        g.outs[2001] = vec![3000];
        // Unbounded, the search crosses both layers.
        assert_eq!(g.expand(0, &[1], None), vec![3000]);
        // With room for 1500 visited vertices it stops in the second layer,
        // even for the start vertex (no cap), and finds nothing.
        assert_eq!(g.expand_bounded(0, &[1], None, 1500), Vec::<usize>::new());
        assert_eq!(g.expand_full(0, &[1], None, 1500), (vec![], false));
    }

    #[test]
    fn pocket_without_exits() {
        // 1 -> 2 -> 3 -> 1 (all deleted), 3 -> 0 (u itself), 2 -> 9 (missing)
        let mut g = Graph::new(4);
        g.live[0] = true;
        g.outs[0] = vec![1];
        g.outs[1] = vec![2];
        g.outs[2] = vec![3, 9];
        g.outs[3] = vec![1, 0];
        // The region is exhausted and `u` is not a candidate of its own.
        assert_eq!(g.expand_full(0, &[1], None, usize::MAX), (vec![], true));
    }

    fn check(
        start: usize,
        before: &[(usize, &[usize])],
        after: &[(usize, bool, &[usize])],
    ) -> (u64, u64, u64) {
        let before = before
            .iter()
            .map(|&(v, outs)| (v, outs.to_vec()))
            .collect::<HashMap<_, _>>();
        let after = after
            .iter()
            .map(|&(v, live, outs)| (v, (live, outs.to_vec())))
            .collect::<HashMap<_, _>>();
        match compare_reachability(start, &before, &after) {
            Reachability::Checked {
                disconnected,
                unreachable_before,
                live,
            } => (disconnected, unreachable_before, live),
            Reachability::Skipped(_) => unreachable!(),
        }
    }

    #[test]
    fn reachability() {
        // Before: 0 (start) -> 1 (deleted) -> {2, 3}; 3 -> 4; 5 has no in-edges.
        let before: &[(usize, &[usize])] = &[
            (0, &[1]),
            (1, &[2, 3]),
            (2, &[]),
            (3, &[4]),
            (4, &[]),
            (5, &[2]),
        ];
        // The repair links 0 to 2 only: 3 and 4 are cut off, 5 already was.
        let after: &[(usize, bool, &[usize])] = &[
            (0, false, &[2]),
            (2, true, &[]),
            (3, true, &[4]),
            (4, true, &[]),
            (5, true, &[2]),
        ];
        assert_eq!(check(0, before, after), (2, 1, 4));
        // A repair that keeps everything reachable reports nothing new.
        let after: &[(usize, bool, &[usize])] = &[
            (0, true, &[2, 3]),
            (2, true, &[]),
            (3, true, &[4, 9]), // 9 is unknown (freed): not followed
            (4, true, &[]),
            (5, true, &[2]),
        ];
        assert_eq!(check(0, before, after), (0, 1, 5));
    }

    #[test]
    fn recorder_memory_limit() {
        let mut r = Recorder::new(2 * (Recorder::ENTRY + 16));
        r.record_before((1, 1), vec![(1, 2)]);
        r.record_after((1, 1), true, vec![(1, 2)]);
        assert!(matches!(r.finish((1, 1)), Reachability::Checked { .. }));
        let mut r = Recorder::new(2 * (Recorder::ENTRY + 16));
        r.record_before((1, 1), vec![(1, 2)]);
        r.record_after((1, 1), true, vec![(1, 2)]);
        r.record_before((1, 2), vec![]);
        assert!(!r.active());
        assert!(matches!(r.finish((1, 1)), Reachability::Skipped(_)));
    }

    /// Neighbours of vertex `n` (0-based) of a `test_support::graph`.
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
    fn spare_degree_reconnects_a_bridge_beside_a_live_neighbour() {
        use crate::test_support::{TestOp, graph};
        // 1 -> live 2 and deleted 3 -> deleted 4 -> live 5.
        // Keeping only the nearby vertex 2 loses the far side of the bridge.
        let r = graph(
            &[
                (true, &[2, 3]),
                (true, &[1]),
                (false, &[4]),
                (false, &[5]),
                (true, &[]),
            ],
            4,
        );
        let maintained = maintain::<_, TestOp>(&r, || {}, 1024 * 1024);
        let mut outs = outs_of(&r, 0);
        outs.sort_unstable();
        assert_eq!(outs, vec![(1, 2), (1, 5)]);
        assert!(matches!(
            maintained.reachability,
            Reachability::Checked {
                disconnected: 0,
                ..
            }
        ));
        assert!(maintained.retained.is_none());
        assert!(r.read(1).get(3).is_none());
        assert!(r.read(1).get(4).is_none());
        assert!(!pending(&r));
    }

    #[test]
    fn optional_bridge_expansion_does_not_prevent_reclamation() {
        use crate::test_support::{TestOp, graph};
        let r = graph(
            &[
                (true, &[2, 3]),
                (true, &[1]),
                (false, &[4]),
                (false, &[5]),
                (true, &[]),
            ],
            4,
        );
        // A bounded ANN repair may miss this exit; an existing live route
        // must not make the optional expansion retain every tombstone.
        let maintained = maintain::<_, TestOp>(&r, || {}, 2 * 2 * EXPANSION_BYTES_PER_VERTEX);
        assert!(maintained.retained.is_none());
        assert_eq!(outs_of(&r, 0), vec![(1, 2)]);
        assert!(r.read(1).get(3).is_none());
        assert!(r.read(1).get(4).is_none());
        assert!(!pending(&r));
    }

    #[test]
    fn start_vertex_cut_off_keeps_deleted_vertices() {
        use crate::test_support::{TestOp, graph};
        // start 0 -> 1 -> 2 -> 3 (deleted) -> 4 (live); IDs are (1, n + 1).
        let vertices: &[(bool, &[u16])] = &[
            (true, &[2]),
            (false, &[3]),
            (false, &[4]),
            (false, &[5]),
            (true, &[]),
        ];
        let r = graph(vertices, 4);
        // Room for two visited vertices: the start's expansion stops inside
        // the deleted chain without finding vertex 4.
        let maintained = maintain::<_, TestOp>(&r, || {}, 2 * 2 * EXPANSION_BYTES_PER_VERTEX);
        assert!(maintained.retained.is_some());
        assert_eq!(maintained.live, 2);
        assert_eq!(outs_of(&r, 0), vec![(1, 2)]);
        for i in 2..=4 {
            assert!(r.read(1).get(i).is_some(), "deleted vertex {i} was freed");
        }
        assert!(pending(&r));
        // With enough memory the start vertex reaches vertex 4 and the chain
        // is freed.
        let maintained = maintain::<_, TestOp>(&r, || {}, 1024 * 1024);
        assert!(maintained.retained.is_none());
        assert_eq!(maintained.live, 2);
        assert_eq!(outs_of(&r, 0), vec![(1, 5)]);
        for i in 2..=4 {
            assert!(r.read(1).get(i).is_none(), "deleted vertex {i} was kept");
        }
        assert!(!pending(&r));
    }

    #[test]
    fn stale_edges_are_dropped_and_counted() {
        use crate::test_support::{TestOp, graph};
        // IDs are (1, n + 1): 0 (start) -> {(1, 2), (1, 3)}, (1, 2) is
        // deleted and points at (1, 3), whose slot is freed below, so both
        // edges to it point at an empty slot. Only the start's edge is a
        // live vertex's stale edge.
        let vertices: &[(bool, &[u16])] =
            &[(true, &[2, 3]), (false, &[3]), (true, &[]), (true, &[])];
        let r = graph(vertices, 4);
        r.write(1, false).free(3);
        let maintained = maintain::<_, TestOp>(&r, || {}, 1024 * 1024);
        assert!(maintained.retained.is_none());
        assert_eq!(maintained.stale_edges, 1);
        assert_eq!(maintained.broken, 0);
        assert!(!outs_of(&r, 0).contains(&(1, 3)));
        assert!(r.read(1).get(2).is_none());
        assert!(!pending(&r));
    }
}
