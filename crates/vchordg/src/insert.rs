// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use crate::Opaque;
use crate::candidates::Candidates;
use crate::operator::{CloneAccessor, Operator, Vector};
use crate::results::Results;
use crate::tuples::*;
use crate::types::DistanceKind;
use crate::vectors::{by_prefetch, by_read, copy_all, copy_nothing, copy_outs, update};
use crate::visited::Visited;
use always_equal::AlwaysEqual;
use distance::Distance;
use index::bump::Bump;
use index::prefetcher::{Prefetcher, PrefetcherSequenceFamily};
use index::relation::{Page, PageGuard, RelationRead, RelationWrite};
use index_accessor::{DefaultWithDimension, LAccess};
use rabitq::bits::Bits;
use std::cmp::Reverse;
use std::collections::VecDeque;
use std::num::{NonZero, Wrapping};
use vector::{VectorBorrowed, VectorOwned};

pub fn insert<'b, R: RelationRead + RelationWrite, O: Operator>(
    index: &'b R,
    vector: <O::Vector as VectorOwned>::Borrowed<'b>,
    payload: NonZero<u64>,
    bump: &'b impl Bump,
    mut prefetch_vertices: impl PrefetcherSequenceFamily<'b, R> + 'b,
    prefetch_vectors: impl PrefetcherSequenceFamily<'b, R> + 'b,
) where
    R::Page: Page<Opaque = Opaque>,
{
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
    let dim = meta_tuple.dim();
    assert_eq!(dim, vector.dim(), "unmatched dimensions");
    let start = meta_tuple.start();
    let bits = Bits::try_from(meta_tuple.bits()).expect("data corruption");
    let m = meta_tuple.m();
    let alpha = meta_tuple.alpha().to_vec();
    let ef = meta_tuple.ef_construction();
    let beam = meta_tuple.beam_construction();
    let skip = meta_tuple.skip();
    drop(meta_guard);
    let version_t = Wrapping(rand::random());
    let (pointers_t, t) = {
        let list_of_vector_bytes = {
            let (left, right) = O::Vector::split(vector, m as _);
            let left = left.into_iter().enumerate().map(|(index, elements)| {
                VectorTuple::serialize(&VectorTuple::<O::Vector>::_1 {
                    payload: Some(payload),
                    elements: elements.to_vec(),
                    index: index as u32,
                })
            });
            let right = VectorTuple::serialize(&VectorTuple::<O::Vector>::_0 {
                payload: Some(payload),
                elements: right.0.to_vec(),
                metadata: right.1,
                neighbours: vec![OptionNeighbour::NONE; m as usize],
                version: version_t,
            });
            left.chain(std::iter::once(right)).collect::<Vec<_>>()
        };
        let mut vertex_bytes = {
            let code = O::Vector::code(bits, vector);
            VertexTuple::serialize(&VertexTuple {
                metadata: code.0.into_array(),
                payload: Some(payload),
                elements: rabitq::bits::pack_code(bits, &code.1),
                pointers: vec![Pointer::new((u32::MAX, 0)); list_of_vector_bytes.len()], // a sentinel value
            })
        };
        let mut vertex_guard = if let Some(guard) = index.search(vertex_bytes.len()) {
            guard
        } else {
            append_vertex_tuple(index, skip, vertex_bytes.len())
        };
        let link = vertex_guard.get_opaque().link;
        let pointers_t = list_of_vector_bytes
            .iter()
            .map(|vector_bytes| {
                let mut vector_guard = append_vector_tuple(index, link, vector_bytes.len());
                let i = vector_guard
                    .alloc(vector_bytes)
                    .expect("implementation: a free page cannot accommodate a single tuple");
                Pointer::new((vector_guard.id(), i))
            })
            .collect::<Vec<_>>();
        {
            let mut vertex_tuple = VertexTuple::deserialize_mut(&mut vertex_bytes);
            vertex_tuple.pointers().copy_from_slice(&pointers_t);
        }
        let t = {
            let i = vertex_guard
                .alloc(&vertex_bytes)
                .expect("implementation: a free page cannot accommodate a single tuple");
            (vertex_guard.id(), i)
        };
        drop(vertex_guard);
        if skip < t.0 {
            let mut meta_guard = index.write(0, false);
            let meta_bytes = meta_guard.get_mut(1).expect("data corruption");
            let mut meta_tuple = MetaTuple::deserialize_mut(meta_bytes);
            *meta_tuple.skip() = (*meta_tuple.skip()).max(t.0);
        }
        (pointers_t, t)
    };
    let start = if start.into_inner().is_none() {
        let mut meta_guard = index.write(0, false);
        let meta_bytes = meta_guard.get_mut(1).expect("data corruption");
        let mut meta_tuple = MetaTuple::deserialize_mut(meta_bytes);
        let dst = meta_tuple.start();
        if dst.into_inner().is_none() {
            *dst = OptionPointer::some(t);
            return;
        } else {
            *dst
        }
    } else {
        start
    };
    let lut = O::Vector::preprocess(vector);
    let mut visited = Visited::new();
    let mut candidates = Candidates::new(beam as usize, prefetch_vectors);
    let Some(s) = start.into_inner() else {
        return;
    };
    {
        visited.insert(s);
        let vertex_guard = index.read(s.0);
        let Some(vertex_bytes) = vertex_guard.get(s.1) else {
            // the link is broken
            return;
        };
        let vertex_tuple = VertexTuple::deserialize_ref(vertex_bytes);
        let pointers_s: &[_] = bump.alloc_slice(vertex_tuple.pointers());
        let payload_s = vertex_tuple.payload();
        let score_s = O::process(
            bits,
            dim,
            (vertex_tuple.metadata(), vertex_tuple.elements()),
            &lut,
        );
        candidates.push((Reverse(score_s), AlwaysEqual((pointers_s, (s, payload_s)))));
    }
    let mut iter = std::iter::from_fn(|| {
        while let Some(((_, AlwaysEqual((pointers_u, (u, payload_u)))), guards)) = candidates.pop()
        {
            let Ok((dis_u, outs_u, _, _)) = crate::vectors::read::<R, O, _, _>(
                by_prefetch::<R>(guards, pointers_u.iter().copied()),
                LAccess::new(
                    O::Vector::unpack(vector),
                    O::DistanceAccessor::default_with_dimension(dim),
                ),
                copy_outs,
            ) else {
                // the link is broken
                continue;
            };
            let mut iterator = prefetch_vertices.prefetch(
                outs_u
                    .into_iter()
                    .filter(|&x| !visited.contains(x))
                    .collect::<VecDeque<_>>(),
            );
            while let Some((v, guards)) = iterator.next() {
                visited.insert(v);
                let vertex_guard = {
                    let mut guards = guards;
                    let r = guards.next().expect("internal");
                    assert!(guards.next().is_none(), "internal");
                    drop(guards);
                    r
                };
                let Some(vertex_bytes) = vertex_guard.get(v.1) else {
                    // the link is broken
                    continue;
                };
                let vertex_tuple = VertexTuple::deserialize_ref(vertex_bytes);
                let pointers_v: &[_] = bump.alloc_slice(vertex_tuple.pointers());
                let payload_v = vertex_tuple.payload();
                let score_v = O::process(
                    bits,
                    dim,
                    (vertex_tuple.metadata(), vertex_tuple.elements()),
                    &lut,
                );
                candidates.push((Reverse(score_v), AlwaysEqual((pointers_v, (v, payload_v)))));
            }
            return Some((Reverse(dis_u), AlwaysEqual((pointers_u, (u, payload_u)))));
        }
        None
    });
    // Deleted vertices only guide the search (#26): VACUUM frees them, so
    // neither an out-edge of the new vertex nor its backlink may rely on
    // one. They do not count towards `ef`, so a region of them cannot end
    // the search before live vertices are found; a separate limit bounds how
    // many the search may pass. The start vertex is never freed and stays
    // eligible, deleted or not.
    let mut results = Results::new(ef as _);
    let mut passed = 0_usize;
    for element @ (Reverse(dis_c), AlwaysEqual((_, (u, payload_u)))) in iter.by_ref() {
        if payload_u.is_none() && u != s {
            passed += 1;
            if passed > DELETED_TRAVERSAL_FACTOR * ef as usize {
                break;
            }
            continue;
        }
        results.push(element);
        if results
            .peek_ef_th()
            .map(|dis_e| dis_e < dis_c)
            .unwrap_or_default()
        {
            break;
        }
    }
    let trace = {
        let (left, right) = results.into_inner();
        left.into_iter()
            .map(|(dis_v, v)| (Reverse(dis_v), v))
            .chain(right)
            .flat_map(|item| {
                let (Reverse(dis_u), AlwaysEqual((pointers_u, (u, payload_u)))) = item;
                let Ok((vector_u, _, _, _)) = crate::vectors::read::<R, O, _, _>(
                    by_read(index, pointers_u.iter().copied()),
                    CloneAccessor::<O::Vector>::default_with_dimension(dim),
                    copy_nothing,
                ) else {
                    // the link is broken
                    return None;
                };
                Some((
                    (Reverse(dis_u), AlwaysEqual((pointers_u, u, payload_u))),
                    vector_u,
                ))
            })
            .collect::<Vec<_>>()
    };
    // The start vertex is popped first, so it is in the trace unless its
    // vectors could not be read.
    let start_distance = trace
        .iter()
        .find(|((_, AlwaysEqual((_, u, _))), _)| *u == s)
        .map(|((Reverse(dis_s), _), _)| *dis_s);
    let outs = crate::prune::prune(
        |x, y| O::distance(x.as_borrowed(), y.as_borrowed()),
        (bump.alloc_slice(&pointers_t) as &[_], t, Some(payload)),
        trace.into_iter(),
        m,
        &alpha,
        |(_, u, _)| *u,
        O::DISTANCE == DistanceKind::L2S,
    );
    let _ = update::<R, O>(
        (index, pointers_t.as_slice()),
        (version_t, VecDeque::new()),
        outs.iter()
            .map(|&(Reverse(dis_u), AlwaysEqual((_, u, _)))| (u, dis_u)),
    );
    let mut stale = Vec::new();
    for &(Reverse(dis_t), AlwaysEqual((pointers_u, u, payload_u))) in outs.iter() {
        if backlink::<R, O>(
            index,
            (dim, m, &alpha),
            (pointers_u, u, payload_u),
            (t, dis_t),
        ) == Backlink::Stale
        {
            stale.push(u);
        }
    }
    // A neighbour validated by its backlink was live after this list was
    // written. Freeing it later needs a VACUUM that marks it and then runs a
    // complete edge-repair pass before freeing anything; that pass visits
    // this vertex and repairs the edge. So only the stale ones need care.
    if stale.is_empty() {
        return;
    }
    // Some selected neighbours were marked deleted, freed or replaced by a
    // concurrent VACUUM before their backlinks were placed: take them out of
    // the new vertex's list again, so it keeps no edge to a freed vertex. If
    // none is left, link through the start vertex, which is never freed.
    // (Matching by slot can also drop an edge to a vertex that reused a
    // freed slot meanwhile; that edge was never chosen, so nothing is lost.)
    let mut fallback = None;
    if stale.len() == outs.len()
        && let Some(dis_s) = start_distance
        && let Some((pointers_s, payload_s)) = read_vertex(index, s)
        && backlink::<R, O>(
            index,
            (dim, m, &alpha),
            (&pointers_s, s, payload_s),
            (t, dis_s),
        ) == Backlink::Done
    {
        fallback = Some((s, dis_s));
    }
    // Retried while the list changes under us, a bounded number of times;
    // stops if its tuple is gone.
    for _ in 0..MAX_RETRIES {
        let Ok((neighbours_t, _, version)) =
            crate::vectors::read_without_accessor::<R, O, _>((index, &pointers_t), copy_all)
        else {
            break;
        };
        let mut list = neighbours_t
            .iter()
            .copied()
            .filter(|(v, _)| !stale.contains(v))
            .collect::<Vec<_>>();
        if let Some((s, dis_s)) = fallback
            && !list.iter().any(|&(v, _)| v == s)
        {
            list.insert(0, (s, dis_s));
        }
        if update::<R, O>(
            (index, &pointers_t),
            (version, neighbours_t),
            list.into_iter(),
        ) != Ok(false)
        {
            break;
        }
    }
}

/// An insertion may pass through `DELETED_TRAVERSAL_FACTOR * ef` deleted
/// vertices besides the `ef` live ones it collects (#26). This also bounds
/// the candidates and visited vertices those passes add.
const DELETED_TRAVERSAL_FACTOR: usize = 4;

/// Recomputations of one neighbour list after concurrent changes before an
/// insertion gives up on it (the edge is then left out, or left in for
/// edge repair to handle).
const MAX_RETRIES: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Backlink {
    /// The neighbour list of the target was recomputed with the new vertex.
    Done,
    /// The target is no longer the vertex that was selected.
    Stale,
}

fn read_vertex<R: RelationRead>(
    index: &R,
    u: (u32, u16),
) -> Option<(Vec<Pointer>, Option<NonZero<u64>>)>
where
    R::Page: Page<Opaque = Opaque>,
{
    let vertex_guard = index.read(u.0);
    let vertex_tuple = VertexTuple::deserialize_ref(vertex_guard.get(u.1)?);
    Some((vertex_tuple.pointers().to_vec(), vertex_tuple.payload()))
}

/// Adds the new vertex `t` to the neighbour list of `u` (pruning it), if `u`
/// is still the vertex that was selected, in the same state.
fn backlink<R: RelationRead + RelationWrite, O: Operator>(
    index: &R,
    (dim, m, alpha): (u32, u32, &[f32]),
    (pointers_u, u, payload_u): (&[Pointer], (u32, u16), Option<NonZero<u64>>),
    (t, dis_t): ((u32, u16), Distance),
) -> Backlink
where
    R::Page: Page<Opaque = Opaque>,
{
    for _ in 0..MAX_RETRIES {
        let Ok((neighbours_u, _, version)) =
            crate::vectors::read_without_accessor::<R, O, _>((index, pointers_u), copy_all)
        else {
            // the link is broken: `u` was freed
            return Backlink::Stale;
        };
        let trace = neighbours_u
            .iter()
            .copied()
            .chain(std::iter::once((t, dis_t)))
            .map(|(v, dis_v)| (Reverse(dis_v), AlwaysEqual(v)))
            .flat_map(|item| {
                let (Reverse(dis_u), AlwaysEqual(u)) = item;
                let vertex_guard = index.read(u.0);
                let Some(vertex_bytes) = vertex_guard.get(u.1) else {
                    // the link is broken
                    return None;
                };
                let vertex_tuple = VertexTuple::deserialize_ref(vertex_bytes);
                let pointers_u = vertex_tuple.pointers().to_vec();
                Some((Reverse(dis_u), AlwaysEqual((pointers_u, u))))
            })
            .flat_map(|item| {
                let (Reverse(dis_u), AlwaysEqual((pointers_u, u))) = item;
                let Ok((vector_u, _, _, _)) = crate::vectors::read::<R, O, _, _>(
                    by_read(index, pointers_u.iter().copied()),
                    CloneAccessor::<O::Vector>::default_with_dimension(dim),
                    copy_nothing,
                ) else {
                    // the link is broken
                    return None;
                };
                Some(((Reverse(dis_u), AlwaysEqual((pointers_u, u))), vector_u))
            })
            .collect::<Vec<_>>();
        let outs = crate::prune::prune(
            |x, y| O::distance(x.as_borrowed(), y.as_borrowed()),
            (pointers_u.to_vec(), u),
            trace.into_iter(),
            m,
            alpha,
            |(_, u)| *u,
            O::DISTANCE == DistanceKind::L2S,
        );
        // Publish the backlink only while `u` is still the vertex that was
        // selected, in the same state (#26): a VACUUM may have marked it
        // deleted since, or freed it and reused its slot. Its vertex page
        // stays share-locked through the update, so it cannot be marked or
        // freed in between (vertex before vector, as in insert and maintain);
        // the version check in `update`, also for an unchanged result, covers
        // changes to the list since it was read.
        let vertex_guard = index.read(u.0);
        let Some(vertex_bytes) = vertex_guard.get(u.1) else {
            return Backlink::Stale;
        };
        let vertex_tuple = VertexTuple::deserialize_ref(vertex_bytes);
        if vertex_tuple.payload() != payload_u || vertex_tuple.pointers() != pointers_u {
            return Backlink::Stale;
        }
        let updated = update::<R, O>(
            (index, pointers_u),
            (version, neighbours_u),
            outs.iter()
                .map(|&(Reverse(dis_u), AlwaysEqual((_, u)))| (u, dis_u)),
        );
        drop(vertex_guard);
        match updated {
            Ok(true) => return Backlink::Done,
            Ok(false) => continue,
            // the link is broken
            Err(()) => return Backlink::Stale,
        }
    }
    // The list kept changing; `u` is live, so the edge from the new vertex
    // stays and only the backlink is skipped.
    Backlink::Done
}

fn append_vertex_tuple<'b, R: RelationRead + RelationWrite>(
    index: &'b R,
    first: u32,
    size: usize,
) -> R::WriteGuard<'b>
where
    R::Page: Page<Opaque = Opaque>,
{
    assert!(first != u32::MAX);
    let mut current = first;
    loop {
        let read = index.read(current);
        if read.freespace() as usize >= size || read.get_opaque().next == u32::MAX {
            drop(read);
            let mut write = index.write(current, true);
            if write.freespace() as usize >= size {
                return write;
            }
            if write.get_opaque().next == u32::MAX {
                // Initialize and WAL-log the new pages before the predecessor
                // links to them, so neither an error nor a crash can leave the
                // chain pointing at an uninitialized page. The new vertex page
                // does not track free space yet: the free space map must not
                // hand it out before it is reachable from the chain.
                let link = index
                    .extend(
                        Opaque {
                            next: u32::MAX,
                            link: u32::MAX,
                        },
                        false,
                    )
                    .id();
                let extend = index.extend(
                    Opaque {
                        next: u32::MAX,
                        link,
                    },
                    false,
                );
                if (extend.freespace() as usize) < size {
                    panic!("implementation: a clear page cannot accommodate a single tuple");
                }
                let id = extend.id();
                drop(extend);
                write.get_opaque_mut().next = id;
                drop(write);
                // Take the new page through the ordinary path. Another backend
                // may have used its space in the meantime.
                current = id;
                continue;
            }
            current = write.get_opaque().next;
        } else {
            current = read.get_opaque().next;
        }
    }
}

fn append_vector_tuple<'b, R: RelationRead + RelationWrite>(
    index: &'b R,
    first: u32,
    size: usize,
) -> R::WriteGuard<'b>
where
    R::Page: Page<Opaque = Opaque>,
{
    assert!(first != u32::MAX);
    let mut current = first;
    loop {
        let read = index.read(current);
        if read.freespace() as usize >= size || read.get_opaque().next == u32::MAX {
            drop(read);
            let mut write = index.write(current, false);
            if write.freespace() as usize >= size {
                return write;
            }
            if write.get_opaque().next == u32::MAX {
                // Initialize and WAL-log the new page before the predecessor
                // links to it (see `append_vertex_tuple`).
                let extend = index.extend(
                    Opaque {
                        next: u32::MAX,
                        link: u32::MAX,
                    },
                    false,
                );
                if (extend.freespace() as usize) < size {
                    panic!("implementation: a clear page cannot accommodate a single tuple");
                }
                let id = extend.id();
                drop(extend);
                write.get_opaque_mut().next = id;
                drop(write);
                current = id;
                continue;
            }
            current = write.get_opaque().next
        } else {
            current = read.get_opaque().next
        }
    }
}
