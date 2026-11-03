// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use crate::Opaque;
use crate::candidates::Candidates;
use crate::operator::{Operator, Vector};
use crate::results::Results;
use crate::tuples::*;
use crate::vectors::{by_prefetch, copy_outs};
use crate::visited::Visited;
use always_equal::AlwaysEqual;
use distance::Distance;
use index::bump::Bump;
use index::prefetcher::{Prefetcher, PrefetcherSequenceFamily};
use index::relation::{Page, RelationRead};
use index_accessor::{DefaultWithDimension, LAccess};
use rabitq::bits::Bits;
use std::cmp::Reverse;
use std::collections::VecDeque;
use std::num::NonZero;
use vector::{VectorBorrowed, VectorOwned};

pub fn search<'b, R: RelationRead, O: Operator>(
    index: &'b R,
    vector: <O::Vector as VectorOwned>::Borrowed<'b>,
    ef_search: u32,
    beam_search: u32,
    bump: &'b impl Bump,
    mut prefetch_vertices: impl PrefetcherSequenceFamily<'b, R> + 'b,
    prefetch_vectors: impl PrefetcherSequenceFamily<'b, R> + 'b,
) -> Box<dyn Iterator<Item = (Distance, NonZero<u64>)> + 'b>
where
    R::Page: Page<Opaque = Opaque>,
{
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
    let dim = meta_tuple.dim();
    let start = meta_tuple.start();
    let bits = Bits::try_from(meta_tuple.bits()).expect("data corruption");
    assert_eq!(dim, vector.dim(), "unmatched dimensions");
    let ef = ef_search;
    let beam = beam_search;
    drop(meta_guard);
    let lut = O::Vector::preprocess(vector);
    let mut visited = Visited::new();
    let mut candidates = Candidates::new(beam as usize, prefetch_vectors);
    let Some(s) = start.into_inner() else {
        return Box::new(std::iter::empty());
    };
    {
        visited.insert(s);
        let vertex_guard = index.read(s.0);
        let Some(vertex_bytes) = vertex_guard.get(s.1) else {
            // the link is broken
            return Box::new(std::iter::empty());
        };
        let vertex_tuple = VertexTuple::deserialize_ref(vertex_bytes);
        let pointers_s: &[_] = bump.alloc_slice(vertex_tuple.pointers());
        // The vertex tuple holds the authoritative payload: `bulkdelete` clears
        // it, but not the copy in the vector tuple. A deleted vertex (including
        // the start vertex, which `maintain` never frees) is still traversed,
        // but it is never emitted.
        let payload_s = vertex_tuple.payload();
        let score_s = O::process(
            bits,
            dim,
            (vertex_tuple.metadata(), vertex_tuple.elements()),
            &lut,
        );
        candidates.push((Reverse(score_s), AlwaysEqual((pointers_s, payload_s))));
    }
    let mut iter = std::iter::from_fn(move || {
        while let Some(((_, AlwaysEqual((pointers_u, payload_u))), guards)) = candidates.pop() {
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
                candidates.push((Reverse(score_v), AlwaysEqual((pointers_v, payload_v))));
            }
            return Some((Reverse(dis_u), AlwaysEqual(payload_u)));
        }
        None
    });
    let mut results = Results::new(ef as _);
    let search = std::iter::from_fn(move || {
        for element @ (Reverse(dis_c), _) in iter.by_ref() {
            results.push(element);
            if results
                .peek_ef_th()
                .map(|dis_e| dis_e < dis_c)
                .unwrap_or_default()
            {
                break;
            }
        }
        results.pop_min()
    });
    Box::new(search.filter_map(|(dis_u, AlwaysEqual(payload_u))| Some((dis_u, payload_u?))))
}
