// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use crate::index::fetcher::*;
use crate::index::opclass::Sphere;
use crate::index::scanners::{Io, SearchBuilder};
use crate::index::vchordrq::dispatch::*;
use crate::index::vchordrq::opclass::Opfamily;
use crate::index::vchordrq::scanners::{FromOwned, SearchOptions};
use crate::recorder::{Recorder, text};
use always_equal::AlwaysEqual;
use distance::Distance;
use index::bump::Bump;
use index::filter::filter;
use index::packed::PackedRefMut4;
use index::prefetcher::{PlainPrefetcher, SimplePrefetcher, StreamPrefetcher};
use index::relation::{Hints, Page, RelationPrefetch, RelationRead, RelationReadStream};
use index_accessor::{Dot, L2S};
use simd::f16;
use std::borrow::Borrow;
use std::num::NonZero;
use std::rc::Rc;
use vchordrq::operator::Operator;
use vchordrq::types::{DistanceKind, OwnedVector, VectorKind};
use vchordrq::{
    Candidates, RerankMethod, TableAhead, default_search, exact_distance, how, rerank_heap,
    rerank_index, unit_distance,
};
use vector::VectorOwned;
use vector::rabitq4::Rabitq4Owned;
use vector::rabitq8::Rabitq8Owned;
use vector::vect::VectOwned;

pub struct DefaultBuilder {
    opfamily: Opfamily,
    orderbys: Vec<Option<OwnedVector>>,
    spheres: Vec<Option<Sphere<OwnedVector>>>,
}

impl SearchBuilder for DefaultBuilder {
    type Options = SearchOptions;

    type Opfamily = Opfamily;

    type Opaque = vchordrq::Opaque;

    fn new(opfamily: Opfamily) -> Self {
        assert!(matches!(
            opfamily,
            Opfamily::HalfvecCosine
                | Opfamily::HalfvecIp
                | Opfamily::HalfvecL2
                | Opfamily::VectorCosine
                | Opfamily::VectorIp
                | Opfamily::VectorL2
                | Opfamily::Rabitq8Cosine
                | Opfamily::Rabitq8Ip
                | Opfamily::Rabitq8L2
                | Opfamily::Rabitq4Cosine
                | Opfamily::Rabitq4Ip
                | Opfamily::Rabitq4L2
        ));
        Self {
            opfamily,
            orderbys: Vec::new(),
            spheres: Vec::new(),
        }
    }

    unsafe fn add(&mut self, strategy: u16, datum: Option<pgrx::pg_sys::Datum>) {
        match strategy {
            1 => {
                let x = unsafe { datum.and_then(|x| self.opfamily.input_vector(x)) };
                self.orderbys.push(x);
            }
            2 => {
                let x = unsafe { datum.and_then(|x| self.opfamily.input_sphere(x)) };
                self.spheres.push(x);
            }
            _ => unreachable!(),
        }
    }

    fn build<'b, R>(
        self,
        index: &'b R,
        options: SearchOptions,
        fetcher: impl Fetcher + 'b,
        bump: &'b impl Bump,
        recorder: impl Recorder,
    ) -> Box<dyn Iterator<Item = (f32, [u16; 3], bool)> + 'b>
    where
        R: RelationRead + RelationPrefetch + RelationReadStream,
        R::Page: Page<Opaque = vchordrq::Opaque>,
    {
        let mut vector = None;
        let mut threshold = None;
        // PostgreSQL rechecks spheres, as for vchordg; only the first one around the search vector
        // bounds the search.
        let recheck = !self.spheres.is_empty()
            || (options.search.filter.as_ref()).is_some_and(|filter| !filter.exact());
        for orderby_vector in self.orderbys.into_iter().flatten() {
            if vector.is_none() {
                vector = Some(orderby_vector);
            } else {
                pgrx::error!("vector search with multiple vectors is not supported");
            }
        }
        for Sphere { center, radius } in self.spheres.into_iter().flatten() {
            if vector.is_none() {
                (vector, threshold) = (Some(center), Some(radius));
            } else if threshold.is_none() && vector.as_ref() == Some(&center) {
                threshold = Some(radius);
            }
        }
        let opfamily = self.opfamily;
        let threshold = threshold.map(|radius| match opfamily {
            Opfamily::VectorL2 | Opfamily::HalfvecL2 if radius > 0.0 => radius.next_up(),
            _ => radius,
        });
        let bound = threshold.map(|radius| opfamily.bound(radius));
        let Some(vector) = vector else {
            return Box::new(std::iter::empty()) as Box<dyn Iterator<Item = (f32, [u16; 3], bool)>>;
        };
        // A zero/non-finite cosine query has no direction. Cosine indexes
        // contain only defined vectors and return no neighbours for it.
        if !opfamily.valid_cosine_query(&vector) {
            return Box::new(std::iter::empty());
        }
        let reranking = RerankOptions {
            method: how(index),
            io: options.io_rerank,
            prefilter: options.prefilter,
            table: options.table,
            threshold,
        };
        let iter: Ranked = if options.max_scan_tuples == Some(0) {
            Box::new(std::iter::empty())
        } else {
            match (opfamily.vector_kind(), opfamily.distance_kind()) {
                (VectorKind::Vecf32, DistanceKind::L2S) => {
                    type Op = vchordrq::operator::Op<VectOwned<f32>, L2S>;
                    let unprojected = VectOwned::<f32>::from_owned(vector.clone());
                    let projected = RandomProject::project(unprojected.as_borrowed());
                    let candidates = prefetched!(
                        index,
                        options.io_search,
                        default_search::<_, Op>,
                        (index, Rc::new(projected), bound, options.search, bump)
                    );
                    rerank::<_, Op>(index, candidates, unprojected, reranking, fetcher, opfamily)
                }
                (VectorKind::Vecf32, DistanceKind::Dot) => {
                    type Op = vchordrq::operator::Op<VectOwned<f32>, Dot>;
                    let unprojected = VectOwned::<f32>::from_owned(vector.clone());
                    let projected = RandomProject::project(unprojected.as_borrowed());
                    let candidates = prefetched!(
                        index,
                        options.io_search,
                        default_search::<_, Op>,
                        (index, Rc::new(projected), bound, options.search, bump)
                    );
                    rerank::<_, Op>(index, candidates, unprojected, reranking, fetcher, opfamily)
                }
                (VectorKind::Vecf16, DistanceKind::L2S) => {
                    type Op = vchordrq::operator::Op<VectOwned<f16>, L2S>;
                    let unprojected = VectOwned::<f16>::from_owned(vector.clone());
                    let projected = RandomProject::project(unprojected.as_borrowed());
                    let candidates = prefetched!(
                        index,
                        options.io_search,
                        default_search::<_, Op>,
                        (index, Rc::new(projected), bound, options.search, bump)
                    );
                    rerank::<_, Op>(index, candidates, unprojected, reranking, fetcher, opfamily)
                }
                (VectorKind::Vecf16, DistanceKind::Dot) => {
                    type Op = vchordrq::operator::Op<VectOwned<f16>, Dot>;
                    let unprojected = VectOwned::<f16>::from_owned(vector.clone());
                    let projected = RandomProject::project(unprojected.as_borrowed());
                    let candidates = prefetched!(
                        index,
                        options.io_search,
                        default_search::<_, Op>,
                        (index, Rc::new(projected), bound, options.search, bump)
                    );
                    rerank::<_, Op>(index, candidates, unprojected, reranking, fetcher, opfamily)
                }
                (VectorKind::Rabitq8, DistanceKind::L2S) => {
                    type Op = vchordrq::operator::Op<Rabitq8Owned, L2S>;
                    let unprojected = Rc::new(Rabitq8Owned::from_owned(vector.clone()));
                    let candidates = prefetched!(
                        index,
                        options.io_search,
                        default_search::<_, Op>,
                        (index, Rc::clone(&unprojected), bound, options.search, bump)
                    );
                    rerank::<_, Op>(index, candidates, unprojected, reranking, fetcher, opfamily)
                }
                (VectorKind::Rabitq8, DistanceKind::Dot) => {
                    type Op = vchordrq::operator::Op<Rabitq8Owned, Dot>;
                    let unprojected = Rc::new(Rabitq8Owned::from_owned(vector.clone()));
                    let candidates = prefetched!(
                        index,
                        options.io_search,
                        default_search::<_, Op>,
                        (index, Rc::clone(&unprojected), bound, options.search, bump)
                    );
                    rerank::<_, Op>(index, candidates, unprojected, reranking, fetcher, opfamily)
                }
                (VectorKind::Rabitq4, DistanceKind::L2S) => {
                    type Op = vchordrq::operator::Op<Rabitq4Owned, L2S>;
                    let unprojected = Rc::new(Rabitq4Owned::from_owned(vector.clone()));
                    let candidates = prefetched!(
                        index,
                        options.io_search,
                        default_search::<_, Op>,
                        (index, Rc::clone(&unprojected), bound, options.search, bump)
                    );
                    rerank::<_, Op>(index, candidates, unprojected, reranking, fetcher, opfamily)
                }
                (VectorKind::Rabitq4, DistanceKind::Dot) => {
                    type Op = vchordrq::operator::Op<Rabitq4Owned, Dot>;
                    let unprojected = Rc::new(Rabitq4Owned::from_owned(vector.clone()));
                    let candidates = prefetched!(
                        index,
                        options.io_search,
                        default_search::<_, Op>,
                        (index, Rc::clone(&unprojected), bound, options.search, bump)
                    );
                    rerank::<_, Op>(index, candidates, unprojected, reranking, fetcher, opfamily)
                }
            }
        };
        let iter = if let Some(max_scan_tuples) = options.max_scan_tuples {
            Box::new(iter.take(max_scan_tuples as _))
        } else {
            iter
        };
        if recorder.is_enabled() {
            match &vector {
                OwnedVector::Vecf32(v) => {
                    recorder.send(&text::vector_out(v.as_borrowed()));
                }
                OwnedVector::Vecf16(v) => {
                    recorder.send(&text::halfvec_out(v.as_borrowed()));
                }
                OwnedVector::Rabitq8(v) => {
                    recorder.send(&text::rabitq8_out(v.as_borrowed()));
                }
                OwnedVector::Rabitq4(v) => {
                    recorder.send(&text::rabitq4_out(v.as_borrowed()));
                }
            }
        }
        Box::new(iter.map(move |(distance, pointer)| {
            let (key, _) = pointer_to_kv(pointer);
            (opfamily.output(distance), key, recheck)
        }))
    }
}

type Ranked<'b> = Box<dyn Iterator<Item = (Distance, NonZero<u64>)> + 'b>;

#[derive(Clone, Copy)]
struct RerankOptions {
    method: RerankMethod,
    io: Io,
    prefilter: bool,
    table: TablePrefetch,
    threshold: Option<f32>,
}

/// The table rerank of `rerank`, with `d` the distance from the query to a row's vector.
fn rerank_table<'b, R, O, D>(
    index: &'b R,
    candidates: Candidates<'b>,
    vector: impl Borrow<O::Vector> + 'b,
    (io, prefilter, table): (Io, bool, TablePrefetch),
    mut fetcher: impl Fetcher + 'b,
    d: D,
) -> Ranked<'b>
where
    R: RelationRead,
    O: Operator,
    O::Vector: FromOwned,
    D: Fn(
            <O::Vector as VectorOwned>::Borrowed<'_>,
            <O::Vector as VectorOwned>::Borrowed<'_>,
        ) -> Distance
        + 'b,
{
    let distance = move |payload| {
        let (key, _) = pointer_to_kv(payload);
        let mut tuple = fetcher.fetch(key)?;
        if prefilter && !tuple.filter() {
            return None;
        }
        let (datums, is_nulls) = tuple.build();
        assert!(!is_nulls[0], "a reranked row has no vector");
        let query = vector.borrow().as_borrowed();
        Some(unsafe { O::Vector::in_datum(datums[0], |row| d(query, row)) })
    };
    match io {
        Io::Plain => Box::new(rerank_heap(
            PlainPrefetcher::new(index, candidates),
            distance,
        )),
        Io::Simple | Io::Stream => {
            let prefetcher = PlainPrefetcher::new(index, TableAhead::new(candidates, table));
            Box::new(rerank_heap(prefetcher, distance))
        }
    }
}

/// Reranks `candidates` by exact distance, reading vectors from the index or the table as it was
/// built; with `prefilter`, rows the table's quals reject are dropped first. Unless `io` is plain,
/// table pages are read ahead: of candidates before the table rerank, else of rows about to be
/// returned unless the prefilter has read them.
fn rerank<'b, R, O>(
    index: &'b R,
    candidates: Candidates<'b>,
    vector: impl Borrow<O::Vector> + 'b,
    options: RerankOptions,
    mut fetcher: impl Fetcher + 'b,
    opfamily: Opfamily,
) -> Ranked<'b>
where
    R: RelationRead + RelationPrefetch + RelationReadStream,
    R::Page: Page<Opaque = vchordrq::Opaque>,
    O: Operator,
    O::Vector: FromOwned,
{
    let can_refill = candidates.can_refill();
    let rows: Ranked = if let RerankMethod::Heap = options.method {
        let reranking = (options.io, options.prefilter, options.table);
        if opfamily.is_cosine() {
            rerank_table::<_, O, _>(
                index,
                candidates,
                vector,
                reranking,
                fetcher,
                unit_distance::<O>,
            )
        } else {
            rerank_table::<_, O, _>(
                index,
                candidates,
                vector,
                reranking,
                fetcher,
                exact_distance::<O>,
            )
        }
    } else {
        let predicate = id_0(move |(_, AlwaysEqual(PackedRefMut4((pointer, _, _))))| {
            let (key, _) = pointer_to_kv(*pointer);
            let Some(mut tuple) = fetcher.fetch(key) else {
                return false;
            };
            tuple.filter()
        });
        let hints = Hints::default().full(false);
        match (options.io, options.prefilter) {
            (Io::Plain, false) => {
                let prefetcher = PlainPrefetcher::new(index, candidates);
                Box::new(rerank_index::<O, _, _, _>(vector, prefetcher, ()))
            }
            (Io::Plain, true) => {
                let prefetcher = PlainPrefetcher::new(index, filter(candidates, predicate));
                Box::new(rerank_index::<O, _, _, _>(vector, prefetcher, ()))
            }
            (Io::Simple, false) => {
                let prefetcher = SimplePrefetcher::new(index, candidates);
                Box::new(rerank_index::<O, _, _, _>(
                    vector,
                    prefetcher,
                    options.table,
                ))
            }
            (Io::Simple, true) => {
                let prefetcher = SimplePrefetcher::new(index, filter(candidates, predicate));
                Box::new(rerank_index::<O, _, _, _>(vector, prefetcher, ()))
            }
            (Io::Stream, false) => {
                let prefetcher = StreamPrefetcher::new(index, candidates, hints);
                Box::new(rerank_index::<O, _, _, _>(
                    vector,
                    prefetcher,
                    options.table,
                ))
            }
            (Io::Stream, true) => {
                let prefetcher = StreamPrefetcher::new(index, filter(candidates, predicate), hints);
                Box::new(rerank_index::<O, _, _, _>(vector, prefetcher, ()))
            }
        }
    };
    let Some(threshold) = options.threshold else {
        return rows;
    };
    // `<=`: the index's distance may round up to the radius where the operator's is under it.
    let within =
        move |&(distance, _): &(Distance, NonZero<u64>)| opfamily.output(distance) <= threshold;
    if can_refill {
        // Out-of-radius rows in one round must not end the scan before further lists are read.
        Box::new(rows.filter(within))
    } else {
        Box::new(rows.take_while(within))
    }
}

#[inline(always)]
pub fn id_0<F, A: ?Sized, B: ?Sized, C: ?Sized, D: ?Sized, R: ?Sized>(f: F) -> F
where
    F: for<'a> FnMut(&(A, AlwaysEqual<PackedRefMut4<'a, (B, C, D)>>)) -> R,
{
    f
}
