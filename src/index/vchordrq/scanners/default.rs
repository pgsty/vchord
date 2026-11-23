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
use index::prefetcher::*;
use index::relation::{Hints, Page, RelationPrefetch, RelationRead, RelationReadStream};
use index_accessor::{Accessor2, CosineAccessor, DefaultWithDimension, Dot, L2S};
use simd::f16;
use std::borrow::Borrow;
use std::num::NonZero;
use std::rc::Rc;
use vchordrq::KKind;
use vchordrq::operator::{Operator, Vector};
use vchordrq::types::{DistanceKind, OwnedVector, VectorKind};
use vchordrq::{
    Candidates, Hinted, KHint, RerankMethod, Table, TableAhead, cosine_distance, default_search,
    exact_distance, how, rerank_heap, rerank_index_by, unit_distance,
};
use vector::rabitq4::Rabitq4Owned;
use vector::rabitq8::Rabitq8Owned;
use vector::vect::{MIN_FAST_X2, VectBorrowed, VectOwned};
use vector::{VectorBorrowed, VectorOwned};

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
        // the first sphere around the search vector bounds the search; the others are rechecked
        let mut others = (options.search.filter.as_ref()).is_some_and(|filter| !filter.exact());
        let opfamily = self.opfamily;
        // each vector as the index takes it, and as given
        let input = |raw: OwnedVector| (opfamily.normalized(&raw), raw);
        for orderby_vector in self.orderbys.into_iter().flatten() {
            if vector.is_none() {
                vector = Some(input(orderby_vector));
            } else {
                pgrx::error!("vector search with multiple vectors is not supported");
            }
        }
        for sphere in self.spheres {
            match sphere.map(|Sphere { center, radius }| (input(center), radius)) {
                Some((center, radius)) if vector.is_none() => {
                    (vector, threshold) = (Some(center), Some(radius));
                }
                Some((center, radius))
                    if threshold.is_none() && vector.as_ref().map(|x| &x.0) == Some(&center.0) =>
                {
                    threshold = Some(radius);
                }
                _ => others = true,
            }
        }
        // a halfvec copy normalized in the index is within its rounding of the operator's cosine
        let rounded = matches!(opfamily, Opfamily::HalfvecCosine)
            && !matches!(how(index), RerankMethod::Heap);
        let owned = options.owned && !others && !rounded;
        let recheck = others || (threshold.is_some() && !owned);
        let k = if recheck {
            options.k.and(KKind::AtLeast)
        } else {
            options.k
        };
        let bound = threshold.map(|radius| opfamily.bound(radius));
        let Some((vector, raw)) = vector else {
            return Box::new(std::iter::empty()) as Box<dyn Iterator<Item = (f32, [u16; 3], bool)>>;
        };
        // A zero/non-finite cosine query has no direction. Cosine indexes
        // contain only defined vectors and return no neighbours for it.
        if !opfamily.valid_cosine_query(&vector) {
            return Box::new(std::iter::empty());
        }
        let reranking = (options.io_rerank, options.prefilter, options.table, k);
        let iter: Box<dyn Iterator<Item = (f32, NonZero<u64>)>> =
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
                    // halfvec rounds the normalized query: rerank by the query as given
                    let raw = VectOwned::<f16>::from_owned(raw);
                    let y2 = <Op as Operator>::Vector::squared_norm(raw.as_borrowed());
                    if opfamily.is_cosine() && (MIN_FAST_X2..f32::INFINITY).contains(&y2) {
                        let accessor = move || CosineAccessor::new(y2);
                        let d = move |query: VectBorrowed<'_, f16>, row: VectBorrowed<'_, f16>| {
                            cosine_distance::<Op>(query, row, y2)
                        };
                        let reranking = (reranking, accessor, d);
                        rerank_by::<_, Op, _, _>(
                            index, candidates, raw, reranking, fetcher, opfamily,
                        )
                    } else {
                        rerank::<_, Op>(
                            index,
                            candidates,
                            unprojected,
                            reranking,
                            fetcher,
                            opfamily,
                        )
                    }
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
            };
        // owned, the sphere is the operator's `<` on the scan's distance; else `<=` and the recheck
        let iter: Box<dyn Iterator<Item = (f32, NonZero<u64>)>> = match threshold {
            Some(threshold) if owned => Box::new(iter.take_while(move |(x, _)| *x < threshold)),
            Some(threshold) => Box::new(iter.take_while(move |(x, _)| *x <= threshold)),
            None => iter,
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
            (distance, key, recheck)
        }))
    }
}

/// The table rerank of `rerank`, with `d` the distance from the query to a row's vector.
fn rerank_table<'b, R, O, D>(
    index: &'b R,
    candidates: Candidates<'b>,
    vector: impl Borrow<O::Vector> + 'b,
    (io, prefilter, table): (Io, bool, impl Table + 'b),
    mut fetcher: impl Fetcher + 'b,
    (opfamily, d): (Opfamily, D),
) -> Box<dyn Iterator<Item = (f32, NonZero<u64>)> + 'b>
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
    let f = move |(distance, payload)| (opfamily.output(distance), payload);
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
        Io::Plain => {
            Box::new(rerank_heap(PlainPrefetcher::new(index, candidates), distance).map(f))
        }
        Io::Simple | Io::Stream => {
            let prefetcher = PlainPrefetcher::new(index, TableAhead::new(candidates, table));
            Box::new(rerank_heap(prefetcher, distance).map(f))
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
    reranking: (Io, bool, TablePrefetch, KHint),
    fetcher: impl Fetcher + 'b,
    opfamily: Opfamily,
) -> Box<dyn Iterator<Item = (f32, NonZero<u64>)> + 'b>
where
    R: RelationRead + RelationPrefetch + RelationReadStream,
    R::Page: Page<Opaque = vchordrq::Opaque>,
    O: Operator,
    O::Vector: FromOwned,
{
    let dim = vector.borrow().as_borrowed().dim();
    let accessor = move || O::DistanceAccessor::default_with_dimension(dim);
    if opfamily.is_cosine() {
        let reranking = (reranking, accessor, unit_distance::<O>);
        rerank_by::<_, O, _, _>(index, candidates, vector, reranking, fetcher, opfamily)
    } else {
        let reranking = (reranking, accessor, exact_distance::<O>);
        rerank_by::<_, O, _, _>(index, candidates, vector, reranking, fetcher, opfamily)
    }
}

/// [`rerank`] measuring copies in the index with `accessor()` and rows in the table with `d`.
fn rerank_by<'b, R, O, A, D>(
    index: &'b R,
    candidates: Candidates<'b>,
    vector: impl Borrow<O::Vector> + 'b,
    ((io, prefilter, table, k), accessor, d): (
        (Io, bool, TablePrefetch, KHint),
        impl Fn() -> A + Copy + 'b,
        D,
    ),
    fetcher: impl Fetcher + 'b,
    opfamily: Opfamily,
) -> Box<dyn Iterator<Item = (f32, NonZero<u64>)> + 'b>
where
    R: RelationRead + RelationPrefetch + RelationReadStream,
    R::Page: Page<Opaque = vchordrq::Opaque>,
    O: Operator,
    O::Vector: FromOwned,
    A: Accessor2<
            <O::Vector as Vector>::Element,
            <O::Vector as Vector>::Element,
            <O::Vector as Vector>::Metadata,
            <O::Vector as Vector>::Metadata,
            Output = Distance,
        > + 'b,
    D: Fn(
            <O::Vector as VectorOwned>::Borrowed<'_>,
            <O::Vector as VectorOwned>::Borrowed<'_>,
        ) -> Distance
        + 'b,
{
    if k.kind == KKind::Unknown {
        let reranking = ((io, prefilter, table), accessor, d);
        rerank_on::<_, O, _, _>(index, candidates, vector, reranking, fetcher, opfamily)
    } else {
        let reranking = ((io, prefilter, Hinted(table, k)), accessor, d);
        rerank_on::<_, O, _, _>(index, candidates, vector, reranking, fetcher, opfamily)
    }
}

/// [`rerank_by`] with the table read ahead as `table` does.
fn rerank_on<'b, R, O, A, D>(
    index: &'b R,
    candidates: Candidates<'b>,
    vector: impl Borrow<O::Vector> + 'b,
    ((io, prefilter, table), accessor, d): (
        (Io, bool, impl Table + 'b),
        impl Fn() -> A + Copy + 'b,
        D,
    ),
    mut fetcher: impl Fetcher + 'b,
    opfamily: Opfamily,
) -> Box<dyn Iterator<Item = (f32, NonZero<u64>)> + 'b>
where
    R: RelationRead + RelationPrefetch + RelationReadStream,
    R::Page: Page<Opaque = vchordrq::Opaque>,
    O: Operator,
    O::Vector: FromOwned,
    A: Accessor2<
            <O::Vector as Vector>::Element,
            <O::Vector as Vector>::Element,
            <O::Vector as Vector>::Metadata,
            <O::Vector as Vector>::Metadata,
            Output = Distance,
        > + 'b,
    D: Fn(
            <O::Vector as VectorOwned>::Borrowed<'_>,
            <O::Vector as VectorOwned>::Borrowed<'_>,
        ) -> Distance
        + 'b,
{
    if let RerankMethod::Heap = how(index) {
        let reranking = (io, prefilter, table);
        return rerank_table::<_, O, _>(
            index,
            candidates,
            vector,
            reranking,
            fetcher,
            (opfamily, d),
        );
    }
    let f = move |(distance, payload)| (opfamily.output(distance), payload);
    let predicate = id_0(move |(_, AlwaysEqual(PackedRefMut4((pointer, _, _))))| {
        let (key, _) = pointer_to_kv(*pointer);
        let Some(mut tuple) = fetcher.fetch(key) else {
            return false;
        };
        tuple.filter()
    });
    let hints = Hints::default().full(false);
    match (io, prefilter) {
        (Io::Plain, false) => {
            let prefetcher = PlainPrefetcher::new(index, candidates);
            Box::new(rerank_index_by::<O, _, _, _, _>(vector, prefetcher, (), accessor).map(f))
        }
        (Io::Plain, true) => {
            let prefetcher = PlainPrefetcher::new(index, filter(candidates, predicate));
            Box::new(rerank_index_by::<O, _, _, _, _>(vector, prefetcher, (), accessor).map(f))
        }
        (Io::Simple, false) => {
            let prefetcher = SimplePrefetcher::new(index, candidates);
            Box::new(rerank_index_by::<O, _, _, _, _>(vector, prefetcher, table, accessor).map(f))
        }
        (Io::Simple, true) => {
            let prefetcher = SimplePrefetcher::new(index, filter(candidates, predicate));
            Box::new(rerank_index_by::<O, _, _, _, _>(vector, prefetcher, (), accessor).map(f))
        }
        (Io::Stream, false) => {
            let prefetcher = StreamPrefetcher::new(index, candidates, hints);
            Box::new(rerank_index_by::<O, _, _, _, _>(vector, prefetcher, table, accessor).map(f))
        }
        (Io::Stream, true) => {
            let prefetcher = StreamPrefetcher::new(index, filter(candidates, predicate), hints);
            Box::new(rerank_index_by::<O, _, _, _, _>(vector, prefetcher, (), accessor).map(f))
        }
    }
}

#[inline(always)]
pub fn id_0<F, A: ?Sized, B: ?Sized, C: ?Sized, D: ?Sized, R: ?Sized>(f: F) -> F
where
    F: for<'a> FnMut(&(A, AlwaysEqual<PackedRefMut4<'a, (B, C, D)>>)) -> R,
{
    f
}
