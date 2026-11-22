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
        let ordered = !self.orderbys.is_empty();
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
        let bound = threshold.map(|radius| opfamily.bound(radius));
        let Some(vector) = vector else {
            return Box::new(std::iter::empty()) as Box<dyn Iterator<Item = (f32, [u16; 3], bool)>>;
        };
        // A zero/non-finite cosine query has no direction. Cosine indexes
        // contain only defined vectors and return no neighbours for it.
        if !opfamily.valid_cosine_query(&vector) {
            return Box::new(std::iter::empty());
        }
        let reranking = (options.io_rerank, options.prefilter, options.table);
        let finish = Finish {
            ordered,
            threshold,
            context: options.scan_context,
            max_scan_tuples: options.max_scan_tuples,
        };
        let iter = match (opfamily.vector_kind(), opfamily.distance_kind()) {
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
                finish.rows::<_, Op>(index, candidates, unprojected, reranking, fetcher, opfamily)
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
                finish.rows::<_, Op>(index, candidates, unprojected, reranking, fetcher, opfamily)
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
                finish.rows::<_, Op>(index, candidates, unprojected, reranking, fetcher, opfamily)
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
                finish.rows::<_, Op>(index, candidates, unprojected, reranking, fetcher, opfamily)
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
                finish.rows::<_, Op>(index, candidates, unprojected, reranking, fetcher, opfamily)
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
                finish.rows::<_, Op>(index, candidates, unprojected, reranking, fetcher, opfamily)
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
                finish.rows::<_, Op>(index, candidates, unprojected, reranking, fetcher, opfamily)
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
                finish.rows::<_, Op>(index, candidates, unprojected, reranking, fetcher, opfamily)
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

struct Finish {
    ordered: bool,
    threshold: Option<f32>,
    context: pgrx::pg_sys::MemoryContext,
    max_scan_tuples: Option<u32>,
}

impl Finish {
    fn rows<'b, R, O>(
        self,
        index: &'b R,
        search: vchordrq::Search<'b>,
        vector: impl Borrow<O::Vector> + 'b,
        options: (Io, bool, TablePrefetch),
        mut fetcher: impl Fetcher + 'b,
        opfamily: Opfamily,
    ) -> Ranked<'b>
    where
        R: RelationRead + RelationPrefetch + RelationReadStream,
        R::Page: Page<Opaque = vchordrq::Opaque>,
        O: Operator,
        O::Vector: FromOwned,
    {
        if self.max_scan_tuples == Some(0) {
            return Box::new(std::iter::empty());
        }
        // `<=`: the index's distance may round up to the radius where the operator's is under it.
        let within = move |(distance, _): &(Distance, NonZero<u64>)| {
            self.threshold
                .is_none_or(|r| opfamily.output(*distance) <= r)
        };
        match search {
            vchordrq::Search::Single(candidates) => {
                let rows = rerank::<_, O>(index, candidates, vector, options, fetcher, opfamily);
                if self.threshold.is_some() {
                    Box::new(rows.take_while(within))
                } else {
                    rows
                }
            }
            vchordrq::Search::Refill(mut refill) => {
                let mut scratch = bumpalo::Bump::new();
                if self.ordered {
                    let mut sorted = unsafe { SortedRows::new(self.context, self.max_scan_tuples) };
                    while let Some(candidates) = refill(&scratch) {
                        let rows = rerank::<_, O>(
                            index,
                            candidates,
                            vector.borrow(),
                            options,
                            &mut fetcher,
                            opfamily,
                        );
                        for (distance, payload) in rows.filter(within) {
                            sorted.push(distance, payload);
                        }
                        scratch.reset();
                    }
                    unsafe {
                        pgrx::pg_sys::tuplesort_performsort(sorted.sort);
                    }
                    Box::new(sorted)
                } else {
                    let mut rows = Vec::new().into_iter();
                    Box::new(std::iter::from_fn(move || {
                        loop {
                            if let Some(row) = rows.next() {
                                return Some(row);
                            }
                            scratch.reset();
                            let candidates = refill(&scratch)?;
                            let list = rerank::<_, O>(
                                index,
                                candidates,
                                vector.borrow(),
                                options,
                                &mut fetcher,
                                opfamily,
                            );
                            rows = list.filter(within).collect::<Vec<_>>().into_iter();
                        }
                    }))
                }
            }
        }
    }
}

struct SortedRows {
    sort: *mut pgrx::pg_sys::Tuplesortstate,
    slot: *mut pgrx::pg_sys::TupleTableSlot,
    desc: pgrx::pg_sys::TupleDesc,
}

impl SortedRows {
    unsafe fn new(context: pgrx::pg_sys::MemoryContext, bound: Option<u32>) -> Self {
        use pgrx::pg_sys;
        use std::ptr::{null, null_mut};
        unsafe {
            let _switch = crate::index::MemoryContextSwitch::new(context);
            let desc = pg_sys::CreateTemplateTupleDesc(2);
            pg_sys::TupleDescInitEntry(desc, 1, null(), pg_sys::FLOAT4OID, -1, 0);
            pg_sys::TupleDescInitEntry(desc, 2, null(), pg_sys::INT8OID, -1, 0);
            #[cfg(feature = "pg19")]
            pg_sys::TupleDescFinalize(desc);
            let slot =
                pg_sys::MakeSingleTupleTableSlot(desc, &raw const pg_sys::TTSOpsMinimalTuple);
            let float = pg_sys::lookup_type_cache(pg_sys::FLOAT4OID, pg_sys::TYPECACHE_LT_OPR as _);
            let sort = pg_sys::tuplesort_begin_heap(
                desc,
                2,
                [1, 2].as_mut_ptr(),
                [(*float).lt_opr, pg_sys::Oid::from(pg_sys::Int8LessOperator)].as_mut_ptr(),
                [pg_sys::InvalidOid; 2].as_mut_ptr(),
                [false; 2].as_mut_ptr(),
                pg_sys::work_mem,
                null_mut(),
                pg_sys::TUPLESORT_ALLOWBOUNDED as _,
            );
            if let Some(bound) = bound {
                pg_sys::tuplesort_set_bound(sort, bound.into());
            }
            Self { sort, slot, desc }
        }
    }

    fn push(&mut self, distance: Distance, payload: NonZero<u64>) {
        use pgrx::pg_sys;
        unsafe {
            pg_sys::ExecClearTuple(self.slot);
            (*self.slot)
                .tts_values
                .write(pg_sys::Datum::from(distance.to_f32().to_bits()));
            (*self.slot)
                .tts_values
                .add(1)
                .write(pg_sys::Datum::from(payload.get()));
            (*self.slot).tts_isnull.write(false);
            (*self.slot).tts_isnull.add(1).write(false);
            pg_sys::ExecStoreVirtualTuple(self.slot);
            pg_sys::tuplesort_puttupleslot(self.sort, self.slot);
        }
    }
}

impl Iterator for SortedRows {
    type Item = (Distance, NonZero<u64>);

    fn next(&mut self) -> Option<Self::Item> {
        use pgrx::pg_sys;
        unsafe {
            if !pg_sys::tuplesort_gettupleslot(
                self.sort,
                true,
                false,
                self.slot,
                std::ptr::null_mut(),
            ) {
                return None;
            }
            pg_sys::slot_getallattrs(self.slot);
            let values = (*self.slot).tts_values;
            let distance = f32::from_bits((*values).value() as u32);
            let payload = NonZero::new((*values.add(1)).value() as u64).unwrap();
            Some((Distance::from_f32(distance), payload))
        }
    }
}

impl Drop for SortedRows {
    fn drop(&mut self) {
        if crate::index::cleanup::pg_cleanup_allowed() {
            unsafe {
                pgrx::pg_sys::ExecDropSingleTupleTableSlot(self.slot);
                pgrx::pg_sys::tuplesort_end(self.sort);
                pgrx::pg_sys::FreeTupleDesc(self.desc);
            }
        }
    }
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
    (io, prefilter, table): (Io, bool, TablePrefetch),
    mut fetcher: impl Fetcher + 'b,
    opfamily: Opfamily,
) -> Ranked<'b>
where
    R: RelationRead + RelationPrefetch + RelationReadStream,
    R::Page: Page<Opaque = vchordrq::Opaque>,
    O: Operator,
    O::Vector: FromOwned,
{
    if let RerankMethod::Heap = how(index) {
        let reranking = (io, prefilter, table);
        return if opfamily.is_cosine() {
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
        };
    }
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
            Box::new(rerank_index::<O, _, _, _>(vector, prefetcher, ()))
        }
        (Io::Plain, true) => {
            let prefetcher = PlainPrefetcher::new(index, filter(candidates, predicate));
            Box::new(rerank_index::<O, _, _, _>(vector, prefetcher, ()))
        }
        (Io::Simple, false) => {
            let prefetcher = SimplePrefetcher::new(index, candidates);
            Box::new(rerank_index::<O, _, _, _>(vector, prefetcher, table))
        }
        (Io::Simple, true) => {
            let prefetcher = SimplePrefetcher::new(index, filter(candidates, predicate));
            Box::new(rerank_index::<O, _, _, _>(vector, prefetcher, ()))
        }
        (Io::Stream, false) => {
            let prefetcher = StreamPrefetcher::new(index, candidates, hints);
            Box::new(rerank_index::<O, _, _, _>(vector, prefetcher, table))
        }
        (Io::Stream, true) => {
            let prefetcher = StreamPrefetcher::new(index, filter(candidates, predicate), hints);
            Box::new(rerank_index::<O, _, _, _>(vector, prefetcher, ()))
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
