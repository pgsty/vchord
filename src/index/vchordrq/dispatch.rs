// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use crate::index::vchordrq::build::{Normalize, Normalized};
use crate::index::vchordrq::opclass::Opfamily;
use index::bump::Bump;
use index::fetch::Fetch;
use index::prefetcher::*;
use index::relation::{
    Hints, Page, RelationPrefetch, RelationRead, RelationReadStream, RelationWrite,
};
use index_accessor::{Dot, L2S};
use simd::f16;
use std::collections::BinaryHeap;
use std::num::NonZero;
use vchordrq::operator::{Op, Operator};
use vchordrq::types::*;
use vchordrq::{FastHeap, InsertChooser};
use vector::VectorOwned;
use vector::rabitq4::Rabitq4Owned;
use vector::rabitq8::Rabitq8Owned;
use vector::vect::{VectBorrowed, VectOwned};

/// The page ids that every vector of `dim` dimensions keeps, to prefetch it.
pub fn prefetch(vector_kind: VectorKind, dim: u32) -> usize {
    use vchordrq::operator::Vector;
    (match vector_kind {
        VectorKind::Vecf32 => VectOwned::<f32>::count(dim),
        VectorKind::Vecf16 => VectOwned::<f16>::count(dim),
        VectorKind::Rabitq8 => Rabitq8Owned::count(dim),
        VectorKind::Rabitq4 => Rabitq4Owned::count(dim),
    }) as usize
}

pub fn prewarm<R>(opfamily: Opfamily, index: &R, height: i32) -> String
where
    R: RelationRead,
    R::Page: Page<Opaque = vchordrq::Opaque>,
{
    let make_h0_plain_prefetcher = MakeH0PlainPrefetcher { index };
    match (opfamily.vector_kind(), opfamily.distance_kind()) {
        (VectorKind::Vecf32, DistanceKind::L2S) => {
            vchordrq::prewarm::<_, Op<VectOwned<f32>, L2S>>(index, height, make_h0_plain_prefetcher)
        }
        (VectorKind::Vecf32, DistanceKind::Dot) => {
            vchordrq::prewarm::<_, Op<VectOwned<f32>, Dot>>(index, height, make_h0_plain_prefetcher)
        }
        (VectorKind::Vecf16, DistanceKind::L2S) => {
            vchordrq::prewarm::<_, Op<VectOwned<f16>, L2S>>(index, height, make_h0_plain_prefetcher)
        }
        (VectorKind::Vecf16, DistanceKind::Dot) => {
            vchordrq::prewarm::<_, Op<VectOwned<f16>, Dot>>(index, height, make_h0_plain_prefetcher)
        }
        (VectorKind::Rabitq8, DistanceKind::L2S) => {
            vchordrq::prewarm::<_, Op<Rabitq8Owned, L2S>>(index, height, make_h0_plain_prefetcher)
        }
        (VectorKind::Rabitq8, DistanceKind::Dot) => {
            vchordrq::prewarm::<_, Op<Rabitq8Owned, Dot>>(index, height, make_h0_plain_prefetcher)
        }
        (VectorKind::Rabitq4, DistanceKind::L2S) => {
            vchordrq::prewarm::<_, Op<Rabitq4Owned, L2S>>(index, height, make_h0_plain_prefetcher)
        }
        (VectorKind::Rabitq4, DistanceKind::Dot) => {
            vchordrq::prewarm::<_, Op<Rabitq4Owned, Dot>>(index, height, make_h0_plain_prefetcher)
        }
    }
}

/// Removes the index tuples whose payload `callback` selects and returns how
/// many were removed.
pub fn bulkdelete<R>(
    opfamily: Opfamily,
    index: &R,
    check: impl Fn(),
    callback: impl Fn(NonZero<u64>) -> bool,
) -> u64
where
    R: RelationRead + RelationWrite,
    R::Page: Page<Opaque = vchordrq::Opaque>,
{
    match (opfamily.vector_kind(), opfamily.distance_kind()) {
        (VectorKind::Vecf32, DistanceKind::L2S) => {
            let removed =
                vchordrq::bulkdelete::<_, Op<VectOwned<f32>, L2S>>(index, &check, &callback);
            vchordrq::bulkdelete_vectors::<_, Op<VectOwned<f32>, L2S>>(index, &check, &callback);
            removed
        }
        (VectorKind::Vecf32, DistanceKind::Dot) => {
            let removed =
                vchordrq::bulkdelete::<_, Op<VectOwned<f32>, Dot>>(index, &check, &callback);
            vchordrq::bulkdelete_vectors::<_, Op<VectOwned<f32>, Dot>>(index, &check, &callback);
            removed
        }
        (VectorKind::Vecf16, DistanceKind::L2S) => {
            let removed =
                vchordrq::bulkdelete::<_, Op<VectOwned<f16>, L2S>>(index, &check, &callback);
            vchordrq::bulkdelete_vectors::<_, Op<VectOwned<f16>, L2S>>(index, &check, &callback);
            removed
        }
        (VectorKind::Vecf16, DistanceKind::Dot) => {
            let removed =
                vchordrq::bulkdelete::<_, Op<VectOwned<f16>, Dot>>(index, &check, &callback);
            vchordrq::bulkdelete_vectors::<_, Op<VectOwned<f16>, Dot>>(index, &check, &callback);
            removed
        }
        (VectorKind::Rabitq8, DistanceKind::L2S) => {
            let removed =
                vchordrq::bulkdelete::<_, Op<Rabitq8Owned, L2S>>(index, &check, &callback);
            vchordrq::bulkdelete_vectors::<_, Op<Rabitq8Owned, L2S>>(index, &check, &callback);
            removed
        }
        (VectorKind::Rabitq8, DistanceKind::Dot) => {
            let removed =
                vchordrq::bulkdelete::<_, Op<Rabitq8Owned, Dot>>(index, &check, &callback);
            vchordrq::bulkdelete_vectors::<_, Op<Rabitq8Owned, Dot>>(index, &check, &callback);
            removed
        }
        (VectorKind::Rabitq4, DistanceKind::L2S) => {
            let removed =
                vchordrq::bulkdelete::<_, Op<Rabitq4Owned, L2S>>(index, &check, &callback);
            vchordrq::bulkdelete_vectors::<_, Op<Rabitq4Owned, L2S>>(index, &check, &callback);
            removed
        }
        (VectorKind::Rabitq4, DistanceKind::Dot) => {
            let removed =
                vchordrq::bulkdelete::<_, Op<Rabitq4Owned, Dot>>(index, &check, &callback);
            vchordrq::bulkdelete_vectors::<_, Op<Rabitq4Owned, Dot>>(index, &check, &callback);
            removed
        }
    }
}

/// What is done with a row's vector, as its opfamily's operator: the vector as it is stored, and as it is routed.
trait WithVector {
    fn with<O: Operator>(
        self,
        vector: <O::Vector as VectorOwned>::Borrowed<'_>,
        projected: <O::Vector as VectorOwned>::Borrowed<'_>,
    );
}

fn with_vector(opfamily: Opfamily, vector: OwnedVector, f: impl WithVector) {
    match (vector, opfamily.distance_kind()) {
        (OwnedVector::Vecf32(vector), DistanceKind::L2S) => {
            assert!(opfamily.vector_kind() == VectorKind::Vecf32);
            let projected = RandomProject::project(vector.as_borrowed());
            f.with::<Op<VectOwned<f32>, L2S>>(vector.as_borrowed(), projected.as_borrowed())
        }
        (OwnedVector::Vecf32(vector), DistanceKind::Dot) => {
            assert!(opfamily.vector_kind() == VectorKind::Vecf32);
            let projected = RandomProject::project(vector.as_borrowed());
            f.with::<Op<VectOwned<f32>, Dot>>(vector.as_borrowed(), projected.as_borrowed())
        }
        (OwnedVector::Vecf16(vector), DistanceKind::L2S) => {
            assert!(opfamily.vector_kind() == VectorKind::Vecf16);
            let projected = RandomProject::project(vector.as_borrowed());
            f.with::<Op<VectOwned<f16>, L2S>>(vector.as_borrowed(), projected.as_borrowed())
        }
        (OwnedVector::Vecf16(vector), DistanceKind::Dot) => {
            assert!(opfamily.vector_kind() == VectorKind::Vecf16);
            let projected = RandomProject::project(vector.as_borrowed());
            f.with::<Op<VectOwned<f16>, Dot>>(vector.as_borrowed(), projected.as_borrowed())
        }
        (OwnedVector::Rabitq8(vector), DistanceKind::Dot) => {
            assert!(opfamily.vector_kind() == VectorKind::Rabitq8);
            f.with::<Op<Rabitq8Owned, Dot>>(vector.as_borrowed(), vector.as_borrowed())
        }
        (OwnedVector::Rabitq8(vector), DistanceKind::L2S) => {
            assert!(opfamily.vector_kind() == VectorKind::Rabitq8);
            f.with::<Op<Rabitq8Owned, L2S>>(vector.as_borrowed(), vector.as_borrowed())
        }
        (OwnedVector::Rabitq4(vector), DistanceKind::Dot) => {
            assert!(opfamily.vector_kind() == VectorKind::Rabitq4);
            f.with::<Op<Rabitq4Owned, Dot>>(vector.as_borrowed(), vector.as_borrowed())
        }
        (OwnedVector::Rabitq4(vector), DistanceKind::L2S) => {
            assert!(opfamily.vector_kind() == VectorKind::Rabitq4);
            f.with::<Op<Rabitq4Owned, L2S>>(vector.as_borrowed(), vector.as_borrowed())
        }
    }
}

/// Encodes the row as the build sorts it.
pub fn encode<R>(
    opfamily: Opfamily,
    encoder: &mut vchordrq::Encoder<'_, R>,
    payload: NonZero<u64>,
    columns: &[u8],
    vector: OwnedVector,
    bump: &impl Bump,
) where
    R: RelationRead,
    R::Page: Page<Opaque = vchordrq::Opaque>,
{
    struct Encode<'a, 'b, R, B> {
        encoder: &'a mut vchordrq::Encoder<'b, R>,
        payload: NonZero<u64>,
        columns: &'a [u8],
        bump: &'a B,
    }
    impl<R: RelationRead, B: Bump> WithVector for Encode<'_, '_, R, B>
    where
        R::Page: Page<Opaque = vchordrq::Opaque>,
    {
        fn with<O: Operator>(
            self,
            vector: <O::Vector as VectorOwned>::Borrowed<'_>,
            projected: <O::Vector as VectorOwned>::Borrowed<'_>,
        ) {
            let index = self.encoder.index();
            let prefetch = MakeH1PlainPrefetcherForInsertion { index };
            let vectors = (vector, projected);
            let Self {
                encoder,
                payload,
                columns,
                bump,
            } = self;
            encoder.encode::<O>(payload, columns, vectors, bump, prefetch);
        }
    }
    let encode = Encode {
        encoder,
        payload,
        columns,
        bump,
    };
    with_vector(opfamily, vector, encode);
}

/// What is done as an opfamily's operator.
trait WithOperator {
    type Output;
    fn with<O: Operator>(self) -> Self::Output;
}

fn with_operator<F: WithOperator>(opfamily: Opfamily, f: F) -> F::Output {
    match (opfamily.vector_kind(), opfamily.distance_kind()) {
        (VectorKind::Vecf32, DistanceKind::L2S) => f.with::<Op<VectOwned<f32>, L2S>>(),
        (VectorKind::Vecf32, DistanceKind::Dot) => f.with::<Op<VectOwned<f32>, Dot>>(),
        (VectorKind::Vecf16, DistanceKind::L2S) => f.with::<Op<VectOwned<f16>, L2S>>(),
        (VectorKind::Vecf16, DistanceKind::Dot) => f.with::<Op<VectOwned<f16>, Dot>>(),
        (VectorKind::Rabitq8, DistanceKind::Dot) => f.with::<Op<Rabitq8Owned, Dot>>(),
        (VectorKind::Rabitq8, DistanceKind::L2S) => f.with::<Op<Rabitq8Owned, L2S>>(),
        (VectorKind::Rabitq4, DistanceKind::Dot) => f.with::<Op<Rabitq4Owned, Dot>>(),
        (VectorKind::Rabitq4, DistanceKind::L2S) => f.with::<Op<Rabitq4Owned, L2S>>(),
    }
}

/// Writes every list from the rows `encode` appends, sorted, holding at most `memory` bytes of a list's codes.
pub fn write<R>(
    opfamily: Opfamily,
    index: &R,
    rows: impl vchordrq::Sorted,
    memory: usize,
    spill: &mut impl vchordrq::Spill,
) where
    R: RelationRead + RelationWrite,
    R::Page: Page<Opaque = vchordrq::Opaque>,
{
    struct Write<'a, R, S, P> {
        index: &'a R,
        rows: S,
        memory: usize,
        spill: &'a mut P,
    }
    impl<R, S: vchordrq::Sorted, P: vchordrq::Spill> WithOperator for Write<'_, R, S, P>
    where
        R: RelationRead + RelationWrite,
        R::Page: Page<Opaque = vchordrq::Opaque>,
    {
        type Output = ();
        fn with<O: Operator>(self) {
            vchordrq::write::<_, O>(self.index, self.rows, self.memory, self.spill)
        }
    }
    let write = Write {
        index,
        rows,
        memory,
        spill,
    };
    with_operator(opfamily, write)
}

/// What a list's rows may take to be sorted: maintenance memory, as a vacuum reads it.
fn memory() -> usize {
    use pgrx::pg_sys::{BackendType, MyBackendType, autovacuum_work_mem, maintenance_work_mem};
    let kilobytes = unsafe {
        if MyBackendType == BackendType::B_AUTOVAC_WORKER && autovacuum_work_mem != -1 {
            autovacuum_work_mem
        } else {
            maintenance_work_mem
        }
    };
    kilobytes as usize * 1024
}

/// Compacts the lists and returns the number of live index tuples in them.
pub fn maintain<R>(
    opfamily: Opfamily,
    index: &R,
    pending: &vchordrq::Pending,
    check: impl Fn(),
) -> u64
where
    R: RelationRead + RelationWrite,
    R::Page: Page<Opaque = vchordrq::Opaque>,
{
    struct Maintain<'a, R, C> {
        index: &'a R,
        pending: &'a vchordrq::Pending,
        check: C,
    }
    impl<'a, R, C: Fn()> WithOperator for Maintain<'a, R, C>
    where
        R: RelationRead + RelationWrite,
        R::Page: Page<Opaque = vchordrq::Opaque>,
    {
        type Output = vchordrq::Maintain;
        fn with<O: Operator>(self) -> vchordrq::Maintain {
            let Self {
                index,
                pending,
                check,
            } = self;
            let prefetch = MakeH0PlainPrefetcher { index };
            vchordrq::maintain::<_, O>(index, prefetch, memory(), pending, check)
        }
    }
    let maintain = with_operator(
        opfamily,
        Maintain {
            index,
            pending,
            check,
        },
    );
    pgrx::debug1!(
        "maintain: number_of_formerly_allocated_pages = {}",
        maintain.number_of_formerly_allocated_pages
    );
    pgrx::debug1!(
        "maintain: number_of_freshly_allocated_pages = {}",
        maintain.number_of_freshly_allocated_pages
    );
    pgrx::debug1!(
        "maintain: number_of_freed_pages = {}",
        maintain.number_of_freed_pages
    );
    maintain.number_of_tuples
}

pub fn build<R>(
    vector_options: VectorOptions,
    vchordrq_options: VchordrqIndexOptions,
    keys: &vchordrq::keys::Keys,
    index: &R,
    structures: Vec<Structure<Normalized>>,
) where
    R: RelationRead + RelationWrite,
    R::Page: Page<Opaque = vchordrq::Opaque>,
{
    match (vector_options.v, vector_options.d) {
        (VectorKind::Vecf32, DistanceKind::L2S) => vchordrq::build::<_, Op<VectOwned<f32>, L2S>>(
            vector_options,
            vchordrq_options,
            keys,
            index,
            map_structures(structures, Normalize::denormalize),
        ),
        (VectorKind::Vecf32, DistanceKind::Dot) => vchordrq::build::<_, Op<VectOwned<f32>, Dot>>(
            vector_options,
            vchordrq_options,
            keys,
            index,
            map_structures(structures, Normalize::denormalize),
        ),
        (VectorKind::Vecf16, DistanceKind::L2S) => vchordrq::build::<_, Op<VectOwned<f16>, L2S>>(
            vector_options,
            vchordrq_options,
            keys,
            index,
            map_structures(structures, Normalize::denormalize),
        ),
        (VectorKind::Vecf16, DistanceKind::Dot) => vchordrq::build::<_, Op<VectOwned<f16>, Dot>>(
            vector_options,
            vchordrq_options,
            keys,
            index,
            map_structures(structures, Normalize::denormalize),
        ),
        (VectorKind::Rabitq8, DistanceKind::L2S) => vchordrq::build::<_, Op<Rabitq8Owned, L2S>>(
            vector_options,
            vchordrq_options,
            keys,
            index,
            map_structures(structures, Normalize::denormalize),
        ),
        (VectorKind::Rabitq8, DistanceKind::Dot) => vchordrq::build::<_, Op<Rabitq8Owned, Dot>>(
            vector_options,
            vchordrq_options,
            keys,
            index,
            map_structures(structures, Normalize::denormalize),
        ),
        (VectorKind::Rabitq4, DistanceKind::L2S) => vchordrq::build::<_, Op<Rabitq4Owned, L2S>>(
            vector_options,
            vchordrq_options,
            keys,
            index,
            map_structures(structures, Normalize::denormalize),
        ),
        (VectorKind::Rabitq4, DistanceKind::Dot) => vchordrq::build::<_, Op<Rabitq4Owned, Dot>>(
            vector_options,
            vchordrq_options,
            keys,
            index,
            map_structures(structures, Normalize::denormalize),
        ),
    }
}

pub fn insert<R>(
    opfamily: Opfamily,
    index: &R,
    payload: NonZero<u64>,
    columns: &[u8],
    vector: OwnedVector,
    chooser: &mut impl InsertChooser,
    bump: &impl Bump,
) where
    R: RelationRead + RelationWrite,
    R::Page: Page<Opaque = vchordrq::Opaque>,
{
    struct Insert<'a, R, C, B> {
        index: &'a R,
        payload: NonZero<u64>,
        columns: &'a [u8],
        chooser: &'a mut C,
        bump: &'a B,
    }
    impl<R: RelationRead + RelationWrite, C: InsertChooser, B: Bump> WithVector for Insert<'_, R, C, B>
    where
        R::Page: Page<Opaque = vchordrq::Opaque>,
    {
        fn with<O: Operator>(
            self,
            vector: <O::Vector as VectorOwned>::Borrowed<'_>,
            projected: <O::Vector as VectorOwned>::Borrowed<'_>,
        ) {
            let Self {
                index,
                payload,
                columns,
                chooser,
                bump,
            } = self;
            let key = vchordrq::insert_vector::<_, O>(index, payload, vector, chooser);
            let prefetch = MakeH1PlainPrefetcherForInsertion { index };
            vchordrq::insert::<_, O>(index, payload, columns, projected, key, bump, prefetch)
        }
    }
    let insert = Insert {
        index,
        payload,
        columns,
        chooser,
        bump,
    };
    with_vector(opfamily, vector, insert);
}

fn map_structures<T, U>(x: Vec<Structure<T>>, f: impl Fn(T) -> U + Copy) -> Vec<Structure<U>> {
    x.into_iter()
        .map(
            |Structure {
                 centroids,
                 children,
             }| Structure {
                centroids: centroids.into_iter().map(f).collect(),
                children,
            },
        )
        .collect()
}

pub trait RandomProject {
    type Output;
    fn project(self) -> Self::Output;
}

impl RandomProject for VectBorrowed<'_, f32> {
    type Output = VectOwned<f32>;
    fn project(self) -> VectOwned<f32> {
        use rabitq::rotate::rotate;
        let input = self.slice();
        VectOwned::new(rotate(input))
    }
}

impl RandomProject for VectBorrowed<'_, f16> {
    type Output = VectOwned<f16>;
    fn project(self) -> VectOwned<f16> {
        use rabitq::rotate::rotate;
        use simd::Floating;
        let input = f16::vector_to_f32(self.slice());
        VectOwned::new(f16::vector_from_f32(&rotate(&input)))
    }
}

#[derive(Debug)]
pub struct MakeH1PlainPrefetcherForInsertion<'b, R> {
    pub index: &'b R,
}

impl<'b, R> Clone for MakeH1PlainPrefetcherForInsertion<'b, R> {
    fn clone(&self) -> Self {
        Self { index: self.index }
    }
}

impl<'b, R: RelationRead> PrefetcherHeapFamily<'b, R> for MakeH1PlainPrefetcherForInsertion<'b, R> {
    type P<T>
        = PlainPrefetcher<'b, R, FastHeap<T>>
    where
        T: Ord + Fetch<'b>;

    fn prefetch<T>(&mut self, seq: Vec<T>) -> Self::P<T>
    where
        T: Ord + Fetch<'b>,
    {
        PlainPrefetcher::new(self.index, FastHeap::from(seq))
    }

    fn is_not_plain(&self) -> bool {
        false
    }
}

#[derive(Debug)]
pub struct MakeH1PlainPrefetcher<'b, R> {
    pub index: &'b R,
}

impl<'b, R> Clone for MakeH1PlainPrefetcher<'b, R> {
    fn clone(&self) -> Self {
        Self { index: self.index }
    }
}

impl<'b, R: RelationRead> PrefetcherHeapFamily<'b, R> for MakeH1PlainPrefetcher<'b, R> {
    type P<T>
        = PlainPrefetcher<'b, R, BinaryHeap<T>>
    where
        T: Ord + Fetch<'b>;

    fn prefetch<T>(&mut self, seq: Vec<T>) -> Self::P<T>
    where
        T: Ord + Fetch<'b>,
    {
        PlainPrefetcher::new(self.index, BinaryHeap::from(seq))
    }

    fn is_not_plain(&self) -> bool {
        false
    }
}

#[derive(Debug)]
pub struct MakeH0PlainPrefetcher<'b, R> {
    pub index: &'b R,
}

impl<'b, R> Clone for MakeH0PlainPrefetcher<'b, R> {
    fn clone(&self) -> Self {
        Self { index: self.index }
    }
}

impl<'b, R: RelationRead> PrefetcherSequenceFamily<'b, R> for MakeH0PlainPrefetcher<'b, R> {
    type P<S: Sequence>
        = PlainPrefetcher<'b, R, S>
    where
        S::Item: Fetch<'b>;

    fn prefetch<S: Sequence>(&mut self, seq: S) -> Self::P<S>
    where
        S::Item: Fetch<'b>,
    {
        PlainPrefetcher::new(self.index, seq)
    }

    fn is_not_plain(&self) -> bool {
        false
    }
}

#[derive(Debug)]
pub struct MakeH0SimplePrefetcher<'b, R> {
    pub index: &'b R,
}

impl<'b, R> Clone for MakeH0SimplePrefetcher<'b, R> {
    fn clone(&self) -> Self {
        Self { index: self.index }
    }
}

impl<'b, R: RelationRead + RelationPrefetch> PrefetcherSequenceFamily<'b, R>
    for MakeH0SimplePrefetcher<'b, R>
{
    type P<S: Sequence>
        = SimplePrefetcher<'b, R, S>
    where
        S::Item: Fetch<'b>;

    fn prefetch<S: Sequence>(&mut self, seq: S) -> Self::P<S>
    where
        S::Item: Fetch<'b>,
    {
        SimplePrefetcher::new(self.index, seq)
    }

    fn is_not_plain(&self) -> bool {
        true
    }
}

#[derive(Debug)]
pub struct MakeH0MixedPrefetcher<'b, R> {
    pub index: &'b R,
    pub hints: Hints,
}

impl<'b, R> Clone for MakeH0MixedPrefetcher<'b, R> {
    fn clone(&self) -> Self {
        Self {
            index: self.index,
            hints: self.hints,
        }
    }
}

impl<'b, R: RelationRead + RelationPrefetch + RelationReadStream> PrefetcherSequenceFamily<'b, R>
    for MakeH0MixedPrefetcher<'b, R>
{
    type P<S: Sequence>
        = MixedPrefetcher<'b, R, S>
    where
        S::Item: Fetch<'b>;

    fn prefetch<S: Sequence>(&mut self, seq: S) -> Self::P<S>
    where
        S::Item: Fetch<'b>,
    {
        MixedPrefetcher::new(self.index, seq, self.hints)
    }

    fn is_not_plain(&self) -> bool {
        true
    }
}

#[derive(Debug)]
pub struct MakeH0StreamPrefetcher<'b, R> {
    pub index: &'b R,
    pub hints: Hints,
}

impl<'b, R> Clone for MakeH0StreamPrefetcher<'b, R> {
    fn clone(&self) -> Self {
        Self {
            index: self.index,
            hints: self.hints,
        }
    }
}

impl<'b, R: RelationRead + RelationReadStream> PrefetcherSequenceFamily<'b, R>
    for MakeH0StreamPrefetcher<'b, R>
{
    type P<S: Sequence>
        = StreamPrefetcher<'b, R, S>
    where
        S::Item: Fetch<'b>;

    fn prefetch<S: Sequence>(&mut self, seq: S) -> Self::P<S>
    where
        S::Item: Fetch<'b>,
    {
        StreamPrefetcher::new(self.index, seq, self.hints)
    }

    fn is_not_plain(&self) -> bool {
        true
    }
}
