// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use crate::closure_lifetime_binder::id_4;
use crate::operator::*;
use crate::tuples::{MetaTuple, WithReader};
use crate::{RerankMethod, vectors};
use always_equal::AlwaysEqual;
use distance::Distance;
use index::fetch::BorrowedIter;
use index::packed::PackedRefMut;
use index::prefetcher::Prefetcher;
use index::relation::{Page, RelationRead};
use index_accessor::{Accessor2, DefaultWithDimension, LTryAccess};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::marker::PhantomData;
use std::num::NonZero;
use vector::{VectorBorrowed, VectorOwned};

type Result = (Reverse<Distance>, AlwaysEqual<NonZero<u64>>);

pub fn how(index: &impl RelationRead) -> RerankMethod {
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
    let rerank_in_heap = meta_tuple.rerank_in_heap();
    if rerank_in_heap {
        RerankMethod::Heap
    } else {
        RerankMethod::Index
    }
}

pub struct Reranker<T, F, P, W> {
    prefetcher: P,
    cache: BinaryHeap<Result>,
    f: F,
    _phantom: PhantomData<fn(T, W) -> (T, W)>,
}

impl<'b, T, F, P, W> Iterator for Reranker<T, F, P, W>
where
    F: FnMut(NonZero<u64>, P::Guards, u16) -> Option<Distance>,
    P: Prefetcher<'b, Item = ((Reverse<Distance>, AlwaysEqual<T>), AlwaysEqual<W>)>,
    W: 'b + PackedRefMut<T = (NonZero<u64>, u16, BorrowedIter<'b>)>,
{
    type Item = (Distance, NonZero<u64>);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // The previous candidate's guards are released here (#28).
            index::interrupt::check();
            let Some(((_, AlwaysEqual(mut w)), prefetch)) = self
                .prefetcher
                .next_if(|((d, _), ..)| Some(*d) > self.cache.peek().map(|(d, ..)| *d))
            else {
                break;
            };
            let &mut (payload, head, ..) = w.get_mut();
            if let Some(distance) = (self.f)(payload, prefetch, head) {
                self.cache.push((Reverse(distance), AlwaysEqual(payload)));
            };
        }
        let (Reverse(distance), AlwaysEqual(payload)) = self.cache.pop()?;
        Some((distance, payload))
    }
}

impl<T, F, P, W> Reranker<T, F, P, W> {
    pub fn finish(self) -> (P, impl Iterator<Item = Result>) {
        (self.prefetcher, self.cache.into_iter())
    }
}

pub fn rerank_index<
    'b,
    O: Operator,
    T,
    P: Prefetcher<'b, Item = ((Reverse<Distance>, AlwaysEqual<T>), AlwaysEqual<W>)>,
    W: 'b + PackedRefMut<T = (NonZero<u64>, u16, BorrowedIter<'b>)>,
>(
    vector: O::Vector,
    prefetcher: P,
) -> Reranker<T, impl FnMut(NonZero<u64>, P::Guards, u16) -> Option<Distance>, P, W> {
    let dim = vector.as_borrowed().dim();
    Reranker {
        prefetcher,
        cache: BinaryHeap::new(),
        f: id_4::<_, P, _, _, _>(move |payload, prefetch, head| {
            vectors::read::<P::R, O, _>(
                prefetch,
                head,
                payload,
                LTryAccess::new(
                    O::Vector::unpack(vector.as_borrowed()),
                    O::DistanceAccessor::default_with_dimension(dim),
                ),
            )
        }),
        _phantom: PhantomData,
    }
}

pub fn rerank_heap<
    'b,
    O: Operator,
    T,
    P: Prefetcher<'b, Item = ((Reverse<Distance>, AlwaysEqual<T>), AlwaysEqual<W>)>,
    W: 'b + PackedRefMut<T = (NonZero<u64>, u16, BorrowedIter<'b>)>,
>(
    vector: O::Vector,
    prefetcher: P,
    mut fetch: impl FnMut(NonZero<u64>) -> Option<O::Vector> + 'b,
) -> Reranker<T, impl FnMut(NonZero<u64>, P::Guards, u16) -> Option<Distance>, P, W> {
    let dim = vector.as_borrowed().dim();
    Reranker {
        prefetcher,
        cache: BinaryHeap::new(),
        f: id_4::<_, P, _, _, _>(move |payload, _, _| {
            let unpack = O::Vector::unpack(vector.as_borrowed());
            let vector = fetch(payload)?;
            let vector = O::Vector::unpack(vector.as_borrowed());
            let mut accessor = O::DistanceAccessor::default_with_dimension(dim);
            accessor.push(unpack.0, vector.0);
            let distance = accessor.finish(unpack.1, vector.1);
            Some(distance)
        }),
        _phantom: PhantomData,
    }
}
