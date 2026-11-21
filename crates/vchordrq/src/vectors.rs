// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use crate::operator::*;
use crate::tuples::*;
use crate::{Opaque, tape};
use index::relation::{Page, PageGuard, RelationRead, RelationWrite};
use index_accessor::TryAccessor1;
use std::num::NonZero;
use vector::VectorOwned;
use zerocopy::{FromBytes, FromZeros, IntoBytes};

pub fn read<
    'a,
    R: RelationRead + 'a,
    O: Operator,
    A: TryAccessor1<<O::Vector as Vector>::Element, <O::Vector as Vector>::Metadata>,
>(
    mut prefetch: impl Iterator<Item = R::ReadGuard<'a>>,
    head: u16,
    payload: NonZero<u64>,
    accessor: A,
) -> Option<A::Output> {
    let mut cursor = Err(head);
    let mut result = accessor;
    while let Err(head) = cursor {
        let guard = prefetch.next()?;
        let bytes = guard.get(head)?;
        let tuple = VectorTuple::<O::Vector>::deserialize_ref(bytes);
        if tuple.payload().is_none() {
            panic!("data corruption");
        }
        if tuple.payload() != Some(payload) {
            return None;
        }
        result.push(tuple.elements())?;
        cursor = tuple.metadata_or_head();
    }
    if prefetch.next().is_some() {
        return None;
    }
    result.finish(cursor.ok()?)
}

pub fn append<O: Operator, R: RelationRead + RelationWrite>(
    index: &R,
    vectors_first: u32,
    vector: <O::Vector as VectorOwned>::Borrowed<'_>,
    payload: NonZero<u64>,
) -> (Vec<u32>, u16)
where
    R::Page: Page<Opaque = Opaque>,
{
    write::<O>(vector, payload, |tuple| {
        let bytes = VectorTuple::serialize(&tuple);
        if let Some(mut write) = index.search(bytes.len()) {
            let i = write
                .alloc(&bytes)
                .expect("implementation: a free page cannot accommodate a single tuple");
            return (write.id(), i);
        }
        tape::append(index, vectors_first, &bytes, true, None)
    })
}

/// Puts `vector`'s parts, its last first; returns their pages, its first first, and the slot of its first.
pub fn write<O: Operator>(
    vector: <O::Vector as VectorOwned>::Borrowed<'_>,
    payload: NonZero<u64>,
    mut put: impl FnMut(VectorTuple<O::Vector>) -> (u32, u16),
) -> (Vec<u32>, u16) {
    let (slices, metadata) = O::Vector::split(vector);
    let mut chain = Ok(metadata);
    let mut prefetch = Vec::new();
    for i in (0..slices.len()).rev() {
        let (id, head) = put(match chain {
            Ok(metadata) => VectorTuple::_0 {
                elements: slices[i].to_vec(),
                payload: Some(payload),
                metadata,
            },
            Err(head) => VectorTuple::_1 {
                elements: slices[i].to_vec(),
                payload: Some(payload),
                head,
            },
        });
        chain = Err(head);
        prefetch.push(id);
    }
    prefetch.reverse();
    (
        prefetch,
        chain.expect_err("internal error: 0-dimensional vector"),
    )
}

/// Appends `vector` as the build carries it to its list: its metadata, then its elements.
pub fn encode<V: Vector>(vector: V::Borrowed<'_>, out: &mut Vec<u8>) {
    let (elements, metadata) = V::unpack(vector);
    out.extend_from_slice(metadata.as_bytes());
    out.extend_from_slice(elements.as_bytes());
}

/// A vector of `dim` dimensions from what `encode` appends.
pub fn decode<V: Vector>(dim: u32, bytes: &[u8]) -> V {
    let (metadata, bytes) = bytes.split_at(size_of::<V::Metadata>());
    let metadata = V::Metadata::read_from_bytes(metadata).expect("internal error: a bad vector");
    let mut elements = vec![V::Element::new_zeroed(); bytes.len() / size_of::<V::Element>()];
    elements.as_mut_bytes().copy_from_slice(bytes);
    V::pack(dim, elements, metadata)
}
