// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use crate::keys::Keys;
use crate::tape::TapeWriter;
use crate::tuples::*;
use crate::{Branch, Opaque};
use index::relation::{Page, RelationWrite};
use rabitq::packing::{any_pack, padding_pack};
use std::num::NonZero;

pub struct DirectoryTapeWriter<'a, R>
where
    R: RelationWrite + 'a,
{
    tape: TapeWriter<'a, R, DirectoryTuple>,
}

impl<'a, R> DirectoryTapeWriter<'a, R>
where
    R: RelationWrite + 'a,
    R::Page: Page<Opaque = Opaque>,
{
    pub fn create(index: &'a R, tracking_freespace: bool) -> Self {
        Self {
            tape: TapeWriter::create(index, tracking_freespace),
        }
    }
    pub fn push(&mut self, branch: &[u32]) {
        self.tape.push_parts(DirectoryTuple::_0 {
            elements: branch.to_vec(),
        });
    }
    pub fn push_layout(&mut self, layout: &[u32]) {
        self.tape.push_parts(DirectoryTuple::_2 {
            elements: layout.to_vec(),
        });
    }
    pub fn into_inner(self) -> TapeWriter<'a, R, DirectoryTuple> {
        self.tape
    }
}

pub struct H1TapeWriter<'a, R>
where
    R: RelationWrite + 'a,
{
    tape: TapeWriter<'a, R, H1Tuple>,
    branches: Vec<Branch<u32>>,
}

impl<'a, R> H1TapeWriter<'a, R>
where
    R: RelationWrite + 'a,
    R::Page: Page<Opaque = Opaque>,
{
    pub fn create(index: &'a R, tracking_freespace: bool) -> Self {
        Self {
            tape: TapeWriter::create(index, tracking_freespace),
            branches: Vec::new(),
        }
    }
    pub fn push(&mut self, branch: Branch<u32>) {
        self.branches.push(branch);
        if let Ok(chunk) = <&[_; 32]>::try_from(self.branches.as_slice()) {
            self.tape.push_parts(H1Tuple0 {
                metadata: [
                    chunk.each_ref().map(|x| x.code.0.dis_u_2),
                    chunk.each_ref().map(|x| x.code.0.factor_cnt),
                    chunk.each_ref().map(|x| x.code.0.factor_ip),
                    chunk.each_ref().map(|x| x.code.0.factor_err),
                ],
                delta: chunk.each_ref().map(|x| x.delta),
                prefetch: fix_good(chunk.each_ref().map(|x| x.prefetch.as_slice())),
                norm: chunk.each_ref().map(|x| x.norm),
                head: chunk.each_ref().map(|x| x.head),
                first: chunk.each_ref().map(|x| x.extra),
                len: chunk.len() as _,
                elements: padding_pack(
                    chunk.iter().map(|x| rabitq::packing::pack_to_u4(&x.code.1)),
                ),
            });
            self.branches.clear();
        }
    }
    pub fn into_inner(self) -> (TapeWriter<'a, R, H1Tuple>, Vec<Branch<u32>>) {
        (self.tape, self.branches)
    }
    pub fn flush(tape: &mut TapeWriter<'_, R, H1Tuple>, chunk: Vec<Branch<u32>>) {
        if chunk.is_empty() {
            return;
        }
        tape.push_parts(H1Tuple0 {
            metadata: [
                any_pack(chunk.iter().map(|x| x.code.0.dis_u_2)),
                any_pack(chunk.iter().map(|x| x.code.0.factor_cnt)),
                any_pack(chunk.iter().map(|x| x.code.0.factor_ip)),
                any_pack(chunk.iter().map(|x| x.code.0.factor_err)),
            ],
            delta: any_pack(chunk.iter().map(|x| x.delta)),
            prefetch: fix_bad(chunk.iter().map(|x| x.prefetch.as_slice())),
            head: any_pack(chunk.iter().map(|x| x.head)),
            norm: any_pack(chunk.iter().map(|x| x.norm)),
            first: any_pack(chunk.iter().map(|x| x.extra)),
            len: chunk.len() as _,
            elements: padding_pack(chunk.iter().map(|x| rabitq::packing::pack_to_u4(&x.code.1))),
        });
    }
}

pub struct FrozenTapeWriter<'a, R>
where
    R: RelationWrite + 'a,
{
    tape: TapeWriter<'a, R, FrozenTuple>,
    branches: Vec<Branch<NonZero<u64>>>,
    keys: &'a Keys,
    /// The branches' filter key columns, a row's after another's.
    columns: Vec<u8>,
}

impl<'a, R> FrozenTapeWriter<'a, R>
where
    R: RelationWrite + 'a,
    R::Page: Page<Opaque = Opaque>,
{
    pub fn create(index: &'a R, keys: &'a Keys, tracking_freespace: bool) -> Self {
        Self {
            tape: TapeWriter::create(index, tracking_freespace),
            branches: Vec::new(),
            keys,
            columns: Vec::new(),
        }
    }
    pub fn last(&self) -> u32 {
        self.tape.last()
    }
    pub fn unlock(&mut self) {
        self.tape.unlock();
    }
    /// `columns`: the branch's, as an appendable tuple stores them.
    pub fn push(&mut self, branch: Branch<NonZero<u64>>, columns: &[u8]) {
        self.branches.push(branch);
        self.columns.extend_from_slice(columns);
        if let Ok(chunk) = <&[_; 32]>::try_from(self.branches.as_slice()) {
            self.tape.push_parts(FrozenTuple0 {
                metadata: [
                    chunk.each_ref().map(|x| x.code.0.dis_u_2),
                    chunk.each_ref().map(|x| x.code.0.factor_cnt),
                    chunk.each_ref().map(|x| x.code.0.factor_ip),
                    chunk.each_ref().map(|x| x.code.0.factor_err),
                ],
                delta: chunk.each_ref().map(|x| x.delta),
                prefetch: fix_good(chunk.each_ref().map(|x| x.prefetch.as_slice())),
                head: chunk.each_ref().map(|x| x.head),
                payload: chunk.each_ref().map(|x| Some(x.extra)),
                elements: padding_pack(
                    chunk.iter().map(|x| rabitq::packing::pack_to_u4(&x.code.1)),
                ),
                columns: self.keys.block(&self.columns),
            });
            self.branches.clear();
            self.columns.clear();
        }
    }
    pub fn into_inner(
        self,
    ) -> (
        TapeWriter<'a, R, FrozenTuple>,
        Vec<Branch<NonZero<u64>>>,
        Vec<u8>,
    ) {
        (self.tape, self.branches, self.columns)
    }
}

fn fix_good<'a>(into_iter: impl IntoIterator<Item = &'a [u32]>) -> Vec<[u32; 32]> {
    let slices = into_iter.into_iter().collect::<Vec<_>>();
    if slices.len() != 32 {
        panic!("too many or too few slices");
    }
    let min = slices.iter().map(|x| x.len()).min().unwrap_or_default();
    let max = slices.iter().map(|x| x.len()).max().unwrap_or_default();
    if min != max {
        panic!("the number of pages for prefetching is not a constant");
    }
    let flattened = slices.into_iter().flatten().copied().collect::<Vec<_>>();
    let (arrays, remainder) = flattened.as_chunks::<32>();
    assert!(remainder.is_empty());
    arrays.to_vec()
}

fn fix_bad<'a>(into_iter: impl IntoIterator<Item = &'a [u32]>) -> Vec<[u32; 32]> {
    let mut slices = into_iter.into_iter().collect::<Vec<_>>();
    if slices.len() > 32 {
        panic!("too many slices");
    }
    let min = slices.iter().map(|x| x.len()).min().unwrap_or_default();
    let max = slices.iter().map(|x| x.len()).max().unwrap_or_default();
    if min != max {
        panic!("the number of pages for prefetching is not a constant");
    }
    let temp = vec![u32::MAX; max];
    slices.resize(32, temp.as_slice());
    let flattened = slices.into_iter().flatten().copied().collect::<Vec<_>>();
    let (arrays, remainder) = flattened.as_chunks::<32>();
    assert!(remainder.is_empty());
    arrays.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use index::test_support::MemoryRelation;

    fn branch<T>(i: usize, dim: usize, extra: T) -> Branch<T> {
        Branch {
            code: (
                rabitq::bit::CodeMetadata::from_array([1.0; 4]),
                (0..dim).map(|j| (i + j) % 3 == 0).collect(),
            ),
            delta: 0.0,
            prefetch: vec![i as u32],
            head: 0,
            norm: 0.0,
            extra,
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn parts() {
        for dim in [64, 768, 1000, 1536, 3000] {
            let relation = MemoryRelation::<Opaque>::new([]);
            let keys = Keys::new([]);
            let mut tape = FrozenTapeWriter::create(&relation, &keys, false);
            let n = 32 * 40;
            for i in 0..n {
                tape.push(branch(i, dim, NonZero::new(1 + i as u64).unwrap()), &[]);
            }
            let first = tape.into_inner().0.first();
            let mut rows = 0;
            for guard in crate::tape::by_next(&relation, first) {
                for i in 1..=guard.len() {
                    match FrozenTuple::deserialize_ref(guard.get(i).unwrap()) {
                        FrozenTupleReader::_0(tuple) => rows += tuple.payload().len(),
                        FrozenTupleReader::_1(tuple) => {
                            assert!(!tuple.elements().is_empty(), "dim {dim}: an empty part")
                        }
                    }
                }
            }
            assert_eq!(rows, n);
            let relation = MemoryRelation::<Opaque>::new([]);
            let mut tape = H1TapeWriter::create(&relation, false);
            let n = 32 * 40 + 7;
            for i in 0..n {
                tape.push(branch(i, dim, i as u32));
            }
            let (mut tape, chunk) = tape.into_inner();
            H1TapeWriter::flush(&mut tape, chunk);
            let first = { tape }.first();
            let mut rows = 0;
            for guard in crate::tape::by_next(&relation, first) {
                for i in 1..=guard.len() {
                    match H1Tuple::deserialize_ref(guard.get(i).unwrap()) {
                        H1TupleReader::_0(tuple) => rows += tuple.len() as usize,
                        H1TupleReader::_1(tuple) => {
                            assert!(!tuple.elements().is_empty(), "dim {dim}: an empty part")
                        }
                    }
                }
            }
            assert_eq!(rows, n);
        }
        for n in [1, 100, 2040, 2041, 5000, 20000] {
            let relation = MemoryRelation::<Opaque>::new([]);
            let mut tape = DirectoryTapeWriter::create(&relation, false);
            let ids = (0..n).collect::<Vec<u32>>();
            tape.push(&ids);
            tape.push(&ids);
            tape.push_layout(&ids);
            let first = tape.into_inner().first();
            let mut read = Vec::new();
            for guard in crate::tape::by_next(&relation, first) {
                for i in 1..=guard.len() {
                    let elements = match DirectoryTuple::deserialize_ref(guard.get(i).unwrap()) {
                        DirectoryTupleReader::_0(tuple) => tuple.elements(),
                        DirectoryTupleReader::_1(tuple) | DirectoryTupleReader::_2(tuple) => {
                            assert!(!tuple.elements().is_empty(), "{n}: an empty part");
                            tuple.elements()
                        }
                    };
                    read.extend_from_slice(elements);
                }
            }
            assert_eq!(
                read,
                [ids.as_slice(), ids.as_slice(), ids.as_slice()].concat()
            );
        }
    }
}
