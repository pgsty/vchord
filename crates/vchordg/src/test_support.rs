// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

// Real serialized graph pages with deterministic interleaving at the write-lock boundary.
use crate::Opaque;
use crate::operator::Op;
use crate::tuples::*;
use distance::Distance;
use index::relation::{
    Page, PageGuard, Relation, RelationRead, RelationReadTypes, RelationWrite, RelationWriteTypes,
};
use index_accessor::L2S;
use std::cell::{Cell, Ref, RefCell, RefMut};
use std::num::{NonZero, Wrapping};
use std::ops::{Deref, DerefMut};
use vector::vect::VectOwned;

pub(crate) type TestOp = Op<VectOwned<f32>, L2S>;
pub(crate) struct TestPage {
    opaque: Opaque,
    items: Vec<Option<Vec<u8>>>,
}
impl Page for TestPage {
    type Opaque = Opaque;
    fn get_opaque(&self) -> &Opaque {
        &self.opaque
    }
    fn get_opaque_mut(&mut self) -> &mut Opaque {
        &mut self.opaque
    }
    fn len(&self) -> u16 {
        self.items.len() as u16
    }
    fn get(&self, i: u16) -> Option<&[u8]> {
        self.items.get(i as usize - 1)?.as_deref()
    }
    fn get_mut(&mut self, i: u16) -> Option<&mut [u8]> {
        self.items.get_mut(i as usize - 1)?.as_deref_mut()
    }
    fn alloc(&mut self, data: &[u8]) -> Option<u16> {
        self.items.push(Some(data.to_vec()));
        Some(self.len())
    }
    fn free(&mut self, i: u16) {
        self.items[i as usize - 1] = None;
    }
    fn freespace(&self) -> u16 {
        7000
    }
    fn clear(&mut self, opaque: Opaque) {
        self.items.clear();
        self.opaque = opaque;
    }
}
pub(crate) struct Read<'a> {
    id: u32,
    page: Ref<'a, TestPage>,
}
impl Deref for Read<'_> {
    type Target = TestPage;
    fn deref(&self) -> &TestPage {
        &self.page
    }
}
impl PageGuard for Read<'_> {
    fn id(&self) -> u32 {
        self.id
    }
}
pub(crate) struct Write<'a> {
    id: u32,
    page: RefMut<'a, TestPage>,
}
impl Deref for Write<'_> {
    type Target = TestPage;
    fn deref(&self) -> &TestPage {
        &self.page
    }
}
impl DerefMut for Write<'_> {
    fn deref_mut(&mut self) -> &mut TestPage {
        &mut self.page
    }
}
impl PageGuard for Write<'_> {
    fn id(&self) -> u32 {
        self.id
    }
}
pub(crate) struct TestRelation {
    pages: Vec<RefCell<TestPage>>,
    pub inject: Cell<bool>,
    pub conflicts: Cell<usize>,
}
impl Relation for TestRelation {
    type Page = TestPage;
}
impl RelationReadTypes for TestRelation {
    type ReadGuard<'b> = Read<'b>;
}
impl RelationRead for TestRelation {
    fn read(&self, id: u32) -> Read<'_> {
        Read {
            id,
            page: self.pages[id as usize].borrow(),
        }
    }
}
impl RelationWriteTypes for TestRelation {
    type WriteGuard<'b> = Write<'b>;
}
impl RelationWrite for TestRelation {
    fn write(&self, id: u32, _: bool) -> Write<'_> {
        let mut page = self.pages[id as usize].borrow_mut();
        // Emulate intervening successful list writes before taking the write
        // lock: the observed list version is now stale. No thread scheduling.
        if self.inject.get() && id == 2 {
            let VectorTupleWriter::_0(mut tuple) =
                VectorTuple::<VectOwned<f32>>::deserialize_mut(page.get_mut(1).unwrap())
            else {
                panic!()
            };
            *tuple.version() += 1;
            self.conflicts.set(self.conflicts.get() + 1);
        }
        Write { id, page }
    }
    fn extend(&self, _: Opaque, _: bool) -> Write<'_> {
        panic!("unexpected extend")
    }
    fn search(&self, _: usize) -> Option<Write<'_>> {
        None
    }
}
fn page(link: u32, next: u32, items: Vec<Vec<u8>>) -> RefCell<TestPage> {
    RefCell::new(TestPage {
        opaque: Opaque { link, next },
        items: items.into_iter().map(Some).collect(),
    })
}
pub(crate) fn fixture(inject: bool) -> TestRelation {
    let meta = MetaTuple {
        dim: 2,
        bits: 1,
        m: 4,
        alpha: vec![1.2],
        ef_construction: 64,
        beam_construction: 1,
        start: OptionPointer::some((1, 1)),
        skip: 1,
    }
    .serialize();
    let vertices = (1..=2)
        .map(|i| {
            VertexTuple {
                metadata: [0.; 3],
                elements: vec![0],
                payload: NonZero::new(i),
                pointers: vec![Pointer::new((i as u32 + 1, 1))],
            }
            .serialize()
        })
        .collect();
    let u = VectorTuple::<VectOwned<f32>>::_0 {
        payload: NonZero::new(1),
        elements: vec![0., 0.],
        metadata: (),
        neighbours: vec![OptionNeighbour::NONE; 4],
        version: Wrapping(0),
    }
    .serialize();
    let t = VectorTuple::<VectOwned<f32>>::_0 {
        payload: NonZero::new(2),
        elements: vec![1., 0.],
        metadata: (),
        neighbours: vec![
            OptionNeighbour::some((1, 1), Distance::from_f32(1.)),
            OptionNeighbour::NONE,
            OptionNeighbour::NONE,
            OptionNeighbour::NONE,
        ],
        version: Wrapping(0),
    }
    .serialize();
    TestRelation {
        pages: vec![
            page(1, u32::MAX, vec![meta]),
            page(2, u32::MAX, vertices),
            page(u32::MAX, 3, vec![u]),
            page(u32::MAX, u32::MAX, vec![t]),
        ],
        inject: Cell::new(inject),
        conflicts: Cell::new(0),
    }
}

/// One real vector page per vertex; IDs are (1, position + 1).
pub(crate) fn graph(vertices: &[(bool, &[u16])], m: u32) -> TestRelation {
    let meta = MetaTuple {
        dim: 2,
        bits: 1,
        m,
        alpha: vec![1.2],
        ef_construction: 64,
        beam_construction: 1,
        start: OptionPointer::some((1, 1)),
        skip: 1,
    }
    .serialize();
    let mut vector_pages = Vec::new();
    let mut vertex_tuples = Vec::new();
    for (n, &(live, edges)) in vertices.iter().enumerate() {
        let payload = live.then(|| NonZero::new(n as u64 + 1).unwrap());
        let mut neighbours = edges
            .iter()
            .map(|&v| {
                OptionNeighbour::some(
                    (1, v),
                    Distance::from_f32((n as f32 + 1.0 - v as f32).powi(2)),
                )
            })
            .collect::<Vec<_>>();
        assert!(neighbours.len() <= m as usize);
        neighbours.resize(m as usize, OptionNeighbour::NONE);
        let vector = VectorTuple::<VectOwned<f32>>::_0 {
            payload,
            elements: vec![n as f32, 1.0],
            metadata: (),
            neighbours,
            version: Wrapping(0),
        }
        .serialize();
        vector_pages.push(page(
            u32::MAX,
            if n + 1 == vertices.len() {
                u32::MAX
            } else {
                n as u32 + 3
            },
            vec![vector],
        ));
        vertex_tuples.push(
            VertexTuple {
                metadata: [0.; 3],
                elements: vec![0],
                payload,
                pointers: vec![Pointer::new((n as u32 + 2, 1))],
            }
            .serialize(),
        );
    }
    let mut pages = vec![
        page(1, u32::MAX, vec![meta]),
        page(2, u32::MAX, vertex_tuples),
    ];
    pages.extend(vector_pages);
    TestRelation {
        pages,
        inject: Cell::new(false),
        conflicts: Cell::new(0),
    }
}

pub(crate) struct MakePlainPrefetcher<'b, R> {
    pub index: &'b R,
}
impl<'b, R: RelationRead + 'b> index::prefetcher::PrefetcherSequenceFamily<'b, R>
    for MakePlainPrefetcher<'b, R>
{
    type P<S: index::prefetcher::Sequence>
        = index::prefetcher::PlainPrefetcher<'b, R, S>
    where
        S::Item: index::fetch::Fetch<'b>;
    fn prefetch<S: index::prefetcher::Sequence>(&mut self, seq: S) -> Self::P<S>
    where
        S::Item: index::fetch::Fetch<'b>,
    {
        index::prefetcher::PlainPrefetcher::new(self.index, seq)
    }
    fn is_not_plain(&self) -> bool {
        false
    }
}
