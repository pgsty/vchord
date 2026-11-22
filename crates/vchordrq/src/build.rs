// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use crate::insert::{Root, assign};
use crate::keys::{Columns, Keys};
use crate::list::{Fixed, ListWriter, Rows, firsts};
use crate::operator::{Operator, Vector};
use crate::tape::TapeWriter;
use crate::tape_writer::H1TapeWriter;
use crate::tuples::*;
use crate::types::*;
use crate::{Branch, Opaque, vectors};
use index::bump::Bump;
use index::prefetcher::PrefetcherHeapFamily;
use index::relation::{Page, RelationRead, RelationWrite};
use std::num::NonZero;
use vector::{VectorBorrowed, VectorOwned};
use zerocopy::{FromBytes, IntoBytes};

pub fn build<R: RelationWrite, O: Operator>(
    vector_options: VectorOptions,
    vchordrq_options: VchordrqIndexOptions,
    keys: &Keys,
    index: &R,
    structures: Vec<Structure<O::Vector>>,
) where
    R::Page: Page<Opaque = Opaque>,
{
    let dim = vector_options.dim;
    let is_residual = vchordrq_options.residual_quantization;
    let mut meta = TapeWriter::<_, MetaTuple>::create(index, false);
    assert_eq!(meta.first(), 0);
    let mut freepages = TapeWriter::<_, FreepagesTuple>::create(index, false);
    freepages.push(FreepagesTuple {});
    let mut centroids = TapeWriter::<_, CentroidTuple<O::Vector>>::create(index, false);
    let vectors = (0..vchordrq_options.degree_of_parallelism)
        .map(|_| TapeWriter::<_, VectorTuple<O::Vector>>::create(index, true).first())
        .collect::<Vec<_>>();
    let mut pointer_of_centroids = Vec::<Vec<(Vec<u32>, u16)>>::new();
    for i in 0..structures.len() {
        let mut level = Vec::new();
        for j in 0..structures[i].len() {
            let vector = structures[i].centroids[j].as_borrowed();
            let (slices, metadata) = O::Vector::split(vector);
            let mut chain = Ok(metadata);
            let mut prefetch = Vec::new();
            for i in (0..slices.len()).rev() {
                let (id, head) = centroids.push(match chain {
                    Ok(metadata) => CentroidTuple::_0 {
                        elements: slices[i].to_vec(),
                        metadata,
                    },
                    Err(head) => CentroidTuple::_1 {
                        elements: slices[i].to_vec(),
                        head,
                    },
                });
                chain = Err(head);
                prefetch.push(id);
            }
            prefetch.reverse();
            level.push((
                prefetch,
                chain.expect_err("internal error: 0-dimensional vector"),
            ));
        }
        pointer_of_centroids.push(level);
    }
    let mut pointer_of_firsts = Vec::<Vec<u32>>::new();
    for i in 0..structures.len() {
        let mut level = Vec::new();
        for j in 0..structures[i].len() {
            if i == 0 {
                // `write` gives every list its tapes
                let mut jump = TapeWriter::<_, JumpTuple>::create(index, false);
                jump.push(JumpTuple {
                    directory_first: u32::MAX,
                    frozen_first: u32::MAX,
                    appendable_first: u32::MAX,
                    centroid_prefetch: pointer_of_centroids[i][j].0.clone(),
                    centroid_head: pointer_of_centroids[i][j].1,
                    tuples: 0,
                });
                level.push(jump.first());
            } else {
                let mut tape = H1TapeWriter::create(index, false);
                let centroid = structures[i].centroids[j].as_borrowed();
                for child in structures[i].children[j].iter().copied() {
                    let vector = structures[i - 1].centroids[child as usize].as_borrowed();
                    let (code, delta) = O::build(vector, is_residual.then_some(centroid.own()));
                    tape.push(Branch {
                        code,
                        delta,
                        prefetch: pointer_of_centroids[i - 1][child as usize].0.clone(),
                        head: pointer_of_centroids[i - 1][child as usize].1,
                        extra: pointer_of_firsts[i - 1][child as usize],
                        norm: norm::<O::Vector>(vector),
                    });
                }
                let (mut tape, chunk) = tape.into_inner();
                H1TapeWriter::flush(&mut tape, chunk);
                level.push(tape.first());
            }
        }
        pointer_of_firsts.push(level);
    }
    meta.push(MetaTuple {
        keys: (!keys.is_empty()).then(|| keys.descriptors()),
        dim,
        height_of_root: structures.len() as u32,
        is_residual,
        rerank_in_heap: vchordrq_options.rerank_in_table,
        centroids_first: centroids.first(),
        vectors_first: vectors,
        centroid_prefetch: pointer_of_centroids
            .last()
            .expect("internal error: empty structure")[0]
            .0
            .clone(),
        centroid_head: pointer_of_centroids
            .last()
            .expect("internal error: empty structure")[0]
            .1,
        centroid_norm: norm::<O::Vector>(
            structures
                .last()
                .expect("internal error: empty structure")
                .centroids[0]
                .as_borrowed(),
        ),
        first: pointer_of_firsts
            .last()
            .expect("internal error: empty structure")[0],
        freepages_first: freepages.first(),
        cells: structures.iter().map(|s| s.len() as _).collect(),
    });
}

/// Rows as `Encoder::encode` lays them out, in the order of their bytes; each is valid until the next.
pub trait Sorted {
    fn next(&mut self) -> Option<&[u8]>;
}

/// Where a build keeps the codes of a list larger than its memory until the list's vectors are written.
pub trait Spill {
    fn put(&mut self, bytes: &[u8]);
    /// Each piece put since it was last called, in order.
    fn take(&mut self, f: impl FnMut(&[u8]));
}

#[cfg(test)]
impl Spill for Vec<Vec<u8>> {
    fn put(&mut self, bytes: &[u8]) {
        self.push(bytes.to_vec());
    }
    fn take(&mut self, mut f: impl FnMut(&[u8])) {
        std::mem::take(self).iter().for_each(|x| f(x));
    }
}

/// What a build encodes its rows with: where routing begins, the index's filter keys, and whether its vectors are in it.
pub struct Encoder<'b, R> {
    index: &'b R,
    root: Root,
    keys: Keys,
    vectors: bool,
    /// The row last encoded.
    row: Vec<u8>,
}

impl<'b, R: RelationRead> Encoder<'b, R>
where
    R::Page: Page<Opaque = Opaque>,
{
    pub fn new(index: &'b R) -> Self {
        let meta_guard = index.read(0);
        let meta_tuple = MetaTuple::deserialize_ref(meta_guard.get(1).expect("data corruption"));
        Self {
            index,
            root: Root::of(meta_tuple),
            keys: Keys::of_meta(meta_tuple),
            vectors: !meta_tuple.rerank_in_heap(),
            row: Vec::new(),
        }
    }
    pub fn index(&self) -> &'b R {
        self.index
    }
    /// The row last encoded.
    pub fn row(&self) -> &[u8] {
        &self.row
    }
    /// Encodes a row as the build sorts it: by list, then by its keys, then by payload, as `write` lays rows out;
    /// then its code, its columns and its vector. `projected` routes it.
    pub fn encode<'c, O: Operator>(
        &mut self,
        payload: NonZero<u64>,
        columns: &[u8],
        (vector, projected): (
            <O::Vector as VectorOwned>::Borrowed<'_>,
            <O::Vector as VectorOwned>::Borrowed<'_>,
        ),
        bump: &'c impl Bump,
        prefetch_h1_vectors: impl PrefetcherHeapFamily<'c, R>,
    ) where
        'b: 'c,
    {
        let out = &mut self.row;
        out.clear();
        let assigned = assign::<R, O>(self.index, &self.root, projected, bump, prefetch_h1_vectors);
        let (code, delta) = assigned.code;
        out.extend_from_slice(&assigned.first.to_be_bytes());
        self.keys.sortable(Columns::row(columns), out);
        out.extend_from_slice(&payload.get().to_be_bytes());
        let metadata = [
            code.0.dis_u_2,
            code.0.factor_cnt,
            code.0.factor_ip,
            code.0.factor_err,
        ];
        out.extend_from_slice(metadata.as_bytes());
        out.extend_from_slice(delta.as_bytes());
        out.extend_from_slice(rabitq::bit::binary::pack_code(&code.1).as_bytes());
        out.extend_from_slice(columns);
        if self.vectors {
            vectors::encode::<O::Vector>(vector, out);
        }
    }
}

/// A row's parts, as `Encoder::encode` lays them out, after its list.
struct Parts<'a> {
    fixed: Fixed,
    signs: &'a [u8],
    columns: &'a [u8],
    vector: &'a [u8],
}

impl<'a> Parts<'a> {
    fn of(row: &'a [u8], keys: &Keys, dim: u32) -> Self {
        let (_, row) = row.split_at(size_of::<u32>() + keys.sortable_len());
        let (payload, row) = row.split_first_chunk().expect("a row");
        let (metadata, row) = row.split_at(size_of::<[f32; 4]>());
        let (delta, row) = row.split_first_chunk().expect("a row");
        let (signs, row) = row.split_at(dim.div_ceil(64) as usize * size_of::<u64>());
        let (columns, vector) = row.split_at(keys.columns());
        let fixed = Fixed {
            metadata: FromBytes::read_from_bytes(metadata).expect("a row"),
            delta: f32::from_ne_bytes(*delta),
            head: 0,
            _padding: [0; 2],
            payload: u64::from_be_bytes(*payload),
        };
        Self {
            fixed,
            signs,
            columns,
            vector,
        }
    }
}

/// Writes every list from `rows`, each in one run of pages: its vectors, its frozen tape, its appendable tape, its directory.
/// `memory`: the most bytes of a list's codes held; past it they go to `spill` until the list's vectors are written.
/// Without vectors in the index, a list's rows go straight to its tapes.
pub fn write<R: RelationRead + RelationWrite, O: Operator>(
    index: &R,
    mut rows: impl Sorted,
    memory: usize,
    spill: &mut impl Spill,
) where
    R::Page: Page<Opaque = Opaque>,
{
    let meta_guard = index.read(0);
    let meta_tuple = MetaTuple::deserialize_ref(meta_guard.get(1).expect("data corruption"));
    let dim = meta_tuple.dim();
    let rerank_in_heap = meta_tuple.rerank_in_heap();
    let keys = Keys::of_meta(meta_tuple);
    let vectors_first = meta_tuple.vectors_first()[0];
    drop(meta_guard);
    let mut firsts = firsts(index, || ());
    firsts.sort_unstable();
    let mut vectors = None::<TapeWriter<'_, R, VectorTuple<O::Vector>>>;
    let mut signs = Vec::<u64>::new();
    let mut row = rows.next();
    for first in firsts {
        // Between lists nothing is locked (#28).
        index::interrupt::check();
        let mut list = rerank_in_heap.then(|| ListWriter::create(index, &keys));
        let (mut pending, mut tuples, mut spilled) = (Rows::new(dim as _, memory), 0_u64, false);
        while let Some(bytes) = row
            && bytes[..4] == first.to_be_bytes()
        {
            let Parts {
                mut fixed,
                signs: bytes,
                columns,
                vector,
            } = Parts::of(bytes, &keys, dim);
            signs.clear();
            signs.extend(bytes.as_chunks().0.iter().map(|&x| u64::from_ne_bytes(x)));
            tuples += 1;
            // interrupts are served where the pages being written are unlocked: as a tape moves on to a page (#28)
            if let Some(list) = list.as_mut() {
                if list.push(fixed.branch(&signs, &[], dim as _), columns) {
                    list.unlock();
                    index::interrupt::check();
                }
            } else {
                let tape = vectors.get_or_insert_with(|| TapeWriter::create(index, true));
                // a list's vectors begin on a page of their own
                if tuples == 1 {
                    tape.renew();
                }
                let last = tape.last();
                let vector = vectors::decode::<O::Vector>(dim, vector);
                let payload = NonZero::new(fixed.payload).expect("a row");
                let (prefetch, head) =
                    vectors::write::<O>(vector.as_borrowed(), payload, |x| tape.push(x));
                fixed.head = head;
                pending.push_parts(fixed, &signs, &prefetch, columns);
                let full = pending.is_full();
                if tape.last() != last || full {
                    tape.unlock();
                    index::interrupt::check();
                }
                // a temporary file is written and read with no page locked
                if full {
                    pending.save(|piece| spill.put(piece));
                    spilled = true;
                }
            }
            row = rows.next();
        }
        if let Some(tape) = vectors.as_mut() {
            tape.unlock();
        }
        let mut list = list.unwrap_or_else(|| ListWriter::create(index, &keys));
        let mut put = |rows: &mut Rows| {
            rows.flush(&keys, |branch, columns| {
                if list.push(branch, columns) {
                    list.unlock();
                    index::interrupt::check();
                }
            });
            list.unlock();
        };
        if spilled {
            pending.save(|piece| spill.put(piece));
            spill.take(|piece| put(&mut Rows::load(dim as _, keys.columns(), piece)));
        }
        put(&mut pending);
        let written = list.finish(index::interrupt::check);
        let mut jump_guard = index.write(first, false);
        let mut jump_tuple =
            JumpTuple::deserialize_mut(jump_guard.get_mut(1).expect("data corruption"));
        *jump_tuple.directory_first() = written.directory_first;
        *jump_tuple.frozen_first() = written.frozen_first;
        *jump_tuple.appendable_first() = { written.appendable }.first();
        *jump_tuple.tuples() = tuples;
    }
    assert!(row.is_none(), "internal error: a row of no list");
    if let Some(tape) = vectors {
        let (first, last) = (tape.first(), tape.last());
        drop(tape);
        let mut guard = index.write(vectors_first, true);
        guard.get_opaque_mut().next = first;
        guard.get_opaque_mut().skip = last;
    }
}

fn norm<V: Vector>(vector: V::Borrowed<'_>) -> f32 {
    V::squared_norm(vector).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::closure_lifetime_binder::id_1;
    use crate::insert::{InsertChooser, insert, insert_vector};
    use crate::keys::{Summary, Width};
    use crate::list::unpack_code;
    use crate::maintain::{Pending, maintain};
    use crate::operator::Op;
    use crate::tape::{by_next, read_appendable_tape, read_directory_words, read_frozen_tape};
    use crate::{RerankMethod, how};
    use index::relation::PageGuard;
    use index::test_support::{MakePlainPrefetcher, MemoryRelation};
    use index_accessor::{FunctionalAccessor, L2S, TryAccessor1};
    use rabitq::packing::unpack;
    use std::cell::Cell;
    use vector::vect::VectOwned;

    type M = MemoryRelation<Opaque>;
    type O = Op<VectOwned<f32>, L2S>;

    thread_local! {
        /// The relation a write is checked on, the rows it has taken, and the rows taken at each interrupt check.
        static CHECKED: Cell<*const M> = const { Cell::new(std::ptr::null()) };
        static TAKEN: Cell<usize> = const { Cell::new(0) };
        static CHECKS: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
    }

    #[allow(unsafe_code)]
    fn check() {
        let relation = CHECKED.get();
        if !relation.is_null() {
            // Safety: the relation outlives the write that checks it
            let relation = unsafe { &*relation };
            for id in 0..relation.pages() {
                assert!(
                    relation.page(id).try_borrow_mut().is_ok(),
                    "page {id} locked at a check"
                );
            }
            CHECKS.with(|x| x.borrow_mut().push(TAKEN.get()));
        }
    }

    struct Rows(std::vec::IntoIter<Vec<u8>>, Vec<u8>);

    impl Sorted for Rows {
        fn next(&mut self) -> Option<&[u8]> {
            self.1 = self.0.next()?;
            TAKEN.set(TAKEN.get() + 1);
            Some(&self.1)
        }
    }

    struct First;

    impl InsertChooser for First {
        fn choose(&mut self, _: NonZero<usize>) -> usize {
            0
        }
    }

    /// A vector's elements, read back part by part.
    struct Elements(Vec<f32>);

    impl TryAccessor1<f32, ()> for Elements {
        type Output = Vec<f32>;
        fn push(&mut self, input: &[f32]) -> Option<()> {
            self.0.extend_from_slice(input);
            Some(())
        }
        fn finish(self, _: ()) -> Option<Vec<f32>> {
            Some(self.0)
        }
    }

    /// An index of `lists` lists of `dim` dimensions, as `build` leaves it for `write`.
    fn index(keys: &Keys, rerank_in_table: bool, dim: usize, lists: usize) -> M {
        let relation = M::new([]);
        let options = VchordrqIndexOptions {
            rerank_in_table,
            degree_of_parallelism: 2,
            ..Default::default()
        };
        let centroids = (0..lists)
            .map(|i| VectOwned::new((0..dim).map(|j| ((i * 7 + j) % 11) as f32 - 5.0).collect()))
            .collect();
        let mut structures = vec![Structure {
            centroids,
            children: vec![Vec::new(); lists],
        }];
        if lists > 1 {
            structures.push(Structure {
                centroids: vec![VectOwned::new(vec![0.0; dim])],
                children: vec![(0..lists as u32).collect()],
            });
        }
        let (v, d) = (VectorKind::Vecf32, DistanceKind::L2S);
        let vector_options = VectorOptions {
            dim: dim as _,
            v,
            d,
        };
        build::<_, O>(vector_options, options, keys, &relation, structures);
        relation
    }

    /// Rows `payloads`: their columns hold a key of negatives and positives, NULL in every fifth row.
    fn data(
        payloads: std::ops::RangeInclusive<u64>,
        dim: usize,
        keys: &Keys,
    ) -> Vec<(NonZero<u64>, Vec<u8>, VectOwned<f32>)> {
        payloads
            .map(|p| {
                let elements = (0..dim)
                    .map(|j| ((p * 2654435761 + j as u64 * 40503) % 1000) as f32 / 100.0 - 5.0);
                let columns = if keys.is_empty() {
                    Vec::new()
                } else {
                    let (key, null) = ((p as i64 % 7 - 3) as u64 ^ (1 << 63), (p % 5 == 0) as u32);
                    [&key.to_le_bytes()[..], &null.to_le_bytes()[..]].concat()
                };
                (
                    NonZero::new(p).unwrap(),
                    columns,
                    VectOwned::new(elements.collect()),
                )
            })
            .collect()
    }

    /// The rows sorted as a build sorts them.
    fn sorted(relation: &M, rows: &[(NonZero<u64>, Vec<u8>, VectOwned<f32>)]) -> Rows {
        let mut encoder = Encoder::new(relation);
        let mut sorted = rows
            .iter()
            .map(|(payload, columns, vector)| {
                let bump = bumpalo::Bump::new();
                let prefetch = MakePlainPrefetcher { index: relation };
                let v = vector.as_borrowed();
                encoder.encode::<O>(*payload, columns, (v, v), &bump, prefetch);
                encoder.row().to_vec()
            })
            .collect::<Vec<_>>();
        sorted.sort_unstable();
        Rows(sorted.into_iter(), Vec::new())
    }

    /// A row as a list holds it: payload, code, columns and, read back, its vector.
    #[derive(Debug, PartialEq)]
    struct Entry {
        payload: u64,
        code: ([u32; 4], u32, Vec<bool>),
        columns: Vec<u8>,
        vector: Option<Vec<f32>>,
    }

    /// A list: its pages in the order the search reads them, its directory, its vectors' pages and its rows.
    struct List {
        tape: Vec<u32>,
        directory: Vec<u32>,
        vectors: Vec<u32>,
        rows: Vec<Entry>,
    }

    fn lists(relation: &M, keys: &Keys, dim: usize) -> Vec<List> {
        let (mut words, rerank) = (Vec::new(), how(relation));
        let mut firsts = firsts(relation, || ());
        firsts.sort_unstable();
        let accessor = || {
            FunctionalAccessor::new(
                Vec::<[u8; 16]>::new(),
                Vec::<[u8; 16]>::extend_from_slice,
                id_1(
                    |elements: Vec<_>, (metadata, delta): (&[[f32; 32]; 4], &[f32; 32])| {
                        let unpacked = unpack(&elements);
                        std::array::from_fn(|i| {
                            let f = |&x| [x & 1 != 0, x & 2 != 0, x & 4 != 0, x & 8 != 0];
                            let signs = unpacked[i].iter().flat_map(f).collect::<Vec<_>>();
                            let m = [0, 1, 2, 3].map(|k| metadata[k][i]);
                            ((rabitq::bit::CodeMetadata::from_array(m), signs), delta[i])
                        })
                    },
                ),
            )
        };
        firsts
            .into_iter()
            .map(|first| {
                let jump_guard = relation.read(first);
                let jump_tuple = JumpTuple::deserialize_ref(jump_guard.get(1).unwrap());
                let directory = by_next(relation, jump_tuple.directory_first());
                let pages = read_directory_words::<M>(directory, &mut words, false);
                let (mut vectors, mut rows) = (Vec::new(), Vec::new());
                let mut row = |row: crate::tape::Row<'_, (rabitq::bit::Code, f32)>| {
                    let ((m, signs), delta) = row.value;
                    let mut columns = Vec::new();
                    row.columns.extend(keys, &mut columns);
                    vectors.extend_from_slice(row.prefetch);
                    let vector = matches!(rerank, RerankMethod::Index).then(|| {
                        let pages = row.prefetch.iter().map(|&id| relation.read(id));
                        let read = vectors::read::<M, O, _>(
                            pages,
                            row.head,
                            row.payload,
                            Elements(Vec::new()),
                        );
                        read.expect("a vector that is not found")
                    });
                    rows.push(Entry {
                        payload: row.payload.get(),
                        code: (
                            m.into_array().map(f32::to_bits),
                            delta.to_bits(),
                            signs[..dim].to_vec(),
                        ),
                        columns,
                        vector,
                    });
                };
                let frozen = words[..pages].iter().map(|&id| relation.read(id));
                read_frozen_tape::<M, _, _>(frozen, crate::tape::All, accessor, (), &mut row);
                let appendable = by_next(relation, jump_tuple.appendable_first());
                let access = |metadata, elements: &[u64], delta| {
                    (unpack_code(metadata, elements, dim), delta)
                };
                read_appendable_tape::<M, _>(appendable, 0, access, (), &mut row);
                let appendable = by_next(relation, jump_tuple.appendable_first()).map(|g| g.id());
                let directory = by_next(relation, jump_tuple.directory_first()).map(|g| g.id());
                List {
                    tape: words[..pages].iter().copied().chain(appendable).collect(),
                    directory: directory.collect(),
                    vectors,
                    rows,
                }
            })
            .collect()
    }

    fn runs(pages: &[u32]) -> usize {
        1 + pages.windows(2).filter(|x| x[1] != x[0] + 1).count()
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn one_run_per_list() {
        let keyed = Keys::new([(Width::W8, false, Some(Summary::Bounds))]);
        // and a list larger than `memory`
        for (keys, rerank_in_table, dim, n, lists_n, memory) in [
            (Keys::new([]), false, 100, 3000, 7, 1 << 30),
            (keyed, false, 1500, 2000, 5, 1 << 30),
            (keyed, true, 100, 3000, 7, 1 << 30),
            (keyed, false, 100, 0, 3, 1 << 30),
            (Keys::new([]), true, 64, 5, 1, 1 << 30),
            (keyed, false, 100, 3000, 1, 1 << 12),
            (keyed, false, 100, 10000, 1, 1 << 19),
        ] {
            let relation = index(&keys, rerank_in_table, dim, lists_n);
            let rows = sorted(&relation, &data(1..=n, dim, &keys));
            write::<_, O>(&relation, rows, memory, &mut Vec::new());
            let mut payloads = Vec::new();
            for list in lists(&relation, &keys, dim) {
                let tape = [&list.tape[..], &list.directory[..]].concat();
                assert_eq!(runs(&tape), 1, "{tape:?}");
                let order = |e: &Entry| {
                    let mut out = Vec::new();
                    keys.sortable(Columns::row(&e.columns), &mut out);
                    out
                };
                assert!(
                    list.rows.windows(2).all(|x| order(&x[0]) <= order(&x[1])),
                    "rows out of key order"
                );
                if !list.vectors.is_empty() {
                    let mut pages = list.vectors.clone();
                    pages.sort_unstable();
                    pages.dedup();
                    assert_eq!(runs(&pages), 1, "{pages:?}");
                    assert_eq!(pages.last().unwrap() + 1, list.tape[0]);
                }
                payloads.extend(list.rows.iter().map(|x| x.payload));
            }
            payloads.sort_unstable();
            assert_eq!(payloads, (1..=n).collect::<Vec<_>>());
        }
    }

    /// The build writes what inserts and a maintain would; inserts and a VACUUM later find, keep and remove its vectors.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn writes_what_inserts_and_maintain_write() {
        let keyed = Keys::new([(Width::W8, false, Some(Summary::Bounds))]);
        // a two-level tree, two-part vectors, a list larger than `memory`, `rerank_in_table`, one list
        for (keys, rerank_in_table, dim, n, lists_n, memory) in [
            (Keys::new([]), false, 100, 3000, 7, 1 << 30),
            (keyed, false, 3000, 300, 3, 1 << 30),
            (keyed, false, 100, 3000, 3, 1 << 12),
            (keyed, true, 100, 2000, 5, 1 << 12),
            (keyed, false, 100, 1000, 1, 1 << 30),
            (keyed, false, 100, 10000, 1, 1 << 19),
        ] {
            let rows = data(1..=n, dim, &keys);
            let built = index(&keys, rerank_in_table, dim, lists_n);
            write::<_, O>(&built, sorted(&built, &rows), memory, &mut Vec::new());
            let inserted = index(&keys, rerank_in_table, dim, lists_n);
            let none = Rows(Vec::new().into_iter(), Vec::new());
            write::<_, O>(&inserted, none, memory, &mut Vec::new());
            let add = |relation: &M, rows: &[(NonZero<u64>, Vec<u8>, VectOwned<f32>)]| {
                for (payload, columns, vector) in rows {
                    let key =
                        insert_vector::<_, O>(relation, *payload, vector.as_borrowed(), &mut First);
                    let (bump, prefetch) = (
                        bumpalo::Bump::new(),
                        MakePlainPrefetcher { index: relation },
                    );
                    insert::<_, O>(
                        relation,
                        *payload,
                        columns,
                        vector.as_borrowed(),
                        key,
                        &bump,
                        prefetch,
                    );
                }
            };
            add(&inserted, &rows);
            let prefetch = MakePlainPrefetcher { index: &inserted };
            maintain::<_, O>(&inserted, prefetch, memory, &Pending::default(), || ());
            let by_payload = |relation: &M| {
                let mut lists = lists(relation, &keys, dim);
                lists
                    .iter_mut()
                    .for_each(|x| x.rows.sort_by_key(|e| e.payload));
                lists.into_iter().map(|x| x.rows).collect::<Vec<_>>()
            };
            let (a, b) = (by_payload(&built), by_payload(&inserted));
            for (x, y) in std::iter::zip(&a, &b) {
                let strip = |e: &Entry| (e.payload, e.code.clone(), e.columns.clone());
                assert_eq!(
                    x.iter().map(strip).collect::<Vec<_>>(),
                    y.iter().map(strip).collect::<Vec<_>>()
                );
            }
            if !rerank_in_table {
                for e in a.iter().flatten() {
                    assert_eq!(
                        e.vector.as_deref(),
                        Some(rows[e.payload as usize - 1].2.slice())
                    );
                }
            }
            add(&built, &data(n + 1..=n + 50, dim, &keys));
            let gone = |payload: NonZero<u64>| payload.get() % 3 == 0;
            crate::bulkdelete::<_, O>(&built, || (), gone);
            crate::bulkdelete_vectors::<_, O>(&built, || (), gone);
            let prefetch = MakePlainPrefetcher { index: &built };
            maintain::<_, O>(&built, prefetch, memory, &Pending::default(), || ());
            let mut left = by_payload(&built).into_iter().flatten().collect::<Vec<_>>();
            left.sort_by_key(|e| e.payload);
            let kept = (1..=n + 50).filter(|&p| p % 3 != 0).collect::<Vec<_>>();
            assert_eq!(left.iter().map(|e| e.payload).collect::<Vec<_>>(), kept);
            if !rerank_in_table {
                let all = data(1..=n + 50, dim, &keys);
                for e in &left {
                    assert_eq!(
                        e.vector.as_deref(),
                        Some(all[e.payload as usize - 1].2.slice())
                    );
                }
                let meta_guard = built.read(0);
                let tapes = MetaTuple::deserialize_ref(meta_guard.get(1).unwrap())
                    .vectors_first()
                    .to_vec();
                drop(meta_guard);
                let mut stored = Vec::new();
                for guard in tapes.into_iter().flat_map(|first| by_next(&built, first)) {
                    for i in 1..=guard.len() {
                        if let Some(bytes) = guard.get(i) {
                            let tuple = VectorTuple::<VectOwned<f32>>::deserialize_ref(bytes);
                            if let (Some(p), Ok(())) = (tuple.payload(), tuple.metadata_or_head()) {
                                stored.push(p.get());
                            }
                        }
                    }
                }
                stored.sort_unstable();
                assert_eq!(stored, kept, "the vectors that the vector tapes reach");
            }
        }
    }

    /// A write serves interrupts where no page is locked, at most a page's worth of rows apart.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn serves_interrupts_while_it_writes() {
        index::interrupt::set(check);
        let keys = Keys::new([]);
        for rerank_in_table in [false, true] {
            let (dim, n) = (100, 3000);
            let relation = index(&keys, rerank_in_table, dim, 1);
            let rows = sorted(&relation, &data(1..=n, dim, &keys));
            CHECKED.set(&relation);
            TAKEN.set(0);
            CHECKS.with(|x| x.borrow_mut().clear());
            write::<_, O>(&relation, rows, 1 << 30, &mut Vec::new());
            CHECKED.set(std::ptr::null());
            let checks = CHECKS.with(|x| x.take());
            let apart = std::iter::once(0)
                .chain(checks.iter().copied())
                .collect::<Vec<_>>();
            let most = apart
                .windows(2)
                .map(|x| x[1] - x[0])
                .max()
                .unwrap_or(n as usize);
            // vectors move on to a page every few rows; frozen rows, every few blocks of 32
            let bound = if rerank_in_table { 32 * 8 } else { 32 };
            assert!(most <= bound, "{most} rows between checks, of {n}");
        }
    }

    /// The relation; each write records whether the list's jump page is share-locked, and the first runs `meanwhile`.
    struct Watched<'a, F>(&'a M, u32, std::cell::RefCell<(Vec<bool>, Option<F>)>);

    impl<F> index::relation::Relation for Watched<'_, F> {
        type Page = <M as index::relation::Relation>::Page;
    }

    impl<F> index::relation::RelationReadTypes for Watched<'_, F> {
        type ReadGuard<'b> = <M as index::relation::RelationReadTypes>::ReadGuard<'b>;
    }

    impl<F> RelationRead for Watched<'_, F> {
        fn read(&self, id: u32) -> Self::ReadGuard<'_> {
            self.0.read(id)
        }
    }

    impl<F> index::relation::RelationWriteTypes for Watched<'_, F> {
        type WriteGuard<'b> = <M as index::relation::RelationWriteTypes>::WriteGuard<'b>;
    }

    impl<F: FnOnce()> RelationWrite for Watched<'_, F> {
        fn write(&self, id: u32, tracking_freespace: bool) -> Self::WriteGuard<'_> {
            self.watch();
            self.0.write(id, tracking_freespace)
        }
        fn extend(&self, opaque: Opaque, tracking_freespace: bool) -> Self::WriteGuard<'_> {
            self.watch();
            self.0.extend(opaque, tracking_freespace)
        }
        fn search(&self, freespace: usize) -> Option<Self::WriteGuard<'_>> {
            self.0.search(freespace)
        }
        fn commit_pair(&self, first: Self::WriteGuard<'_>, second: Self::WriteGuard<'_>) {
            self.0.commit_pair(first, second)
        }
    }

    impl<F: FnOnce()> Watched<'_, F> {
        fn watch(&self) {
            let locked = self.0.page(self.1).try_borrow_mut().is_err();
            let meanwhile = {
                let mut state = self.2.borrow_mut();
                state.0.push(locked);
                state.1.take()
            };
            if let Some(meanwhile) = meanwhile {
                meanwhile();
            }
        }
    }

    /// An insert appends under its list's jump page share lock, which a maintain takes exclusively to swap the list's
    /// tapes: a maintain between the insert's routing and its append cannot swap them.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn inserts_append_under_the_jump_page() {
        let (keys, dim) = (Keys::new([]), 100);
        let relation = index(&keys, false, dim, 1);
        write::<_, O>(
            &relation,
            sorted(&relation, &data(1..=3000, dim, &keys)),
            1 << 30,
            &mut Vec::new(),
        );
        let first = firsts(&relation, || ())[0];
        let rows = data(3001..=3100, dim, &keys);
        // from the 41st on, the list holds rows to compact
        let swapped = Cell::new(None);
        for (i, (payload, columns, vector)) in rows.iter().enumerate() {
            let meanwhile = || {
                let maintain = std::panic::AssertUnwindSafe(|| {
                    let prefetch = MakePlainPrefetcher { index: &relation };
                    maintain::<_, O>(&relation, prefetch, 1 << 30, &Pending::default(), || ())
                });
                swapped.set(Some(std::panic::catch_unwind(maintain).is_ok()));
            };
            let watched = Watched(
                &relation,
                first,
                (Vec::new(), (i == 40).then_some(meanwhile)).into(),
            );
            let key = insert_vector::<_, O>(&relation, *payload, vector.as_borrowed(), &mut First);
            let (bump, prefetch) = (
                bumpalo::Bump::new(),
                MakePlainPrefetcher { index: &watched },
            );
            insert::<_, O>(
                &watched,
                *payload,
                columns,
                vector.as_borrowed(),
                key,
                &bump,
                prefetch,
            );
            let (locked, _) = watched.2.into_inner();
            assert!(
                locked.iter().all(|&x| x),
                "an insert writes with its jump page unlocked"
            );
        }
        assert_eq!(
            swapped.get(),
            Some(false),
            "a maintain swapped a list's tapes under an insert"
        );
        let prefetch = MakePlainPrefetcher { index: &relation };
        maintain::<_, O>(&relation, prefetch, 1 << 30, &Pending::default(), || ());
        let mut payloads = lists(&relation, &keys, dim)
            .into_iter()
            .flat_map(|list| list.rows.into_iter().map(|x| x.payload))
            .collect::<Vec<_>>();
        payloads.sort_unstable();
        assert_eq!(payloads, (1..=3100).collect::<Vec<_>>());
    }
}
