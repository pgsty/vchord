// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use crate::insert::assign;
use crate::keys::{Columns, Keys};
use crate::list::{ListWriter, Rows, firsts};
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
use zerocopy::IntoBytes;

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

/// Rows as `encode` appends them, in the order of their bytes; each is valid until the next.
pub trait Sorted {
    fn next(&mut self) -> Option<&[u8]>;
}

/// Appends a row as the build sorts it: by list, then by its keys, then by payload, as `write` lays rows out.
#[allow(clippy::too_many_arguments)]
pub fn encode<'b, R: RelationRead, O: Operator>(
    index: &'b R,
    payload: NonZero<u64>,
    columns: &[u8],
    vector: <O::Vector as VectorOwned>::Borrowed<'_>,
    projected: <O::Vector as VectorOwned>::Borrowed<'_>,
    bump: &'b impl Bump,
    prefetch_h1_vectors: impl PrefetcherHeapFamily<'b, R>,
    out: &mut Vec<u8>,
) where
    R::Page: Page<Opaque = Opaque>,
{
    let (first, (code, delta)) = assign::<R, O>(index, projected, bump, prefetch_h1_vectors);
    let meta_guard = index.read(0);
    let meta_tuple = MetaTuple::deserialize_ref(meta_guard.get(1).expect("data corruption"));
    let keys = Keys::of_meta(meta_tuple);
    let rerank_in_heap = meta_tuple.rerank_in_heap();
    drop(meta_guard);
    out.extend_from_slice(&first.to_be_bytes());
    keys.sortable(Columns::row(columns), out);
    out.extend_from_slice(&payload.get().to_be_bytes());
    let tuple = AppendableTuple::serialize(&AppendableTuple {
        metadata: [
            code.0.dis_u_2,
            code.0.factor_cnt,
            code.0.factor_ip,
            code.0.factor_err,
        ],
        delta,
        payload: Some(payload),
        prefetch: Vec::new(),
        head: 0,
        elements: rabitq::bit::binary::pack_code(&code.1),
        columns: columns.to_vec(),
    });
    out.extend_from_slice(&(tuple.len() as u32).to_ne_bytes());
    out.extend_from_slice(&tuple);
    if !rerank_in_heap {
        vectors::encode::<O::Vector>(vector, out);
    }
}

/// Writes every list from `rows`, each in one run of pages: its vectors, its frozen tape, its appendable tape, its directory.
/// `memory`: the most bytes of a list's rows held before they are written.
pub fn write<R: RelationRead + RelationWrite, O: Operator>(
    index: &R,
    mut rows: impl Sorted,
    memory: usize,
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
    let prefix = size_of::<u32>() + keys.sortable_len() + size_of::<u64>();
    let mut firsts = firsts(index, || ());
    firsts.sort_unstable();
    let mut vectors = None::<TapeWriter<'_, R, VectorTuple<O::Vector>>>;
    let mut aligned = Vec::<u64>::new();
    let mut row = rows.next();
    for first in firsts {
        let (mut list, mut pending, mut tuples) = (None, Rows::new(dim as _), 0_u64);
        while let Some(bytes) = row
            && bytes[..4] == first.to_be_bytes()
        {
            let (len, rest) = bytes[prefix..].split_at(size_of::<u32>());
            let (tuple, vector) = rest.split_at(u32::from_ne_bytes(len.try_into().unwrap()) as _);
            aligned.clear();
            aligned.resize(tuple.len().div_ceil(8), 0);
            aligned.as_mut_bytes()[..tuple.len()].copy_from_slice(tuple);
            let tuple = AppendableTuple::deserialize_ref(&aligned.as_bytes()[..tuple.len()]);
            let (prefetch, head) = if rerank_in_heap {
                (Vec::new(), 0)
            } else {
                let tape = vectors.get_or_insert_with(|| TapeWriter::create(index, true));
                // a list's vectors begin on a page of their own
                if tuples == 0 {
                    tape.renew();
                }
                let payload = tuple.payload().expect("a live row");
                let vector = vectors::decode::<O::Vector>(dim, vector);
                vectors::write::<O>(vector.as_borrowed(), payload, |x| tape.push(x))
            };
            pending.push_appendable(tuple, &prefetch, head);
            tuples += 1;
            if pending.bytes() > memory {
                let list = list.get_or_insert_with(|| ListWriter::create(index, &keys));
                pending.flush(&keys, |branch, columns| _ = list.push(branch, columns));
            }
            row = rows.next();
        }
        if let Some(tape) = vectors.as_mut() {
            tape.unlock();
        }
        let mut list = list.unwrap_or_else(|| ListWriter::create(index, &keys));
        pending.flush(&keys, |branch, columns| _ = list.push(branch, columns));
        let written = list.finish(|| ());
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
    use crate::closure_lifetime_binder::{id_0, id_1};
    use crate::keys::{Summary, Width};
    use crate::operator::Op;
    use crate::tape::{by_next, read_appendable_tape, read_directory_words, read_frozen_tape};
    use index::relation::PageGuard;
    use index::test_support::{MakePlainPrefetcher, MemoryRelation};
    use index_accessor::{FunctionalAccessor, L2S};
    use vector::vect::VectOwned;

    type M = MemoryRelation<Opaque>;
    type O = Op<VectOwned<f32>, L2S>;

    struct Rows(std::vec::IntoIter<Vec<u8>>, Vec<u8>);

    impl Sorted for Rows {
        fn next(&mut self) -> Option<&[u8]> {
            self.1 = self.0.next()?;
            Some(&self.1)
        }
    }

    fn random(dim: usize) -> VectOwned<f32> {
        VectOwned::new((0..dim).map(|_| rand::random::<f32>() - 0.5).collect())
    }

    /// Per list: its vectors' pages, in row order, then its frozen, appendable and directory pages and its rows' payloads and keys.
    fn lists(
        relation: &M,
        keys: &Keys,
    ) -> Vec<(Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, Vec<(u64, u128)>)> {
        let mut words = Vec::new();
        let mut firsts = firsts(relation, || ());
        firsts.sort_unstable();
        let accessor = || FunctionalAccessor::new((), id_0(|_, _| ()), id_1(|_, _| [(); _]));
        firsts
            .into_iter()
            .map(|first| {
                let jump_guard = relation.read(first);
                let jump_tuple = JumpTuple::deserialize_ref(jump_guard.get(1).unwrap());
                let directory = by_next(relation, jump_tuple.directory_first()).map(|g| g.id());
                let pages = read_directory_words::<M>(
                    by_next(relation, jump_tuple.directory_first()),
                    &mut words,
                    false,
                );
                let (mut vectors, mut rows) = (Vec::new(), Vec::new());
                let mut row = |row: crate::tape::Row<'_, ()>| {
                    vectors.extend_from_slice(row.prefetch);
                    let key = keys.as_slice().first().map_or(0, |k| k.get(row.columns));
                    rows.push((row.payload.get(), key));
                };
                read_frozen_tape::<M, _, _>(
                    words[..pages].iter().map(|&id| relation.read(id)),
                    crate::tape::All,
                    accessor,
                    (),
                    &mut row,
                );
                let appendable = by_next(relation, jump_tuple.appendable_first());
                read_appendable_tape::<M, _>(appendable, 0, |_, _, _| (), (), &mut row);
                let appendable = by_next(relation, jump_tuple.appendable_first()).map(|g| g.id());
                (
                    vectors,
                    words[..pages].to_vec(),
                    appendable.collect(),
                    directory.collect(),
                    rows,
                )
            })
            .collect()
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn one_run_per_list() {
        let keyed = Keys::new([(Width::W8, false, Some(Summary::Bounds))]);
        for (keys, rerank_in_table, dim, n, lists_n) in [
            (Keys::new([]), false, 100, 3000, 7),
            (keyed, false, 1500, 2000, 5),
            (keyed, true, 100, 3000, 7),
            (keyed, false, 100, 0, 3),
            (Keys::new([]), true, 64, 5, 1),
        ] {
            let relation = M::new([]);
            let (v, d) = (VectorKind::Vecf32, DistanceKind::L2S);
            let options = VchordrqIndexOptions {
                rerank_in_table,
                degree_of_parallelism: 4,
                ..Default::default()
            };
            let centroids = (0..lists_n).map(|_| random(dim)).collect::<Vec<_>>();
            let mut structures = vec![Structure {
                centroids,
                children: vec![Vec::new(); lists_n],
            }];
            if lists_n > 1 {
                structures.push(Structure {
                    centroids: vec![VectOwned::new(vec![0.0; dim])],
                    children: vec![(0..lists_n as u32).collect()],
                });
            }
            let vector_options = VectorOptions {
                dim: dim as _,
                v,
                d,
            };
            build::<_, O>(vector_options, options, &keys, &relation, structures);
            let mut sorted = (1..=n as u64)
                .map(|payload| {
                    let vector = random(dim);
                    let columns = if keys.is_empty() {
                        Vec::new()
                    } else {
                        [&(payload % 13).to_le_bytes()[..], &[0; 4]].concat()
                    };
                    let (mut out, bump) = (Vec::new(), bumpalo::Bump::new());
                    let prefetch = MakePlainPrefetcher { index: &relation };
                    let payload = NonZero::new(payload).unwrap();
                    let v = vector.as_borrowed();
                    encode::<_, O>(
                        &relation, payload, &columns, v, v, &bump, prefetch, &mut out,
                    );
                    out
                })
                .collect::<Vec<_>>();
            sorted.sort_unstable();
            write::<_, O>(&relation, Rows(sorted.into_iter(), Vec::new()), 1 << 30);
            let mut payloads = Vec::new();
            for (vectors, frozen, appendable, directory, rows) in lists(&relation, &keys) {
                let run = [&frozen[..], &appendable[..1], &directory[..]].concat();
                assert!(run.windows(2).all(|x| x[1] == x[0] + 1), "{run:?}");
                assert!(
                    rows.windows(2).all(|x| x[0].1 <= x[1].1),
                    "rows out of key order"
                );
                if !rerank_in_table && !vectors.is_empty() {
                    let mut pages = vectors.clone();
                    pages.sort_unstable();
                    pages.dedup();
                    assert!(pages.windows(2).all(|x| x[1] == x[0] + 1), "{pages:?}");
                    assert_eq!(pages.last().unwrap() + 1, frozen[0]);
                }
                payloads.extend(rows.iter().map(|x| x.0));
            }
            payloads.sort_unstable();
            assert_eq!(payloads, (1..=n as u64).collect::<Vec<_>>());
        }
    }
}
