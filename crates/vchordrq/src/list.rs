// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use crate::closure_lifetime_binder::{id_0, id_1};
use crate::keys::{Columns, Keys};
use crate::tape::{Layout, TapeWriter, by_next, paced};
use crate::tape_writer::{DirectoryTapeWriter, FrozenTapeWriter};
use crate::tuples::*;
use crate::{Branch, Opaque, tape};
use index::relation::{Page, PageGuard, RelationRead, RelationWrite};
use index_accessor::FunctionalAccessor;
use std::num::NonZero;
use zerocopy::{FromBytes, IntoBytes};

/// The lists' jump pages, in the tree's order; `check` runs after each tape of the tree, where no page is locked.
pub fn firsts<R: RelationRead>(index: &R, check: impl Fn()) -> Vec<u32>
where
    R::Page: Page<Opaque = Opaque>,
{
    walk(index, check, |_| ())
}

/// The lists' jump pages, as `firsts`; `page` sees each page of the tree's tapes, as it is read.
pub fn walk<R: RelationRead>(index: &R, check: impl Fn(), mut page: impl FnMut(u32)) -> Vec<u32>
where
    R::Page: Page<Opaque = Opaque>,
{
    let meta_guard = index.read(0);
    let meta_tuple = MetaTuple::deserialize_ref(meta_guard.get(1).expect("data corruption"));
    let height_of_root = meta_tuple.height_of_root();
    let mut state = vec![meta_tuple.first()];
    drop(meta_guard);
    for _ in (1..height_of_root).rev() {
        let mut results = Vec::new();
        for first in state {
            tape::read_h1_tape::<R, _, _>(
                by_next(index, first).inspect(|guard| page(guard.id())),
                || FunctionalAccessor::new((), id_0(|_, _| ()), id_1(|_, _| [(); _])),
                |(), _, _, first, _| results.push(first),
            );
            // `check` may sleep or serve interrupts, which LWLocks hold off: never under a buffer lock (#30)
            check();
        }
        state = results;
    }
    state
}

/// Writes a list's rows, in the order given, as its frozen tape, then its appendable tape, then its directory.
pub struct ListWriter<'a, R: RelationWrite + 'a> {
    index: &'a R,
    keys: &'a Keys,
    frozen: FrozenTapeWriter<'a, R>,
}

/// A written list; its appendable tape is still open.
pub struct Written<'a, R: RelationWrite + 'a> {
    pub frozen_first: u32,
    pub directory_first: u32,
    pub appendable: TapeWriter<'a, R, AppendableTuple>,
}

impl<'a, R: RelationRead + RelationWrite> ListWriter<'a, R>
where
    R::Page: Page<Opaque = Opaque>,
{
    pub fn create(index: &'a R, keys: &'a Keys) -> Self {
        Self {
            index,
            keys,
            frozen: FrozenTapeWriter::create(index, keys, false),
        }
    }
    /// Returns whether the frozen tape moved on to a new page.
    pub fn push(&mut self, branch: Branch<NonZero<u64>>, columns: &[u8]) -> bool {
        let last = self.frozen.last();
        self.frozen.push(branch, columns);
        self.frozen.last() != last
    }
    /// Unlocks the frozen tape's last page; the next push locks it again.
    pub fn unlock(&mut self) {
        self.frozen.unlock();
    }
    /// The rows short of a frozen block go to the appendable tape; `pace` runs where no page is locked.
    pub fn finish(self, pace: impl Fn()) -> Written<'a, R> {
        let Self {
            index,
            keys,
            frozen,
        } = self;
        let (frozen, branches, columns) = frozen.into_inner();
        let frozen_first = { frozen }.first();
        let mut appendable = TapeWriter::create(index, false);
        let width = keys.columns();
        for (i, branch) in branches.into_iter().enumerate() {
            appendable.push(appendable_tuple(branch, &columns[i * width..][..width]));
        }
        appendable.unlock();
        let summarized = keys.summaries() != 0;
        let mut layout = Layout::new(keys.summaries());
        for guard in paced(by_next(index, frozen_first), &pace) {
            layout.page(guard.id());
            if summarized {
                for i in 1..=guard.len() {
                    let bytes = guard.get(i).expect("data corruption");
                    layout.tuple(match FrozenTuple::deserialize_ref(bytes) {
                        FrozenTupleReader::_0(x) => Some(keys.summarize(x.payload(), x.columns())),
                        FrozenTupleReader::_1(_) => None,
                    });
                }
            }
        }
        let mut directory = DirectoryTapeWriter::create(index, false);
        directory.push(layout.ids.as_slice());
        if summarized {
            directory.push_layout(&layout.words());
        }
        let directory_first = { directory.into_inner() }.first();
        Written {
            frozen_first,
            directory_first,
            appendable,
        }
    }
}

/// A row as an appendable tuple stores it.
pub fn appendable_tuple(branch: Branch<NonZero<u64>>, columns: &[u8]) -> AppendableTuple {
    AppendableTuple {
        metadata: [
            branch.code.0.dis_u_2,
            branch.code.0.factor_cnt,
            branch.code.0.factor_ip,
            branch.code.0.factor_err,
        ],
        elements: rabitq::bit::binary::pack_code(&branch.code.1),
        delta: branch.delta,
        prefetch: branch.prefetch,
        head: branch.head,
        payload: Some(branch.extra),
        columns: columns.to_vec(),
    }
}

/// A list's row, but for its signs, pages and columns.
#[derive(Clone, Copy)]
pub struct Fixed {
    pub metadata: [f32; 4],
    pub delta: f32,
    /// Where its vector's first part is, on the first of its pages.
    pub head: u16,
    pub payload: NonZero<u64>,
}

/// A list's rows, packed, to be written in the order of their keys.
pub struct Rows {
    dim: usize,
    fixed: Vec<Fixed>,
    /// per row: the signs of the code, a bit each
    signs: Vec<u64>,
    /// per row: the same number of pages
    prefetch: Vec<u32>,
    /// per row: the filter key columns, as an appendable tuple stores them
    columns: Vec<u8>,
}

impl Rows {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            fixed: Vec::new(),
            signs: Vec::new(),
            prefetch: Vec::new(),
            columns: Vec::new(),
        }
    }
    pub fn len(&self) -> usize {
        self.fixed.len()
    }
    pub fn bytes(&self) -> usize {
        let fixed = size_of::<Fixed>() + size_of::<u32>();
        let words = self.signs.len() * size_of::<u64>() + self.prefetch.len() * 4;
        self.len() * fixed + words + self.columns.len()
    }
    pub fn push(&mut self, row: tape::Row<'_, (rabitq::bit::Code, f32)>, keys: &Keys) {
        let ((m, signs), delta) = row.value;
        let fixed = Fixed {
            metadata: [m.dis_u_2, m.factor_cnt, m.factor_ip, m.factor_err],
            delta,
            head: row.head,
            payload: row.payload,
        };
        // a frozen code is padded to a multiple of 4 signs, with zeros
        let signs = rabitq::bit::binary::pack_code(&signs[..self.dim]);
        self.fixed(fixed, &signs, row.prefetch);
        row.columns.extend(keys, &mut self.columns);
    }
    /// A row from its parts, its signs and columns as an appendable tuple stores them.
    pub fn push_parts(&mut self, fixed: Fixed, signs: &[u64], prefetch: &[u32], columns: &[u8]) {
        self.fixed(fixed, signs, prefetch);
        self.columns.extend_from_slice(columns);
    }
    /// Hands the rows to `put` as `load` reads them, a few thousand at a time, and clears them.
    pub fn save(&mut self, mut put: impl FnMut(&[u8])) {
        let words = self.dim.div_ceil(64);
        let n = self.prefetch.len().checked_div(self.len()).unwrap_or(0);
        let width = self.columns.len().checked_div(self.len()).unwrap_or(0);
        let mut out = Vec::new();
        for start in (0..self.len()).step_by(4096) {
            let end = self.len().min(start + 4096);
            out.clear();
            out.extend(((end - start) as u32).to_ne_bytes());
            for x in &self.fixed[start..end] {
                out.extend(x.metadata.as_bytes());
                out.extend(x.delta.to_ne_bytes());
                out.extend(x.head.to_ne_bytes());
                out.extend(x.payload.get().to_ne_bytes());
            }
            out.extend(self.signs[start * words..end * words].as_bytes());
            out.extend(self.prefetch[start * n..end * n].as_bytes());
            out.extend(&self.columns[start * width..end * width]);
            put(&out);
        }
        *self = Self::new(self.dim);
    }
    /// The rows `save` wrote, of `width` bytes of columns each.
    pub fn load(dim: usize, width: usize, bytes: &[u8]) -> Self {
        let (n, mut bytes) = bytes.split_first_chunk().expect("saved rows");
        let n = u32::from_ne_bytes(*n) as usize;
        let mut rows = Self::new(dim);
        for _ in 0..n {
            let fixed;
            (fixed, bytes) = bytes.split_first_chunk::<30>().expect("saved rows");
            let (metadata, fixed) = fixed.split_first_chunk::<16>().expect("saved rows");
            let (delta, fixed) = fixed.split_first_chunk().expect("saved rows");
            let (head, payload) = fixed.split_first_chunk().expect("saved rows");
            rows.fixed.push(Fixed {
                metadata: FromBytes::read_from_bytes(metadata).expect("saved rows"),
                delta: f32::from_ne_bytes(*delta),
                head: u16::from_ne_bytes(*head),
                payload: NonZero::new(u64::from_ne_bytes(payload.try_into().expect("saved rows")))
                    .expect("saved rows"),
            });
        }
        let (signs, rest) = bytes.split_at(n * dim.div_ceil(64) * size_of::<u64>());
        let (prefetch, columns) = rest.split_at(rest.len() - n * width);
        rows.signs
            .extend(signs.as_chunks().0.iter().map(|&x| u64::from_ne_bytes(x)));
        rows.prefetch.extend(
            prefetch
                .as_chunks()
                .0
                .iter()
                .map(|&x| u32::from_ne_bytes(x)),
        );
        rows.columns.extend_from_slice(columns);
        rows
    }
    fn fixed(&mut self, fixed: Fixed, signs: &[u64], prefetch: &[u32]) {
        self.fixed.push(fixed);
        self.signs.extend_from_slice(signs);
        self.prefetch.extend_from_slice(prefetch);
    }
    /// Writes the rows to `sink` in the order of their keys, stable, and clears them.
    pub fn flush(&mut self, keys: &Keys, mut sink: impl FnMut(Branch<NonZero<u64>>, &[u8])) {
        let width = keys.columns();
        let row = |i: u32| Columns::row(&self.columns[i as usize * width..][..width]);
        let mut order = (0..self.len() as u32).collect::<Vec<_>>();
        order.sort_unstable_by(|&i, &j| keys.cmp(row(i), row(j)).then(i.cmp(&j)));
        for i in order {
            let (branch, columns) = self.branch(i as _);
            sink(branch, columns);
        }
        *self = Self::new(self.dim);
    }
    fn branch(&self, i: usize) -> (Branch<NonZero<u64>>, &[u8]) {
        let Fixed {
            metadata,
            delta,
            head,
            payload,
            ..
        } = self.fixed[i];
        let words = self.dim.div_ceil(64);
        let n = self.prefetch.len() / self.len();
        let width = self.columns.len() / self.len();
        let branch = Branch {
            code: unpack_code(metadata, &self.signs[i * words..][..words], self.dim),
            delta,
            prefetch: self.prefetch[i * n..][..n].to_vec(),
            head,
            norm: 0.0,
            extra: payload,
        };
        (branch, &self.columns[i * width..][..width])
    }
}

/// A code from its metadata and its first `dim` signs, as an appendable tuple stores them.
pub fn unpack_code(metadata: [f32; 4], words: &[u64], dim: usize) -> rabitq::bit::Code {
    let signs = words
        .iter()
        .flat_map(|x| std::array::from_fn::<_, 64, _>(|j| x & (1 << j) != 0))
        .take(dim)
        .collect();
    (rabitq::bit::CodeMetadata::from_array(metadata), signs)
}
