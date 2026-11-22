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
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

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
#[repr(C)]
#[derive(Clone, Copy, FromBytes, IntoBytes, Immutable, KnownLayout)]
pub struct Fixed {
    pub metadata: [f32; 4],
    pub delta: f32,
    /// Where its vector's first part is, on the first of its pages.
    pub head: u16,
    pub _padding: [u8; 2],
    pub payload: u64,
}

impl Fixed {
    /// The row as a list writer takes it.
    pub fn branch(self, signs: &[u64], prefetch: &[u32], dim: usize) -> Branch<NonZero<u64>> {
        Branch {
            code: unpack_code(self.metadata, signs, dim),
            delta: self.delta,
            prefetch: prefetch.to_vec(),
            head: self.head,
            norm: 0.0,
            extra: NonZero::new(self.payload).expect("data corruption"),
        }
    }
}

/// A list's rows, packed, to be written in the order of their keys, in at most `memory` bytes.
pub struct Rows {
    dim: usize,
    memory: usize,
    fixed: Vec<Fixed>,
    /// per row: the signs of the code, a bit each
    signs: Vec<u64>,
    /// per row: the same number of pages
    prefetch: Vec<u32>,
    /// per row: the filter key columns, as an appendable tuple stores them
    columns: Vec<u8>,
}

impl Rows {
    pub fn new(dim: usize, memory: usize) -> Self {
        Self {
            dim,
            memory,
            fixed: Vec::new(),
            signs: Vec::new(),
            prefetch: Vec::new(),
            columns: Vec::new(),
        }
    }
    pub fn len(&self) -> usize {
        self.fixed.len()
    }
    /// Whether they take their memory, with what `flush` adds.
    pub fn is_full(&self) -> bool {
        !self.fixed.is_empty() && self.len() >= self.most()
    }
    /// As many rows as `memory` holds, of the size of those it has.
    fn most(&self) -> usize {
        let words = self.signs.len() * size_of::<u64>() + self.prefetch.len() * size_of::<u32>();
        let bytes =
            self.len() * (size_of::<Fixed>() + size_of::<u32>()) + words + self.columns.len();
        (self.memory / bytes.div_ceil(self.len())).max(1)
    }
    pub fn push(&mut self, row: tape::Row<'_, (rabitq::bit::Code, f32)>, keys: &Keys) {
        let ((m, signs), delta) = row.value;
        let fixed = Fixed {
            metadata: [m.dis_u_2, m.factor_cnt, m.factor_ip, m.factor_err],
            delta,
            head: row.head,
            _padding: [0; 2],
            payload: row.payload.get(),
        };
        // a frozen code is padded to a multiple of 4 signs, with zeros
        let signs = rabitq::bit::binary::pack_code(&signs[..self.dim]);
        self.grow();
        self.fixed.push(fixed);
        self.signs.extend_from_slice(&signs);
        self.prefetch.extend_from_slice(row.prefetch);
        row.columns.extend(keys, &mut self.columns);
    }
    /// A row from its parts, its signs and columns as an appendable tuple stores them.
    pub fn push_parts(&mut self, fixed: Fixed, signs: &[u64], prefetch: &[u32], columns: &[u8]) {
        self.grow();
        self.fixed.push(fixed);
        self.signs.extend_from_slice(signs);
        self.prefetch.extend_from_slice(prefetch);
        self.columns.extend_from_slice(columns);
    }
    /// Room for one more row: they grow by doubling, but past their memory only by a few rows at a time.
    fn grow(&mut self) {
        let n = self.len();
        if n == 0 || n < self.fixed.capacity() {
            return;
        }
        let more = n.min(self.most().saturating_sub(n)).max(64);
        self.fixed.reserve_exact(more);
        self.signs.reserve_exact(more * (self.signs.len() / n));
        self.prefetch
            .reserve_exact(more * (self.prefetch.len() / n));
        self.columns.reserve_exact(more * (self.columns.len() / n));
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
            out.extend(self.fixed[start..end].as_bytes());
            out.extend(self.signs[start * words..end * words].as_bytes());
            out.extend(self.prefetch[start * n..end * n].as_bytes());
            out.extend(&self.columns[start * width..end * width]);
            put(&out);
        }
        *self = Self::new(self.dim, self.memory);
    }
    /// The rows `save` wrote, of `width` bytes of columns each.
    pub fn load(dim: usize, width: usize, bytes: &[u8]) -> Self {
        let (n, bytes) = bytes.split_first_chunk().expect("saved rows");
        let n = u32::from_ne_bytes(*n) as usize;
        let (fixed, bytes) = bytes.split_at(n * size_of::<Fixed>());
        let (signs, bytes) = bytes.split_at(n * dim.div_ceil(64) * size_of::<u64>());
        let (prefetch, columns) = bytes.split_at(bytes.len() - n * width);
        let read = |x: &[u8; size_of::<Fixed>()]| Fixed::read_from_bytes(x).expect("saved rows");
        Self {
            dim,
            memory: usize::MAX,
            fixed: fixed.as_chunks().0.iter().map(read).collect(),
            signs: signs
                .as_chunks()
                .0
                .iter()
                .map(|&x| u64::from_ne_bytes(x))
                .collect(),
            prefetch: prefetch
                .as_chunks()
                .0
                .iter()
                .map(|&x| u32::from_ne_bytes(x))
                .collect(),
            columns: columns.to_vec(),
        }
    }
    /// Writes the rows to `sink` in the order of their keys, stable, and clears them.
    pub fn flush(&mut self, keys: &Keys, mut sink: impl FnMut(Branch<NonZero<u64>>, &[u8])) {
        let width = keys.columns();
        let row = |i: u32| Columns::row(&self.columns[i as usize * width..][..width]);
        let mut order = (0..self.len() as u32).collect::<Vec<_>>();
        order.sort_unstable_by(|&i, &j| keys.cmp(row(i), row(j)).then(i.cmp(&j)));
        let (words, n) = (
            self.dim.div_ceil(64),
            self.prefetch.len() / self.len().max(1),
        );
        for i in order.into_iter().map(|i| i as usize) {
            let (signs, prefetch) = (
                &self.signs[i * words..][..words],
                &self.prefetch[i * n..][..n],
            );
            sink(
                self.fixed[i].branch(signs, prefetch, self.dim),
                &self.columns[i * width..][..width],
            );
        }
        *self = Self::new(self.dim, self.memory);
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Rows grow to their memory, and past it by no more than a few rows.
    #[test]
    fn rows_stay_within_their_memory() {
        let (signs, prefetch, columns) = ([u64::MAX; 12], [7_u32], [3_u8; 9]);
        let fixed = Fixed {
            metadata: [1.0; 4],
            delta: 0.5,
            head: 3,
            _padding: [0; 2],
            payload: 1,
        };
        let row = size_of::<Fixed>() + size_of::<u32>() + size_of_val(&signs) + 4 + columns.len();
        for memory in [1000, 1 << 16, (1 << 20) + 1, 5 << 20] {
            let mut rows = Rows::new(768, memory);
            for _ in 0..2 {
                while !rows.is_full() {
                    rows.push_parts(fixed, &signs, &prefetch, &columns);
                }
                let held = rows.fixed.capacity() * (size_of::<Fixed>() + size_of::<u32>())
                    + rows.signs.capacity() * size_of::<u64>()
                    + rows.prefetch.capacity() * size_of::<u32>()
                    + rows.columns.capacity();
                assert!(
                    held <= memory.max(row) + 64 * row,
                    "{held} bytes held for {memory}"
                );
                let mut n = 0;
                rows.flush(&Keys::new([]), |_, _| n += 1);
                assert_eq!(n, (memory / row).max(1));
            }
        }
    }
}
