// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use crate::keys::{Columns, Keys};
use crate::tape::{Layout, TapeWriter, by_next, paced};
use crate::tape_writer::{DirectoryTapeWriter, FrozenTapeWriter};
use crate::tuples::*;
use crate::{Branch, Opaque, tape};
use index::relation::{Page, PageGuard, RelationRead, RelationWrite};
use std::num::NonZero;

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
struct Fixed {
    metadata: [f32; 4],
    delta: f32,
    head: u16,
    payload: NonZero<u64>,
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
    fn len(&self) -> usize {
        self.fixed.len()
    }
    pub fn bytes(&self) -> usize {
        let fixed = size_of::<Fixed>() + size_of::<u32>();
        let words = self.signs.len() * size_of::<u64>() + self.prefetch.len() * 4;
        self.len() * fixed + words + self.columns.len()
    }
    pub fn push(&mut self, row: tape::Row<'_, (rabitq::bit::Code, f32)>, keys: &Keys) {
        let ((m, signs), delta) = row.value;
        let metadata = [m.dis_u_2, m.factor_cnt, m.factor_ip, m.factor_err];
        let (head, payload) = (row.head, row.payload);
        self.fixed.push(Fixed {
            metadata,
            delta,
            head,
            payload,
        });
        // a frozen code is padded to a multiple of 4 signs, with zeros
        let signs = rabitq::bit::binary::pack_code(&signs[..self.dim]);
        self.signs.extend_from_slice(&signs);
        self.prefetch.extend_from_slice(row.prefetch);
        row.columns.extend(keys, &mut self.columns);
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
