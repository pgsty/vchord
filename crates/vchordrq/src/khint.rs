// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

/// How `k` bounds the rows a scan will be asked for.
///
/// `k` counts the rows the scan returns whose heap tuples are visible: the
/// executor skips the others and asks for more. Nodes above the Limit may
/// stop reading early (a cursor, an outer Limit, a semi join), so a scan can
/// always be asked for fewer rows than its hint says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KKind {
    /// Nothing is known (`k` is 0).
    Unknown,
    /// No more than `k` rows, and `k` unless the statement stops early.
    Exact,
    /// Rows may be dropped above the scan (a filter, a join, a recheck): at
    /// least `k` rows unless the statement stops early, maybe more.
    AtLeast,
    /// Other inputs share the Limit, or rows may be multiplied above the
    /// scan: no more than `k` rows, maybe fewer.
    AtMost,
}

impl KKind {
    /// The bound left after passing through `self`, then `other`.
    pub fn and(self, other: Self) -> Self {
        use KKind::*;
        match (self, other) {
            (Exact, kind) | (kind, Exact) => kind,
            (AtLeast, AtLeast) => AtLeast,
            (AtMost, AtMost) => AtMost,
            _ => Unknown,
        }
    }
}

/// A hint, never a promise: a scan must stay correct when asked for any
/// number of rows. It is resolved again after every rescan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KHint {
    pub k: u64,
    pub kind: KKind,
}

impl KHint {
    pub const UNKNOWN: Self = Self {
        k: 0,
        kind: KKind::Unknown,
    };

    /// The hint once rows pass a node of `kind`.
    pub fn and(self, kind: KKind) -> Self {
        match self.kind.and(kind) {
            KKind::Unknown => Self::UNKNOWN,
            kind => Self { kind, ..self },
        }
    }
}
