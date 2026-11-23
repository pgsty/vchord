// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

/// How `k`, counting rows with visible heap tuples, bounds the rows a scan will be asked for.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KKind {
    /// Nothing is known (`k` is 0).
    Unknown = 0,
    /// No more than `k` rows, and `k` unless the statement stops early.
    Exact = 1,
    /// Rows may be dropped above the scan: at least `k` unless the statement stops early.
    AtLeast = 2,
    /// Other inputs share the Limit, or rows multiply above the scan: no more than `k`.
    AtMost = 3,
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

/// A hint, never a promise: a scan stays correct when asked for any number of rows.
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
