// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use crate::Id;
use std::collections::HashSet;

pub struct Visited {
    inner: HashSet<Id>,
}

impl Visited {
    pub fn new() -> Self {
        Self {
            inner: HashSet::new(),
        }
    }
    pub fn insert(&mut self, x: Id) {
        self.inner.insert(x);
    }
    pub fn contains(&mut self, x: Id) -> bool {
        self.inner.contains(&x)
    }
}
