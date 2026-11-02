// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.

use std::num::NonZero;

#[derive(Debug, Clone, Copy)]
pub enum Typmod {
    Any,
    Dims(NonZero<u32>),
}

impl Typmod {
    pub fn new(x: i32) -> Option<Self> {
        use Typmod::*;
        if x == -1 {
            Some(Any)
        } else if x >= 1 {
            Some(Dims(NonZero::new(x as u32).unwrap()))
        } else {
            None
        }
    }
    pub fn dim(self) -> Option<NonZero<u32>> {
        use Typmod::*;
        match self {
            Any => None,
            Dims(dim) => Some(dim),
        }
    }
}
