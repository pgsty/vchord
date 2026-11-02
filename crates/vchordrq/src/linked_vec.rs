// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

pub struct LinkedVec<T> {
    inner: Vec<Vec<T>>,
    last: Vec<T>,
}

impl<T> LinkedVec<T> {
    pub fn new() -> Self {
        Self {
            inner: Vec::new(),
            last: Vec::with_capacity(4096),
        }
    }
    pub fn push(&mut self, value: T) {
        if self.last.len() == self.last.capacity() {
            self.reserve();
        }
        #[allow(unsafe_code)]
        unsafe {
            std::hint::assert_unchecked(self.last.len() != self.last.capacity());
        }
        self.last.push(value);
    }
    #[cold]
    fn reserve(&mut self) {
        let fresh = Vec::with_capacity(self.last.capacity() * 4);
        self.inner.push(core::mem::replace(&mut self.last, fresh));
    }
    pub fn into_vec(self) -> Vec<T> {
        let mut last = self.last;
        last.reserve(self.inner.iter().map(Vec::len).sum::<usize>());
        for x in self.inner {
            last.extend(x);
        }
        last
    }
}
