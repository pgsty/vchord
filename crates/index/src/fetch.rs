// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use crate::packed::PackedRefMut;
use always_equal::AlwaysEqual;

pub type BorrowedIter<'b> = small_iter::borrowed::Iter<'b, u32, 1>;

pub trait Fetch<'b> {
    type Iter: ExactSizeIterator<Item = u32> + 'b;
    #[must_use]
    fn fetch(&self) -> Self::Iter;
}

impl Fetch<'_> for u32 {
    type Iter = std::iter::Once<u32>;
    #[inline(always)]
    fn fetch(&self) -> std::iter::Once<u32> {
        std::iter::once(*self)
    }
}

impl<'b, T, A, B, W: 'b + PackedRefMut<T = (A, B, BorrowedIter<'b>)>> Fetch<'b>
    for (T, AlwaysEqual<W>)
{
    type Iter = BorrowedIter<'b>;
    #[inline(always)]
    fn fetch(&self) -> BorrowedIter<'b> {
        let (.., list) = self.1.0.get();
        *list
    }
}

impl<'b, T, A, B, C> Fetch<'b> for (T, AlwaysEqual<&mut (A, B, C, BorrowedIter<'b>)>) {
    type Iter = BorrowedIter<'b>;
    #[inline(always)]
    fn fetch(&self) -> BorrowedIter<'b> {
        let (_, AlwaysEqual((.., list))) = self;
        *list
    }
}

impl<T> Fetch<'_> for (T, AlwaysEqual<(u32, u16)>) {
    type Iter = std::iter::Once<u32>;
    #[inline(always)]
    fn fetch(&self) -> std::iter::Once<u32> {
        let (_, AlwaysEqual((x, _))) = self;
        std::iter::once(*x)
    }
}

impl Fetch<'_> for (u32, u16) {
    type Iter = std::iter::Once<u32>;
    #[inline(always)]
    fn fetch(&self) -> std::iter::Once<u32> {
        std::iter::once(self.0)
    }
}

impl<T> Fetch<'_> for (T, AlwaysEqual<((u32, u16), (u32, u16))>) {
    type Iter = std::iter::Once<u32>;
    #[inline(always)]
    fn fetch(&self) -> std::iter::Once<u32> {
        let (_, AlwaysEqual(((x, _), _))) = self;
        std::iter::once(*x)
    }
}

pub trait Fetch1 {
    fn fetch_1(&self) -> u32;
}

#[repr(transparent)]
pub struct Fetch1Iter<I> {
    iter: I,
}

impl<I: Iterator> Iterator for Fetch1Iter<I>
where
    I::Item: Fetch1,
{
    type Item = u32;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|x| x.fetch_1())
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<I: ExactSizeIterator> ExactSizeIterator for Fetch1Iter<I> where I::Item: Fetch1 {}

impl<'b, T, F: Fetch1 + Copy> Fetch<'b> for (T, AlwaysEqual<&'b [F]>) {
    type Iter = Fetch1Iter<std::iter::Copied<std::slice::Iter<'b, F>>>;

    #[inline(always)]
    fn fetch(&self) -> Self::Iter {
        Fetch1Iter {
            iter: self.1.0.iter().copied(),
        }
    }
}

impl<'b, T, U, F: Fetch1 + Copy> Fetch<'b> for (T, AlwaysEqual<(&'b [F], U)>) {
    type Iter = Fetch1Iter<std::iter::Copied<std::slice::Iter<'b, F>>>;

    #[inline(always)]
    fn fetch(&self) -> Self::Iter {
        Fetch1Iter {
            iter: self.1.0.0.iter().copied(),
        }
    }
}
