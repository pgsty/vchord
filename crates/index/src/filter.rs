// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use crate::prefetcher::Sequence;

/// A [`Sequence`] that skips items rejected by `predicate`.
///
/// The predicate runs at most once per item: a head that was accepted by
/// `peek` is remembered until `next` consumes it (#34). With prefiltering, the
/// predicate fetches the heap tuple and evaluates the executor's qual, so
/// evaluating it again for every `peek`/`next` pair cost extra heap fetches
/// and changed the semantics of volatile quals. A new `Filter` is built for
/// every rescan.
pub struct Filter<S, P> {
    sequence: S,
    predicate: P,
    head_accepted: bool,
}

impl<S, P> Filter<S, P>
where
    S: Sequence,
    P: FnMut(&S::Item) -> bool,
{
    /// Advances past rejected items; returns whether an accepted head exists.
    fn settle(&mut self) -> bool {
        if self.head_accepted {
            return true;
        }
        loop {
            let Some(head) = self.sequence.peek() else {
                return false;
            };
            if (self.predicate)(head) {
                self.head_accepted = true;
                return true;
            }
            let _ = self.sequence.next();
            crate::interrupt::check();
        }
    }
}

impl<S, P> Sequence for Filter<S, P>
where
    S: Sequence,
    P: FnMut(&S::Item) -> bool,
{
    type Item = S::Item;

    type Inner = S::Inner;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.settle() {
            return None;
        }
        self.head_accepted = false;
        self.sequence.next()
    }

    fn peek(&mut self) -> Option<&Self::Item> {
        if !self.settle() {
            return None;
        }
        self.sequence.peek()
    }

    fn into_inner(self) -> Self::Inner {
        self.sequence.into_inner()
    }
}

pub fn filter<S, P>(sequence: S, predicate: P) -> Filter<S, P> {
    Filter {
        sequence,
        predicate,
        head_accepted: false,
    }
}

#[cfg(test)]
mod tests {
    use super::filter;
    use crate::prefetcher::Sequence;
    use std::cell::Cell;
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    // (value, identity): equal values, distinct items.
    fn heap(items: &[(u32, u32)]) -> BinaryHeap<Reverse<(u32, u32)>> {
        items.iter().copied().map(Reverse).collect()
    }

    #[test]
    fn predicate_runs_once_per_item() {
        let calls = Cell::new(0);
        let mut f = filter(
            heap(&[(1, 0), (1, 1), (2, 2), (3, 3)]),
            |Reverse((_, id)): &Reverse<(u32, u32)>| {
                calls.set(calls.get() + 1);
                id % 2 == 1
            },
        );
        // peek -> peek -> next on the same head evaluates it once.
        assert_eq!(f.peek(), Some(&Reverse((1, 1))));
        assert_eq!(f.peek(), Some(&Reverse((1, 1))));
        assert_eq!(calls.get(), 2); // (1, 0) rejected, (1, 1) accepted
        assert_eq!(f.next(), Some(Reverse((1, 1))));
        assert_eq!(calls.get(), 2);
        // The next head is a different item with a different identity.
        assert_eq!(f.next(), Some(Reverse((3, 3))));
        assert_eq!(calls.get(), 4); // (2, 2) rejected, (3, 3) accepted
        assert_eq!(f.peek(), None);
        assert_eq!(f.next(), None);
        assert_eq!(calls.get(), 4);
    }

    #[test]
    fn equal_values_distinct_heads() {
        let calls = Cell::new(0);
        let mut f = filter(
            heap(&[(5, 0), (5, 1), (5, 2)]),
            |_: &Reverse<(u32, u32)>| {
                calls.set(calls.get() + 1);
                true
            },
        );
        let mut seen = Vec::new();
        while let Some(Reverse((_, id))) = f.next() {
            let _ = f.peek();
            seen.push(id);
        }
        seen.sort_unstable();
        assert_eq!(seen, vec![0, 1, 2]);
        assert_eq!(calls.get(), 3);
    }

    #[test]
    fn a_new_filter_per_rescan_reevaluates() {
        let calls = Cell::new(0);
        let predicate = |_: &Reverse<(u32, u32)>| {
            calls.set(calls.get() + 1);
            true
        };
        let mut f = filter(heap(&[(1, 0)]), predicate);
        assert!(f.peek().is_some());
        drop(f);
        let mut g = filter(heap(&[(1, 0)]), predicate);
        assert!(g.peek().is_some());
        assert_eq!(calls.get(), 2);
    }
}
