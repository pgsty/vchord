// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.

use index::prefetcher::Sequence;

pub struct Filter<S, P> {
    sequence: S,
    predicate: P,
}

impl<S, P> Sequence for Filter<S, P>
where
    S: Sequence,
    P: FnMut(&S::Item) -> bool,
{
    type Item = S::Item;

    type Inner = S::Inner;

    fn next(&mut self) -> Option<Self::Item> {
        while !(self.predicate)(self.sequence.peek()?) {
            let _ = self.sequence.next();
        }
        self.sequence.next()
    }

    fn peek(&mut self) -> Option<&Self::Item> {
        while !(self.predicate)(self.sequence.peek()?) {
            let _ = self.sequence.next();
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
    }
}
