// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use crate::operator::*;
use crate::tuples::*;
use index::relation::{Page, RelationRead};
use index_accessor::Accessor1;

pub fn read<
    'a,
    R: RelationRead + 'a,
    O: Operator,
    A: Accessor1<<O::Vector as Vector>::Element, <O::Vector as Vector>::Metadata>,
>(
    mut prefetch: impl Iterator<Item = R::ReadGuard<'a>>,
    head: u16,
    accessor: A,
) -> A::Output {
    let mut cursor = Err(head);
    let mut result = accessor;
    while let Err(head) = cursor {
        let guard = prefetch.next().expect("data corruption");
        let bytes = guard.get(head).expect("data corruption");
        let tuple = CentroidTuple::<O::Vector>::deserialize_ref(bytes);
        result.push(tuple.elements());
        cursor = tuple.metadata_or_head();
    }
    if prefetch.next().is_some() {
        panic!("data corruption");
    }
    result.finish(cursor.expect("data corruption"))
}
