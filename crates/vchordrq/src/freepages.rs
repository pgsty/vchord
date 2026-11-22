// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use crate::Opaque;
use crate::tuples::*;
use index::relation::{Page, RelationRead, RelationWrite};

pub fn alloc<R: RelationWrite>(index: &R, freepages_first: u32) -> Option<R::WriteGuard<'_>>
where
    R::Page: Page<Opaque = Opaque>,
{
    let mut freepages_guard = index.write(freepages_first, false);
    let freepages_bytes = freepages_guard.get_mut(1).expect("data corruption");
    let mut freepages_tuple = FreepagesTuple::deserialize_mut(freepages_bytes);
    let id = *freepages_tuple.first();
    if id != u32::MAX {
        let mut guard = index.write(id, false);
        *freepages_tuple.first() = guard.get_opaque_mut().next;
        drop(freepages_guard); // write log of freespaces_guard
        Some(guard)
    } else {
        // else the last page of the shortest kept run, which leaves the longest for the lists a maintain places
        let mut tuple = FreeRunsTuple::deserialize_mut(freepages_guard.get_mut(2)?);
        let mut runs = tuple.runs().to_vec();
        let i = (0..runs.len()).min_by_key(|&i| runs[i].len())?;
        let id = shrink(&mut runs, i, false);
        tuple.set(&runs);
        let guard = index.write(id, false);
        drop(freepages_guard); // write log of freespaces_guard
        Some(guard)
    }
}

/// Takes the first page or else the last of the `i`th run.
fn shrink(runs: &mut Vec<Run>, i: usize, first: bool) -> u32 {
    let run = &mut runs[i];
    let id = if first {
        run.start += 1;
        run.start - 1
    } else {
        run.end -= 1;
        run.end
    };
    if run.len() == 0 {
        runs.remove(i);
    }
    id
}

// the page must be inaccessible in the graph
pub fn free<R: RelationWrite>(index: &R, freepages_first: u32, id: u32)
where
    R::Page: Page<Opaque = Opaque>,
{
    let mut guard = index.write(id, false);
    let mut freepages_guard = index.write(freepages_first, false);
    let freepages_bytes = freepages_guard.get_mut(1).expect("data corruption");
    let mut freepages_tuple = FreepagesTuple::deserialize_mut(freepages_bytes);
    guard.get_opaque_mut().next = *freepages_tuple.first();
    drop(guard); // write log of guard
    *freepages_tuple.first() = id;
    drop(freepages_guard); // write log of freepages_guard
}

/// The pages at the head of the free list that follow one another, at most `most` of them.
pub fn head<R: RelationRead>(index: &R, freepages_first: u32, most: usize) -> Vec<u32>
where
    R::Page: Page<Opaque = Opaque>,
{
    let freepages_guard = index.read(freepages_first);
    let freepages_bytes = freepages_guard.get(1).expect("data corruption");
    let mut id = FreepagesTuple::deserialize_ref(freepages_bytes).first();
    drop(freepages_guard);
    let mut run = Vec::new();
    while id != u32::MAX && run.len() < most && run.last().is_none_or(|&last| id == last + 1) {
        run.push(id);
        id = index.read(id).get_opaque().next;
    }
    run
}

/// The runs of free pages that the free pages' page keeps.
pub fn runs<R: RelationRead>(index: &R, freepages_first: u32) -> Vec<Run>
where
    R::Page: Page<Opaque = Opaque>,
{
    let guard = index.read(freepages_first);
    let tuple = guard.get(2).map(FreeRunsTuple::deserialize_ref);
    tuple.map_or_else(Vec::new, |x| x.runs().to_vec())
}

/// Takes page `id`, if a kept run begins with it.
pub fn take<R: RelationWrite>(index: &R, freepages_first: u32, id: u32) -> Option<R::WriteGuard<'_>>
where
    R::Page: Page<Opaque = Opaque>,
{
    let mut freepages_guard = index.write(freepages_first, false);
    let mut tuple = FreeRunsTuple::deserialize_mut(freepages_guard.get_mut(2)?);
    let mut runs = tuple.runs().to_vec();
    let i = runs.iter().position(|run| run.start == id)?;
    shrink(&mut runs, i, true);
    tuple.set(&runs);
    let guard = index.write(id, false);
    drop(freepages_guard); // write log of freespaces_guard
    Some(guard)
}

/// Keeps `runs` with those kept already, joined where they meet, the longest that fit; returns the others.
pub fn put_runs<R: RelationRead + RelationWrite>(
    index: &R,
    freepages_first: u32,
    runs: &[Run],
) -> Vec<Run>
where
    R::Page: Page<Opaque = Opaque>,
{
    if runs.is_empty() {
        return Vec::new();
    }
    let mut guard = index.write(freepages_first, false);
    if guard.get(2).is_none() {
        // as many runs as the page holds
        let capacity = FreeRunsTuple::capacity((guard.freespace() as usize).saturating_sub(8));
        let bytes = FreeRunsTuple::serialize(&FreeRunsTuple { capacity });
        assert_eq!(guard.alloc(&bytes), Some(2), "data corruption");
    }
    let mut tuple = FreeRunsTuple::deserialize_mut(guard.get_mut(2).expect("data corruption"));
    let mut all = [tuple.runs(), runs].concat();
    all.sort_unstable();
    let mut joined = Vec::<Run>::new();
    for run in all {
        let end = joined.last().map(|last| last.end);
        assert!(
            end.is_none_or(|end| end <= run.start),
            "data corruption: a page is freed twice"
        );
        match joined.last_mut() {
            Some(last) if last.end == run.start => last.end = run.end,
            _ => joined.push(run),
        }
    }
    joined.sort_by_key(|run| std::cmp::Reverse(run.len()));
    let kept = tuple.set(&joined);
    joined.split_off(kept)
}
