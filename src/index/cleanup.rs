// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use std::cell::Cell;

std::thread_local! {
    static ABANDONING: Cell<bool> = const { Cell::new(false) };
}

/// Whether a destructor may release its PostgreSQL resources.
///
/// After an error PostgreSQL owns resource cleanup. A healthy transaction does
/// not imply that an old scan's resources survived a subtransaction abort.
pub fn pg_cleanup_allowed() -> bool {
    !std::thread::panicking() && !ABANDONING.get() && unsafe { pgrx::pg_sys::IsTransactionState() }
}

/// Drop an abandoned scan's Rust objects without touching PostgreSQL resources.
///
/// All PostgreSQL-facing destructors reachable from the scan must honor
/// `pg_cleanup_allowed()`. Those objects must not escape the Scanner's lifetime.
/// This scope only destroys objects; it must not advance iterators or evaluate
/// user expressions. MemoryContext and ResourceOwner cleanup may already have
/// invalidated their PostgreSQL pointers.
pub fn abandon<T>(value: T) {
    struct Restore(bool);

    impl Drop for Restore {
        fn drop(&mut self) {
            ABANDONING.set(self.0);
        }
    }

    let _restore = Restore(ABANDONING.replace(true));
    drop(value);
}
