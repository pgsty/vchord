// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

//! A hook for interrupt checks in long-running loops of the algorithm crates.
//!
//! These crates do not depend on PostgreSQL. The extension installs a plain
//! Rust function that performs PostgreSQL's interrupt check on the backend's
//! main thread. Callers must only call [`check`] where no buffer content lock
//! is held (PostgreSQL holds off interrupts while LWLocks are held) (#28).

use std::sync::OnceLock;

static HOOK: OnceLock<fn()> = OnceLock::new();

/// Installs the interrupt check. Called once at library load; later calls are
/// ignored.
pub fn set(hook: fn()) {
    let _ = HOOK.set(hook);
}

/// Runs the installed interrupt check, if any.
#[inline]
pub fn check() {
    if let Some(hook) = HOOK.get() {
        hook();
    }
}
