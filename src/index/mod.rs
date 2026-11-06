// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

mod cleanup;
mod fetcher;
mod functions;
mod gucs;
mod hook;
mod opclass;
mod sample;
mod scanners;
mod storage;
mod traverse;
mod vchordg;
mod vchordrq;

pub fn init() {
    gucs::init();
    hook::init();
    vchordrq::am::init();
    vchordg::am::init();
}

/// Cost given to index paths the access methods cannot execute (no ORDER BY
/// vector and no index clause, or index-only scans). Finite on purpose:
/// `inf - inf` in planner arithmetic yields NaN. This is a mitigation, not a
/// structural exclusion (#32).
pub const UNUSABLE_PATH_COST: f64 = 1.0e30;
