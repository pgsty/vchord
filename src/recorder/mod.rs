// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.

pub use types::{DefaultRecorder, Recorder};
pub use worker::dump;

mod hook;
mod types;
mod worker;

pub mod text;

pub fn init() {
    hook::init();
}
