// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.

use serde::{Deserialize, Serialize};
use validator::Validate;
use vchordg::types::VchordgIndexOptions;

#[derive(Debug, Clone, Default, Serialize, Deserialize, Validate)]
#[serde(deny_unknown_fields)]
pub struct VchordgIndexingOptions {
    #[serde(flatten)]
    #[validate(nested)]
    pub index: VchordgIndexOptions,
}
