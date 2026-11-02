// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.

use std::env::var;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo::rerun-if-changed=cshim");
    let target_arch = var("CARGO_CFG_TARGET_ARCH")?;
    let target_endian = var("CARGO_CFG_TARGET_ENDIAN")?;
    #[allow(clippy::single_match)]
    match target_arch.as_str() {
        "aarch64" => {
            let mut build = cc::Build::new();
            build.file("./cshim/aarch64_byte.c");
            build.file("./cshim/aarch64_halfbyte.c");
            if target_endian == "little" {
                build.file("./cshim/aarch64_sve_fp16.c");
                build.file("./cshim/aarch64_sve_fp32.c");
            }
            build.opt_level(3);
            build.compile("simd_cshim");
        }
        _ => {}
    }
    Ok(())
}
