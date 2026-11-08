// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use crate::datatype::memory_rabitq8::{Rabitq8Input, Rabitq8Output};
use pgrx::datum::Array;
use std::num::NonZero;
use vector::VectorBorrowed;
use vector::rabitq8::Rabitq8Borrowed;

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchord_rabitq8_operator_l2(lhs: Rabitq8Input<'_>, rhs: Rabitq8Input<'_>) -> f32 {
    let lhs = lhs.as_borrowed();
    let rhs = rhs.as_borrowed();
    if lhs.dim() != rhs.dim() {
        pgrx::error!("dimension is not matched");
    }
    Rabitq8Borrowed::operator_l2s(lhs, rhs)
        .to_f32()
        .clamp(0.0, f32::INFINITY)
        .sqrt()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchord_rabitq8_operator_ip(lhs: Rabitq8Input<'_>, rhs: Rabitq8Input<'_>) -> f32 {
    let lhs = lhs.as_borrowed();
    let rhs = rhs.as_borrowed();
    if lhs.dim() != rhs.dim() {
        pgrx::error!("dimension is not matched");
    }
    Rabitq8Borrowed::operator_dot(lhs, rhs).to_f32()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchord_rabitq8_operator_cosine(lhs: Rabitq8Input<'_>, rhs: Rabitq8Input<'_>) -> f32 {
    let lhs = lhs.as_borrowed();
    let rhs = rhs.as_borrowed();
    if lhs.dim() != rhs.dim() {
        pgrx::error!("dimension is not matched");
    }
    Rabitq8Borrowed::operator_cos(lhs, rhs).to_f32()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchord_rabitq8_sphere_l2_in(
    lhs: Rabitq8Input<'_>,
    rhs: pgrx::composite_type!("sphere_rabitq8"),
) -> bool {
    let center: Rabitq8Output = match rhs.get_by_index(NonZero::new(1).unwrap()) {
        Ok(Some(s)) => s,
        Ok(None) => pgrx::error!("Bad input: empty center at sphere"),
        Err(_) => unreachable!(),
    };
    let radius: f32 = match rhs.get_by_index(NonZero::new(2).unwrap()) {
        Ok(Some(s)) => s,
        Ok(None) => pgrx::error!("Bad input: empty radius at sphere"),
        Err(_) => unreachable!(),
    };
    let lhs = lhs.as_borrowed();
    let center = center.as_borrowed();
    if lhs.dim() != center.dim() {
        pgrx::error!("dimension is not matched");
    }
    let d = Rabitq8Borrowed::operator_l2s(lhs, center)
        .to_f32()
        .clamp(0.0, f32::INFINITY)
        .sqrt();
    d < radius
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchord_rabitq8_sphere_ip_in(
    lhs: Rabitq8Input<'_>,
    rhs: pgrx::composite_type!("sphere_rabitq8"),
) -> bool {
    let center: Rabitq8Output = match rhs.get_by_index(NonZero::new(1).unwrap()) {
        Ok(Some(s)) => s,
        Ok(None) => pgrx::error!("Bad input: empty center at sphere"),
        Err(_) => unreachable!(),
    };
    let radius: f32 = match rhs.get_by_index(NonZero::new(2).unwrap()) {
        Ok(Some(s)) => s,
        Ok(None) => pgrx::error!("Bad input: empty radius at sphere"),
        Err(_) => unreachable!(),
    };
    let lhs = lhs.as_borrowed();
    let center = center.as_borrowed();
    if lhs.dim() != center.dim() {
        pgrx::error!("dimension is not matched");
    }
    let d = Rabitq8Borrowed::operator_dot(lhs, center).to_f32();
    d < radius
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchord_rabitq8_sphere_cosine_in(
    lhs: Rabitq8Input<'_>,
    rhs: pgrx::composite_type!("sphere_rabitq8"),
) -> bool {
    let center: Rabitq8Output = match rhs.get_by_index(NonZero::new(1).unwrap()) {
        Ok(Some(s)) => s,
        Ok(None) => pgrx::error!("Bad input: empty center at sphere"),
        Err(_) => unreachable!(),
    };
    let radius: f32 = match rhs.get_by_index(NonZero::new(2).unwrap()) {
        Ok(Some(s)) => s,
        Ok(None) => pgrx::error!("Bad input: empty radius at sphere"),
        Err(_) => unreachable!(),
    };
    let lhs = lhs.as_borrowed();
    let center = center.as_borrowed();
    if lhs.dim() != center.dim() {
        pgrx::error!("dimension is not matched");
    }
    let d = Rabitq8Borrowed::operator_cos(lhs, center).to_f32();
    d < radius
}

#[pgrx::pg_extern(sql = "")]
fn _vchord_rabitq8_operator_maxsim(
    lhs: Array<'_, Rabitq8Input<'_>>,
    rhs: Array<'_, Rabitq8Input<'_>>,
) -> f32 {
    let mut maxsim = 0.0f32;
    for rhs in rhs.iter().flatten() {
        let mut d = f32::INFINITY;
        for lhs in lhs.iter().flatten() {
            let lhs = lhs.as_borrowed();
            let rhs = rhs.as_borrowed();
            // The kernels only assert equal storage lengths, which rabitq4
            // shares between odd and even dimensions (#31).
            if lhs.dim() != rhs.dim() {
                pgrx::error!("dimension is not matched");
            }
            d = d.min(Rabitq8Borrowed::operator_dot(lhs, rhs).to_f32());
        }
        maxsim += d;
    }
    maxsim
}
