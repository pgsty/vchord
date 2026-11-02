// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordg_support_vector_l2_ops() -> String {
    "vchordg_vector_l2_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordg_support_vector_cosine_ops() -> String {
    "vchordg_vector_cosine_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordg_support_vector_ip_ops() -> String {
    "vchordg_vector_ip_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordg_support_halfvec_l2_ops() -> String {
    "vchordg_halfvec_l2_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordg_support_halfvec_cosine_ops() -> String {
    "vchordg_halfvec_cosine_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordg_support_halfvec_ip_ops() -> String {
    "vchordg_halfvec_ip_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordg_support_rabitq8_l2_ops() -> String {
    "vchordg_rabitq8_l2_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordg_support_rabitq8_cosine_ops() -> String {
    "vchordg_rabitq8_cosine_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordg_support_rabitq8_ip_ops() -> String {
    "vchordg_rabitq8_ip_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordg_support_rabitq4_l2_ops() -> String {
    "vchordg_rabitq4_l2_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordg_support_rabitq4_cosine_ops() -> String {
    "vchordg_rabitq4_cosine_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordg_support_rabitq4_ip_ops() -> String {
    "vchordg_rabitq4_ip_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_support_vector_l2_ops() -> String {
    "vchordrq_vector_l2_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_support_vector_ip_ops() -> String {
    "vchordrq_vector_ip_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_support_vector_cosine_ops() -> String {
    "vchordrq_vector_cosine_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_support_halfvec_l2_ops() -> String {
    "vchordrq_halfvec_l2_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_support_halfvec_ip_ops() -> String {
    "vchordrq_halfvec_ip_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_support_halfvec_cosine_ops() -> String {
    "vchordrq_halfvec_cosine_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_support_rabitq8_l2_ops() -> String {
    "vchordrq_rabitq8_l2_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_support_rabitq8_ip_ops() -> String {
    "vchordrq_rabitq8_ip_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_support_rabitq8_cosine_ops() -> String {
    "vchordrq_rabitq8_cosine_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_support_rabitq4_l2_ops() -> String {
    "vchordrq_rabitq4_l2_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_support_rabitq4_ip_ops() -> String {
    "vchordrq_rabitq4_ip_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_support_rabitq4_cosine_ops() -> String {
    "vchordrq_rabitq4_cosine_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_support_vector_maxsim_ops() -> String {
    "vchordrq_vector_maxsim_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_support_halfvec_maxsim_ops() -> String {
    "vchordrq_halfvec_maxsim_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_support_rabitq8_maxsim_ops() -> String {
    "vchordrq_rabitq8_maxsim_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_support_rabitq4_maxsim_ops() -> String {
    "vchordrq_rabitq4_maxsim_ops".to_string()
}

pub struct Sphere<T> {
    pub center: T,
    pub radius: f32,
}
