// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use crate::datatype::memory_halfvec::{HalfvecInput, HalfvecOutput};
use crate::datatype::memory_rabitq4::{Rabitq4Input, Rabitq4Output};
use crate::datatype::memory_rabitq8::{Rabitq8Input, Rabitq8Output};
use crate::datatype::memory_vector::{VectorInput, VectorOutput};
use crate::index::opclass::Sphere;
use distance::Distance;
use pgrx::datum::FromDatum;
use pgrx::heap_tuple::PgHeapTuple;
use pgrx::pg_sys::Datum;
use std::num::NonZero;
use vchordg::types::*;
use vector::{VectorBorrowed, VectorOwned};

#[derive(Debug, Clone, Copy)]
pub enum Opfamily {
    VectorL2,
    VectorCosine,
    VectorIp,
    HalfvecL2,
    HalfvecCosine,
    HalfvecIp,
    Rabitq8L2,
    Rabitq8Cosine,
    Rabitq8Ip,
    Rabitq4L2,
    Rabitq4Cosine,
    Rabitq4Ip,
}

impl Opfamily {
    pub fn is_cosine(self) -> bool {
        matches!(
            self,
            Self::VectorCosine | Self::HalfvecCosine | Self::Rabitq8Cosine | Self::Rabitq4Cosine
        )
    }
    /// Uses the same test as `function_normalize`, so every vector a cosine
    /// index stores is normalized.
    fn has_direction(vector: BorrowedVector<'_>) -> bool {
        match vector {
            BorrowedVector::Vecf32(v) => v.has_direction(),
            BorrowedVector::Vecf16(v) => v.has_direction(),
            BorrowedVector::Rabitq8(v) => v.has_direction(),
            BorrowedVector::Rabitq4(v) => v.has_direction(),
        }
    }
    pub fn valid_cosine_query(self, vector: &OwnedVector) -> bool {
        if !self.is_cosine() {
            return true;
        }
        match vector {
            OwnedVector::Vecf32(v) => Self::has_direction(BorrowedVector::Vecf32(v.as_borrowed())),
            OwnedVector::Vecf16(v) => Self::has_direction(BorrowedVector::Vecf16(v.as_borrowed())),
            OwnedVector::Rabitq8(v) => {
                Self::has_direction(BorrowedVector::Rabitq8(v.as_borrowed()))
            }
            OwnedVector::Rabitq4(v) => {
                Self::has_direction(BorrowedVector::Rabitq4(v.as_borrowed()))
            }
        }
    }
    fn input(self, vector: BorrowedVector<'_>) -> OwnedVector {
        use BorrowedVector as B;
        use OwnedVector as O;
        match (self, vector) {
            (Self::VectorL2, B::Vecf32(x)) => O::Vecf32(x.own()),
            (Self::VectorL2, _) => unreachable!(),
            (Self::VectorCosine, B::Vecf32(x)) => O::Vecf32(x.function_normalize()),
            (Self::VectorCosine, _) => unreachable!(),
            (Self::VectorIp, B::Vecf32(x)) => O::Vecf32(x.own()),
            (Self::VectorIp, _) => unreachable!(),
            (Self::HalfvecL2, B::Vecf16(x)) => O::Vecf16(x.own()),
            (Self::HalfvecL2, _) => unreachable!(),
            (Self::HalfvecCosine, B::Vecf16(x)) => O::Vecf16(x.function_normalize()),
            (Self::HalfvecCosine, _) => unreachable!(),
            (Self::HalfvecIp, B::Vecf16(x)) => O::Vecf16(x.own()),
            (Self::HalfvecIp, _) => unreachable!(),
            (Self::Rabitq8L2, B::Rabitq8(x)) => O::Rabitq8(x.own()),
            (Self::Rabitq8L2, _) => unreachable!(),
            (Self::Rabitq8Cosine, B::Rabitq8(x)) => O::Rabitq8(x.function_normalize()),
            (Self::Rabitq8Cosine, _) => unreachable!(),
            (Self::Rabitq8Ip, B::Rabitq8(x)) => O::Rabitq8(x.own()),
            (Self::Rabitq8Ip, _) => unreachable!(),
            (Self::Rabitq4L2, B::Rabitq4(x)) => O::Rabitq4(x.own()),
            (Self::Rabitq4L2, _) => unreachable!(),
            (Self::Rabitq4Cosine, B::Rabitq4(x)) => O::Rabitq4(x.function_normalize()),
            (Self::Rabitq4Cosine, _) => unreachable!(),
            (Self::Rabitq4Ip, B::Rabitq4(x)) => O::Rabitq4(x.own()),
            (Self::Rabitq4Ip, _) => unreachable!(),
        }
    }
    pub unsafe fn store(self, datum: Datum) -> Option<Vec<(OwnedVector, u16)>> {
        if datum.is_null() {
            return None;
        }
        let store = match self {
            Self::VectorL2 | Self::VectorCosine | Self::VectorIp => {
                let vector = unsafe { VectorInput::from_datum(datum, false).unwrap() };
                if matches!(self, Self::VectorCosine)
                    && !Self::has_direction(BorrowedVector::Vecf32(vector.as_borrowed()))
                {
                    return None;
                }
                vec![(self.input(BorrowedVector::Vecf32(vector.as_borrowed())), 0)]
            }
            Self::HalfvecL2 | Self::HalfvecCosine | Self::HalfvecIp => {
                let vector = unsafe { HalfvecInput::from_datum(datum, false).unwrap() };
                if matches!(self, Self::HalfvecCosine)
                    && !Self::has_direction(BorrowedVector::Vecf16(vector.as_borrowed()))
                {
                    return None;
                }
                vec![(self.input(BorrowedVector::Vecf16(vector.as_borrowed())), 0)]
            }
            Self::Rabitq8L2 | Self::Rabitq8Cosine | Self::Rabitq8Ip => {
                let vector = unsafe { Rabitq8Input::from_datum(datum, false).unwrap() };
                if matches!(self, Self::Rabitq8Cosine)
                    && !Self::has_direction(BorrowedVector::Rabitq8(vector.as_borrowed()))
                {
                    return None;
                }
                vec![(self.input(BorrowedVector::Rabitq8(vector.as_borrowed())), 0)]
            }
            Self::Rabitq4L2 | Self::Rabitq4Cosine | Self::Rabitq4Ip => {
                let vector = unsafe { Rabitq4Input::from_datum(datum, false).unwrap() };
                if matches!(self, Self::Rabitq4Cosine)
                    && !Self::has_direction(BorrowedVector::Rabitq4(vector.as_borrowed()))
                {
                    return None;
                }
                vec![(self.input(BorrowedVector::Rabitq4(vector.as_borrowed())), 0)]
            }
        };
        Some(store)
    }
    pub unsafe fn input_sphere(self, datum: Datum) -> Option<Sphere<OwnedVector>> {
        // A NULL sphere matches nothing; PostgreSQL rechecks it. A sphere
        // with a NULL center or radius is malformed, as for the operators.
        if datum.is_null() {
            return None;
        }
        let attno_1 = NonZero::new(1_usize).unwrap();
        let attno_2 = NonZero::new(2_usize).unwrap();
        let tuple = unsafe { PgHeapTuple::from_composite_datum(datum) };
        let center = match self {
            Self::VectorL2 | Self::VectorCosine | Self::VectorIp => {
                let vector = tuple
                    .get_by_index::<VectorOutput>(attno_1)
                    .unwrap()
                    .unwrap_or_else(|| pgrx::error!("Bad input: empty center at sphere"));
                self.input(BorrowedVector::Vecf32(vector.as_borrowed()))
            }
            Self::HalfvecL2 | Self::HalfvecCosine | Self::HalfvecIp => {
                let vector = tuple
                    .get_by_index::<HalfvecOutput>(attno_1)
                    .unwrap()
                    .unwrap_or_else(|| pgrx::error!("Bad input: empty center at sphere"));
                self.input(BorrowedVector::Vecf16(vector.as_borrowed()))
            }
            Self::Rabitq8L2 | Self::Rabitq8Cosine | Self::Rabitq8Ip => {
                let vector = tuple
                    .get_by_index::<Rabitq8Output>(attno_1)
                    .unwrap()
                    .unwrap_or_else(|| pgrx::error!("Bad input: empty center at sphere"));
                self.input(BorrowedVector::Rabitq8(vector.as_borrowed()))
            }
            Self::Rabitq4L2 | Self::Rabitq4Cosine | Self::Rabitq4Ip => {
                let vector = tuple
                    .get_by_index::<Rabitq4Output>(attno_1)
                    .unwrap()
                    .unwrap_or_else(|| pgrx::error!("Bad input: empty center at sphere"));
                self.input(BorrowedVector::Rabitq4(vector.as_borrowed()))
            }
        };
        let radius = tuple
            .get_by_index::<f32>(attno_2)
            .unwrap()
            .unwrap_or_else(|| pgrx::error!("Bad input: empty radius at sphere"));
        Some(Sphere { center, radius })
    }
    pub unsafe fn input_vector(self, datum: Datum) -> Option<OwnedVector> {
        if datum.is_null() {
            return None;
        }
        let vector = match self {
            Self::VectorL2 | Self::VectorCosine | Self::VectorIp => {
                let vector = unsafe { VectorInput::from_datum(datum, false).unwrap() };
                self.input(BorrowedVector::Vecf32(vector.as_borrowed()))
            }
            Self::HalfvecL2 | Self::HalfvecCosine | Self::HalfvecIp => {
                let vector = unsafe { HalfvecInput::from_datum(datum, false).unwrap() };
                self.input(BorrowedVector::Vecf16(vector.as_borrowed()))
            }
            Self::Rabitq8L2 | Self::Rabitq8Cosine | Self::Rabitq8Ip => {
                let vector = unsafe { Rabitq8Input::from_datum(datum, false).unwrap() };
                self.input(BorrowedVector::Rabitq8(vector.as_borrowed()))
            }
            Self::Rabitq4L2 | Self::Rabitq4Cosine | Self::Rabitq4Ip => {
                let vector = unsafe { Rabitq4Input::from_datum(datum, false).unwrap() };
                self.input(BorrowedVector::Rabitq4(vector.as_borrowed()))
            }
        };
        Some(vector)
    }
    pub fn output(self, x: Distance) -> f32 {
        match self {
            Self::VectorCosine
            | Self::HalfvecCosine
            | Self::Rabitq8Cosine
            | Self::Rabitq4Cosine => x.to_f32() * 0.5,
            Self::VectorL2 | Self::HalfvecL2 | Self::Rabitq8L2 | Self::Rabitq4L2 => {
                x.to_f32().clamp(0.0, f32::INFINITY).sqrt()
            }
            Self::VectorIp | Self::HalfvecIp | Self::Rabitq8Ip | Self::Rabitq4Ip => x.to_f32(),
        }
    }
    pub const fn distance_kind(self) -> DistanceKind {
        match self {
            Self::VectorL2 | Self::HalfvecL2 | Self::Rabitq8L2 | Self::Rabitq4L2 => {
                DistanceKind::L2S
            }
            Self::VectorCosine
            | Self::HalfvecCosine
            | Self::Rabitq8Cosine
            | Self::Rabitq4Cosine => DistanceKind::L2S,
            Self::VectorIp | Self::HalfvecIp | Self::Rabitq8Ip | Self::Rabitq4Ip => {
                DistanceKind::Dot
            }
        }
    }
    pub const fn vector_kind(self) -> VectorKind {
        match self {
            Self::VectorL2 | Self::VectorCosine | Self::VectorIp => VectorKind::Vecf32,
            Self::HalfvecL2 | Self::HalfvecCosine | Self::HalfvecIp => VectorKind::Vecf16,
            Self::Rabitq8L2 | Self::Rabitq8Cosine | Self::Rabitq8Ip => VectorKind::Rabitq8,
            Self::Rabitq4L2 | Self::Rabitq4Cosine | Self::Rabitq4Ip => VectorKind::Rabitq4,
        }
    }
}

pub unsafe fn opfamily(index_relation: pgrx::pg_sys::Relation) -> Opfamily {
    use pgrx::pg_sys::Oid;

    let proc = unsafe { pgrx::pg_sys::index_getprocid(index_relation, 1, 1) };

    if proc == Oid::INVALID {
        pgrx::error!("support function 1 is not found");
    }

    let mut flinfo = pgrx::pg_sys::FmgrInfo::default();

    unsafe {
        pgrx::pg_sys::fmgr_info(proc, &mut flinfo);
    }

    let fn_addr = flinfo.fn_addr.expect("null function pointer");

    let mut fcinfo = unsafe { std::mem::zeroed::<pgrx::pg_sys::FunctionCallInfoBaseData>() };
    fcinfo.flinfo = &mut flinfo;
    fcinfo.fncollation = pgrx::pg_sys::DEFAULT_COLLATION_OID;
    fcinfo.context = std::ptr::null_mut();
    fcinfo.resultinfo = std::ptr::null_mut();
    fcinfo.isnull = true;
    fcinfo.nargs = 0;

    let result_datum = unsafe {
        #[allow(ffi_unwind_calls, reason = "protected by pg_guard_ffi_boundary")]
        pgrx::pg_sys::ffi::pg_guard_ffi_boundary(|| fn_addr(&mut fcinfo))
    };

    let result_option = unsafe { String::from_datum(result_datum, fcinfo.isnull) };

    let result_string = result_option.expect("null return value");

    let result = match result_string.as_str() {
        "vchordg_vector_l2_ops" => Opfamily::VectorL2,
        "vchordg_vector_ip_ops" => Opfamily::VectorIp,
        "vchordg_vector_cosine_ops" => Opfamily::VectorCosine,
        "vchordg_halfvec_l2_ops" => Opfamily::HalfvecL2,
        "vchordg_halfvec_ip_ops" => Opfamily::HalfvecIp,
        "vchordg_halfvec_cosine_ops" => Opfamily::HalfvecCosine,
        "vchordg_rabitq8_l2_ops" => Opfamily::Rabitq8L2,
        "vchordg_rabitq8_ip_ops" => Opfamily::Rabitq8Ip,
        "vchordg_rabitq8_cosine_ops" => Opfamily::Rabitq8Cosine,
        "vchordg_rabitq4_l2_ops" => Opfamily::Rabitq4L2,
        "vchordg_rabitq4_ip_ops" => Opfamily::Rabitq4Ip,
        "vchordg_rabitq4_cosine_ops" => Opfamily::Rabitq4Cosine,
        _ => pgrx::error!("unknown operator class"),
    };

    unsafe {
        pgrx::pg_sys::pfree(result_datum.cast_mut_ptr());
    }

    result
}
