// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

/// Calls `$search` with `$args` and the prefetchers that `$io` asks for.
macro_rules! prefetched {
    ($index:expr, $io:expr, $search:path, ($($args:expr),* $(,)?)) => {{
        use $crate::index::scanners::{Io, SearchIo};
        use $crate::index::vchordrq::dispatch::*;
        let index = $index;
        let vectors = MakeH1PlainPrefetcher { index };
        let hints = ::index::relation::Hints::default().full(true);
        match $io {
            SearchIo::Io(Io::Plain) => $search($($args,)* vectors, MakeH0PlainPrefetcher { index }),
            SearchIo::Io(Io::Simple) => $search($($args,)* vectors, MakeH0SimplePrefetcher { index }),
            SearchIo::Io(Io::Stream) => {
                $search($($args,)* vectors, MakeH0StreamPrefetcher { index, hints })
            }
            SearchIo::Mixed => $search($($args,)* vectors, MakeH0MixedPrefetcher { index, hints }),
        }
    }};
}

mod default;
mod maxsim;

use crate::datatype::memory_halfvec::HalfvecInput;
use crate::datatype::memory_rabitq4::Rabitq4Input;
use crate::datatype::memory_rabitq8::Rabitq8Input;
use crate::datatype::memory_vector::VectorInput;
use crate::index::scanners::{Io, SearchIo};
use pgrx::datum::FromDatum;
use pgrx::pg_sys::Datum;
use simd::f16;
use vchordrq::types::OwnedVector;
use vector::VectorOwned;
use vector::rabitq4::Rabitq4Owned;
use vector::rabitq8::Rabitq8Owned;
use vector::vect::VectOwned;

pub use default::DefaultBuilder;
pub use maxsim::MaxsimBuilder;

#[derive(Debug)]
pub struct SearchOptions {
    pub search: vchordrq::SearchParams,
    pub max_scan_tuples: Option<u32>,
    pub maxsim_refine: u32,
    pub maxsim_threshold: u32,
    pub io_search: SearchIo,
    pub io_rerank: Io,
    pub prefilter: bool,
    pub table: crate::index::fetcher::TablePrefetch,
    /// The scan node above owns the sphere around the vector.
    pub owned: bool,
    /// How many rows the scan node above will take, and then what the scan used.
    pub k: std::rc::Rc<std::cell::Cell<vchordrq::KHint>>,
}

/// An opfamily's own vector type, taken out of an `OwnedVector` or read in place from a datum.
pub trait FromOwned: VectorOwned {
    fn from_owned(vector: OwnedVector) -> Self;
    /// Calls `f` with the vector in `datum`, which must not be null; only a toasted value is copied.
    unsafe fn in_datum<R>(datum: Datum, f: impl FnOnce(Self::Borrowed<'_>) -> R) -> R;
}

impl FromOwned for VectOwned<f32> {
    fn from_owned(vector: OwnedVector) -> Self {
        let OwnedVector::Vecf32(vector) = vector else {
            unreachable!()
        };
        vector
    }

    unsafe fn in_datum<R>(datum: Datum, f: impl FnOnce(Self::Borrowed<'_>) -> R) -> R {
        f(unsafe { VectorInput::from_datum(datum, false) }
            .unwrap()
            .as_borrowed())
    }
}

impl FromOwned for VectOwned<f16> {
    fn from_owned(vector: OwnedVector) -> Self {
        let OwnedVector::Vecf16(vector) = vector else {
            unreachable!()
        };
        vector
    }

    unsafe fn in_datum<R>(datum: Datum, f: impl FnOnce(Self::Borrowed<'_>) -> R) -> R {
        f(unsafe { HalfvecInput::from_datum(datum, false) }
            .unwrap()
            .as_borrowed())
    }
}

impl FromOwned for Rabitq8Owned {
    fn from_owned(vector: OwnedVector) -> Self {
        let OwnedVector::Rabitq8(vector) = vector else {
            unreachable!()
        };
        vector
    }

    unsafe fn in_datum<R>(datum: Datum, f: impl FnOnce(Self::Borrowed<'_>) -> R) -> R {
        f(unsafe { Rabitq8Input::from_datum(datum, false) }
            .unwrap()
            .as_borrowed())
    }
}

impl FromOwned for Rabitq4Owned {
    fn from_owned(vector: OwnedVector) -> Self {
        let OwnedVector::Rabitq4(vector) = vector else {
            unreachable!()
        };
        vector
    }

    unsafe fn in_datum<R>(datum: Datum, f: impl FnOnce(Self::Borrowed<'_>) -> R) -> R {
        f(unsafe { Rabitq4Input::from_datum(datum, false) }
            .unwrap()
            .as_borrowed())
    }
}
