// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

#[derive(
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    IntoBytes,
    FromBytes,
    Immutable,
    KnownLayout,
)]
#[repr(transparent)]
pub struct Distance(i32);

impl std::fmt::Debug for Distance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

impl Distance {
    pub const ZERO: Self = Distance::from_f32(0.0f32);
    pub const INFINITY: Self = Distance::from_f32(f32::INFINITY);
    pub const NEG_INFINITY: Self = Distance::from_f32(f32::NEG_INFINITY);
    pub const NAN: Self = Distance::from_f32(f32::NAN);

    /// Every NaN maps to the same value, which orders after `+inf`.
    ///
    /// The sign and payload of a NaN produced by arithmetic differ between CPUs
    /// (`0.0 / 0.0` is `+NaN` on aarch64 and `-NaN` on x86_64), and a negative
    /// NaN would otherwise order before `-inf`. Ordering every NaN last matches
    /// PostgreSQL, where NaN sorts after all other values.
    #[inline(always)]
    pub const fn from_f32(value: f32) -> Self {
        let value = if value.is_nan() { f32::NAN } else { value };
        let bits = value.to_bits() as i32;
        let mask = ((bits >> 31) as u32) >> 1;
        let res = bits ^ (mask as i32);
        Self(res)
    }

    #[inline(always)]
    pub const fn to_f32(self) -> f32 {
        let bits = self.0;
        let mask = ((bits >> 31) as u32) >> 1;
        let res = bits ^ (mask as i32);
        f32::from_bits(res as u32)
    }

    #[inline(always)]
    pub const fn to_i32(self) -> i32 {
        self.0
    }
}

impl From<f32> for Distance {
    #[inline(always)]
    fn from(value: f32) -> Self {
        Distance::from_f32(value)
    }
}

impl From<Distance> for f32 {
    #[inline(always)]
    fn from(value: Distance) -> Self {
        Distance::to_f32(value)
    }
}

#[test]
fn distance_conversions() {
    assert_eq!(Distance::from(0.0f32), Distance::ZERO);
    assert_eq!(Distance::from(f32::INFINITY), Distance::INFINITY);
    assert_eq!(Distance::from(f32::NEG_INFINITY), Distance::NEG_INFINITY);
    for i in -100..100 {
        let val = (i as f32) * 0.1;
        assert_eq!(f32::from(Distance::from(val)).to_bits(), val.to_bits());
    }
    assert_eq!(
        f32::from(Distance::from(0.0f32)).to_bits(),
        0.0f32.to_bits()
    );
    assert_eq!(
        f32::from(Distance::from(-0.0f32)).to_bits(),
        (-0.0f32).to_bits()
    );
    assert_eq!(
        f32::from(Distance::from(f32::NAN)).to_bits(),
        f32::NAN.to_bits()
    );
    assert_eq!(
        f32::from(Distance::from(-f32::NAN)).to_bits(),
        f32::NAN.to_bits()
    );
    assert_eq!(
        f32::from(Distance::from(f32::INFINITY)).to_bits(),
        f32::INFINITY.to_bits()
    );
    assert_eq!(
        f32::from(Distance::from(-f32::INFINITY)).to_bits(),
        (-f32::INFINITY).to_bits()
    );
}

#[test]
fn distance_nan_ordering() {
    let nans = [
        f32::NAN,
        -f32::NAN,
        f32::from_bits(0x7f80_0001),
        f32::from_bits(0xff80_0001),
        f32::from_bits(0x7fff_ffff),
        f32::from_bits(0xffff_ffff),
        f32::from_bits(0xffc0_0000),
        std::hint::black_box(0.0f32) / std::hint::black_box(0.0f32),
        std::hint::black_box(f32::INFINITY) - std::hint::black_box(f32::INFINITY),
        (std::hint::black_box(-1.0f32)).sqrt(),
    ];
    for nan in nans {
        assert!(nan.is_nan());
        let d = Distance::from_f32(nan);
        // Canonical, so independent of the sign and payload the CPU produced.
        assert_eq!(d, Distance::NAN);
        assert!(d.to_f32().is_nan());
        assert!(d.to_f32().is_sign_positive());
        // NaN orders after every number, including `+inf`, like in PostgreSQL.
        assert!(d > Distance::INFINITY);
        assert!(d > Distance::from_f32(f32::MAX));
        assert!(d > Distance::NEG_INFINITY);
        assert!(d > Distance::ZERO);
    }
    let mut v = [
        Distance::from_f32(-f32::NAN),
        Distance::from_f32(1.0),
        Distance::from_f32(f32::NAN),
        Distance::INFINITY,
        Distance::from_f32(-1.0),
        Distance::NEG_INFINITY,
        Distance::from_f32(-0.0),
        Distance::ZERO,
    ];
    v.sort();
    let sorted = v.map(Distance::to_f32);
    assert_eq!(sorted[0], f32::NEG_INFINITY);
    assert_eq!(sorted[1], -1.0);
    assert_eq!(sorted[2].to_bits(), (-0.0f32).to_bits());
    assert_eq!(sorted[3].to_bits(), 0.0f32.to_bits());
    assert_eq!(sorted[4], 1.0);
    assert_eq!(sorted[5], f32::INFINITY);
    assert!(sorted[6].is_nan() && sorted[7].is_nan());
    // Ordering of non-NaN values is unchanged.
    for i in -1000..1000 {
        let (a, b) = ((i as f32) * 0.37, ((i + 1) as f32) * 0.37);
        assert!(Distance::from_f32(a) < Distance::from_f32(b));
    }
}
