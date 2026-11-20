// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

//! The rows of a block whose value is in a set or a range; values order as little-endian integers.

pub trait Value: Copy + Eq + Ord + Default + std::fmt::Debug + 'static {
    /// The rows whose value is one of `keys`: bit `i` is row `i`.
    fn matches(column: &[Self; 32], keys: &[Self]) -> u32;
    /// The rows whose value lies in `lo..=hi`.
    fn between(column: &[Self; 32], lo: Self, hi: Self) -> u32;
    /// The unsigned integer that a stored value encodes.
    fn ordinal(self) -> u128;
    /// The exclusive or of the value's 16-bit lanes.
    fn fold_one(self) -> u16;
    fn wide(self) -> u64;
    fn from_u64(x: u64) -> Self;
    #[inline(always)]
    fn fold(column: &[Self; 32]) -> [u16; 32] {
        let mut folds = [0; 32];
        for i in 0..32 {
            folds[i] = column[i].fold_one();
        }
        folds
    }
}

impl Value for u16 {
    #[inline(always)]
    fn matches(column: &[Self; 32], keys: &[Self]) -> u32 {
        u16s::matches(column, keys)
    }
    #[inline(always)]
    fn between(column: &[Self; 32], lo: Self, hi: Self) -> u32 {
        u16s::between::between(column, lo, hi)
    }
    #[inline(always)]
    fn ordinal(self) -> u128 {
        u16::from_le(self) as u128
    }
    #[inline(always)]
    fn fold_one(self) -> u16 {
        self
    }
    #[inline(always)]
    fn wide(self) -> u64 {
        self as u64
    }
    #[inline(always)]
    fn from_u64(x: u64) -> Self {
        x as u16
    }
}

impl Value for u32 {
    #[inline(always)]
    fn matches(column: &[Self; 32], keys: &[Self]) -> u32 {
        u32s::matches(column, keys)
    }
    #[inline(always)]
    fn between(column: &[Self; 32], lo: Self, hi: Self) -> u32 {
        u32s::between::between(column, lo, hi)
    }
    #[inline(always)]
    fn ordinal(self) -> u128 {
        u32::from_le(self) as u128
    }
    #[inline(always)]
    fn fold_one(self) -> u16 {
        (self ^ self >> 16) as u16
    }
    #[inline(always)]
    fn wide(self) -> u64 {
        self as u64
    }
    #[inline(always)]
    fn from_u64(x: u64) -> Self {
        x as u32
    }
}

impl Value for u64 {
    #[inline(always)]
    fn matches(column: &[Self; 32], keys: &[Self]) -> u32 {
        u64s::matches(column, keys)
    }
    #[inline(always)]
    fn between(column: &[Self; 32], lo: Self, hi: Self) -> u32 {
        u64s::between::between(column, lo, hi)
    }
    #[inline(always)]
    fn ordinal(self) -> u128 {
        u64::from_le(self) as u128
    }
    #[inline(always)]
    fn fold_one(self) -> u16 {
        fold_u64(self)
    }
    #[inline(always)]
    fn wide(self) -> u64 {
        self
    }
    #[inline(always)]
    fn from_u64(x: u64) -> Self {
        x
    }
}

impl Value for [u64; 2] {
    #[inline(always)]
    fn matches(column: &[Self; 32], keys: &[Self]) -> u32 {
        u128s::matches(column, keys)
    }
    #[inline(always)]
    fn between(column: &[Self; 32], lo: Self, hi: Self) -> u32 {
        u128s::between::between(column, lo, hi)
    }
    #[inline(always)]
    fn ordinal(self) -> u128 {
        (u64::from_le(self[1]) as u128) << 64 | u64::from_le(self[0]) as u128
    }
    #[inline(always)]
    fn fold_one(self) -> u16 {
        fold_u64(self[0] ^ self[1])
    }
    #[inline(always)]
    fn wide(self) -> u64 {
        self[0] ^ self[1].rotate_left(32)
    }
    #[inline(always)]
    fn from_u64(x: u64) -> Self {
        [x, 0]
    }
}

#[inline(always)]
const fn fold_u64(x: u64) -> u16 {
    let x = x ^ x >> 32;
    (x ^ x >> 16) as u16
}

/// A bitmap of the values' folds, then a table of the values (none for 16-bit values).
#[derive(Debug)]
pub struct Set<T> {
    /// Inline: a set is kept behind a pointer, and this saves a load.
    bitmap: [u64; 1024],
    table: Box<[T]>,
    bits: u32,
    /// A value that is not in the set, which marks the free slots.
    free: T,
}

impl<T: Value> Set<T> {
    /// `values`: sorted and deduplicated.
    pub fn new(values: &[T]) -> Self {
        let mut bitmap = [0_u64; 1024];
        for value in values {
            let x = value.fold_one();
            bitmap[x as usize / 64] |= 1 << (x % 64);
        }
        if size_of::<T>() == 2 {
            let table = Box::new([]);
            return Self {
                bitmap,
                table,
                bits: 0,
                free: T::default(),
            };
        }
        let free = (0..)
            .map(T::from_u64)
            .find(|x| values.binary_search(x).is_err())
            .expect("a set of every value");
        let bits = (2 * values.len())
            .next_power_of_two()
            .trailing_zeros()
            .max(4);
        let mut table = vec![free; 1 << bits].into_boxed_slice();
        let mask = (1 << bits) - 1;
        for &value in values {
            let mut i = Self::slot(value, bits);
            while table[i] != free {
                i = (i + 1) & mask;
            }
            table[i] = value;
        }
        Self {
            bitmap,
            table,
            bits,
            free,
        }
    }

    #[inline(always)]
    pub fn contains(&self, x: T) -> bool {
        let f = x.fold_one();
        if self.bitmap[f as usize / 64] & 1 << (f % 64) == 0 {
            return false;
        }
        self.table.is_empty() || self.lookup(x)
    }

    #[inline(always)]
    fn slot(x: T, bits: u32) -> usize {
        (x.wide().wrapping_mul(0x9e37_79b9_7f4a_7c15) >> (64 - bits)) as usize
    }

    #[inline(always)]
    fn lookup(&self, x: T) -> bool {
        let mask = (1 << self.bits) - 1;
        let mut i = Self::slot(x, self.bits);
        loop {
            // a free slot first: `x` may be the value that marks them
            let y = self.table[i];
            if y == self.free {
                return false;
            }
            if y == x {
                return true;
            }
            i = (i + 1) & mask;
        }
    }

    #[inline(always)]
    pub fn matches(&self, column: &[T; 32]) -> u32 {
        let mut mask = matches_bitmap(&T::fold(column), &self.bitmap);
        if self.table.is_empty() {
            return mask;
        }
        let mut rest = mask;
        while rest != 0 {
            let i = rest.trailing_zeros();
            rest &= rest - 1;
            if !self.lookup(column[i as usize]) {
                mask &= !(1 << i);
            }
        }
        mask
    }
}

mod matches_bitmap {
    #[crate::multiversion(
        "v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn matches_bitmap(keys: &[u16; 32], bitmap: &[u64; 1024]) -> u32 {
        let mut mask = 0;
        for i in 0..32 {
            let x = keys[i] as usize;
            mask |= ((bitmap[x / 64] >> (x % 64)) as u32 & 1) << i;
        }
        mask
    }
}

#[inline(always)]
pub fn matches_bitmap(keys: &[u16; 32], bitmap: &[u64; 1024]) -> u32 {
    matches_bitmap::matches_bitmap(keys, bitmap)
}

#[inline(always)]
fn matches_portable<T: Value>(column: &[T; 32], keys: &[T]) -> u32 {
    let mut hit = [false; 32];
    for &key in keys {
        for i in 0..32 {
            hit[i] |= column[i] == key;
        }
    }
    let mut mask = 0;
    for i in 0..32 {
        mask |= (hit[i] as u32) << i;
    }
    mask
}

#[inline(always)]
fn between_portable<T: Value>(column: &[T; 32], lo: T, hi: T) -> u32 {
    let (lo, hi) = (lo.ordinal(), hi.ordinal());
    let mut mask = 0;
    for i in 0..32 {
        let x = column[i].ordinal();
        mask |= ((lo <= x && x <= hi) as u32) << i;
    }
    mask
}

/// Checks a range kernel against [`between_portable`].
#[cfg(test)]
fn check_between<T: Value>(random: impl Fn() -> T, kernel: impl Fn(&[T; 32], T, T) -> u32) {
    for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
        let (column, keys) = sample(&random)();
        let pick = || match rand::random::<u8>() % 3 {
            0 => column[rand::random::<u32>() as usize % 32],
            1 => keys[rand::random::<u32>() as usize % keys.len()],
            _ => random(),
        };
        for _ in 0..8 {
            let (a, b) = (pick(), pick());
            for (lo, hi) in [(a, b), (b, a), (a, a)] {
                assert_eq!(kernel(&column, lo, hi), between_portable(&column, lo, hi));
            }
        }
    }
}

#[cfg(test)]
fn sample<T: Value>(random: impl Fn() -> T) -> impl Fn() -> ([T; 32], Vec<T>) {
    move || {
        let keys = (0..20).map(|_| random()).collect::<Vec<_>>();
        let column = std::array::from_fn(|_| {
            if rand::random::<u8>() < 64 {
                keys[rand::random::<u32>() as usize % keys.len()]
            } else {
                random()
            }
        });
        (column, keys)
    }
}

#[cfg(test)]
fn check<C, K>(
    sample: impl Fn() -> (C, Vec<K>),
    kernel: impl Fn(&C, &[K]) -> u32,
    fallback: impl Fn(&C, &[K]) -> u32,
) {
    for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
        let (column, keys) = sample();
        for n in 0..=keys.len() {
            assert_eq!(kernel(&column, &keys[..n]), fallback(&column, &keys[..n]));
        }
    }
}

#[cfg_attr(feature = "internal", simd_macros::public)]
mod u16s {
    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    fn matches_v4(column: &[u16; 32], keys: &[u16]) -> u32 {
        use core::arch::x86_64::*;
        let x = unsafe { _mm512_loadu_si512(column.as_ptr().cast()) };
        let mut mask = 0;
        for &key in keys {
            mask |= _mm512_cmpeq_epi16_mask(x, _mm512_set1_epi16(key as i16));
        }
        mask
    }

    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v3")]
    fn matches_v3(column: &[u16; 32], keys: &[u16]) -> u32 {
        use core::arch::x86_64::*;
        let p = column.as_ptr();
        let x0 = unsafe { _mm256_loadu_si256(p.cast()) };
        let x1 = unsafe { _mm256_loadu_si256(p.add(16).cast()) };
        let (mut y0, mut y1) = (_mm256_setzero_si256(), _mm256_setzero_si256());
        for &key in keys {
            let k = _mm256_set1_epi16(key as i16);
            y0 = _mm256_or_si256(y0, _mm256_cmpeq_epi16(x0, k));
            y1 = _mm256_or_si256(y1, _mm256_cmpeq_epi16(x1, k));
        }
        // packs interleaves the halves by lane, which the permute undoes
        let y = _mm256_packs_epi16(y0, y1);
        _mm256_movemask_epi8(_mm256_permute4x64_epi64(y, 0b11_01_10_00)) as u32
    }

    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "aarch64")]
    #[crate::target_cpu(enable = "a2")]
    fn matches_a2(column: &[u16; 32], keys: &[u16]) -> u32 {
        use core::arch::aarch64::*;
        let p = column.as_ptr();
        let x: [uint16x8_t; 4] = std::array::from_fn(|i| unsafe { vld1q_u16(p.add(8 * i)) });
        let mut y = [vdupq_n_u16(0); 4];
        for &key in keys {
            let k = vdupq_n_u16(key);
            for i in 0..4 {
                y[i] = vorrq_u16(y[i], vceqq_u16(x[i], k));
            }
        }
        let bits = unsafe { vld1q_u16([1_u16, 2, 4, 8, 16, 32, 64, 128].as_ptr()) };
        let mut mask = 0;
        for i in 0..4 {
            mask |= (vaddvq_u16(vandq_u16(y[i], bits)) as u32) << (8 * i);
        }
        mask
    }

    #[cfg(test)]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn kernels() {
        use super::{check, matches_portable, sample};
        let random = rand::random::<u16>;
        #[cfg(target_arch = "x86_64")]
        if crate::is_cpu_detected!("v4") {
            check(
                sample(random),
                |c, k| unsafe { matches_v4(c, k) },
                matches_portable,
            );
        }
        #[cfg(target_arch = "x86_64")]
        if crate::is_cpu_detected!("v3") {
            check(
                sample(random),
                |c, k| unsafe { matches_v3(c, k) },
                matches_portable,
            );
        }
        #[cfg(target_arch = "aarch64")]
        if crate::is_cpu_detected!("a2") {
            check(
                sample(random),
                |c, k| unsafe { matches_a2(c, k) },
                matches_portable,
            );
        }
        check(sample(random), matches, matches_portable);
    }

    #[crate::multiversion(@"v4", @"v3", "v2", @"a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1")]
    pub fn matches(column: &[u16; 32], keys: &[u16]) -> u32 {
        super::matches_portable(column, keys)
    }

    #[cfg_attr(feature = "internal", simd_macros::public)]
    pub mod between {
        #[cfg_attr(feature = "internal", simd_macros::public)]
        #[inline]
        #[cfg(target_arch = "x86_64")]
        #[crate::target_cpu(enable = "v4")]
        fn between_v4(column: &[u16; 32], lo: u16, hi: u16) -> u32 {
            use core::arch::x86_64::*;
            let x = unsafe { _mm512_loadu_si512(column.as_ptr().cast()) };
            let ge = _mm512_cmpge_epu16_mask(x, _mm512_set1_epi16(lo as i16));
            _mm512_mask_cmple_epu16_mask(ge, x, _mm512_set1_epi16(hi as i16))
        }

        #[cfg_attr(feature = "internal", simd_macros::public)]
        #[inline]
        #[cfg(target_arch = "x86_64")]
        #[crate::target_cpu(enable = "v3")]
        fn between_v3(column: &[u16; 32], lo: u16, hi: u16) -> u32 {
            use core::arch::x86_64::*;
            let p = column.as_ptr();
            let x0 = unsafe { _mm256_loadu_si256(p.cast()) };
            let x1 = unsafe { _mm256_loadu_si256(p.add(16).cast()) };
            let (l, h) = (_mm256_set1_epi16(lo as i16), _mm256_set1_epi16(hi as i16));
            // `x` is in range where raising it to `lo` and lowering it to `hi` both leave it
            let y0 = _mm256_and_si256(
                _mm256_cmpeq_epi16(_mm256_max_epu16(x0, l), x0),
                _mm256_cmpeq_epi16(_mm256_min_epu16(x0, h), x0),
            );
            let y1 = _mm256_and_si256(
                _mm256_cmpeq_epi16(_mm256_max_epu16(x1, l), x1),
                _mm256_cmpeq_epi16(_mm256_min_epu16(x1, h), x1),
            );
            // packs interleaves the halves by lane, which the permute undoes
            let y = _mm256_packs_epi16(y0, y1);
            _mm256_movemask_epi8(_mm256_permute4x64_epi64(y, 0b11_01_10_00)) as u32
        }

        #[cfg_attr(feature = "internal", simd_macros::public)]
        #[inline]
        #[cfg(target_arch = "aarch64")]
        #[crate::target_cpu(enable = "a2")]
        fn between_a2(column: &[u16; 32], lo: u16, hi: u16) -> u32 {
            use core::arch::aarch64::*;
            let p = column.as_ptr();
            let x: [uint16x8_t; 4] = std::array::from_fn(|i| unsafe { vld1q_u16(p.add(8 * i)) });
            let (l, h) = (vdupq_n_u16(lo), vdupq_n_u16(hi));
            let bits = unsafe { vld1q_u16([1_u16, 2, 4, 8, 16, 32, 64, 128].as_ptr()) };
            let mut mask = 0;
            for i in 0..4 {
                let y = vandq_u16(vcgeq_u16(x[i], l), vcleq_u16(x[i], h));
                mask |= (vaddvq_u16(vandq_u16(y, bits)) as u32) << (8 * i);
            }
            mask
        }

        #[cfg(test)]
        #[test]
        #[cfg_attr(miri, ignore)]
        fn kernels() {
            use super::super::check_between;
            let random = rand::random::<u16>;
            #[cfg(target_arch = "x86_64")]
            if crate::is_cpu_detected!("v4") {
                check_between(random, |c, l, h| unsafe { between_v4(c, l, h) });
            }
            #[cfg(target_arch = "x86_64")]
            if crate::is_cpu_detected!("v3") {
                check_between(random, |c, l, h| unsafe { between_v3(c, l, h) });
            }
            #[cfg(target_arch = "aarch64")]
            if crate::is_cpu_detected!("a2") {
                check_between(random, |c, l, h| unsafe { between_a2(c, l, h) });
            }
            check_between(random, between);
        }

        #[crate::multiversion(@"v4", @"v3", "v2", @"a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1")]
        pub fn between(column: &[u16; 32], lo: u16, hi: u16) -> u32 {
            super::super::between_portable(column, lo, hi)
        }
    }
}

#[cfg_attr(feature = "internal", simd_macros::public)]
mod u32s {
    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    fn matches_v4(column: &[u32; 32], keys: &[u32]) -> u32 {
        use core::arch::x86_64::*;
        let p = column.as_ptr();
        let x0 = unsafe { _mm512_loadu_si512(p.cast()) };
        let x1 = unsafe { _mm512_loadu_si512(p.add(16).cast()) };
        let (mut m0, mut m1) = (0, 0);
        for &key in keys {
            let k = _mm512_set1_epi32(key as i32);
            m0 |= _mm512_cmpeq_epi32_mask(x0, k);
            m1 |= _mm512_cmpeq_epi32_mask(x1, k);
        }
        m0 as u32 | (m1 as u32) << 16
    }

    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v3")]
    fn matches_v3(column: &[u32; 32], keys: &[u32]) -> u32 {
        use core::arch::x86_64::*;
        let p = column.as_ptr();
        let x: [__m256i; 4] =
            std::array::from_fn(|i| unsafe { _mm256_loadu_si256(p.add(8 * i).cast()) });
        let mut y = [_mm256_setzero_si256(); 4];
        for &key in keys {
            let k = _mm256_set1_epi32(key as i32);
            for i in 0..4 {
                y[i] = _mm256_or_si256(y[i], _mm256_cmpeq_epi32(x[i], k));
            }
        }
        let mut mask = 0;
        for i in 0..4 {
            mask |= (_mm256_movemask_ps(_mm256_castsi256_ps(y[i])) as u32) << (8 * i);
        }
        mask
    }

    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "aarch64")]
    #[crate::target_cpu(enable = "a2")]
    fn matches_a2(column: &[u32; 32], keys: &[u32]) -> u32 {
        use core::arch::aarch64::*;
        let p = column.as_ptr();
        let x: [uint32x4_t; 8] = std::array::from_fn(|i| unsafe { vld1q_u32(p.add(4 * i)) });
        let mut y = [vdupq_n_u32(0); 8];
        for &key in keys {
            let k = vdupq_n_u32(key);
            for i in 0..8 {
                y[i] = vorrq_u32(y[i], vceqq_u32(x[i], k));
            }
        }
        let bits = unsafe { vld1q_u32([1_u32, 2, 4, 8].as_ptr()) };
        let mut mask = 0;
        for i in 0..8 {
            mask |= vaddvq_u32(vandq_u32(y[i], bits)) << (4 * i);
        }
        mask
    }

    #[cfg(test)]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn kernels() {
        use super::{check, matches_portable, sample};
        let random = rand::random::<u32>;
        #[cfg(target_arch = "x86_64")]
        if crate::is_cpu_detected!("v4") {
            check(
                sample(random),
                |c, k| unsafe { matches_v4(c, k) },
                matches_portable,
            );
        }
        #[cfg(target_arch = "x86_64")]
        if crate::is_cpu_detected!("v3") {
            check(
                sample(random),
                |c, k| unsafe { matches_v3(c, k) },
                matches_portable,
            );
        }
        #[cfg(target_arch = "aarch64")]
        if crate::is_cpu_detected!("a2") {
            check(
                sample(random),
                |c, k| unsafe { matches_a2(c, k) },
                matches_portable,
            );
        }
        check(sample(random), matches, matches_portable);
    }

    #[crate::multiversion(@"v4", @"v3", "v2", @"a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1")]
    pub fn matches(column: &[u32; 32], keys: &[u32]) -> u32 {
        super::matches_portable(column, keys)
    }

    #[cfg_attr(feature = "internal", simd_macros::public)]
    pub mod between {
        #[cfg_attr(feature = "internal", simd_macros::public)]
        #[inline]
        #[cfg(target_arch = "x86_64")]
        #[crate::target_cpu(enable = "v4")]
        fn between_v4(column: &[u32; 32], lo: u32, hi: u32) -> u32 {
            use core::arch::x86_64::*;
            let p = column.as_ptr();
            let x0 = unsafe { _mm512_loadu_si512(p.cast()) };
            let x1 = unsafe { _mm512_loadu_si512(p.add(16).cast()) };
            let (l, h) = (_mm512_set1_epi32(lo as i32), _mm512_set1_epi32(hi as i32));
            let m0 = _mm512_mask_cmple_epu32_mask(_mm512_cmpge_epu32_mask(x0, l), x0, h);
            let m1 = _mm512_mask_cmple_epu32_mask(_mm512_cmpge_epu32_mask(x1, l), x1, h);
            m0 as u32 | (m1 as u32) << 16
        }

        #[cfg_attr(feature = "internal", simd_macros::public)]
        #[inline]
        #[cfg(target_arch = "x86_64")]
        #[crate::target_cpu(enable = "v3")]
        fn between_v3(column: &[u32; 32], lo: u32, hi: u32) -> u32 {
            use core::arch::x86_64::*;
            let p = column.as_ptr();
            let (l, h) = (_mm256_set1_epi32(lo as i32), _mm256_set1_epi32(hi as i32));
            let mut mask = 0;
            for i in 0..4 {
                let x = unsafe { _mm256_loadu_si256(p.add(8 * i).cast()) };
                // `x` is in range where raising it to `lo` and lowering it to `hi` both leave it
                let y = _mm256_and_si256(
                    _mm256_cmpeq_epi32(_mm256_max_epu32(x, l), x),
                    _mm256_cmpeq_epi32(_mm256_min_epu32(x, h), x),
                );
                mask |= (_mm256_movemask_ps(_mm256_castsi256_ps(y)) as u32) << (8 * i);
            }
            mask
        }

        #[cfg_attr(feature = "internal", simd_macros::public)]
        #[inline]
        #[cfg(target_arch = "aarch64")]
        #[crate::target_cpu(enable = "a2")]
        fn between_a2(column: &[u32; 32], lo: u32, hi: u32) -> u32 {
            use core::arch::aarch64::*;
            let p = column.as_ptr();
            let (l, h) = (vdupq_n_u32(lo), vdupq_n_u32(hi));
            let bits = unsafe { vld1q_u32([1_u32, 2, 4, 8].as_ptr()) };
            let mut mask = 0;
            for i in 0..8 {
                let x = unsafe { vld1q_u32(p.add(4 * i)) };
                let y = vandq_u32(vcgeq_u32(x, l), vcleq_u32(x, h));
                mask |= vaddvq_u32(vandq_u32(y, bits)) << (4 * i);
            }
            mask
        }

        #[cfg(test)]
        #[test]
        #[cfg_attr(miri, ignore)]
        fn kernels() {
            use super::super::check_between;
            let random = rand::random::<u32>;
            #[cfg(target_arch = "x86_64")]
            if crate::is_cpu_detected!("v4") {
                check_between(random, |c, l, h| unsafe { between_v4(c, l, h) });
            }
            #[cfg(target_arch = "x86_64")]
            if crate::is_cpu_detected!("v3") {
                check_between(random, |c, l, h| unsafe { between_v3(c, l, h) });
            }
            #[cfg(target_arch = "aarch64")]
            if crate::is_cpu_detected!("a2") {
                check_between(random, |c, l, h| unsafe { between_a2(c, l, h) });
            }
            check_between(random, between);
        }

        #[crate::multiversion(@"v4", @"v3", "v2", @"a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1")]
        pub fn between(column: &[u32; 32], lo: u32, hi: u32) -> u32 {
            super::super::between_portable(column, lo, hi)
        }
    }
}

#[cfg_attr(feature = "internal", simd_macros::public)]
mod u64s {
    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    fn matches_v4(column: &[u64; 32], keys: &[u64]) -> u32 {
        use core::arch::x86_64::*;
        let p = column.as_ptr();
        let x: [__m512i; 4] =
            std::array::from_fn(|i| unsafe { _mm512_loadu_si512(p.add(8 * i).cast()) });
        let mut m = [0_u8; 4];
        for &key in keys {
            let k = _mm512_set1_epi64(key as i64);
            for i in 0..4 {
                m[i] |= _mm512_cmpeq_epi64_mask(x[i], k);
            }
        }
        u32::from_le_bytes(m)
    }

    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v3")]
    fn matches_v3(column: &[u64; 32], keys: &[u64]) -> u32 {
        use core::arch::x86_64::*;
        let p = column.as_ptr();
        let x: [__m256i; 8] =
            std::array::from_fn(|i| unsafe { _mm256_loadu_si256(p.add(4 * i).cast()) });
        let mut y = [_mm256_setzero_si256(); 8];
        for &key in keys {
            let k = _mm256_set1_epi64x(key as i64);
            for i in 0..8 {
                y[i] = _mm256_or_si256(y[i], _mm256_cmpeq_epi64(x[i], k));
            }
        }
        let mut mask = 0;
        for i in 0..8 {
            mask |= (_mm256_movemask_pd(_mm256_castsi256_pd(y[i])) as u32) << (4 * i);
        }
        mask
    }

    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "aarch64")]
    #[crate::target_cpu(enable = "a2")]
    fn matches_a2(column: &[u64; 32], keys: &[u64]) -> u32 {
        use core::arch::aarch64::*;
        let p = column.as_ptr();
        let x: [uint64x2_t; 16] = std::array::from_fn(|i| unsafe { vld1q_u64(p.add(2 * i)) });
        let mut y = [vdupq_n_u64(0); 16];
        for &key in keys {
            let k = vdupq_n_u64(key);
            for i in 0..16 {
                y[i] = vorrq_u64(y[i], vceqq_u64(x[i], k));
            }
        }
        let bits = unsafe { vld1q_u32([1_u32, 2, 4, 8].as_ptr()) };
        let mut mask = 0;
        for i in 0..8 {
            let rows = vcombine_u32(vmovn_u64(y[2 * i]), vmovn_u64(y[2 * i + 1]));
            mask |= vaddvq_u32(vandq_u32(rows, bits)) << (4 * i);
        }
        mask
    }

    #[cfg(test)]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn kernels() {
        use super::{check, matches_portable, sample};
        let random = rand::random::<u64>;
        #[cfg(target_arch = "x86_64")]
        if crate::is_cpu_detected!("v4") {
            check(
                sample(random),
                |c, k| unsafe { matches_v4(c, k) },
                matches_portable,
            );
        }
        #[cfg(target_arch = "x86_64")]
        if crate::is_cpu_detected!("v3") {
            check(
                sample(random),
                |c, k| unsafe { matches_v3(c, k) },
                matches_portable,
            );
        }
        #[cfg(target_arch = "aarch64")]
        if crate::is_cpu_detected!("a2") {
            check(
                sample(random),
                |c, k| unsafe { matches_a2(c, k) },
                matches_portable,
            );
        }
        check(sample(random), matches, matches_portable);
    }

    #[crate::multiversion(@"v4", @"v3", "v2", @"a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1")]
    pub fn matches(column: &[u64; 32], keys: &[u64]) -> u32 {
        super::matches_portable(column, keys)
    }

    #[cfg_attr(feature = "internal", simd_macros::public)]
    pub mod between {
        #[cfg_attr(feature = "internal", simd_macros::public)]
        #[inline]
        #[cfg(target_arch = "x86_64")]
        #[crate::target_cpu(enable = "v4")]
        fn between_v4(column: &[u64; 32], lo: u64, hi: u64) -> u32 {
            use core::arch::x86_64::*;
            let p = column.as_ptr();
            let (l, h) = (_mm512_set1_epi64(lo as i64), _mm512_set1_epi64(hi as i64));
            let mut m = [0_u8; 4];
            for i in 0..4 {
                let x = unsafe { _mm512_loadu_si512(p.add(8 * i).cast()) };
                m[i] = _mm512_mask_cmple_epu64_mask(_mm512_cmpge_epu64_mask(x, l), x, h);
            }
            u32::from_le_bytes(m)
        }

        #[cfg_attr(feature = "internal", simd_macros::public)]
        #[inline]
        #[cfg(target_arch = "x86_64")]
        #[crate::target_cpu(enable = "v3")]
        fn between_v3(column: &[u64; 32], lo: u64, hi: u64) -> u32 {
            use core::arch::x86_64::*;
            let p = column.as_ptr();
            // biased by the sign bit, so that signed compares order them
            let bias = _mm256_set1_epi64x(i64::MIN);
            let l = _mm256_set1_epi64x((lo ^ 1 << 63) as i64);
            let h = _mm256_set1_epi64x((hi ^ 1 << 63) as i64);
            let mut mask = 0;
            for i in 0..8 {
                let x = _mm256_xor_si256(unsafe { _mm256_loadu_si256(p.add(4 * i).cast()) }, bias);
                let out = _mm256_or_si256(_mm256_cmpgt_epi64(l, x), _mm256_cmpgt_epi64(x, h));
                mask |= ((!_mm256_movemask_pd(_mm256_castsi256_pd(out)) & 0xf) as u32) << (4 * i);
            }
            mask
        }

        #[cfg_attr(feature = "internal", simd_macros::public)]
        #[inline]
        #[cfg(target_arch = "aarch64")]
        #[crate::target_cpu(enable = "a2")]
        fn between_a2(column: &[u64; 32], lo: u64, hi: u64) -> u32 {
            use core::arch::aarch64::*;
            let p = column.as_ptr();
            let (l, h) = (vdupq_n_u64(lo), vdupq_n_u64(hi));
            let bits = unsafe { vld1q_u32([1_u32, 2, 4, 8].as_ptr()) };
            let mut mask = 0;
            for i in 0..8 {
                let y = |j: usize| {
                    let x = unsafe { vld1q_u64(p.add(4 * i + 2 * j)) };
                    vmovn_u64(vandq_u64(vcgeq_u64(x, l), vcleq_u64(x, h)))
                };
                let rows = vcombine_u32(y(0), y(1));
                mask |= vaddvq_u32(vandq_u32(rows, bits)) << (4 * i);
            }
            mask
        }

        #[cfg(test)]
        #[test]
        #[cfg_attr(miri, ignore)]
        fn kernels() {
            use super::super::check_between;
            let random = rand::random::<u64>;
            #[cfg(target_arch = "x86_64")]
            if crate::is_cpu_detected!("v4") {
                check_between(random, |c, l, h| unsafe { between_v4(c, l, h) });
            }
            #[cfg(target_arch = "x86_64")]
            if crate::is_cpu_detected!("v3") {
                check_between(random, |c, l, h| unsafe { between_v3(c, l, h) });
            }
            #[cfg(target_arch = "aarch64")]
            if crate::is_cpu_detected!("a2") {
                check_between(random, |c, l, h| unsafe { between_a2(c, l, h) });
            }
            check_between(random, between);
        }

        #[crate::multiversion(@"v4", @"v3", "v2", @"a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1")]
        pub fn between(column: &[u64; 32], lo: u64, hi: u64) -> u32 {
            super::super::between_portable(column, lo, hi)
        }
    }
}

#[cfg_attr(feature = "internal", simd_macros::public)]
mod u128s {
    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    fn matches_v4(column: &[[u64; 2]; 32], keys: &[[u64; 2]]) -> u32 {
        use core::arch::x86_64::*;
        let p = column.as_ptr();
        let x: [__m512i; 8] =
            std::array::from_fn(|i| unsafe { _mm512_loadu_si512(p.add(4 * i).cast()) });
        let even = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
        let odd = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
        let lo: [__m512i; 4] =
            std::array::from_fn(|i| _mm512_permutex2var_epi64(x[2 * i], even, x[2 * i + 1]));
        let hi: [__m512i; 4] =
            std::array::from_fn(|i| _mm512_permutex2var_epi64(x[2 * i], odd, x[2 * i + 1]));
        let mut m = [0_u8; 4];
        for key in keys {
            let (k0, k1) = (
                _mm512_set1_epi64(key[0] as i64),
                _mm512_set1_epi64(key[1] as i64),
            );
            for i in 0..4 {
                m[i] |= _mm512_cmpeq_epi64_mask(lo[i], k0) & _mm512_cmpeq_epi64_mask(hi[i], k1);
            }
        }
        u32::from_le_bytes(m)
    }

    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v3")]
    fn matches_v3(column: &[[u64; 2]; 32], keys: &[[u64; 2]]) -> u32 {
        use core::arch::x86_64::*;
        let p = column.as_ptr();
        let x: [__m256i; 16] =
            std::array::from_fn(|i| unsafe { _mm256_loadu_si256(p.add(2 * i).cast()) });
        // the low and the high halves of 4 rows each, rows 0 2 1 3
        let lo: [__m256i; 8] =
            std::array::from_fn(|i| _mm256_unpacklo_epi64(x[2 * i], x[2 * i + 1]));
        let hi: [__m256i; 8] =
            std::array::from_fn(|i| _mm256_unpackhi_epi64(x[2 * i], x[2 * i + 1]));
        let mut y = [_mm256_setzero_si256(); 8];
        for key in keys {
            let (k0, k1) = (
                _mm256_set1_epi64x(key[0] as i64),
                _mm256_set1_epi64x(key[1] as i64),
            );
            for i in 0..8 {
                let both =
                    _mm256_and_si256(_mm256_cmpeq_epi64(lo[i], k0), _mm256_cmpeq_epi64(hi[i], k1));
                y[i] = _mm256_or_si256(y[i], both);
            }
        }
        let mut mask = 0;
        for i in 0..8 {
            // bits: rows 0 2 1 3
            let m = _mm256_movemask_pd(_mm256_castsi256_pd(y[i])) as u32;
            let m = (m & 0b1001) | (m & 0b0010) << 1 | (m & 0b0100) >> 1;
            mask |= m << (4 * i);
        }
        mask
    }

    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "aarch64")]
    #[crate::target_cpu(enable = "a2")]
    fn matches_a2(column: &[[u64; 2]; 32], keys: &[[u64; 2]]) -> u32 {
        use core::arch::aarch64::*;
        let p = column.as_ptr().cast::<u64>();
        let x: [uint64x2_t; 32] = std::array::from_fn(|i| unsafe { vld1q_u64(p.add(2 * i)) });
        let mut y = [vdupq_n_u32(0); 8];
        for key in keys {
            let k = unsafe { vld1q_u64(key.as_ptr()) };
            for i in 0..8 {
                let a = vcombine_u32(
                    vmovn_u64(vceqq_u64(x[4 * i], k)),
                    vmovn_u64(vceqq_u64(x[4 * i + 1], k)),
                );
                let b = vcombine_u32(
                    vmovn_u64(vceqq_u64(x[4 * i + 2], k)),
                    vmovn_u64(vceqq_u64(x[4 * i + 3], k)),
                );
                let rows = vandq_u32(vuzp1q_u32(a, b), vuzp2q_u32(a, b));
                y[i] = vorrq_u32(y[i], rows);
            }
        }
        let bits = unsafe { vld1q_u32([1_u32, 2, 4, 8].as_ptr()) };
        let mut mask = 0;
        for i in 0..8 {
            mask |= vaddvq_u32(vandq_u32(y[i], bits)) << (4 * i);
        }
        mask
    }

    #[cfg(test)]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn kernels() {
        use super::{check, matches_portable, sample};
        let random = || {
            let lo = rand::random::<u64>();
            let hi = if rand::random::<bool>() {
                lo
            } else {
                rand::random()
            };
            [lo, hi]
        };
        #[cfg(target_arch = "x86_64")]
        if crate::is_cpu_detected!("v4") {
            check(
                sample(random),
                |c, k| unsafe { matches_v4(c, k) },
                matches_portable,
            );
        }
        #[cfg(target_arch = "x86_64")]
        if crate::is_cpu_detected!("v3") {
            check(
                sample(random),
                |c, k| unsafe { matches_v3(c, k) },
                matches_portable,
            );
        }
        #[cfg(target_arch = "aarch64")]
        if crate::is_cpu_detected!("a2") {
            check(
                sample(random),
                |c, k| unsafe { matches_a2(c, k) },
                matches_portable,
            );
        }
        check(sample(random), matches, matches_portable);
        // a row that matches one half of a key and the other half of another
        let column = std::array::from_fn(|i| [i as u64, 100 + i as u64]);
        let keys = [[3, 0], [0, 103], [5, 105]];
        assert_eq!(matches(&column, &keys), 1 << 5);
    }

    #[crate::multiversion(@"v4", @"v3", "v2", @"a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1")]
    pub fn matches(column: &[[u64; 2]; 32], keys: &[[u64; 2]]) -> u32 {
        super::matches_portable(column, keys)
    }

    #[cfg_attr(feature = "internal", simd_macros::public)]
    pub mod between {
        #[cfg_attr(feature = "internal", simd_macros::public)]
        #[inline]
        #[cfg(target_arch = "x86_64")]
        #[crate::target_cpu(enable = "v4")]
        fn between_v4(column: &[[u64; 2]; 32], lo: [u64; 2], hi: [u64; 2]) -> u32 {
            use core::arch::x86_64::*;
            let p = column.as_ptr();
            let even = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
            let odd = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
            let [l0, l1, h0, h1] =
                [lo[0], lo[1], hi[0], hi[1]].map(|x| _mm512_set1_epi64(x as i64));
            let mut m = [0_u8; 4];
            for i in 0..4 {
                let a = unsafe { _mm512_loadu_si512(p.add(8 * i).cast()) };
                let b = unsafe { _mm512_loadu_si512(p.add(8 * i + 4).cast()) };
                // the rows' low and high words
                let (x0, x1) = (
                    _mm512_permutex2var_epi64(a, even, b),
                    _mm512_permutex2var_epi64(a, odd, b),
                );
                let ge = _mm512_cmpgt_epu64_mask(x1, l1)
                    | _mm512_mask_cmpge_epu64_mask(_mm512_cmpeq_epi64_mask(x1, l1), x0, l0);
                let le = _mm512_cmplt_epu64_mask(x1, h1)
                    | _mm512_mask_cmple_epu64_mask(_mm512_cmpeq_epi64_mask(x1, h1), x0, h0);
                m[i] = ge & le;
            }
            u32::from_le_bytes(m)
        }

        #[cfg_attr(feature = "internal", simd_macros::public)]
        #[inline]
        #[cfg(target_arch = "x86_64")]
        #[crate::target_cpu(enable = "v3")]
        fn between_v3(column: &[[u64; 2]; 32], lo: [u64; 2], hi: [u64; 2]) -> u32 {
            use core::arch::x86_64::*;
            let p = column.as_ptr();
            // biased by the sign bit, so that signed compares order them
            let bias = _mm256_set1_epi64x(i64::MIN);
            let [l0, l1, h0, h1] =
                [lo[0], lo[1], hi[0], hi[1]].map(|x| _mm256_set1_epi64x((x ^ 1 << 63) as i64));
            let mut mask = 0;
            for i in 0..8 {
                let a = unsafe { _mm256_loadu_si256(p.add(4 * i).cast()) };
                let b = unsafe { _mm256_loadu_si256(p.add(4 * i + 2).cast()) };
                // the rows' low and high words, the middle two rows swapped
                let x0 = _mm256_xor_si256(_mm256_unpacklo_epi64(a, b), bias);
                let x1 = _mm256_xor_si256(_mm256_unpackhi_epi64(a, b), bias);
                // below `lo` or above `hi`, by the high word and then the low word
                let below = _mm256_or_si256(
                    _mm256_cmpgt_epi64(l1, x1),
                    _mm256_and_si256(_mm256_cmpeq_epi64(x1, l1), _mm256_cmpgt_epi64(l0, x0)),
                );
                let above = _mm256_or_si256(
                    _mm256_cmpgt_epi64(x1, h1),
                    _mm256_and_si256(_mm256_cmpeq_epi64(x1, h1), _mm256_cmpgt_epi64(x0, h0)),
                );
                let m = !_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_or_si256(below, above)))
                    as u32
                    & 0xf;
                let m = (m & 0b1001) | (m & 0b0010) << 1 | (m & 0b0100) >> 1;
                mask |= m << (4 * i);
            }
            mask
        }

        #[cfg_attr(feature = "internal", simd_macros::public)]
        #[inline]
        #[cfg(target_arch = "aarch64")]
        #[crate::target_cpu(enable = "a2")]
        fn between_a2(column: &[[u64; 2]; 32], lo: [u64; 2], hi: [u64; 2]) -> u32 {
            use core::arch::aarch64::*;
            let p = column.as_ptr().cast::<u64>();
            let [l0, l1, h0, h1] = [lo[0], lo[1], hi[0], hi[1]].map(|x| vdupq_n_u64(x));
            let bits = unsafe { vld1q_u32([1_u32, 2, 4, 8].as_ptr()) };
            let mut mask = 0;
            for i in 0..8 {
                let y = |j: usize| {
                    // the rows' low and high words
                    let x = unsafe { vld2q_u64(p.add(8 * i + 4 * j)) };
                    let ge = vorrq_u64(
                        vcgtq_u64(x.1, l1),
                        vandq_u64(vceqq_u64(x.1, l1), vcgeq_u64(x.0, l0)),
                    );
                    let le = vorrq_u64(
                        vcltq_u64(x.1, h1),
                        vandq_u64(vceqq_u64(x.1, h1), vcleq_u64(x.0, h0)),
                    );
                    vmovn_u64(vandq_u64(ge, le))
                };
                let rows = vcombine_u32(y(0), y(1));
                mask |= vaddvq_u32(vandq_u32(rows, bits)) << (4 * i);
            }
            mask
        }

        #[cfg(test)]
        #[test]
        #[cfg_attr(miri, ignore)]
        fn kernels() {
            use super::super::check_between;
            let random = || {
                let lo = rand::random::<u64>();
                [
                    lo,
                    if rand::random::<bool>() {
                        lo
                    } else {
                        rand::random()
                    },
                ]
            };
            #[cfg(target_arch = "x86_64")]
            if crate::is_cpu_detected!("v4") {
                check_between(random, |c, l, h| unsafe { between_v4(c, l, h) });
            }
            #[cfg(target_arch = "x86_64")]
            if crate::is_cpu_detected!("v3") {
                check_between(random, |c, l, h| unsafe { between_v3(c, l, h) });
            }
            #[cfg(target_arch = "aarch64")]
            if crate::is_cpu_detected!("a2") {
                check_between(random, |c, l, h| unsafe { between_a2(c, l, h) });
            }
            check_between(random, between);
            // the high word decides first, then the low word
            let column = std::array::from_fn(|i| [u64::MAX - i as u64, i as u64]);
            assert_eq!(between(&column, [u64::MAX - 3, 3], [0, 7]), 0b0111_1000);
        }

        #[crate::multiversion(@"v4", @"v3", "v2", @"a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1")]
        pub fn between(column: &[[u64; 2]; 32], lo: [u64; 2], hi: [u64; 2]) -> u32 {
            super::super::between_portable(column, lo, hi)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fold() {
        let x = 0x0123_4567_89ab_cdef_u64;
        let expected = 0x0123 ^ 0x4567 ^ 0x89ab ^ 0xcdef;
        assert_eq!(x.fold_one(), expected);
        assert_eq!([x, 0].fold_one(), expected);
        assert_eq!(((x ^ x >> 32) as u32).fold_one(), expected);
    }

    fn sets<T: Value>(random: impl Fn() -> T) {
        for n in [1, 10, 33, 200, 5000] {
            let mut values = (0..n).map(|_| random()).collect::<Vec<_>>();
            values.sort_unstable();
            values.dedup();
            let set = Set::new(&values);
            for _ in 0..if cfg!(not(miri)) { 64 } else { 1 } {
                let column: [T; 32] = std::array::from_fn(|_| {
                    if rand::random::<bool>() {
                        values[rand::random::<u32>() as usize % values.len()]
                    } else {
                        random()
                    }
                });
                let mask = set.matches(&column);
                for i in 0..32 {
                    let expected = values.binary_search(&column[i]).is_ok();
                    assert_eq!(mask >> i & 1 != 0, expected);
                    assert_eq!(set.contains(column[i]), expected);
                }
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn set() {
        sets(rand::random::<u16>);
        sets(|| rand::random::<u32>() % 100_000);
        sets(rand::random::<u32>);
        sets(rand::random::<u64>);
        sets(|| [rand::random::<u64>() % 7, rand::random::<u64>() % 7]);
        sets(rand::random::<[u64; 2]>);
        // the value that marks free slots by default is in the set
        let set = Set::new(&[0_u64, 1, 2]);
        assert!(set.contains(0) && set.contains(2) && !set.contains(3));
    }
}
