// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

pub fn feistel<I>(width: u32, x: I, round: u32, secret: impl Fn(u32, I) -> I) -> I
where
    I: Copy,
    I: Ord,
    I: std::ops::Add<Output = I>,
    I: std::ops::Sub<Output = I>,
    I: std::ops::BitAnd<Output = I>,
    I: std::ops::BitOr<Output = I>,
    I: std::ops::BitXor<Output = I>,
    I: std::ops::Shl<u32, Output = I>,
    I: std::ops::Shr<u32, Output = I>,
    I: Zero,
    I: One,
    I: std::fmt::Debug,
{
    assert_eq!(width % 2, 0);
    assert_eq!(x >> width, I::zero());

    let half_width = width >> 1;
    let half_mask = (I::one() << half_width) - I::one();

    let mut left = (x >> half_width) & half_mask;
    let mut right = x & half_mask;

    for i in 0..round {
        (left, right) = (right, left ^ (secret(i, right) & half_mask));
    }

    (left << half_width) | right
}

/// Lazily enumerates a pseudo-random permutation of `0..n` (the order in which
/// index builds sample heap blocks).
///
/// The permutation runs a Feistel network over the smallest power-of-two domain
/// of even width that contains `n`, skipping values `>= n`. The domain is
/// computed in `u64`: for `n >= 2^30` its width is 32, and `1 << 32` overflowed
/// a `u32` (release builds then sampled at most one block).
pub fn sample(n: u32, key: [u8; 8]) -> impl Iterator<Item = u32> + Clone {
    let width = if n == 0 {
        0
    } else {
        (n.ilog2() + 1).next_multiple_of(2)
    };
    let end: u64 = if n == 0 { 0 } else { 1 << width };
    let secret = move |round: u32, x: u64| {
        let mut buffer = [0_u8; 20];
        buffer[0..4].copy_from_slice(&round.to_le_bytes());
        buffer[4..12].copy_from_slice(&x.to_le_bytes());
        buffer[12..20].copy_from_slice(&key);
        wyhash::wyhash(&buffer, 0)
    };
    (0..end)
        .map(move |i| feistel::<u64>(width, i, 8, secret))
        .filter(move |&x| x < u64::from(n))
        .map(|x| x as u32)
}

pub trait Zero {
    fn zero() -> Self;
}

pub trait One {
    fn one() -> Self;
}

macro_rules! impl_traits {
    ($t:ty) => {
        impl Zero for $t {
            fn zero() -> Self {
                0
            }
        }

        impl One for $t {
            fn one() -> Self {
                1
            }
        }
    };
}

impl_traits!(u8);
impl_traits!(u16);
impl_traits!(u32);
impl_traits!(u64);
impl_traits!(u128);

// This is a standalone crate, simply because we want to run tests for it.

#[test]
fn is_a_permutation() {
    let key_0 = [7u8; _];
    let key_1 = [8u8; _];
    let secret = move |round: u32, x: u32| {
        let buffer = [round.to_le_bytes(), x.to_le_bytes(), key_0, key_1];
        wyhash::wyhash(buffer.as_flattened(), 0) as u32
    };
    for width in (0..if cfg!(not(miri)) { 20 } else { 10 }).step_by(2) {
        let mut y = Vec::new();
        for x in 0..1 << width {
            y.push(feistel::<u32>(width, x, 8, secret));
        }
        if width <= 8 {
            eprintln!("feistel({width}, _, 8, *) = {y:?}");
        }
        y.sort_unstable();
        for x in 0..1 << width {
            assert_eq!(y[x as usize], x);
        }
    }
}

#[test]
fn small_u32_domain() {
    let n = if cfg!(not(miri)) { 6370_u32 } else { 637_u32 };
    let width = (n.ilog2() + 1).next_multiple_of(2);
    let key_0 = rand::RngExt::random(&mut rand::rng());
    let key_1 = rand::RngExt::random(&mut rand::rng());
    let secret = move |round: u32, x: u32| {
        let buffer = [round.to_le_bytes(), x.to_le_bytes(), key_0, key_1];
        wyhash::wyhash(buffer.as_flattened(), 0) as u32
    };
    let mut permutation = (0..1 << width)
        .map(move |i| feistel(width, i, 8, secret))
        .filter(move |&x| x < n);
    for _ in 0..n {
        assert!(permutation.next().is_some());
    }
    assert_eq!(permutation.next(), None);
}

#[test]
fn sample_is_a_permutation() {
    let key = [3_u8; 8];
    assert_eq!(sample(0, key).next(), None);
    let check = |n: u32| {
        let mut y = sample(n, key).collect::<Vec<_>>();
        y.sort_unstable();
        assert_eq!(y, (0..n).collect::<Vec<_>>(), "n = {n}");
    };
    let limit = if cfg!(not(miri)) { 1024 } else { 64 };
    for n in 1..=limit {
        check(n);
    }
    if cfg!(not(miri)) {
        for n in (1025..=4096).step_by(97) {
            check(n);
        }
    }
}

#[test]
fn sample_large_domains() {
    if cfg!(miri) {
        return;
    }
    let key = [5_u8; 8];
    for n in [(1_u32 << 30) - 1, 1 << 30, 1 << 31, u32::MAX] {
        let y = sample(n, key).take(10_000).collect::<Vec<_>>();
        assert_eq!(y.len(), 10_000, "n = {n}");
        assert!(y.iter().all(|&x| x < n), "n = {n}");
        let mut z = y.clone();
        z.sort_unstable();
        z.dedup();
        assert_eq!(z.len(), y.len(), "n = {n}");
    }
}
