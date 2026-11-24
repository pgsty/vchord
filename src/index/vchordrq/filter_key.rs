// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

//! The filter key columns of a vchordrq index: their stored keys and scan keys.

use crate::index::storage::PostgresRelation;
use pgrx::pg_sys::{self, BuiltinOid, Datum};
use vchordrq::keys::{
    Condition, Intervals, Key, Keys, Summary, Width, fold, hash_int, hash_uuid, raw_f32, raw_f64,
    raw_int, raw_uuid,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Type {
    Uuid,
    Int2,
    Int4,
    Int8,
    Date,
    Timestamp,
    Timestamptz,
    Oid,
    Bool,
    Float4,
    Float8,
    Enum,
}

/// A day, in the unit of timestamps.
const DAY: i64 = 86_400_000_000;

impl Type {
    pub fn width(self) -> Width {
        match self {
            Self::Int2 | Self::Bool => Width::W2,
            Self::Int4 | Self::Date | Self::Oid | Self::Float4 | Self::Enum => Width::W4,
            Self::Int8 | Self::Timestamp | Self::Timestamptz | Self::Float8 => Width::W8,
            Self::Uuid => Width::W16,
        }
    }

    fn of_oid(oid: pg_sys::Oid) -> Option<Self> {
        use BuiltinOid::*;
        Some(match BuiltinOid::from_u32(oid.to_u32()).ok()? {
            UUIDOID => Self::Uuid,
            INT2OID => Self::Int2,
            INT4OID => Self::Int4,
            INT8OID => Self::Int8,
            DATEOID => Self::Date,
            TIMESTAMPOID => Self::Timestamp,
            TIMESTAMPTZOID => Self::Timestamptz,
            OIDOID => Self::Oid,
            BOOLOID => Self::Bool,
            FLOAT4OID => Self::Float4,
            FLOAT8OID => Self::Float8,
            ANYENUMOID => Self::Enum,
            _ => return None,
        })
    }

    /// Where `datum`, of type `self`, lies among the stored forms of `key`, of type `column`.
    unsafe fn bound(self, datum: Datum, column: Type, key: &Key) -> Bound {
        use Type::*;
        let int = |x: i64| match raw_int(x, key.width) {
            Some(stored) => Bound::At(stored as u128),
            None if x < 0 => Bound::Below,
            None => Bound::Above,
        };
        // from `x - d` to `x + d`, a day number or a time far inside the range of `key`
        let within = |x: i64, d: i64| {
            let raw = |x| raw_int(x, key.width).expect("a date or a time in range") as u128;
            Bound::Within(raw(x.saturating_sub(d)), raw(x.saturating_add(d)))
        };
        let value = datum.value();
        match (self, column) {
            (Uuid, _) => {
                let bytes = unsafe { &*datum.cast_mut_ptr::<[u8; 16]>() };
                Bound::At(if key.hashed {
                    fold(hash_uuid(bytes), key.width) as u128
                } else {
                    raw_uuid(bytes)
                })
            }
            (Int2 | Int4 | Int8, _) => {
                let x = match self {
                    Int2 => value as i16 as i64,
                    Int4 => value as i32 as i64,
                    _ => value as i64,
                };
                match key.hashed {
                    true => Bound::At(fold(hash_int(x), key.width) as u128),
                    false => int(x),
                }
            }
            (Oid | Enum, _) => Bound::At(value as u32 as u128),
            (Bool, _) => Bound::At((value != 0) as u128),
            (Float4, Float8) => Bound::At(raw_f64(f32::from_bits(value as u32) as f64) as u128),
            (Float4, _) => Bound::At(raw_f32(f32::from_bits(value as u32)) as u128),
            (Float8, Float4) => {
                let x = f64::from_bits(value as u64);
                // the greatest float4 not above `x`
                let below = match x as f32 {
                    y if y as f64 > x => y.next_down(),
                    y => y,
                };
                match x.is_nan() || below as f64 == x {
                    true => Bound::At(raw_f32(x as f32) as u128),
                    false => Bound::After(raw_f32(below) as u128),
                }
            }
            (Float8, _) => Bound::At(raw_f64(f64::from_bits(value as u64)) as u128),
            (Date, column) => {
                let x = value as i32;
                let infinite = x == i32::MIN || x == i32::MAX;
                let time = match infinite {
                    true => [i64::MIN, i64::MAX][(x > 0) as usize],
                    // a date past every timestamp still orders below 'infinity'
                    false => (x as i64).saturating_mul(DAY).min(i64::MAX - 1),
                };
                match column {
                    Date => int(x as i64),
                    Timestamp => int(time),
                    _ if infinite => int(time),
                    _ => within(time, DAY),
                }
            }
            (Timestamp | Timestamptz, column) => {
                let x = value as i64;
                let infinite = x == i64::MIN || x == i64::MAX;
                match column {
                    Date if infinite => Bound::Within(0, key.max()),
                    Date if self == Timestamptz => within(x.div_euclid(DAY), 1),
                    Date => match x.rem_euclid(DAY) {
                        0 => int(x / DAY),
                        _ => Bound::After(raw_int(x.div_euclid(DAY), key.width).unwrap() as u128),
                    },
                    _ if column == self || infinite => int(x),
                    _ => within(x, DAY),
                }
            }
        }
    }
}

/// Where a constant lies among the stored forms of a key, which order like the key.
#[derive(Debug, Clone, Copy)]
enum Bound {
    Below,
    At(u128),
    /// Between a stored form and the next.
    After(u128),
    /// Somewhere from one stored form to another: the rows are rechecked.
    Within(u128, u128),
    Above,
}

const LESS: u16 = pg_sys::BTLessStrategyNumber as u16;
const LESS_EQUAL: u16 = pg_sys::BTLessEqualStrategyNumber as u16;
const EQUAL: u16 = pg_sys::BTEqualStrategyNumber as u16;
const GREATER_EQUAL: u16 = pg_sys::BTGreaterEqualStrategyNumber as u16;
const GREATER: u16 = pg_sys::BTGreaterStrategyNumber as u16;

/// The stored forms, up to `max`, that a scan key of `strategy` accepts against `bound`.
fn interval(strategy: u16, bound: Bound, max: u128) -> Option<[u128; 2]> {
    use Bound::*;
    match (strategy, bound) {
        (LESS, At(x)) => x.checked_sub(1).map(|x| [0, x]),
        (LESS_EQUAL, At(x)) | (LESS | LESS_EQUAL, After(x) | Within(_, x)) => Some([0, x]),
        (EQUAL, At(x)) => Some([x, x]),
        (EQUAL, Within(x, y)) => Some([x, y]),
        (GREATER_EQUAL, At(x)) | (GREATER | GREATER_EQUAL, Within(x, _)) => Some([x, max]),
        (GREATER, At(x)) | (GREATER | GREATER_EQUAL, After(x)) => (x < max).then(|| [x + 1, max]),
        (LESS | LESS_EQUAL, Above) | (GREATER_EQUAL | GREATER, Below) => Some([0, max]),
        _ => None,
    }
}

unsafe fn types(index_relation: pg_sys::Relation) -> impl Iterator<Item = Type> {
    let n = unsafe { (*(*index_relation).rd_index).indnkeyatts } as usize;
    (1..n).map(move |i| {
        let oid = unsafe { *(*index_relation).rd_opcintype.add(i) };
        Type::of_oid(oid)
            .unwrap_or_else(|| pgrx::error!("unknown operator class of a filter key column"))
    })
}

#[repr(C)]
struct Options {
    vl_len_: i32,
    bytes: i32,
}

#[pgrx::pg_guard]
unsafe extern "C-unwind" fn validate(
    parsed: *mut std::ffi::c_void,
    _values: *mut pg_sys::relopt_value,
    _n: std::ffi::c_int,
) {
    let bytes = unsafe { (*parsed.cast::<Options>()).bytes };
    if Width::of_bytes(bytes as usize).is_none() {
        pgrx::error!("invalid value for option \"bytes\": {bytes}, not 2, 4 or 8");
    }
}

/// The options support function of a `*_hash_ops` filter key column: `bytes`, at most half the key.
pub unsafe fn options(relopts: pgrx::Internal, ty: Type) {
    let most = ty.width().bytes() / 2;
    unsafe {
        let relopts = relopts
            .unwrap()
            .expect("no options")
            .cast_mut_ptr::<pg_sys::local_relopts>();
        pg_sys::init_local_reloptions(relopts, size_of::<Options>());
        pg_sys::add_local_int_reloption(
            relopts,
            c"bytes".as_ptr(),
            c"bytes of the hash that a row stores of the key".as_ptr(),
            most.min(4) as _,
            2,
            most as _,
            std::mem::offset_of!(Options, bytes) as _,
        );
        pg_sys::register_reloptions_validator(relopts, Some(validate));
    }
}

/// The filter keys of an index being built; `zone_maps`: how many leading keys get block bounds.
pub unsafe fn of_options(index_relation: pg_sys::Relation, zone_maps: usize) -> Keys {
    let options = unsafe { pg_sys::RelationGetIndexAttOptions(index_relation, false) };
    Keys::new(unsafe { types(index_relation) }.enumerate().map(|(i, ty)| {
        let hash = unsafe { options.add(i + 1).read().cast::<Options>().as_ref() };
        let width = hash.map_or(ty.width(), |x| Width::of_bytes(x.bytes as usize).unwrap());
        (
            width,
            hash.is_some(),
            (i < zone_maps).then_some(Summary::Bounds),
        )
    }))
}

pub struct FilterKeys {
    pub keys: Keys,
    index_relation: pg_sys::Relation,
}

impl FilterKeys {
    pub fn new(index_relation: pg_sys::Relation, keys: Keys) -> Self {
        Self {
            keys,
            index_relation,
        }
    }

    /// Relcache invalidation can free `rd_amcache` on the next catalog access.
    pub unsafe fn of_index(index_relation: pg_sys::Relation) -> Self {
        unsafe {
            if (*index_relation).rd_amcache.is_null() {
                let keys = if (*(*index_relation).rd_index).indnkeyatts < 2 {
                    Keys::default()
                } else {
                    let index = PostgresRelation::<vchordrq::Opaque>::new(index_relation);
                    let keys = Keys::of(&index);
                    if keys.as_slice().len() != types(index_relation).count() {
                        pgrx::error!("the filter keys of the index do not match its columns");
                    }
                    keys
                };
                let memory =
                    pg_sys::MemoryContextAlloc((*index_relation).rd_indexcxt, size_of::<Keys>());
                memory.cast::<Keys>().write(keys);
                (*index_relation).rd_amcache = memory;
            }
            Self {
                keys: (*index_relation).rd_amcache.cast::<Keys>().read(),
                index_relation,
            }
        }
    }

    fn ty(&self, i: usize) -> Type {
        let oid = unsafe { *(*self.index_relation).rd_opcintype.add(i + 1) };
        Type::of_oid(oid).unwrap()
    }

    /// The row's columns: a NULL is stored as zeros and marked in the NULL mask.
    pub unsafe fn row(&self, values: *const Datum, is_null: *const bool) -> Vec<u8> {
        let mut columns = Vec::with_capacity(self.keys.columns());
        let mut nulls = 0_u32;
        for (i, key) in self.keys.as_slice().iter().enumerate() {
            let stored = unsafe {
                if is_null.add(i + 1).read() {
                    nulls |= 1 << i;
                    0
                } else {
                    match self.ty(i).bound(values.add(i + 1).read(), self.ty(i), key) {
                        Bound::At(stored) => stored,
                        _ => unreachable!("a key out of the range of its own type"),
                    }
                }
            };
            columns.extend_from_slice(&stored.to_le_bytes()[..key.width.bytes()]);
        }
        if !self.keys.is_empty() {
            columns.extend_from_slice(&nulls.to_le_bytes());
        }
        columns
    }

    /// What a scan key on filter key column `i` accepts; with `ANY(array)`, what an element does.
    pub unsafe fn of_scan_key(&self, i: usize, scan_key: &pg_sys::ScanKeyData) -> Condition {
        let (column, key) = (self.ty(i), &self.keys.as_slice()[i]);
        let flag = |x: u32| scan_key.sk_flags & x as i32 != 0;
        let null = flag(pg_sys::SK_SEARCHNULL);
        if flag(pg_sys::SK_ISNULL) {
            let values = flag(pg_sys::SK_SEARCHNOTNULL).then_some([0, key.max()]);
            let values = Intervals::new(values.into_iter().collect());
            return Condition {
                values,
                null,
                exact: true,
            };
        }
        let mut exact = true;
        let mut accepts = |datum| {
            if column == Type::Enum && scan_key.sk_strategy != EQUAL {
                return unsafe { self.enum_values(i, datum, scan_key.sk_strategy) };
            }
            // the right type of the operator, which differs for `int4 = int8`
            let ty = Type::of_oid(scan_key.sk_subtype).unwrap_or(column);
            let bound = unsafe { ty.bound(datum, column, key) };
            exact &= !matches!(bound, Bound::Within(..));
            interval(scan_key.sk_strategy, bound, key.max())
                .into_iter()
                .collect()
        };
        let values = unsafe { values(scan_key, &mut accepts) };
        Condition {
            values,
            null,
            exact,
        }
    }

    /// The values of the enum of filter key column `i` that a range of `strategy` accepts against
    /// `value`, by the type's order when the scan starts: their OIDs, which the rows store.
    unsafe fn enum_values(&self, i: usize, value: Datum, strategy: u16) -> Vec<[u128; 2]> {
        unsafe {
            let ty = pg_sys::get_atttype((*self.index_relation).rd_id, i as i16 + 2);
            let ty = pg_sys::getBaseType(ty);
            let cache = pg_sys::lookup_type_cache(ty, 0);
            let id = pg_sys::SysCacheIdentifier::ENUMTYPOIDNAME as _;
            let list = pg_sys::SearchSysCacheList(id, 1, ty.into(), Datum::null(), Datum::null());
            let members = (*list).members.as_slice((*list).n_members as usize);
            let value = pg_sys::Oid::from(value.value() as u32);
            let accepted = (members.iter())
                .map(|&x| {
                    (*pg_sys::GETSTRUCT!(&raw mut (*x).tuple)
                        .get()
                        .cast::<pg_sys::FormData_pg_enum>())
                    .oid
                })
                .filter(|&x| {
                    let order = pg_sys::compare_values_of_enum(cache, x, value);
                    match strategy {
                        LESS => order < 0,
                        LESS_EQUAL => order <= 0,
                        GREATER_EQUAL => order >= 0,
                        _ => order > 0,
                    }
                })
                .map(|x| [x.to_u32() as u128; 2])
                .collect();
            pg_sys::ReleaseCatCacheList(list);
            accepted
        }
    }
}

/// The stored forms a scan key accepts, by what `accepts` makes of its constant or of each element.
unsafe fn values(
    scan_key: &pg_sys::ScanKeyData,
    accepts: &mut impl FnMut(Datum) -> Vec<[u128; 2]>,
) -> Intervals {
    if scan_key.sk_flags & pg_sys::SK_SEARCHARRAY as i32 == 0 {
        return Intervals::new(accepts(scan_key.sk_argument));
    }
    unsafe {
        let datum = scan_key.sk_argument.cast_mut_ptr::<pg_sys::varlena>();
        let array = pg_sys::pg_detoast_datum(datum).cast::<pg_sys::ArrayType>();
        let elemtype = (*array).elemtype;
        let (mut typlen, mut typbyval, mut typalign) = (0, false, 0);
        pg_sys::get_typlenbyvalalign(elemtype, &mut typlen, &mut typbyval, &mut typalign);
        let mut elements = std::ptr::null_mut();
        let mut nulls = std::ptr::null_mut();
        let mut n = 0;
        pg_sys::deconstruct_array(
            array,
            elemtype,
            typlen as _,
            typbyval,
            typalign,
            &mut elements,
            &mut nulls,
            &mut n,
        );
        let accepted = (0..n as usize)
            .filter(|&i| !nulls.add(i).read())
            .flat_map(|i| accepts(elements.add(i).read()))
            .collect::<Vec<_>>();
        pg_sys::pfree(elements.cast());
        pg_sys::pfree(nulls.cast());
        if array.cast() != datum {
            pg_sys::pfree(array.cast());
        }
        Intervals::new(accepted)
    }
}
