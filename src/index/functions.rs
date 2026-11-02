// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use crate::index::storage::PostgresRelation;
use crate::recorder::dump;
use pgrx::datum::IntoDatum;
use pgrx::iter::SetOfIterator;
use pgrx::pg_sys::Oid;
use std::ffi::{CStr, c_char};

struct ClassInfo {
    relname: String,
    relam: Oid,
    relkind: c_char,
}

unsafe fn heap_tuple_struct<T>(tuple: pgrx::pg_sys::HeapTuple) -> *const T {
    let tuple = unsafe { &*tuple };
    let data = tuple.t_data;
    unsafe { data.cast::<u8>().add((*data).t_hoff as usize).cast::<T>() }
}

fn am_oid_by_name(amname: &CStr) -> Option<Oid> {
    let tuple = unsafe {
        pgrx::pg_sys::SearchSysCache1(
            pgrx::pg_sys::SysCacheIdentifier::AMNAME as _,
            amname.into_datum()?,
        )
    };
    if tuple.is_null() {
        return None;
    }
    let oid = unsafe {
        let am = heap_tuple_struct::<pgrx::pg_sys::FormData_pg_am>(tuple);
        (*am).oid
    };
    unsafe { pgrx::pg_sys::ReleaseSysCache(tuple) };
    Some(oid)
}

fn class_info(reloid: Oid) -> Option<ClassInfo> {
    let tuple = unsafe {
        pgrx::pg_sys::SearchSysCache1(
            pgrx::pg_sys::SysCacheIdentifier::RELOID as _,
            reloid.into_datum()?,
        )
    };
    if tuple.is_null() {
        return None;
    }
    let info = unsafe {
        let class = heap_tuple_struct::<pgrx::pg_sys::FormData_pg_class>(tuple);
        ClassInfo {
            relname: CStr::from_ptr((*class).relname.data.as_ptr())
                .to_string_lossy()
                .into_owned(),
            relam: (*class).relam,
            relkind: (*class).relkind,
        }
    };
    unsafe { pgrx::pg_sys::ReleaseSysCache(tuple) };
    Some(info)
}

fn validate_index(indexrelid: Oid, amname: &CStr, am_label: &str) {
    let Some(pg_am_oid) = am_oid_by_name(amname) else {
        pgrx::error!("vchord is not installed");
    };
    let Some(pg_class) = class_info(indexrelid) else {
        pgrx::error!("the relation does not exist");
    };
    if pg_class.relkind != pgrx::pg_sys::RELKIND_INDEX as c_char {
        pgrx::error!("the relation {:?} is not an index", pg_class.relname);
    }
    if pg_class.relam != pg_am_oid {
        pgrx::error!(
            "the index {:?} is not a {} index",
            pg_class.relname,
            am_label
        );
    }
}

#[pgrx::pg_extern(sql = "")]
fn _vchordg_prewarm(indexrelid: Oid) -> String {
    validate_index(indexrelid, c"vchordg", "vchordg");
    let relation = Index::open(indexrelid, pgrx::pg_sys::AccessShareLock as _);
    let opfamily = unsafe { crate::index::vchordg::opclass::opfamily(relation.raw()) };
    let index = unsafe { PostgresRelation::new(relation.raw()) };
    crate::index::vchordg::dispatch::prewarm(opfamily, &index)
}

#[pgrx::pg_extern(sql = "")]
fn _vchordrq_prewarm(indexrelid: Oid, height: i32) -> String {
    validate_index(indexrelid, c"vchordrq", "vchordrq");
    let relation = Index::open(indexrelid, pgrx::pg_sys::AccessShareLock as _);
    let opfamily = unsafe { crate::index::vchordrq::opclass::opfamily(relation.raw()) };
    let index = unsafe { PostgresRelation::new(relation.raw()) };
    crate::index::vchordrq::dispatch::prewarm(opfamily, &index, height)
}

struct Index {
    raw: *mut pgrx::pg_sys::RelationData,
    lockmode: pgrx::pg_sys::LOCKMODE,
}

impl Index {
    fn open(indexrelid: Oid, lockmode: pgrx::pg_sys::LOCKMASK) -> Self {
        Self {
            raw: unsafe { pgrx::pg_sys::index_open(indexrelid, lockmode) },
            lockmode,
        }
    }
    fn raw(&self) -> *mut pgrx::pg_sys::RelationData {
        self.raw
    }
}

impl Drop for Index {
    fn drop(&mut self) {
        unsafe {
            pgrx::pg_sys::index_close(self.raw, self.lockmode);
        }
    }
}

#[pgrx::pg_extern(sql = "")]
fn _vchordrq_sampled_values(indexrelid: Oid) -> SetOfIterator<'static, String> {
    validate_index(indexrelid, c"vchordrq", "vchordrq");
    // The user must have access to the index, if not, raise an error from Postgres.
    let _relation = Index::open(indexrelid, pgrx::pg_sys::AccessShareLock as _);
    let queries = dump(indexrelid.to_u32());
    SetOfIterator::new(queries)
}
