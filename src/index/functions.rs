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
use std::ffi::{CStr, CString, c_char};

struct ClassInfo {
    relname: String,
    relam: Oid,
    relkind: c_char,
    relowner: Oid,
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
            relowner: (*class).relowner,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IndexAccess {
    /// Loading index pages, like `pg_prewarm`: requires `SELECT` on the table.
    Prewarm,
    /// Reading other users' query vectors requires `pg_read_all_stats`.
    SampledValues,
}

/// Checks table access for prewarm and statistics privileges for query samples.
/// Table ownership/SELECT cannot authorize samples from other users or tenants.
///
/// The check is performed before any lock is taken on the index, so an
/// unprivileged user cannot queue locks on another user's relations.
fn check_index_access(indexrelid: Oid, access: IndexAccess) -> Oid {
    let heaprelid = unsafe { pgrx::pg_sys::IndexGetRelation(indexrelid, true) };
    if heaprelid == Oid::INVALID {
        pgrx::error!("the relation does not exist");
    }
    let Some(heap) = class_info(heaprelid) else {
        pgrx::error!("the relation does not exist");
    };
    let userid = unsafe { pgrx::pg_sys::GetUserId() };
    let allowed = unsafe {
        match access {
            IndexAccess::SampledValues => pgrx::pg_sys::has_privs_of_role(
                userid,
                Oid::from(pgrx::pg_sys::ROLE_PG_READ_ALL_STATS),
            ),
            IndexAccess::Prewarm => {
                pgrx::pg_sys::has_privs_of_role(userid, heap.relowner)
                    || pgrx::pg_sys::pg_class_aclcheck(
                        heaprelid,
                        userid,
                        pgrx::pg_sys::ACL_SELECT as pgrx::pg_sys::AclMode,
                    ) == pgrx::pg_sys::AclResult::ACLCHECK_OK
            }
        }
    };
    if !allowed {
        if access == IndexAccess::SampledValues {
            pgrx::ereport!(
                pgrx::PgLogLevel::ERROR,
                pgrx::PgSqlErrorCode::ERRCODE_INSUFFICIENT_PRIVILEGE,
                "permission denied to read sampled queries",
                "Membership in pg_read_all_stats is required."
            );
        }
        let relname = CString::new(heap.relname).unwrap_or_default();
        unsafe {
            pgrx::pg_sys::aclcheck_error(
                pgrx::pg_sys::AclResult::ACLCHECK_NO_PRIV,
                pgrx::pg_sys::ObjectType::OBJECT_TABLE,
                relname.as_ptr(),
            );
        }
        unreachable!("aclcheck_error returned");
    }
    heaprelid
}

/// Opens the index after the access check, and verifies that the index was
/// not dropped and replaced between the check and the lock acquisition.
fn open_checked(indexrelid: Oid, access: IndexAccess) -> Index {
    let heaprelid = check_index_access(indexrelid, access);
    let relation = Index::open(indexrelid, pgrx::pg_sys::AccessShareLock as _);
    if unsafe { pgrx::pg_sys::IndexGetRelation(indexrelid, true) } != heaprelid {
        pgrx::error!("the relation does not exist");
    }
    relation
}

#[pgrx::pg_extern(sql = "")]
fn _vchordg_prewarm(indexrelid: Oid) -> String {
    validate_index(indexrelid, c"vchordg", "vchordg");
    let relation = open_checked(indexrelid, IndexAccess::Prewarm);
    let opfamily = unsafe { crate::index::vchordg::opclass::opfamily(relation.raw()) };
    let index = unsafe { PostgresRelation::new(relation.raw()) };
    crate::index::vchordg::dispatch::prewarm(opfamily, &index)
}

#[pgrx::pg_extern(sql = "")]
fn _vchordrq_prewarm(indexrelid: Oid, height: i32) -> String {
    validate_index(indexrelid, c"vchordrq", "vchordrq");
    let relation = open_checked(indexrelid, IndexAccess::Prewarm);
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
    // Samples may include queries issued by other users under different RLS
    // policies. Enforce this in C as well as in the SQL view, including upgrades.
    let _relation = open_checked(indexrelid, IndexAccess::SampledValues);
    let queries = dump(indexrelid.to_u32());
    SetOfIterator::new(queries)
}
