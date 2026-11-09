// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

use crate::recorder::types::PgRefCell;
use crate::recorder::worker::{delete_database, delete_index};

enum Cleanup {
    Database(u32),
    Index(u32),
}

// SQLite is diagnostic storage, outside PostgreSQL's transaction manager.
// Keep pending deletions backend-local until their PostgreSQL DDL commits.
static PENDING: PgRefCell<Vec<(pgrx::pg_sys::SubTransactionId, Cleanup)>> =
    PgRefCell::new(Vec::new());

static mut PREV_OBJECT_ACCESS: pgrx::pg_sys::object_access_hook_type = None;

#[pgrx::pg_guard]
unsafe extern "C-unwind" fn recorder_object_access(
    access: pgrx::pg_sys::ObjectAccessType::Type,
    class_id: pgrx::pg_sys::Oid,
    object_id: pgrx::pg_sys::Oid,
    sub_id: ::std::os::raw::c_int,
    arg: *mut ::std::os::raw::c_void,
) {
    unsafe {
        use pgrx::pg_sys::submodules::ffi::pg_guard_ffi_boundary;
        if let Some(prev_object_access_hook) = PREV_OBJECT_ACCESS {
            #[allow(ffi_unwind_calls, reason = "protected by pg_guard_ffi_boundary")]
            pg_guard_ffi_boundary(|| {
                prev_object_access_hook(access, class_id, object_id, sub_id, arg)
            });
        }
        if access == pgrx::pg_sys::ObjectAccessType::OAT_DROP
            && class_id == pgrx::pg_sys::DatabaseRelationId
        {
            PENDING.borrow_mut().push((
                pgrx::pg_sys::GetCurrentSubTransactionId(),
                Cleanup::Database(object_id.to_u32()),
            ));
        } else if access == pgrx::pg_sys::ObjectAccessType::OAT_DROP
            && class_id == pgrx::pg_sys::RelationRelationId
            && sub_id == 0
        {
            PENDING.borrow_mut().push((
                pgrx::pg_sys::GetCurrentSubTransactionId(),
                Cleanup::Index(object_id.to_u32()),
            ));
        }
    }
}

#[pgrx::pg_guard]
unsafe extern "C-unwind" fn recorder_xact(
    event: pgrx::pg_sys::XactEvent::Type,
    _arg: *mut std::ffi::c_void,
) {
    use pgrx::pg_sys::XactEvent::*;
    match event {
        XACT_EVENT_COMMIT => {
            let pending = std::mem::take(&mut *PENDING.borrow_mut());
            // Best effort only: no SQL/SPI and no ERROR after commit.
            for (_, cleanup) in pending {
                match cleanup {
                    Cleanup::Database(oid) => delete_database(oid),
                    Cleanup::Index(oid) => delete_index(oid),
                }
            }
        }
        // Do not delete at PREPARE: ROLLBACK PREPARED must retain samples.
        // COMMIT PREPARED may leave unreachable diagnostics until DB cleanup.
        XACT_EVENT_ABORT | XACT_EVENT_PARALLEL_ABORT | XACT_EVENT_PREPARE => {
            PENDING.borrow_mut().clear();
        }
        _ => (),
    }
}

#[pgrx::pg_guard]
unsafe extern "C-unwind" fn recorder_subxact(
    event: pgrx::pg_sys::SubXactEvent::Type,
    subid: pgrx::pg_sys::SubTransactionId,
    parent: pgrx::pg_sys::SubTransactionId,
    _arg: *mut std::ffi::c_void,
) {
    use pgrx::pg_sys::SubXactEvent::*;
    match event {
        SUBXACT_EVENT_ABORT_SUB => PENDING.borrow_mut().retain(|(id, _)| *id != subid),
        SUBXACT_EVENT_COMMIT_SUB => {
            for (id, _) in PENDING.borrow_mut().iter_mut() {
                if *id == subid {
                    *id = parent;
                }
            }
        }
        _ => (),
    }
}

pub fn init() {
    assert!(crate::is_main());
    unsafe {
        PREV_OBJECT_ACCESS = pgrx::pg_sys::object_access_hook;
        pgrx::pg_sys::object_access_hook = Some(recorder_object_access);
        pgrx::pg_sys::RegisterXactCallback(Some(recorder_xact), std::ptr::null_mut());
        pgrx::pg_sys::RegisterSubXactCallback(Some(recorder_subxact), std::ptr::null_mut());
    }
}
