// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

pub(super) unsafe fn usable_path(path: *mut pgrx::pg_sys::IndexPath) -> bool {
    use pgrx::pg_sys::{self, IndexClause, NodeTag, NullTest, NullTestType, RestrictInfo};
    unsafe {
        let index = (*path).indexinfo;
        if (*path).path.pathtype != NodeTag::T_IndexScan {
            return false;
        }
        if (*path)
            .indexorderbys
            .as_ref()
            .is_some_and(|list| list.length > 1)
        {
            (*path).indexorderbys = pg_sys::list_copy_head((*path).indexorderbys, 1);
            (*path).indexorderbycols = pg_sys::list_copy_head((*path).indexorderbycols, 1);
            (*path).path.pathkeys = pg_sys::list_copy_head((*path).path.pathkeys, 1);
        }
        let base_quals = (*(*index).rel).baserestrictinfo;
        let join_quals = (*path)
            .path
            .param_info
            .as_ref()
            .map_or(std::ptr::null_mut(), |p| p.ppi_clauses);
        let quals = [base_quals, join_quals, (*index).indpred];
        if quals.iter().any(|q| !q.is_null()) {
            let mut test = NullTest::default();
            test.xpr.type_ = NodeTag::T_NullTest;
            let column = pgrx::PgList::<pg_sys::TargetEntry>::from_pg((*index).indextlist)
                .get_ptr(0)
                .unwrap();
            test.arg = (*column).expr;
            test.nulltesttype = NullTestType::IS_NULL;
            let mut predicate = pgrx::PgList::new();
            predicate.push(&raw mut test);
            if quals
                .into_iter()
                .any(|q| !q.is_null() && pg_sys::predicate_implied_by(predicate.as_ptr(), q, false))
            {
                return false;
            }
        }
        !(*path).indexorderbys.is_null()
            || pgrx::PgList::<IndexClause>::from_pg((*path).indexclauses)
                .iter_ptr()
                .any(|clause| {
                    (*clause).indexcol == 0
                        && pgrx::PgList::<RestrictInfo>::from_pg((*clause).indexquals)
                            .iter_ptr()
                            .any(|qual| (*(*qual).clause).type_ == NodeTag::T_OpExpr)
                })
    }
}

static mut PREV_SET_REL_PATHLIST: pgrx::pg_sys::set_rel_pathlist_hook_type = None;

#[pgrx::pg_guard]
unsafe extern "C-unwind" fn set_rel_pathlist(
    root: *mut pgrx::pg_sys::PlannerInfo,
    rel: *mut pgrx::pg_sys::RelOptInfo,
    rti: pgrx::pg_sys::Index,
    rte: *mut pgrx::pg_sys::RangeTblEntry,
) {
    use pgrx::pg_sys;
    unsafe {
        let paths = pgrx::PgList::<pg_sys::Path>::from_pg((*rel).pathlist);
        let mut removed = false;
        const ESTIMATORS: [pg_sys::amcostestimate_function; 2] = [
            Some(super::vchordrq::am::amcostestimate),
            Some(super::vchordg::am::amcostestimate),
        ];
        for pos in (0..paths.len()).rev() {
            let path = paths.get_ptr(pos).unwrap();
            if (*path).type_ != pg_sys::NodeTag::T_IndexPath {
                continue;
            }
            let index_path = path.cast::<pg_sys::IndexPath>();
            if (*index_path).indextotalcost != super::UNUSABLE_PATH_COST {
                continue;
            }
            if let Some(cost) = (*(*index_path).indexinfo).amcostestimate
                && ESTIMATORS
                    .into_iter()
                    .flatten()
                    .any(|am| std::ptr::fn_addr_eq(cost, am))
            {
                (*rel).pathlist = pg_sys::list_delete_nth_cell((*rel).pathlist, pos as _);
                removed = true;
            }
        }
        if removed
            && !pgrx::PgList::<pg_sys::Path>::from_pg((*rel).pathlist)
                .iter_ptr()
                .any(|p| (*p).pathtype == pg_sys::NodeTag::T_SeqScan)
        {
            let seq = pg_sys::create_seqscan_path(root, rel, (*rel).lateral_relids, 0);
            pg_sys::add_path(rel, seq);
        }
        if let Some(prev) = PREV_SET_REL_PATHLIST {
            #[allow(ffi_unwind_calls, reason = "protected by pg_guard_ffi_boundary")]
            pg_sys::ffi::pg_guard_ffi_boundary(|| prev(root, rel, rti, rte));
        }
    }
}

#[pgrx::pg_guard]
unsafe extern "C-unwind" fn rewrite_plan_state(
    node: *mut pgrx::pg_sys::PlanState,
    context: *mut core::ffi::c_void,
) -> bool {
    unsafe fn dirty_check_vchordg(index_relation: *mut pgrx::pg_sys::RelationData) -> Option<bool> {
        type FnPtr = unsafe extern "C-unwind" fn(
            *mut pgrx::pg_sys::RelationData,
            i32,
            i32,
        ) -> *mut pgrx::pg_sys::IndexScanDescData;
        unsafe {
            let index_relation = index_relation.as_ref()?;
            let indam = index_relation.rd_indam.as_ref()?;
            let ambeginscan = indam.ambeginscan.as_ref()?;
            Some(core::ptr::fn_addr_eq::<FnPtr, FnPtr>(
                *ambeginscan,
                crate::index::vchordg::am::ambeginscan,
            ))
        }
    }

    unsafe fn dirty_check_vchordrq(
        index_relation: *mut pgrx::pg_sys::RelationData,
    ) -> Option<bool> {
        type FnPtr = unsafe extern "C-unwind" fn(
            *mut pgrx::pg_sys::RelationData,
            i32,
            i32,
        ) -> *mut pgrx::pg_sys::IndexScanDescData;
        unsafe {
            let index_relation = index_relation.as_ref()?;
            let indam = index_relation.rd_indam.as_ref()?;
            let ambeginscan = indam.ambeginscan.as_ref()?;
            Some(core::ptr::fn_addr_eq::<FnPtr, FnPtr>(
                *ambeginscan,
                crate::index::vchordrq::am::ambeginscan,
            ))
        }
    }

    unsafe {
        if (*node).type_ == pgrx::pg_sys::NodeTag::T_IndexScanState {
            let node = node as *mut pgrx::pg_sys::IndexScanState;
            let index_relation = (*node).iss_RelationDesc;
            if (*node).iss_ScanDesc.is_null() {
                if Some(true) == dirty_check_vchordg(index_relation) {
                    use crate::index::vchordg::am::Scanner;

                    // `standard_ExecutorStart` has switched back to the caller's
                    // context. Begin the scan in the query's context, as
                    // `IndexNext` would; otherwise the Scanner lives as long as
                    // the caller's context (for an SQL function on PG14-17, the
                    // whole outer query) (#25).
                    let _switch =
                        crate::index::MemoryContextSwitch::new((*(*node).ss.ps.state).es_query_cxt);
                    #[cfg(feature = "pg18")]
                    let instrument = if (*(*node).ss.ps.state).es_instrument != 0 {
                        &raw mut (*node).iss_Instrument
                    } else {
                        std::ptr::null_mut()
                    };
                    #[cfg(feature = "pg19")]
                    let flags: u32 = if pgrx::pg_sys::ScanRelIsReadOnly(&raw mut (*node).ss) {
                        pgrx::pg_sys::ScanOptions::SO_HINT_REL_READ_ONLY as _
                    } else {
                        pgrx::pg_sys::ScanOptions::SO_NONE as _
                    };
                    (*node).iss_ScanDesc = pgrx::pg_sys::index_beginscan(
                        (*node).ss.ss_currentRelation,
                        (*node).iss_RelationDesc,
                        (*(*node).ss.ps.state).es_snapshot,
                        #[cfg(feature = "pg18")]
                        instrument,
                        #[cfg(feature = "pg19")]
                        (*node).iss_Instrument,
                        (*node).iss_NumScanKeys,
                        (*node).iss_NumOrderByKeys,
                        #[cfg(feature = "pg19")]
                        flags,
                    );

                    let scanner = &mut *((*(*node).iss_ScanDesc).opaque as *mut Scanner);
                    scanner.hack = std::ptr::NonNull::new(node);

                    if (*node).iss_NumRuntimeKeys == 0 || (*node).iss_RuntimeKeysReady {
                        pgrx::pg_sys::index_rescan(
                            (*node).iss_ScanDesc,
                            (*node).iss_ScanKeys,
                            (*node).iss_NumScanKeys,
                            (*node).iss_OrderByKeys,
                            (*node).iss_NumOrderByKeys,
                        );
                    }
                }
                if Some(true) == dirty_check_vchordrq(index_relation) {
                    use crate::index::vchordrq::am::Scanner;

                    // `standard_ExecutorStart` has switched back to the caller's
                    // context. Begin the scan in the query's context, as
                    // `IndexNext` would; otherwise the Scanner lives as long as
                    // the caller's context (for an SQL function on PG14-17, the
                    // whole outer query) (#25).
                    let _switch =
                        crate::index::MemoryContextSwitch::new((*(*node).ss.ps.state).es_query_cxt);
                    #[cfg(feature = "pg18")]
                    let instrument = if (*(*node).ss.ps.state).es_instrument != 0 {
                        &raw mut (*node).iss_Instrument
                    } else {
                        std::ptr::null_mut()
                    };
                    #[cfg(feature = "pg19")]
                    let flags: u32 = if pgrx::pg_sys::ScanRelIsReadOnly(&raw mut (*node).ss) {
                        pgrx::pg_sys::ScanOptions::SO_HINT_REL_READ_ONLY as _
                    } else {
                        pgrx::pg_sys::ScanOptions::SO_NONE as _
                    };
                    (*node).iss_ScanDesc = pgrx::pg_sys::index_beginscan(
                        (*node).ss.ss_currentRelation,
                        (*node).iss_RelationDesc,
                        (*(*node).ss.ps.state).es_snapshot,
                        #[cfg(feature = "pg18")]
                        instrument,
                        #[cfg(feature = "pg19")]
                        (*node).iss_Instrument,
                        (*node).iss_NumScanKeys,
                        (*node).iss_NumOrderByKeys,
                        #[cfg(feature = "pg19")]
                        flags,
                    );

                    let scanner = &mut *((*(*node).iss_ScanDesc).opaque as *mut Scanner);
                    scanner.hack = std::ptr::NonNull::new(node);

                    if (*node).iss_NumRuntimeKeys == 0 || (*node).iss_RuntimeKeysReady {
                        pgrx::pg_sys::index_rescan(
                            (*node).iss_ScanDesc,
                            (*node).iss_ScanKeys,
                            (*node).iss_NumScanKeys,
                            (*node).iss_OrderByKeys,
                            (*node).iss_NumOrderByKeys,
                        );
                    }
                }
            }
        }
        pgrx::pg_sys::planstate_tree_walker(node, Some(rewrite_plan_state), context)
    }
}

static mut PREV_EXECUTOR_START: pgrx::pg_sys::ExecutorStart_hook_type = None;

#[pgrx::pg_guard]
unsafe extern "C-unwind" fn executor_start(
    query_desc: *mut pgrx::pg_sys::QueryDesc,
    eflags: core::ffi::c_int,
) {
    unsafe {
        use core::ptr::null_mut;
        use pgrx::pg_sys::submodules::ffi::pg_guard_ffi_boundary;
        if let Some(prev_executor_start) = PREV_EXECUTOR_START {
            #[allow(ffi_unwind_calls, reason = "protected by pg_guard_ffi_boundary")]
            pg_guard_ffi_boundary(|| prev_executor_start(query_desc, eflags))
        } else {
            pgrx::pg_sys::standard_ExecutorStart(query_desc, eflags)
        }
        if eflags & pgrx::pg_sys::EXEC_FLAG_EXPLAIN_ONLY as core::ffi::c_int == 0 {
            pg_guard_ffi_boundary(|| rewrite_plan_state((*query_desc).planstate, null_mut()));
        }
    }
}

pub fn init() {
    assert!(crate::is_main());
    unsafe {
        PREV_SET_REL_PATHLIST = pgrx::pg_sys::set_rel_pathlist_hook;
        pgrx::pg_sys::set_rel_pathlist_hook = Some(set_rel_pathlist);
        PREV_EXECUTOR_START = pgrx::pg_sys::ExecutorStart_hook;
        pgrx::pg_sys::ExecutorStart_hook = Some(executor_start);
    }
}
