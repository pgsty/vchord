// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

//! A custom scan over a vchordrq scan: hands up its distance, owns its sphere, passes it the Limit.

mod plan;

use crate::index::vchordrq::am::Scanner;
use crate::index::vchordrq::scanners::Above;
use pgrx::pg_sys::submodules::ffi::pg_guard_ffi_boundary;
use pgrx::pg_sys::{self, List, Node, Plan, TupleTableSlot};
use std::ffi::{CString, c_char, c_int};
use std::ptr::null_mut;
use vchordrq::{KHint, KKind};

static mut SCAN_METHODS: pg_sys::CustomScanMethods = pg_sys::CustomScanMethods {
    CustomName: c"VchordrqScan".as_ptr(),
    CreateCustomScanState: Some(create_state),
};

/// by `direct`, then by `ordered`, as `begin` chooses for each scan
static mut EXEC_METHODS: [[pg_sys::CustomExecMethods; 2]; 2] = [
    [methods(exec::<false>), methods(exec::<true>)],
    [methods(direct::<false>), methods(direct::<true>)],
];

type Exec = unsafe extern "C-unwind" fn(*mut pg_sys::CustomScanState) -> *mut TupleTableSlot;

const fn methods(exec: Exec) -> pg_sys::CustomExecMethods {
    pg_sys::CustomExecMethods {
        CustomName: c"VchordrqScan".as_ptr(),
        BeginCustomScan: Some(begin),
        ExecCustomScan: Some(exec),
        EndCustomScan: Some(end),
        ReScanCustomScan: Some(rescan),
        MarkPosCustomScan: None,
        RestrPosCustomScan: None,
        EstimateDSMCustomScan: None,
        InitializeDSMCustomScan: None,
        ReInitializeDSMCustomScan: None,
        InitializeWorkerCustomScan: None,
        ShutdownCustomScan: None,
        ExplainCustomScan: Some(explain),
    }
}

static mut PREV_PLANNER: pg_sys::planner_hook_type = None;

pub fn init() {
    unsafe {
        pg_sys::RegisterCustomScanMethods(&raw const SCAN_METHODS);
        PREV_PLANNER = pg_sys::planner_hook;
        pg_sys::planner_hook = Some(planner);
    }
}

// pg_guard passes the arguments by its own span: the macro names them, including PG 19's `es`
macro_rules! planner {
    ($($explain:ty)?) => {
        #[pgrx::pg_guard]
        unsafe extern "C-unwind" fn planner(
            parse: *mut pg_sys::Query,
            query_string: *const c_char,
            cursor_options: c_int,
            bound_params: pg_sys::ParamListInfo,
            $(es: $explain,)?
        ) -> *mut pg_sys::PlannedStmt {
            unsafe {
                let (query, options, params) = (query_string, cursor_options, bound_params);
                let stmt = if let Some(prev) = PREV_PLANNER {
                    #[allow(ffi_unwind_calls, reason = "protected by pg_guard_ffi_boundary")]
                    pg_guard_ffi_boundary(|| {
                        prev(parse, query, options, params $(, es as $explain)?)
                    })
                } else {
                    pg_sys::standard_planner(parse, query, options, params $(, es as $explain)?)
                };
                if crate::index::gucs::vchordrq_scan_node() {
                    plan::rewrite(stmt);
                }
                stmt
            }
        }
    };
}

#[cfg(not(feature = "pg19"))]
planner!();
#[cfg(feature = "pg19")]
planner!(*mut pg_sys::ExplainState);

unsafe fn cells<'a>(list: *mut List) -> &'a mut [pg_sys::ListCell] {
    unsafe {
        if list.is_null() {
            &mut []
        } else {
            std::slice::from_raw_parts_mut((*list).elements, (*list).length as usize)
        }
    }
}

unsafe fn nth(list: *mut List, i: usize) -> *mut Node {
    unsafe { cells(list)[i].ptr_value.cast() }
}

// called per row without pgrx's guard: these frames drop nothing, so an error unwinds past them
unsafe extern "C" {
    fn ExecScan(
        node: *mut pg_sys::ScanState,
        access: pg_sys::ExecScanAccessMtd,
        recheck: pg_sys::ExecScanRecheckMtd,
    ) -> *mut TupleTableSlot;
    fn slot_getsomeattrs_int(slot: *mut TupleTableSlot, attnum: c_int);
}

/// `custom_exprs`: the Limit's count, offset, EvalPlanQual's distance; `custom_private`: kind, ties
#[repr(C)]
struct State {
    css: pg_sys::CustomScanState,
    child: *mut pg_sys::IndexScanState,
    columns: usize,
    /// the child's scan has started since the last rescan
    started: bool,
    kind: KKind,
    ties: bool,
    count: *mut pg_sys::ExprState,
    offset: *mut pg_sys::ExprState,
    distance: *mut pg_sys::ExprState,
}

#[pgrx::pg_guard]
unsafe extern "C-unwind" fn create_state(_custom: *mut pg_sys::CustomScan) -> *mut Node {
    unsafe {
        let state = pg_sys::palloc0(size_of::<State>()).cast::<State>();
        (*state).css.ss.ps.type_ = pg_sys::NodeTag::T_CustomScanState;
        (*state).css.methods = &raw const EXEC_METHODS[0][0];
        state.cast()
    }
}

#[pgrx::pg_guard]
unsafe extern "C-unwind" fn begin(
    css: *mut pg_sys::CustomScanState,
    estate: *mut pg_sys::EState,
    eflags: c_int,
) {
    unsafe {
        let state = css.cast::<State>();
        let custom = (*css).ss.ps.plan.cast::<pg_sys::CustomScan>();
        let plan = nth((*custom).custom_plans, 0).cast::<Plan>();
        let child = pg_sys::ExecInitNode(plan, estate, eflags).cast::<pg_sys::IndexScanState>();
        (*css).custom_ps = pg_sys::lappend(null_mut(), child.cast());
        // `WHERE CURRENT OF` finds the scan of a table by its relation
        (*css).ss.ss_currentRelation = (*child).ss.ss_currentRelation;
        let columns = pg_sys::list_length((*plan).targetlist) as usize;
        let epq = !(*estate).es_epq_active.is_null();
        let expr = |i| match nth((*custom).custom_exprs, i) {
            expr if expr.is_null() => null_mut(),
            expr => pg_sys::ExecInitExpr(expr.cast(), &raw mut (*css).ss.ps),
        };
        let private = |i| (*nth((*custom).custom_private, i).cast::<pg_sys::Integer>()).ival;
        (*state).child = child;
        (*state).columns = columns;
        // the scan tuple ends with the distance
        let ordered = pg_sys::list_length((*custom).custom_scan_tlist) as usize > columns;
        // no projection and no EvalPlanQual: rows go up without `ExecScan`
        let direct = (*css).ss.ps.ps_ProjInfo.is_null() && !epq;
        (*state).started = false;
        // the plan holds a kind its rewrite wrote
        (*state).kind = std::mem::transmute::<c_int, KKind>(private(0));
        (*state).ties = private(1) != 0;
        (*state).count = expr(0);
        (*state).offset = expr(1);
        (*state).distance = if epq { expr(2) } else { null_mut() };
        (*css).methods = &raw const EXEC_METHODS[direct as usize][ordered as usize];
        if eflags & pg_sys::EXEC_FLAG_EXPLAIN_ONLY as c_int == 0 {
            let scanner = crate::index::hook::begin_scan(child).cast::<Scanner>();
            (*scanner).hack = std::ptr::NonNull::new(child);
            (*scanner).above = Some(Above::new((*child).iss_ScanDesc));
        }
    }
}

/// Evaluates an int8 expression of the plan; `None` if it is null.
unsafe fn eval(expr: *mut pg_sys::ExprState, econtext: *mut pg_sys::ExprContext) -> Option<i64> {
    unsafe {
        let _switch = crate::index::MemoryContextSwitch::new((*econtext).ecxt_per_tuple_memory);
        let mut isnull = false;
        let evalfunc = (*expr).evalfunc.unwrap_unchecked();
        #[allow(ffi_unwind_calls, reason = "protected by pg_guard_ffi_boundary")]
        let value = pg_guard_ffi_boundary(|| evalfunc(expr, econtext, &mut isnull));
        (!isnull).then_some(value.value() as i64)
    }
}

/// The rows the Limit takes, as it computed them before it asked for the first row.
unsafe fn hint(state: &State) -> KHint {
    unsafe {
        if state.kind == KKind::Unknown {
            return KHint::UNKNOWN;
        }
        let econtext = state.css.ss.ps.ps_ExprContext;
        // the Limit has rejected negative values; it takes a null count as all, a null offset as 0
        let count = eval(state.count, econtext);
        let offset = match state.offset.is_null() {
            true => None,
            false => eval(state.offset, econtext),
        };
        pg_sys::MemoryContextReset((*econtext).ecxt_per_tuple_memory);
        let Some(count) = count else {
            return KHint::UNKNOWN;
        };
        KHint {
            k: count as u64 + offset.unwrap_or(0) as u64 + state.ties as u64,
            kind: state.kind,
        }
    }
}

/// What the child's scan holds of this node, which `begin` gave it.
unsafe fn above<'a>(child: *mut pg_sys::IndexScanState) -> &'a Above {
    unsafe {
        let scanner = (*(*child).iss_ScanDesc).opaque.cast::<Scanner>();
        (*scanner).above.as_ref().unwrap_unchecked()
    }
}

/// Starts the child's scan with the hint, once the Limit above has computed its count.
#[pgrx::pg_guard]
unsafe extern "C-unwind" fn start(state: *mut State) {
    unsafe {
        (*state).started = true;
        let child = (*state).child;
        above(child).k.set(hint(&*state));
        pg_sys::ExecReScan(child.cast());
    }
}

/// The child's next row and its distance into the scan slot.
#[inline(always)]
unsafe fn fetch<const ORDERED: bool>(state: *mut State) -> *mut TupleTableSlot {
    const EMPTY: u16 = pg_sys::TTS_FLAG_EMPTY as u16;
    const SHOULDFREE: u16 = pg_sys::TTS_FLAG_SHOULDFREE as u16;
    unsafe {
        let child = (*state).child;
        let slot = (*state).css.ss.ss_ScanTupleSlot;
        // clearing a virtual slot is setting its flags, unless a parent has materialized it
        if (*slot).tts_flags & SHOULDFREE != 0 {
            #[allow(ffi_unwind_calls, reason = "unguarded: see `ExecScan`")]
            (*(*slot).tts_ops).clear.unwrap_unchecked()(slot);
        }
        (*slot).tts_flags |= EMPTY;
        (*slot).tts_nvalid = 0;
        if !(*state).started {
            start(state);
        }
        #[allow(ffi_unwind_calls, reason = "unguarded: see `ExecScan`")]
        let from = (*child).ss.ps.ExecProcNode.unwrap_unchecked()(child.cast());
        if from.is_null() || (*from).tts_flags & EMPTY != 0 {
            return slot;
        }
        let n = (*state).columns;
        if ((*from).tts_nvalid as usize) < n {
            slot_getsomeattrs_int(from, n as c_int);
        }
        std::ptr::copy_nonoverlapping((*from).tts_values, (*slot).tts_values, n);
        std::ptr::copy_nonoverlapping((*from).tts_isnull, (*slot).tts_isnull, n);
        (*slot).tts_flags &= !EMPTY;
        (*slot).tts_nvalid = (*(*slot).tts_tupleDescriptor).natts as i16;
        // the row's identity, for `WHERE CURRENT OF`
        let scanned = (*child).ss.ss_ScanTupleSlot;
        (*slot).tts_tid = (*scanned).tts_tid;
        (*slot).tts_tableOid = (*scanned).tts_tableOid;
        if ORDERED {
            let scan = (*child).iss_ScanDesc;
            *(*slot).tts_values.add(n) = *(*scan).xs_orderbyvals;
            *(*slot).tts_isnull.add(n) = *(*scan).xs_orderbynulls;
        }
        slot
    }
}

unsafe extern "C-unwind" fn next<const ORDERED: bool>(
    ss: *mut pg_sys::ScanState,
) -> *mut TupleTableSlot {
    unsafe { fetch::<ORDERED>(ss.cast()) }
}

/// EvalPlanQual: the child returns the row to recheck once; its distance is the expression's.
#[pgrx::pg_guard]
unsafe extern "C-unwind" fn recheck<const ORDERED: bool>(
    ss: *mut pg_sys::ScanState,
    _slot: *mut TupleTableSlot,
) -> bool {
    unsafe {
        let state = ss.cast::<State>();
        let slot = fetch::<ORDERED>(state);
        if (*slot).tts_flags & pg_sys::TTS_FLAG_EMPTY as u16 != 0 {
            return false;
        }
        if ORDERED {
            let n = (*state).columns;
            let econtext = (*ss).ps.ps_ExprContext;
            (*econtext).ecxt_scantuple = slot;
            let expr = (*state).distance;
            let mut isnull = false;
            let evalfunc = (*expr).evalfunc.unwrap_unchecked();
            #[allow(ffi_unwind_calls, reason = "protected by pg_guard_ffi_boundary")]
            let value = pg_guard_ffi_boundary(|| evalfunc(expr, econtext, &mut isnull));
            *(*slot).tts_values.add(n) = value;
            *(*slot).tts_isnull.add(n) = isnull;
        }
        true
    }
}

unsafe extern "C-unwind" fn exec<const ORDERED: bool>(
    css: *mut pg_sys::CustomScanState,
) -> *mut TupleTableSlot {
    unsafe {
        ExecScan(
            &raw mut (*css).ss,
            Some(next::<ORDERED>),
            Some(recheck::<ORDERED>),
        )
    }
}

unsafe extern "C-unwind" fn direct<const ORDERED: bool>(
    css: *mut pg_sys::CustomScanState,
) -> *mut TupleTableSlot {
    unsafe { fetch::<ORDERED>(css.cast()) }
}

#[pgrx::pg_guard]
unsafe extern "C-unwind" fn end(css: *mut pg_sys::CustomScanState) {
    unsafe { pg_sys::ExecEndNode((*css.cast::<State>()).child.cast()) }
}

/// The child's rescan waits for the next row, after the Limit has computed its count.
#[pgrx::pg_guard]
unsafe extern "C-unwind" fn rescan(css: *mut pg_sys::CustomScanState) {
    unsafe {
        let state = css.cast::<State>();
        if !(*css).ss.ps.chgParam.is_null() {
            pg_sys::UpdateChangedParamSet((*state).child.cast(), (*css).ss.ps.chgParam);
        }
        (*state).started = false;
    }
}

/// EXPLAIN ANALYZE shows the hint the scan last used, after its own recheck.
#[pgrx::pg_guard]
unsafe extern "C-unwind" fn explain(
    css: *mut pg_sys::CustomScanState,
    _ancestors: *mut List,
    es: *mut pg_sys::ExplainState,
) {
    unsafe {
        // the hint is what the scan used, so ANALYZE shows it; a plain EXPLAIN begins no scan
        if !(*es).analyze {
            return;
        }
        let KHint { k, kind } = above((*css.cast::<State>()).child).k.get();
        let kind = match kind {
            KKind::Unknown => return,
            KKind::Exact => "exact",
            KKind::AtLeast => "at least",
            KKind::AtMost => "at most",
        };
        let hint = CString::new(format!("{kind} {k}")).unwrap();
        pg_sys::ExplainPropertyText(c"Row Hint".as_ptr(), hint.as_ptr(), es);
    }
}
