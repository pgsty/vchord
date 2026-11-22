// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

//! The rewrite of a finished plan: each vchordrq index scan goes under the scan node, whose
//! distance column replaces `col <op> q` above it, and which learns the nearest Limit.

use super::{SCAN_METHODS, cells, nth};
use pgrx::pg_sys::{self, AttrNumber, List, Node, NodeTag, Oid, Plan};
use std::ffi::{c_int, c_void};
use std::ptr::null_mut;
use vchordrq::KKind;

pub unsafe fn rewrite(stmt: *mut pg_sys::PlannedStmt) {
    unsafe {
        let mut rewrite = Rewrite {
            stmt,
            next_id: None,
            // EvalPlanQual runs under these only; it computes the distance by the expression.
            epq: (*stmt).commandType != pg_sys::CmdType::CMD_SELECT
                || (*stmt).hasModifyingCTE
                || !(*stmt).rowMarks.is_null(),
        };
        rewrite.fix(&raw mut (*stmt).planTree, false, None);
        for cell in cells((*stmt).subplans) {
            rewrite.fix((&raw mut cell.ptr_value).cast(), false, None);
        }
    }
}

unsafe fn is_var(node: *mut Node, varno: c_int, attno: AttrNumber) -> bool {
    unsafe {
        (*node).type_ == NodeTag::T_Var && {
            let var = node.cast::<pg_sys::Var>();
            (*var).varno == varno && (*var).varattno == attno && (*var).varlevelsup == 0
        }
    }
}

unsafe fn push_entry(list: &mut *mut List, expr: *mut Node, junk: bool) -> AttrNumber {
    unsafe {
        let resno = pg_sys::list_length(*list) as AttrNumber + 1;
        let entry = pg_sys::makeTargetEntry(expr.cast(), resno, null_mut(), junk);
        *list = pg_sys::lappend(*list, entry.cast());
        resno
    }
}

unsafe fn distance_var(varno: c_int, attno: AttrNumber, ty: Oid) -> *mut Node {
    unsafe { pg_sys::makeVar(varno, attno, ty, -1, pg_sys::InvalidOid, 0).cast() }
}

/// The nearest Limit above a node, and how the node's rows reach it.
#[derive(Clone, Copy)]
struct Limit {
    limit: *mut pg_sys::Limit,
    kind: KKind,
}

impl Limit {
    /// A Limit whose count the scan node can evaluate again.
    unsafe fn of(plan: *mut Plan) -> Option<Self> {
        unsafe {
            let limit = plan.cast::<pg_sys::Limit>();
            let (count, offset) = ((*limit).limitCount, (*limit).limitOffset);
            if count.is_null()
                || pg_sys::contain_volatile_functions(count)
                || pg_sys::contain_volatile_functions(offset)
            {
                return None;
            }
            let ties = (*limit).limitOption == pg_sys::LimitOption::LIMIT_OPTION_WITH_TIES;
            Some(Self {
                limit,
                // it reads on while rows tie with the last one
                kind: if ties { KKind::AtLeast } else { KKind::Exact },
            })
        }
    }

    fn and(self, kind: KKind) -> Option<Self> {
        let kind = self.kind.and(kind);
        (kind != KKind::Unknown).then_some(Self { kind, ..self })
    }
}

/// How a join's outer rows reach its output: dropped (inner and semi joins), multiplied (left
/// joins to several rows) or neither.
unsafe fn through_join(plan: *mut Plan) -> KKind {
    use pg_sys::JoinType::*;
    unsafe {
        let join = plan.cast::<pg_sys::Join>();
        let kind = match ((*join).jointype, (*join).inner_unique) {
            (JOIN_INNER, true) | (JOIN_SEMI | JOIN_ANTI, _) => KKind::AtLeast,
            (JOIN_LEFT, true) => KKind::Exact,
            (JOIN_LEFT, false) => KKind::AtMost,
            _ => KKind::Unknown,
        };
        // with several batches, it reads outer rows it returns only after the Limit may stop
        if (*plan).type_ == NodeTag::T_HashJoin {
            kind.and(KKind::AtLeast)
        } else {
            kind
        }
    }
}

/// What a node's output carries of a wrapped scan below it.
#[derive(Clone, Copy)]
struct Up {
    /// the output column holding the scan's ordering column
    vec: Option<AttrNumber>,
    /// the output column holding the scan's distance
    dist: Option<AttrNumber>,
    ordering: Ordering,
}

#[derive(Clone, Copy)]
struct Ordering {
    column: *mut pg_sys::Var,
    op: Oid,
    query: *mut Node,
    expr: *mut Node,
    ty: Oid,
}

impl Ordering {
    /// The scan's `col <op> q` ORDER BY, unless the column is an array (maxsim).
    unsafe fn of(scan: *mut pg_sys::IndexScan) -> Option<Self> {
        unsafe {
            let list = (*scan).indexorderbyorig;
            if pg_sys::list_length(list) != 1 {
                return None;
            }
            let expr = nth(list, 0);
            if (*expr).type_ != NodeTag::T_OpExpr {
                return None;
            }
            let op = expr.cast::<pg_sys::OpExpr>();
            let column = nth((*op).args, 0);
            if (*column).type_ != NodeTag::T_Var {
                return None;
            }
            let column = column.cast::<pg_sys::Var>();
            let array = pg_sys::get_element_type((*column).vartype) != pg_sys::InvalidOid;
            ((*column).varno == (*scan).scan.scanrelid as c_int && !array).then(|| Self {
                column,
                op: (*op).opno,
                query: nth((*op).args, 1),
                expr,
                ty: pg_sys::exprType(expr),
            })
        }
    }
}

struct Rewrite {
    stmt: *mut pg_sys::PlannedStmt,
    next_id: Option<c_int>,
    epq: bool,
}

impl Rewrite {
    /// Wraps the vchordrq index scans under `*slot`, takes their distance in the nodes between,
    /// and with `want` has `*slot` output it too; `limit` is how `*slot`'s rows reach a Limit.
    unsafe fn fix(&mut self, slot: *mut *mut Plan, want: bool, limit: Option<Limit>) -> Option<Up> {
        use NodeTag::*;
        unsafe {
            let plan = *slot;
            if plan.is_null() {
                return None;
            }
            // rows pass a node as `kind` but for its qual
            let qual = !(*plan).qual.is_null();
            let pass = |kind: KKind| {
                let kind = if qual { kind.and(KKind::AtLeast) } else { kind };
                limit.and_then(|l| l.and(kind))
            };
            match (*plan).type_ {
                T_IndexScan => self.wrap(slot, want, limit),
                T_NestLoop | T_MergeJoin | T_HashJoin => {
                    let outer = self.fix(&raw mut (*plan).lefttree, true, pass(through_join(plan)));
                    let inner = self.fix(&raw mut (*plan).righttree, true, None);
                    let mut up = None;
                    for (varno, side) in [(pg_sys::OUTER_VAR, outer), (pg_sys::INNER_VAR, inner)] {
                        let Some(side) = side else { continue };
                        Subst::above(varno, side).apply((*plan.cast::<pg_sys::Join>()).joinqual);
                        let side = project(plan, varno, side, want && up.is_none());
                        up = up.or(side);
                    }
                    up
                }
                T_Result => {
                    let side = self.fix(&raw mut (*plan).lefttree, true, pass(KKind::Exact))?;
                    project(plan, pg_sys::OUTER_VAR, side, want)
                }
                T_SubqueryScan => {
                    let scan = plan.cast::<pg_sys::SubqueryScan>();
                    let side = self.fix(&raw mut (*scan).subplan, true, pass(KKind::Exact))?;
                    project(plan, (*scan).scan.scanrelid as c_int, side, want)
                }
                T_Limit => {
                    let side = self.fix(&raw mut (*plan).lefttree, want, Limit::of(plan))?;
                    mirror(plan, side)
                }
                // they pass rows on as they come; an incremental sort reads its last group whole
                T_IncrementalSort | T_Unique | T_LockRows => {
                    let side = self.fix(&raw mut (*plan).lefttree, want, pass(KKind::AtLeast))?;
                    mirror(plan, side)
                }
                // they read all rows, or replay them on a rescan without rescanning below
                T_Sort | T_Material | T_Memoize | T_Hash => {
                    let side = self.fix(&raw mut (*plan).lefttree, want, None)?;
                    mirror(plan, side)
                }
                // each input is a scan of its own; the distance stays in the column it already had
                T_Append | T_MergeAppend => {
                    let children = if (*plan).type_ == T_Append {
                        (*plan.cast::<pg_sys::Append>()).appendplans
                    } else {
                        (*plan.cast::<pg_sys::MergeAppend>()).mergeplans
                    };
                    for cell in cells(children) {
                        self.fix((&raw mut cell.ptr_value).cast(), false, pass(KKind::AtMost));
                    }
                    None
                }
                // parallel workers run the plan below; the Limit's count may need what they lack
                T_Gather | T_GatherMerge => {
                    let side = self.fix(&raw mut (*plan).lefttree, true, None)?;
                    project(plan, pg_sys::OUTER_VAR, side, want)
                }
                _ => {
                    self.fix(&raw mut (*plan).lefttree, false, None);
                    self.fix(&raw mut (*plan).righttree, false, None);
                    None
                }
            }
        }
    }

    unsafe fn next_id(&mut self) -> c_int {
        unsafe {
            let id = self.next_id.get_or_insert_with(|| max_id(self.stmt) + 1);
            *id += 1;
            *id - 1
        }
    }

    unsafe fn wrap(
        &mut self,
        slot: *mut *mut Plan,
        want: bool,
        limit: Option<Limit>,
    ) -> Option<Up> {
        unsafe {
            let scan = (*slot).cast::<pg_sys::IndexScan>();
            if !is_vchordrq((*scan).indexid) {
                return None;
            }
            let ordering = Ordering::of(scan);
            if ordering.is_none() && !(*scan).indexorderbyorig.is_null() {
                return None;
            }
            let rti = (*scan).scan.scanrelid as c_int;
            let mut subst = Subst {
                varno: rti,
                vec: ordering.map_or(0, |o| (*o.column).varattno),
                op: ordering.map_or(pg_sys::InvalidOid, |o| o.op),
                query: ordering.map_or(null_mut(), |o| o.query),
                to: (pg_sys::INDEX_VAR, 0),
                ty: ordering.map_or(pg_sys::InvalidOid, |o| o.ty),
                columns: Vec::new(),
            };
            // The scan node's tuple is the columns that the target list reads but for the
            // distance, then the distance.
            let tlist = (*scan).scan.plan.targetlist;
            collect(tlist.cast(), (&raw mut subst).cast());
            let epq = ordering.filter(|_| self.epq);
            if let Some(o) = epq {
                subst.add(o.column);
            }
            let n = subst.columns.len() as AttrNumber;
            subst.to.1 = n + 1;
            let copy = |node: *mut Node| pg_sys::copyObjectImpl(node.cast()).cast::<Node>();
            let (mut child_tlist, mut scan_tlist) = (null_mut(), null_mut());
            for &var in &subst.columns {
                push_entry(&mut child_tlist, copy(var.cast()), false);
                push_entry(&mut scan_tlist, copy(var.cast()), false);
            }
            if let Some(o) = ordering {
                push_entry(&mut scan_tlist, copy(o.expr), false);
            }
            let tlist = subst.mutate(tlist.cast()).cast::<List>();
            let distance = epq.map_or(null_mut(), |o| {
                let op = std::mem::replace(&mut subst.op, pg_sys::InvalidOid);
                let distance = subst.mutate(o.expr);
                subst.op = op;
                distance
            });
            let (count, offset, kind, ties) = match limit {
                Some(Limit { limit, kind }) => {
                    let ties = (*limit).limitOption == pg_sys::LimitOption::LIMIT_OPTION_WITH_TIES;
                    ((*limit).limitCount, (*limit).limitOffset, kind, ties)
                }
                None => (null_mut(), null_mut(), KKind::Unknown, false),
            };
            let custom =
                pg_sys::palloc0(size_of::<pg_sys::CustomScan>()).cast::<pg_sys::CustomScan>();
            (*custom).scan.plan = (*scan).scan.plan;
            (*custom).scan.plan.type_ = NodeTag::T_CustomScan;
            (*custom).scan.plan.plan_node_id = self.next_id();
            (*custom).scan.plan.targetlist = tlist;
            (*custom).scan.plan.qual = null_mut();
            (*custom).scan.plan.initPlan = null_mut();
            (*custom).scan.plan.extParam = pg_sys::bms_copy((*scan).scan.plan.extParam);
            (*custom).scan.plan.allParam = pg_sys::bms_copy((*scan).scan.plan.allParam);
            (*custom).scan.scanrelid = 0;
            (*custom).custom_plans = pg_sys::lappend(null_mut(), scan.cast());
            for expr in [count, offset, distance] {
                (*custom).custom_exprs = pg_sys::lappend((*custom).custom_exprs, copy(expr).cast());
            }
            for value in [kind as c_int, ties as c_int] {
                let value = pg_sys::makeInteger(value).cast();
                (*custom).custom_private = pg_sys::lappend((*custom).custom_private, value);
            }
            (*custom).custom_scan_tlist = scan_tlist;
            (*custom).custom_relids = pg_sys::bms_make_singleton(rti);
            (*custom).methods = &raw const SCAN_METHODS;
            (*scan).scan.plan.targetlist = child_tlist;
            *slot = custom.cast();
            let ordering = ordering?;
            let at = subst.position((*ordering.column).varattno);
            let vec = at.and_then(|at| find(tlist, pg_sys::INDEX_VAR, at));
            let tlist = &mut (*custom).scan.plan.targetlist;
            let dist = want.then(|| output(tlist, pg_sys::INDEX_VAR, n + 1, ordering.ty));
            Some(Up {
                vec,
                dist,
                ordering,
            })
        }
    }
}

unsafe fn is_vchordrq(index: Oid) -> bool {
    unsafe {
        let relation = pg_sys::RelationIdGetRelation(index);
        let vchordrq = crate::index::hook::dirty_check_vchordrq(relation) == Some(true);
        pg_sys::RelationClose(relation);
        vchordrq
    }
}

/// For a node that returns its input's tuples, its target list mirrors the input's.
unsafe fn mirror(plan: *mut Plan, side: Up) -> Option<Up> {
    unsafe {
        let Some(dist) = side.dist else {
            return Some(side);
        };
        let len = pg_sys::list_length((*plan).targetlist) as AttrNumber;
        if dist == len + 1 {
            let var = distance_var(pg_sys::OUTER_VAR, dist, side.ordering.ty);
            push_entry(&mut (*plan).targetlist, var, true);
        }
        (dist <= len + 1).then_some(side)
    }
}

/// The resno of `Var(varno, attno)` in the target list, appended as junk if it is not there.
unsafe fn output(tlist: &mut *mut List, varno: c_int, attno: AttrNumber, ty: Oid) -> AttrNumber {
    unsafe {
        find(*tlist, varno, attno)
            .unwrap_or_else(|| push_entry(tlist, distance_var(varno, attno, ty), true))
    }
}

/// Takes the distance of `side` (Vars `varno`) in `plan`, and with `want` outputs it.
unsafe fn project(plan: *mut Plan, varno: c_int, side: Up, want: bool) -> Option<Up> {
    unsafe {
        let subst = Subst::above(varno, side);
        subst.apply((*plan).targetlist);
        subst.apply((*plan).qual);
        let dist = side.dist.filter(|_| want)?;
        let vec = side
            .vec
            .and_then(|vec| find((*plan).targetlist, varno, vec));
        let dist = output(&mut (*plan).targetlist, varno, dist, side.ordering.ty);
        Some(Up {
            vec,
            dist: Some(dist),
            ..side
        })
    }
}

/// The resno of the target list entry that is `Var(varno, attno)`.
unsafe fn find(tlist: *mut List, varno: c_int, attno: AttrNumber) -> Option<AttrNumber> {
    unsafe {
        cells(tlist).iter().find_map(|cell| {
            let entry = cell.ptr_value.cast::<pg_sys::TargetEntry>();
            is_var((*entry).expr.cast(), varno, attno).then_some((*entry).resno)
        })
    }
}

/// The largest `plan_node_id` of the statement.
unsafe fn max_id(stmt: *mut pg_sys::PlannedStmt) -> c_int {
    unsafe fn walk(plan: *mut Plan) -> c_int {
        use NodeTag::*;
        unsafe {
            if plan.is_null() {
                return -1;
            }
            let children = match (*plan).type_ {
                T_SubqueryScan => {
                    let subplan = (*plan.cast::<pg_sys::SubqueryScan>()).subplan;
                    return (*plan).plan_node_id.max(walk(subplan));
                }
                T_Append => (*plan.cast::<pg_sys::Append>()).appendplans,
                T_MergeAppend => (*plan.cast::<pg_sys::MergeAppend>()).mergeplans,
                T_BitmapAnd => (*plan.cast::<pg_sys::BitmapAnd>()).bitmapplans,
                T_BitmapOr => (*plan.cast::<pg_sys::BitmapOr>()).bitmapplans,
                T_CustomScan => (*plan.cast::<pg_sys::CustomScan>()).custom_plans,
                _ => null_mut(),
            };
            let id = (*plan).plan_node_id;
            let id = id.max(walk((*plan).lefttree)).max(walk((*plan).righttree));
            cells(children)
                .iter()
                .fold(id, |id, cell| id.max(walk(cell.ptr_value.cast())))
        }
    }
    unsafe {
        let id = walk((*stmt).planTree);
        cells((*stmt).subplans)
            .iter()
            .fold(id, |id, cell| id.max(walk(cell.ptr_value.cast())))
    }
}

/// Replaces `Var(varno, vec) <op> query` by `Var(to)`; with `columns`, also renumbers the Vars
/// of `varno` to their place in `columns` under `to.0`.
struct Subst {
    varno: c_int,
    vec: AttrNumber,
    op: Oid,
    query: *mut Node,
    to: (c_int, AttrNumber),
    ty: Oid,
    columns: Vec<*mut pg_sys::Var>,
}

impl Subst {
    fn above(varno: c_int, side: Up) -> Self {
        let traced = side.vec.zip(side.dist);
        Self {
            varno,
            vec: traced.map_or(0, |(vec, _)| vec),
            op: traced.map_or(pg_sys::InvalidOid, |_| side.ordering.op),
            query: side.ordering.query,
            to: (varno, traced.map_or(0, |(_, dist)| dist)),
            ty: side.ordering.ty,
            columns: Vec::new(),
        }
    }

    /// The place of column `attno` in `columns`, from one.
    unsafe fn position(&self, attno: AttrNumber) -> Option<AttrNumber> {
        unsafe {
            let at = self.columns.iter().position(|&v| (*v).varattno == attno)?;
            Some(at as AttrNumber + 1)
        }
    }

    unsafe fn add(&mut self, var: *mut pg_sys::Var) {
        unsafe {
            if self.position((*var).varattno).is_none() {
                self.columns.push(var);
            }
        }
    }

    unsafe fn matches(&self, node: *mut Node) -> bool {
        unsafe {
            if self.op == pg_sys::InvalidOid || (*node).type_ != NodeTag::T_OpExpr {
                return false;
            }
            let e = node.cast::<pg_sys::OpExpr>();
            (*e).opno == self.op
                && is_var(nth((*e).args, 0), self.varno, self.vec)
                && pg_sys::equal(nth((*e).args, 1).cast(), self.query.cast())
        }
    }

    fn context(&self) -> *mut c_void {
        (self as *const Self).cast_mut().cast()
    }

    unsafe fn mutate(&self, node: *mut Node) -> *mut Node {
        unsafe { subst(node, self.context()) }
    }

    /// Replaces the distance in a target list or qual, in place.
    unsafe fn apply(&self, list: *mut List) {
        use NodeTag::*;
        unsafe {
            if self.op == pg_sys::InvalidOid {
                return;
            }
            for cell in cells(list) {
                let node = cell.ptr_value.cast::<Node>();
                let expr = if (*node).type_ == T_TargetEntry {
                    (&raw mut (*node.cast::<pg_sys::TargetEntry>()).expr).cast::<*mut Node>()
                } else {
                    (&raw mut cell.ptr_value).cast::<*mut Node>()
                };
                match (**expr).type_ {
                    T_Var | T_Const | T_Param => (),
                    _ if self.matches(*expr) => *expr = distance_var(self.to.0, self.to.1, self.ty),
                    _ if occurs(*expr, self.context()) => *expr = self.mutate(*expr),
                    _ => (),
                }
            }
        }
    }
}

/// Whether the distance occurs in `node`.
#[pgrx::pg_guard]
unsafe extern "C-unwind" fn occurs(node: *mut Node, context: *mut c_void) -> bool {
    unsafe {
        if node.is_null() {
            return false;
        }
        context
            .cast::<Subst>()
            .as_ref()
            .unwrap_unchecked()
            .matches(node)
            || pg_sys::expression_tree_walker(node, Some(occurs), context)
    }
}

/// Collects the columns of `varno` that `node` reads but for the distance.
#[pgrx::pg_guard]
unsafe extern "C-unwind" fn collect(node: *mut Node, context: *mut c_void) -> bool {
    unsafe {
        if node.is_null() {
            return false;
        }
        let cx = context.cast::<Subst>().as_mut().unwrap_unchecked();
        if cx.matches(node) {
            return false;
        }
        if (*node).type_ == NodeTag::T_Var {
            let var = node.cast::<pg_sys::Var>();
            if (*var).varno == cx.varno && (*var).varlevelsup == 0 {
                cx.add(var);
            }
            return false;
        }
        pg_sys::expression_tree_walker(node, Some(collect), context)
    }
}

#[pgrx::pg_guard]
unsafe extern "C-unwind" fn subst(node: *mut Node, context: *mut c_void) -> *mut Node {
    unsafe {
        if node.is_null() {
            return null_mut();
        }
        let cx = context.cast::<Subst>().as_ref().unwrap_unchecked();
        if cx.matches(node) {
            return distance_var(cx.to.0, cx.to.1, cx.ty);
        }
        if (*node).type_ == NodeTag::T_Var {
            let var = node.cast::<pg_sys::Var>();
            if (*var).varno == cx.varno && (*var).varlevelsup == 0 {
                if let Some(at) = cx.position((*var).varattno) {
                    let copy = pg_sys::copyObjectImpl(var.cast()).cast::<pg_sys::Var>();
                    ((*copy).varno, (*copy).varattno) = (cx.to.0, at);
                    return copy.cast();
                }
            }
        }
        pg_sys::expression_tree_mutator(node, Some(subst), context)
    }
}
