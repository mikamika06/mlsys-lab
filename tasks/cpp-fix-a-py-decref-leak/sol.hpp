#pragma once

// ============================================================================
// Fixed simplified PyObject-like record and its ref-counting ops (FIXED —
// do not modify these three definitions; mirrors Py_INCREF/Py_DECREF).
// ============================================================================
struct MyPyObject {
    long ob_refcnt;
    void* ob_type;
    long value;
};

inline void obj_incref(MyPyObject* obj) { ++obj->ob_refcnt; }
inline void obj_decref(MyPyObject* obj) { --obj->ob_refcnt; }

// ============================================================================
// LEARNER implements this in solve.cpp.
//
// Simulates a C-extension loop over `objs` (`n` of them), modeled on a
// pattern like `PyObject_GetAttrString` (returns a NEW/owned reference) plus
// application logic that can fail:
//
//   for each object:
//     obj_incref(obj)              // acquire a new/owned reference
//     if obj->value < 0:
//       // error path — release the reference you just acquired, THEN
//       // return -1 immediately. Do not touch any later object.
//       ...
//     // success path — release the reference, continue to the next object.
//
// Return -1 the first time an object with value < 0 is seen, otherwise 0
// after processing every object. Every object's ob_refcnt must end up
// EXACTLY where it started (no net change) — the reference acquired on the
// error path must not leak.
// ============================================================================
int process_items(MyPyObject* objs, int n);
