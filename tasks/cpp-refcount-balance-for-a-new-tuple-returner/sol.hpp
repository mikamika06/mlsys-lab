#pragma once

// A minimal Python-object-like reference-counted value, modeling CPython's
// C-API refcounting discipline (Py_INCREF / Py_DECREF / "stolen"
// references) with a small self-contained C++ type -- no CPython headers
// needed to build or grade this task, but the same rules apply.
struct PyObj {
    int refcount;
    int tag;
};

// Instrumented refcount ops, DEFINED in main.cpp.
void incref(PyObj* obj);
void decref(PyObj* obj);

struct PyTuple {
    int refcount;
    int n;
    PyObj* items[4];
};

// Build a NEW tuple owning references to items[0..n) (n <= 4), modeling
// PyTuple_New(n) + PyTuple_SetItem for each slot:
//
//   PyTuple_SetItem STEALS a reference to the item it's given -- it does
//   NOT incref it itself. So:
//     - if `increment_input_refs` is true: incref() each item BEFORE
//       handing it to the tuple. You created a fresh reference for the
//       tuple to steal, so the CALLER's own reference to that item is
//       undisturbed -- net refcount delta on the item: +1.
//     - if `increment_input_refs` is false: hand the items to the tuple
//       WITHOUT increfing them first. The tuple now owns the caller's own
//       reference -- net refcount delta on the item: 0 (a real "stolen
//       reference": the caller must not decref or otherwise use that
//       reference independently afterward).
//
// Either way, return a NEW PyTuple with refcount == 1 (a fresh reference
// the caller now owns), whose items[0..n) hold exactly the given items, in
// order.
PyTuple* make_tuple(PyObj** items, int n, bool increment_input_refs);
