#pragma once

// ---------------------------------------------------------------------------
// PROVIDED infrastructure (do not change): a minimal stand-in for a CPython
// PyObject holding a raw double buffer, plus the two reference-counting
// primitives every C-API extension must use correctly. Py_DECREF frees the
// payload the moment refcount drops to (or below) zero, exactly like
// CPython's real object deallocation.
// ---------------------------------------------------------------------------
struct PyObj {
    double* data;
    int     n;
    long    refcount;
};

void Py_INCREF(PyObj* obj);
void Py_DECREF(PyObj* obj);

// ---------------------------------------------------------------------------
// LEARNER IMPLEMENTS.
//
// C-API-style extension function. `buffer` is a BORROWED reference (the
// caller retains its own ownership of it; you must not change its refcount)
// holding `buffer->n` doubles at `buffer->data`.
//
//   1. Compute sum = buffer->data[0] + ... + buffer->data[n-1].
//   2. Allocate and return a brand-new PyObj* that owns a freshly allocated
//      one-element `double[]` holding `sum`, with refcount == 1 (a NEW
//      reference: the caller now owns exactly one reference to it).
//   3. Do NOT touch buffer's refcount. You only borrowed it and never took
//      ownership, so its refcount on return must equal its refcount on
//      entry -- no leak, no accidental free of the caller's buffer.
// ---------------------------------------------------------------------------
PyObj* array_sum_ext(PyObj* buffer);
