#pragma once

struct PyObject {
    long ob_refcnt;
    void* ob_type;
};

// Manual refcount ops — implement these yourself with direct field access
// (obj->ob_refcnt), mirroring what the real CPython Py_INCREF/Py_DECREF
// macros do. (Deallocation on reaching refcnt == 0 is not modeled here.)
void Py_INCREF(PyObject* obj);
void Py_DECREF(PyObject* obj);

// Mock C-API, provided by the harness (defined in main.cpp) — behaves like
// the real CPython functions:
//   PyList_GetItem  returns a BORROWED reference: you do not own it, so do
//                   not Py_DECREF it unless you first Py_INCREF'd it.
//   PyTuple_SetItem STEALS the reference you pass it: the tuple becomes
//                   the owner of that exact reference, so you must already
//                   own a reference to hand over (never pass it a borrowed
//                   one directly).
PyObject* PyList_GetItem(void* list, int i);
int PyList_Size(void* list);
void* PyTuple_New(int size);
void PyTuple_SetItem(void* tup, int i, PyObject* item);

// Build a new tuple containing every item of `list`, following CPython's
// ownership rules: the list keeps its own reference to each item, and the
// new tuple must get its OWN reference to each item (since PyTuple_SetItem
// steals whatever you pass it).
void* process_list_to_tuple(void* list);
