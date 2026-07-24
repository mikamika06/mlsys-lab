#include "sol.hpp"

void Py_INCREF(PyObject* obj) {
    obj->ob_refcnt += 1;
}

void Py_DECREF(PyObject* obj) {
    obj->ob_refcnt -= 1;
}

void* process_list_to_tuple(void* list) {
    int n = PyList_Size(list);
    void* tup = PyTuple_New(n);
    for (int i = 0; i < n; i++) {
        PyObject* item = PyList_GetItem(list, i); // borrowed reference
        PyTuple_SetItem(tup, i, item);             // steals it — but we never took a ref to give!
        Py_DECREF(item);                            // BUG: over-decref a borrowed reference
    }
    return tup;
}
