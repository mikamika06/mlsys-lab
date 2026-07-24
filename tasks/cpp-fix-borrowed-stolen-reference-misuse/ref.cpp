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
        PyObject* item = PyList_GetItem(list, i); // borrowed: do NOT decref this
        Py_INCREF(item);               // take our own reference before giving it away
        PyTuple_SetItem(tup, i, item); // steals the reference we just took
    }
    return tup;
}
