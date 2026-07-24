#include <cstdio>
#include "sol.hpp"

// PROVIDED. Mirrors CPython's Py_INCREF/Py_DECREF: a DECREF that drops the
// refcount to <= 0 deallocates the payload and nulls the data pointer, so a
// double-free / use-after-free of the caller's buffer shows up immediately
// as buffer_alive=0 in the printed output.
void Py_INCREF(PyObj* obj) {
    if (obj) obj->refcount++;
}
void Py_DECREF(PyObj* obj) {
    if (!obj) return;
    obj->refcount--;
    if (obj->refcount <= 0) {
        delete[] obj->data;
        obj->data = nullptr;
    }
}

// FIXED driver. Do not edit. Builds one fixed 8-element buffer object owned
// by the caller (refcount=1), calls the learner's array_sum_ext, then prints
// the sum plus the full refcount/aliveness picture so both numeric
// correctness and C-API refcounting discipline are visible in the output.
int main() {
    double* raw = new double[8]{1.5, -2.25, 3.0, 4.75, -0.5, 2.0, 0.25, -1.0};
    PyObj buffer{raw, 8, 1};

    PyObj* result = array_sum_ext(&buffer);

    if (result == nullptr || result->data == nullptr) {
        printf("NULL_RESULT buffer_refcount=%ld buffer_alive=%d\n",
               buffer.refcount, buffer.data != nullptr ? 1 : 0);
        return 0;
    }

    printf("sum=%.10f result_refcount=%ld buffer_refcount=%ld buffer_alive=%d\n",
           result->data[0], result->refcount, buffer.refcount,
           buffer.data != nullptr ? 1 : 0);

    Py_DECREF(result);
    return 0;
}
