#include "sol.hpp"

// TODO: sum buffer->data[0..buffer->n), and return a NEW PyObj* (refcount=1)
// wrapping a freshly allocated one-element double[] holding that sum. Must
// not touch buffer's refcount -- it is only borrowed.
PyObj* array_sum_ext(PyObj* buffer) {
    (void)buffer;
    return nullptr;
}
