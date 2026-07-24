#include "sol.hpp"

PyTuple* make_tuple(PyObj** items, int n, bool increment_input_refs) {
    if (increment_input_refs) {
        for (int i = 0; i < n; i++) incref(items[i]);
    }
    // else: hand the items to the tuple as-is -- their reference is STOLEN,
    // not duplicated.

    PyTuple* t = new PyTuple();
    t->refcount = 1;
    t->n = n;
    for (int i = 0; i < n; i++) t->items[i] = items[i];
    return t;
}
