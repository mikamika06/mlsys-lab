#include "sol.hpp"

// TODO: if increment_input_refs, incref() each item before handing it to
// the new tuple (net delta +1 per item); otherwise hand items to the tuple
// as-is (their reference is stolen, net delta 0). Either way, return a NEW
// PyTuple with refcount == 1 holding items[0..n) in order. See sol.hpp.
PyTuple* make_tuple(PyObj** items, int n, bool increment_input_refs) {
    (void)items; (void)n; (void)increment_input_refs;
    // your code here
    return nullptr;
}
