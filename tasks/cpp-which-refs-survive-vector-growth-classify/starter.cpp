#include "sol.hpp"

// TODO: return true iff the reference to the element originally at index refIdx
// is still valid after applying `ops`.
//
// Hints:
//   * A reference/pointer/iterator is invalidated by ANY reallocation.
//     - push_back reallocates iff size == capacity.
//     - reserve(k) reallocates iff k > capacity.
//     - insert reallocates iff the new size would exceed capacity.
//   * Without reallocation, an insert/erase at position p invalidates references
//     to elements at index >= p (they shift); references strictly before p stay.
//     - pop_back erases the last element; clear erases everything.
//
// The stub below always claims the reference dies, so it fails the gate.
bool ref_survives(int n0, int cap0, int refIdx, const std::vector<Op>& ops) {
    (void)n0; (void)cap0; (void)refIdx; (void)ops;
    return false;  // your code here
}
