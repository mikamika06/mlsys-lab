#include "sol.hpp"

// FIXED: returns the constructed Result by value. No reference outlives the
// temporary, so there is nothing to dangle.
Result get_result(int id, float val) {
    return Result{id, val};
}
