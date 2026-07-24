#include "sol.hpp"

// Reference: unique_ptr for sole ownership, shared_ptr otherwise, using the
// real per-type facts main.cpp measured (never a hardcoded 8/16 guess).
OwnershipPlan optimize_ownership(bool is_sole_owned, int transfers, const TypeFacts& facts) {
    if (is_sole_owned) {
        return {"unique_ptr", 0, facts.unique_ptr_bytes, 0, facts.object_bytes};
    }
    return {"shared_ptr", 2 * transfers, facts.shared_ptr_bytes, 16, facts.object_bytes};
}
