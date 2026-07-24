#include "sol.hpp"

// TODO: return {"unique_ptr", 0, facts.unique_ptr_bytes, 0, facts.object_bytes}
// when is_sole_owned, otherwise {"shared_ptr", 2*transfers,
// facts.shared_ptr_bytes, 16, facts.object_bytes}.
OwnershipPlan optimize_ownership(bool is_sole_owned, int transfers, const TypeFacts& facts) {
    (void)is_sole_owned;
    (void)transfers;
    (void)facts;
    // your code here
    return {"", 0, 0, 0, 0};
}
