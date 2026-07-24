#include "sol.hpp"

// TODO: replay ops[0..n) against one raw storage location, tracking
// whether an object is currently alive, its type/const/trivial properties,
// and whether the original pointer is stale (needs std::launder). See
// sol.hpp for the exact rules.
int classify_ub(const Op* ops, int n) {
    (void)ops; (void)n;
    // your code here
    return 0;
}
